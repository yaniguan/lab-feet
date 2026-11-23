"""Idea #20: Quantized Activation-Energy OOD & Clipping Monitor for int8/4-bit Edge Boards.

Inspired by microcontroller edge safety monitors and low-bit quantization saturation
drift threads. Implements an integer-only `int4`/`int8` benchtop sensor classifier
paired with an SRAM-tiny (`220 B < 512 B`) integer `L1` feature-Mahalanobis + `int4`
saturation-clip + activation-energy margin monitor that triggers deterministic
`SAFE_HOLD` abstention when benchtop vibration/optical sensors drift out-of-distribution
instead of emitting overconfident clipped predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SAFE_HOLD_LABEL: str = "SAFE_HOLD"
SAFE_HOLD_CLASS_ID: int = -1
REGIME_NAMES: tuple[str, ...] = (
    "idle_bench",
    "orbital_shaker_nominal",
    "centrifuge_balanced",
    "peristaltic_pump_steady",
    "vortex_mixer_pulse",
    "thermocycler_ramp",
)


@dataclass(frozen=True)
class EdgePrediction:
    """Single-sample inference output from the integer `int4`/`int8` edge runtime."""

    class_id: int
    label: str
    abstained: bool
    l1_mahalanobis_score: int
    int4_clip_count: int
    int8_input_clip_count: int
    energy_margin_score: int


def pack_signed_int4(values_int8: np.ndarray) -> np.ndarray:
    """Pack signed two's-complement 4-bit integers `[-8, +7]` into `uint8` (2 nibbles/byte)."""
    flat = np.asarray(values_int8, dtype=np.int8).ravel()
    if np.any((flat < -8) | (flat > 7)):
        raise ValueError("pack_signed_int4 expects integer values in [-8, +7]")
    if flat.size % 2 != 0:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    lo = (flat[0::2].astype(np.int16) & 0x0F).astype(np.uint8)
    hi = ((flat[1::2].astype(np.int16) & 0x0F) << 4).astype(np.uint8)
    return (lo | hi).astype(np.uint8)


def unpack_signed_int4(packed_u8: np.ndarray, total_elements: int) -> np.ndarray:
    """Unpack `uint8` nibbles back into signed two's-complement `int8` in `[-8, +7]`."""
    raw = np.asarray(packed_u8, dtype=np.uint8).ravel()
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
    out = np.empty(raw.size * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:total_elements]


def _quantize_symmetric_int4_per_channel(w_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric `int4` (`[-7, +7]`) weight quantization."""
    max_abs = np.max(np.abs(w_fp32), axis=0)
    scales = np.maximum(max_abs / 7.0, 1e-6).astype(np.float32)
    w_q = np.clip(np.rint(w_fp32 / scales[None, :]), -8, 7).astype(np.int8)
    return w_q, scales


class DriftAbstainingInt4Classifier:
    """Integer-only `int8`-input / `int4`-weight+activation MLP with `< 512 B` OOD monitor."""

    def __init__(self, n_features: int = 24, n_hidden: int = 32, n_classes: int = 6) -> None:
        self.n_features = int(n_features)
        self.n_hidden = int(n_hidden)
        self.n_classes = int(n_classes)

        # Quantized student parameters
        self.input_scale: float = 1.0
        self.w1_packed: np.ndarray = np.zeros((n_features * n_hidden) // 2, dtype=np.uint8)
        self.b1_i32: np.ndarray = np.zeros(n_hidden, dtype=np.int32)
        self.h1_requant_mult_i16: np.ndarray = np.ones(n_hidden, dtype=np.int16)
        self.h1_requant_shift: int = 10

        self.w2_packed: np.ndarray = np.zeros((n_hidden * n_classes) // 2, dtype=np.uint8)
        self.b2_i32: np.ndarray = np.zeros(n_classes, dtype=np.int32)
        self.logit_mult_i16: np.ndarray = np.ones(n_classes, dtype=np.int16)

        # Compact `< 512 B` SRAM OOD + saturation monitor state (exact integer tables)
        self.class_centroids_i8: np.ndarray = np.zeros((n_classes, n_features), dtype=np.int8)
        self.inv_mad_u8: np.ndarray = np.ones(n_features, dtype=np.uint8)
        self.class_l1_thresh_u16: np.ndarray = np.zeros(n_classes, dtype=np.uint16)
        self.scalar_thresholds_u16: np.ndarray = np.zeros(4, dtype=np.uint16)
        # [max_int4_clips, max_int8_input_clips, min_energy_margin, min_peak_logit]

    @property
    def packed_weight_bytes(self) -> int:
        """Total SRAM bytes occupied by the packed `int4` student weights + integer biases/scales."""
        return int(
            self.w1_packed.nbytes
            + self.b1_i32.nbytes
            + self.h1_requant_mult_i16.nbytes
            + self.w2_packed.nbytes
            + self.b2_i32.nbytes
            + self.logit_mult_i16.nbytes
        )

    @property
    def extra_sram_bytes(self) -> int:
        """Exact SRAM overhead in bytes of the OOD + clipping monitor (tables + 8x int32 scratch)."""
        scratch_registers_bytes = 8 * 4  # 32 bytes for accumulator registers on MCU
        return int(
            self.class_centroids_i8.nbytes
            + self.inv_mad_u8.nbytes
            + self.class_l1_thresh_u16.nbytes
            + self.scalar_thresholds_u16.nbytes
            + scratch_registers_bytes
        )

    def fit_and_calibrate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_calib: np.ndarray,
        y_calib: np.ndarray,
        seed: int = 0,
    ) -> None:
        """Train FP32 1-hidden-layer student, quantize to `int4`, and calibrate integer OOD gates."""
        rng = np.random.default_rng(seed)
        n_samples, d = x_train.shape
        assert d == self.n_features

        # Calibrate symmetric int8 input ADC scale on training split (headroom for 2.2x nominal range)
        p995 = float(np.percentile(np.abs(x_train), 99.5))
        self.input_scale = max(p995 * 1.35 / 127.0, 1e-5)

        # Mini-batch Adam/SGD training of compact FP32 student `24 -> 32 (clipped ReLU) -> 6`
        w1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, self.n_hidden)).astype(np.float64)
        b1 = np.zeros(self.n_hidden, dtype=np.float64)
        w2 = rng.normal(0.0, np.sqrt(2.0 / self.n_hidden), size=(self.n_hidden, self.n_classes)).astype(np.float64)
        b2 = np.zeros(self.n_classes, dtype=np.float64)

        lr = 0.045
        for epoch in range(220):
            # Forward pass with bounded ReLU6-style activation matching int4 [0, 7] dynamic range
            z1 = x_train @ w1 + b1
            h1 = np.clip(z1, 0.0, 3.5)
            logits = h1 @ w2 + b2
            logits_shift = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(logits_shift)
            probs /= np.sum(probs, axis=1, keepdims=True)

            dlogits = probs.copy()
            dlogits[np.arange(n_samples), y_train] -= 1.0
            dlogits /= n_samples

            dw2 = h1.T @ dlogits + 1e-4 * w2
            db2 = np.sum(dlogits, axis=0)
            dh1 = dlogits @ w2.T
            dz1 = dh1 * ((z1 > 0.0) & (z1 < 3.5))
            dw1 = x_train.T @ dz1 + 1e-4 * w1
            db1 = np.sum(dz1, axis=0)

            decay = 1.0 / (1.0 + 0.008 * epoch)
            w1 -= lr * decay * dw1
            b1 -= lr * decay * db1
            w2 -= lr * decay * dw2
            db2_step = lr * decay * db2
            b2 -= db2_step

        # Quantize Layer 1: x_q (int8) * w1_q (int4) -> requantize to int4 [0, 7]
        w1_q, w1_scales = _quantize_symmetric_int4_per_channel(w1.astype(np.float32))
        self.w1_packed = pack_signed_int4(w1_q)
        act_step = 3.5 / 7.0  # maps [0, 3.5] -> [0, 7]
        eff_scale_1 = (self.input_scale * w1_scales) / act_step
        self.b1_i32 = np.rint(b1 / (self.input_scale * w1_scales)).astype(np.int32)
        self.h1_requant_mult_i16 = np.clip(
            np.rint(eff_scale_1 * (1 << self.h1_requant_shift)), 1, 32767
        ).astype(np.int16)

        # Quantize Layer 2: h1_q (int4) * w2_q (int4) -> integer scaled logits
        w2_q, w2_scales = _quantize_symmetric_int4_per_channel(w2.astype(np.float32))
        self.w2_packed = pack_signed_int4(w2_q)
        self.b2_i32 = np.rint(b2 / (act_step * w2_scales)).astype(np.int32)
        max_w2_scale = float(np.max(w2_scales))
        self.logit_mult_i16 = np.clip(
            np.rint((w2_scales / max_w2_scale) * 256.0), 1, 32767
        ).astype(np.int16)

        # Calibrate integer L1-Mahalanobis + int4 saturation monitor strictly on `x_calib`
        x_cal_q, _ = self._quantize_input_adc(x_calib)
        for c in range(self.n_classes):
            mask = y_calib == c
            if np.any(mask):
                med = np.median(x_cal_q[mask], axis=0)
                self.class_centroids_i8[c] = np.clip(np.rint(med), -128, 127).astype(np.int8)

        # Feature-wise inverse MAD scaled to uint8 [4..64]
        residuals = np.abs(
            x_cal_q.astype(np.int32) - self.class_centroids_i8[y_calib].astype(np.int32)
        )
        mad = np.maximum(np.mean(residuals, axis=0), 2.5)
        inv_mad = np.clip(np.rint(32.0 / mad), 4, 64).astype(np.uint8)
        self.inv_mad_u8 = inv_mad

        # Collect calibration-split integer monitor statistics
        cal_l1_per_class: dict[int, list[int]] = {c: [] for c in range(self.n_classes)}
        cal_int4_clips: list[int] = []
        cal_in_clips: list[int] = []
        cal_margins: list[int] = []
        cal_peaks: list[int] = []

        for idx in range(x_calib.shape[0]):
            pred_c, l1_d, h_clips, in_clips, margin, peak_logit = self._forward_integer_core(x_calib[idx])
            cal_l1_per_class[int(y_calib[idx])].append(
                self._compute_class_l1(x_cal_q[idx], int(y_calib[idx]))
            )
            cal_int4_clips.append(h_clips)
            cal_in_clips.append(in_clips)
            if pred_c == int(y_calib[idx]):
                cal_margins.append(margin)
                cal_peaks.append(peak_logit)

        for c in range(self.n_classes):
            vals = cal_l1_per_class[c]
            q99 = float(np.percentile(vals, 99.0)) if vals else 400.0
            self.class_l1_thresh_u16[c] = np.uint16(min(65535, int(round(q99 * 1.28 + 35.0))))

        max_h_clips = int(np.percentile(cal_int4_clips, 99.5)) + 3
        max_in_clips = int(np.percentile(cal_in_clips, 99.5)) + 2
        min_margin = max(8, int(np.percentile(cal_margins, 1.0) * 0.55)) if cal_margins else 10
        min_peak = max(12, int(np.percentile(cal_peaks, 1.0) * 0.45)) if cal_peaks else 15
        self.scalar_thresholds_u16[:] = np.array(
            [max_h_clips, max_in_clips, min_margin, min_peak], dtype=np.uint16
        )
        assert self.extra_sram_bytes < 512

    def _quantize_input_adc(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simulate `int8` sensor ADC quantization and count +/-127 saturation rail hits."""
        raw_q = np.rint(np.asarray(x, dtype=np.float64) / self.input_scale).astype(np.int32)
        in_clips = np.sum((raw_q <= -127) | (raw_q >= 127), axis=-1)
        x_q = np.clip(raw_q, -128, 127).astype(np.int8)
        return x_q, in_clips

    def _compute_class_l1(self, x_q_1d: np.ndarray, class_idx: int) -> int:
        diff = np.abs(
            x_q_1d.astype(np.int32) - self.class_centroids_i8[class_idx].astype(np.int32)
        )
        return int(np.sum(diff * self.inv_mad_u8.astype(np.int32)))

    def _forward_integer_core(self, x_1d: np.ndarray) -> tuple[int, int, int, int, int, int]:
        """Pure integer `int8`/`int4`/`int32` forward pass + OOD/saturation telemetry."""
        x_q, in_clips_arr = self._quantize_input_adc(x_1d)
        in_clips = int(in_clips_arr)

        w1_q = unpack_signed_int4(self.w1_packed, self.n_features * self.n_hidden).reshape(
            self.n_features, self.n_hidden
        )
        w2_q = unpack_signed_int4(self.w2_packed, self.n_hidden * self.n_classes).reshape(
            self.n_hidden, self.n_classes
        )

        # Layer 1 integer MAC (`int32` accumulator) + fixed-point shift to `int4` `[0, 7]`
        acc1_i32 = x_q.astype(np.int32) @ w1_q.astype(np.int32) + self.b1_i32
        scaled1_i32 = (acc1_i32 * self.h1_requant_mult_i16.astype(np.int32)) >> self.h1_requant_shift
        # Count hidden units exceeding upper int4 rail (+7) before clipping
        int4_clips = int(np.sum(scaled1_i32 > 7))
        h1_i4 = np.clip(scaled1_i32, 0, 7).astype(np.int8)

        # Layer 2 integer MAC (`int32` accumulator)
        acc2_i32 = h1_i4.astype(np.int32) @ w2_q.astype(np.int32) + self.b2_i32
        logits_i32 = (acc2_i32 * self.logit_mult_i16.astype(np.int32)) >> 4

        pred_class = int(np.argmax(logits_i32))
        sorted_logits = np.sort(logits_i32)
        peak_logit = int(sorted_logits[-1])
        runner_up = int(sorted_logits[-2])
        energy_margin = max(0, peak_logit - runner_up)

        l1_dist = self._compute_class_l1(x_q, pred_class)
        return pred_class, l1_dist, int4_clips, in_clips, energy_margin, peak_logit

    def predict_naive_int4_argmax(self, x_1d: np.ndarray) -> EdgePrediction:
        """Baseline `int4` classifier emitting `argmax` blindly without drift/clip abstention."""
        pred_c, l1_d, h_clips, in_clips, margin, _ = self._forward_integer_core(x_1d)
        return EdgePrediction(
            class_id=pred_c,
            label=REGIME_NAMES[pred_c],
            abstained=False,
            l1_mahalanobis_score=l1_d,
            int4_clip_count=h_clips,
            int8_input_clip_count=in_clips,
            energy_margin_score=margin,
        )

    def predict_drift_abstaining_int4(self, x_1d: np.ndarray) -> EdgePrediction:
        """Guarded `int4` classifier triggering deterministic `SAFE_HOLD` on OOD or rail clipping."""
        pred_c, l1_d, h_clips, in_clips, margin, peak_logit = self._forward_integer_core(x_1d)
        l1_thresh = int(self.class_l1_thresh_u16[pred_c])
        max_h_clips = int(self.scalar_thresholds_u16[0])
        max_in_clips = int(self.scalar_thresholds_u16[1])
        min_margin = int(self.scalar_thresholds_u16[2])
        min_peak = int(self.scalar_thresholds_u16[3])

        abstain = bool(
            (l1_d > l1_thresh)
            or (h_clips > max_h_clips)
            or (in_clips > max_in_clips)
            or (margin < min_margin)
            or (peak_logit < min_peak)
        )
        if abstain:
            return EdgePrediction(
                class_id=SAFE_HOLD_CLASS_ID,
                label=SAFE_HOLD_LABEL,
                abstained=True,
                l1_mahalanobis_score=l1_d,
                int4_clip_count=h_clips,
                int8_input_clip_count=in_clips,
                energy_margin_score=margin,
            )
        return EdgePrediction(
            class_id=pred_c,
            label=REGIME_NAMES[pred_c],
            abstained=False,
            l1_mahalanobis_score=l1_d,
            int4_clip_count=h_clips,
            int8_input_clip_count=in_clips,
            energy_margin_score=margin,
        )


def _generate_benchtop_sensor_dataset(
    rng: np.random.Generator,
    n_per_class: int,
    n_features: int = 24,
    n_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate distinct 6-class benchtop vibration + optical turbidity sensor vectors."""
    prototypes = np.zeros((n_classes, n_features), dtype=np.float64)
    for c in range(n_classes):
        # Structured harmonic + optical band signature per instrument regime
        active_idx = [(4 * c + k) % n_features for k in range(5)]
        prototypes[c, active_idx] = np.array([2.2, 1.8, 1.5, -1.4, 1.2], dtype=np.float64)
        prototypes[c] += 0.25 * np.cos(np.arange(n_features, dtype=np.float64) * (c + 1) * 0.5)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for c in range(n_classes):
        samples = prototypes[c][None, :] + rng.normal(0.0, 0.34, size=(n_per_class, n_features))
        xs.append(samples)
        ys.extend([c] * n_per_class)

    perm = rng.permutation(n_classes * n_per_class)
    return np.vstack(xs)[perm], np.array(ys, dtype=np.int64)[perm], prototypes


def _generate_ood_and_saturated_inputs(
    rng: np.random.Generator,
    n_ood: int,
    n_features: int = 24,
) -> np.ndarray:
    """Generate 4 realistic benchtop edge OOD / saturation drift modes (40 each = 160)."""
    assert n_ood % 4 == 0
    k = n_ood // 4

    # Mode 1: Optical condensation / ambient lid-open saturation glare (rails +7..+12)
    glare = rng.uniform(5.5, 11.0, size=(k, n_features))

    # Mode 2: Loose accelerometer mount high-amplitude bipolar shock spikes
    shock = rng.choice([-8.5, 8.5], size=(k, n_features)) + rng.normal(0.0, 1.0, size=(k, n_features))

    # Mode 3: Unseen out-of-band EMI comb pattern orthogonal to nominal manifolds (no ADC rail clip!)
    emi = np.tile(np.array([2.3, -2.3] * (n_features // 2), dtype=np.float64), (k, 1))
    emi += rng.normal(0.0, 0.25, size=(k, n_features))

    # Mode 4: Disconnected / grounded sensor cable (near-zero dead flatline + DC offset on ch 0..3)
    dead = rng.normal(0.0, 0.03, size=(k, n_features))
    dead[:, :4] = -3.6

    return np.vstack([glare, shock, emi, dead])


def run_benchmark(seed: int = 20261123) -> dict[str, Any]:
    """Run 400-sample edge benchmark (`240` in-domain test + `160` OOD/saturated drift inputs).

    Compares `naive_int4_argmax` vs `drift_abstaining_int4`, reporting:
      - `silent_ood_error_rate_pct` (fraction of OOD inputs that emit an unhalted class prediction)
      - `in_domain_accuracy_pct`
      - `extra_sram_bytes` (`< 512 B`).
    """
    rng = np.random.default_rng(seed)
    n_features = 24
    n_hidden = 32
    n_classes = 6

    x_all, y_all, _ = _generate_benchtop_sensor_dataset(
        rng, n_per_class=160, n_features=n_features, n_classes=n_classes
    )
    # Stratified/shuffled split: 540 train, 180 calibration, 240 held-out in-domain test
    x_train, y_train = x_all[:540], y_all[:540]
    x_calib, y_calib = x_all[540:720], y_all[540:720]
    x_id_test, y_id_test = x_all[720:960], y_all[720:960]
    assert x_id_test.shape[0] == 240

    x_ood = _generate_ood_and_saturated_inputs(rng, n_ood=160, n_features=n_features)
    total_eval = int(x_id_test.shape[0] + x_ood.shape[0])
    assert total_eval == 400

    clf = DriftAbstainingInt4Classifier(
        n_features=n_features, n_hidden=n_hidden, n_classes=n_classes
    )
    clf.fit_and_calibrate(x_train, y_train, x_calib, y_calib, seed=seed)

    naive_id_correct = 0
    guarded_id_correct = 0
    guarded_id_abstained = 0

    for i in range(x_id_test.shape[0]):
        truth = int(y_id_test[i])
        p_naive = clf.predict_naive_int4_argmax(x_id_test[i])
        p_guard = clf.predict_drift_abstaining_int4(x_id_test[i])
        if p_naive.class_id == truth:
            naive_id_correct += 1
        if p_guard.abstained:
            guarded_id_abstained += 1
        elif p_guard.class_id == truth:
            guarded_id_correct += 1

    naive_silent_ood_errors = 0
    guarded_silent_ood_errors = 0
    guarded_ood_safe_holds = 0

    for i in range(x_ood.shape[0]):
        p_naive = clf.predict_naive_int4_argmax(x_ood[i])
        p_guard = clf.predict_drift_abstaining_int4(x_ood[i])
        if not p_naive.abstained:
            naive_silent_ood_errors += 1
        if p_guard.abstained and p_guard.label == SAFE_HOLD_LABEL:
            guarded_ood_safe_holds += 1
        else:
            guarded_silent_ood_errors += 1

    n_id = x_id_test.shape[0]
    n_ood = x_ood.shape[0]

    return {
        "idea_id": "idea_20_drift_abstaining_int4_edge_monitor",
        "n_total_inputs": total_eval,
        "n_in_domain_inputs": n_id,
        "n_ood_saturated_inputs": n_ood,
        "packed_student_weight_bytes": clf.packed_weight_bytes,
        "extra_sram_bytes": clf.extra_sram_bytes,
        "naive_int4_argmax": {
            "in_domain_accuracy_pct": round(100.0 * naive_id_correct / n_id, 4),
            "silent_ood_error_rate_pct": round(100.0 * naive_silent_ood_errors / n_ood, 4),
            "ood_safe_hold_rate_pct": 0.0,
            "extra_sram_bytes": 0,
        },
        "drift_abstaining_int4": {
            "in_domain_accuracy_pct": round(100.0 * guarded_id_correct / n_id, 4),
            "in_domain_abstain_rate_pct": round(100.0 * guarded_id_abstained / n_id, 4),
            "silent_ood_error_rate_pct": round(100.0 * guarded_silent_ood_errors / n_ood, 4),
            "ood_safe_hold_rate_pct": round(100.0 * guarded_ood_safe_holds / n_ood, 4),
            "extra_sram_bytes": clf.extra_sram_bytes,
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2))
