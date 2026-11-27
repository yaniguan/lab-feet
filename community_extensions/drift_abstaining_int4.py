"""Idea #20: Quantized Activation-Energy OOD & Clipping Monitor for int8/4-bit Edge Boards.

Inspired by microcontroller edge safety monitors and low-bit quantization saturation
drift threads. Implements an integer-only `int4`/`int8` benchtop sensor classifier
paired with an integer `L1` feature-Mahalanobis + `int4` saturation-clip + activation-energy
margin monitor (`220 B` @ `24` features, `276 B` @ `32` features, `948 B` @ `128` features for
`6` classes; `< 512 B` MCU budget up to `61` features @ `6` classes) that triggers deterministic
`SAFE_HOLD` abstention when benchtop vibration/optical sensors drift out-of-distribution
instead of emitting overconfident clipped predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SAFE_HOLD_LABEL: str = "SAFE_HOLD"
SAFE_HOLD_CLASS_ID: int = -1
INT32_MIN: int = -(1 << 31)
INT32_MAX: int = (1 << 31) - 1

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
    peak_logit: int = 0


def pack_signed_int4(values_int8: np.ndarray) -> np.ndarray:
    """Pack signed two's-complement 4-bit integers `[-8, +7]` into `uint8` (2 nibbles/byte).

    When `values_int8.size` is odd, pads the trailing high nibble (`bits 4..7`) with canonical `0`.
    """
    arr = np.asarray(values_int8)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"pack_signed_int4 expects an integer array, got dtype={arr.dtype}")
    flat = arr.ravel()
    if flat.size == 0:
        return np.empty(0, dtype=np.uint8)
    if np.any((flat < -8) | (flat > 7)):
        raise ValueError("pack_signed_int4 expects integer values in [-8, +7]")
    flat_i8 = flat.astype(np.int8, copy=False)
    if flat_i8.size % 2 != 0:
        flat_i8 = np.concatenate([flat_i8, np.zeros(1, dtype=np.int8)])
    lo = (flat_i8[0::2].astype(np.int16) & 0x0F).astype(np.uint8)
    hi = ((flat_i8[1::2].astype(np.int16) & 0x0F) << 4).astype(np.uint8)
    return (lo | hi).astype(np.uint8)


def unpack_signed_int4(
    packed_u8: np.ndarray,
    total_elements: int,
    *,
    strict_padding: bool = True,
) -> np.ndarray:
    """Unpack `uint8` nibbles back into signed two's-complement `int8` in `[-8, +7]`.

    Enforces exact packed byte count `raw.size == (total_elements + 1) // 2` and canonical zero
    high-nibble padding on odd `total_elements`.
    """
    if isinstance(total_elements, bool) or not isinstance(total_elements, (int, np.integer)) or total_elements < 0:
        raise ValueError(f"total_elements must be a non-negative integer, got {total_elements!r}")
    n_elems = int(total_elements)
    arr = np.asarray(packed_u8)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"unpack_signed_int4 expects an integer/byte buffer, got dtype={arr.dtype}")
    raw = arr.astype(np.uint8, copy=False).ravel()
    expected_bytes = (n_elems + 1) // 2 if n_elems > 0 else 0
    if raw.size != expected_bytes:
        raise ValueError(
            f"unpack_signed_int4 expected {expected_bytes} bytes for total_elements={n_elems}, got {raw.size}."
        )
    if n_elems == 0:
        return np.empty(0, dtype=np.int8)
    if strict_padding and (n_elems % 2 == 1):
        pad_hi = (int(raw[-1]) >> 4) & 0x0F
        if pad_hi != 0:
            raise ValueError(
                f"Non-zero high padding nibble 0x{pad_hi:X} in odd-length packed int4 buffer."
            )

    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
    out = np.empty(raw.size * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:n_elems]


def requantize_saturate_int64(
    acc: np.ndarray | int,
    mult: np.ndarray | int,
    shift: int,
    qmin: int,
    qmax: int,
    *,
    rounding: Literal["half_up", "ties_to_even"] = "half_up",
) -> tuple[np.ndarray, np.ndarray]:
    """Widening `int64` fixed-point multiply-shift + saturation matching `LabFeetLean.requantSaturate`.

    Returns `(saturated_i64, unclipped_shifted_i64)` so saturation rail clip counts can be measured
    prior to clamping without 32-bit signed integer wraparound.
    """
    if shift < 0:
        raise ValueError(f"shift must be >= 0, got {shift}")
    if qmin > qmax:
        raise ValueError(f"qmin={qmin} must be <= qmax={qmax}")

    acc_i64 = np.asarray(acc, dtype=np.int64)
    mult_i64 = np.asarray(mult, dtype=np.int64)
    prod = acc_i64 * mult_i64

    if shift == 0:
        shifted = prod
    elif rounding == "half_up":
        offset = np.int64(1 << (shift - 1))
        shifted = (prod + offset) >> shift
    elif rounding == "ties_to_even":
        half = np.int64(1 << (shift - 1))
        mask = np.int64((1 << shift) - 1)
        q_floor = prod >> shift
        rem = prod & mask
        gt = rem > half
        tie = rem == half
        odd = (q_floor & np.int64(1)) == np.int64(1)
        shifted = q_floor + np.where(gt | (tie & odd), np.int64(1), np.int64(0))
    else:
        raise ValueError(f"Unsupported rounding mode {rounding!r}")

    saturated = np.clip(shifted, qmin, qmax).astype(np.int64)
    return saturated, shifted


def decide_drift_abstaining_int4(
    argmax_class: int,
    l1_score: int,
    l1_thresh: int,
    int4_clips: int,
    max_int4_clips: int,
    int8_input_clips: int,
    max_input_clips: int,
    energy_margin: int,
    min_energy_margin: int,
    peak_logit: int,
    min_peak_logit: int,
) -> int:
    """Bit-exact 5-gate decision function mirroring `LabFeetLean.decideDriftAbstainingInt4`.

    Returns `SAFE_HOLD_CLASS_ID` (`-1`) if any of the 5 OOD/saturation/margin/peak gates trigger,
    and `argmax_class` otherwise.
    """
    if (
        (l1_score > l1_thresh)
        or (int4_clips > max_int4_clips)
        or (int8_input_clips > max_input_clips)
        or (energy_margin < min_energy_margin)
        or (peak_logit < min_peak_logit)
    ):
        return SAFE_HOLD_CLASS_ID
    return int(argmax_class)


def _quantize_symmetric_int4_per_channel(w_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric `int4` (`[-7, +7]`) weight quantization."""
    max_abs = np.max(np.abs(w_fp32), axis=0)
    scales = np.maximum(max_abs / 7.0, 1e-6).astype(np.float32)
    w_q = np.clip(np.rint(w_fp32 / scales[None, :]), -8, 7).astype(np.int8)
    return w_q, scales


class DriftAbstainingInt4Classifier:
    """Integer-only `int8`-input / `int4`-weight+activation MLP with integer OOD monitor.

    Monitor SRAM table footprint (`extra_sram_bytes`) for `C` classes and `d` features is
    `(C + 1) * d + 2 * C + 40` bytes (`220 B` @ `d=24, C=6`; `276 B` @ `d=32, C=6`;
    `948 B` @ `d=128, C=6`).
    """

    def __init__(self, n_features: int = 24, n_hidden: int = 32, n_classes: int = 6) -> None:
        if n_features < 1 or n_hidden < 1 or n_classes < 2:
            raise ValueError(
                f"Require n_features >= 1, n_hidden >= 1, n_classes >= 2; got {(n_features, n_hidden, n_classes)}."
            )
        if int(n_features) * 128 * 8 > INT32_MAX or int(n_hidden) * 7 * 8 > INT32_MAX:
            raise ValueError(
                f"Layer fan-in {(n_features, n_hidden)} exceeds worst-case signed int32 MAC accumulator bound."
            )
        self.n_features = int(n_features)
        self.n_hidden = int(n_hidden)
        self.n_classes = int(n_classes)

        # Quantized student parameters (ceiling division `(M * N + 1) // 2` preserves odd weight counts)
        self.input_scale: float = 1.0
        self.w1_packed: np.ndarray = np.zeros((self.n_features * self.n_hidden + 1) // 2, dtype=np.uint8)
        self.b1_i32: np.ndarray = np.zeros(self.n_hidden, dtype=np.int32)
        self.h1_requant_mult_i16: np.ndarray = np.ones(self.n_hidden, dtype=np.int16)
        self.h1_requant_shift: int = 10

        self.w2_packed: np.ndarray = np.zeros((self.n_hidden * self.n_classes + 1) // 2, dtype=np.uint8)
        self.b2_i32: np.ndarray = np.zeros(self.n_classes, dtype=np.int32)
        self.logit_mult_i16: np.ndarray = np.ones(self.n_classes, dtype=np.int16)
        self.logit_scale: float = 1.0 / 16.0
        self.react_clip_h1: int = 6

        # Integer SRAM OOD + saturation monitor state (`(C + 1)*d + 2*C + 40` B total with 32 B scratch)
        self.class_centroids_i8: np.ndarray = np.zeros((self.n_classes, self.n_features), dtype=np.int8)
        self.inv_mad_u8: np.ndarray = np.ones(self.n_features, dtype=np.uint8)
        self.class_l1_thresh_u16: np.ndarray = np.zeros(self.n_classes, dtype=np.uint16)
        self.scalar_thresholds_u16: np.ndarray = np.zeros(4, dtype=np.uint16)
        # [max_int4_clips, max_int8_input_clips, min_energy_margin, min_peak_logit]

    def _class_label(self, class_id: int) -> str:
        if 0 <= class_id < len(REGIME_NAMES):
            return REGIME_NAMES[class_id]
        return f"regime_{class_id}"

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

        # Mini-batch Adam/SGD training of compact FP32 student `d -> n_hidden (clipped ReLU) -> n_classes`
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

        self.load_fp32_weights_and_calibrate(w1, b1, w2, b2, x_calib, y_calib, input_scale=self.input_scale)

    def load_fp32_weights_and_calibrate(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        x_calib: np.ndarray,
        y_calib: np.ndarray,
        *,
        input_scale: float | None = None,
        act_cap: float = 3.5,
    ) -> None:
        """Quantize pre-trained `(w1, b1, w2, b2)` layers with saturating `int32` biases and calibrate OOD gates."""
        w1_f64 = np.asarray(w1, dtype=np.float64)
        b1_f64 = np.asarray(b1, dtype=np.float64)
        w2_f64 = np.asarray(w2, dtype=np.float64)
        b2_f64 = np.asarray(b2, dtype=np.float64)
        if w1_f64.shape != (self.n_features, self.n_hidden) or w2_f64.shape != (self.n_hidden, self.n_classes):
            raise ValueError(
                f"Weight shape mismatch: got {w1_f64.shape}, {w2_f64.shape} for "
                f"({self.n_features}, {self.n_hidden}, {self.n_classes})."
            )

        if input_scale is None:
            p995 = float(np.percentile(np.abs(x_calib), 99.5)) if x_calib.size > 0 else 1.0
            self.input_scale = max(p995 * 1.35 / 127.0, 1e-5)
        else:
            self.input_scale = max(float(input_scale), 1e-12)

        # Quantize Layer 1: x_q (int8) * w1_q (int4) -> requantize to int4 [0, 7]
        w1_q, w1_scales = _quantize_symmetric_int4_per_channel(w1_f64.astype(np.float32))
        self.w1_packed = pack_signed_int4(w1_q)
        act_step = max(float(act_cap) / 7.0, 1e-6)
        denom_1 = np.maximum(self.input_scale * w1_scales.astype(np.float64), 1e-15)
        eff_scale_1 = denom_1 / act_step
        self.b1_i32 = np.clip(
            np.rint(b1_f64 / denom_1), INT32_MIN + 1, INT32_MAX
        ).astype(np.int32)
        self.h1_requant_mult_i16 = np.clip(
            np.rint(eff_scale_1 * (1 << self.h1_requant_shift)), 1, 32767
        ).astype(np.int16)

        # Quantize Layer 2: h1_q (int4) * w2_q (int4) -> integer scaled logits
        w2_q, w2_scales = _quantize_symmetric_int4_per_channel(w2_f64.astype(np.float32))
        self.w2_packed = pack_signed_int4(w2_q)
        denom_2 = np.maximum(act_step * w2_scales.astype(np.float64), 1e-15)
        self.b2_i32 = np.clip(
            np.rint(b2_f64 / denom_2), INT32_MIN + 1, INT32_MAX
        ).astype(np.int32)
        max_w2_scale = max(float(np.max(w2_scales)), 1e-12)
        self.logit_mult_i16 = np.clip(
            np.rint((w2_scales.astype(np.float64) / max_w2_scale) * 256.0), 1, 32767
        ).astype(np.int16)
        # Physical float32 logit step per unit of `logits_i32` (`acc2 * (w2_s/max_w2_s * 256) >> 4`)
        self.logit_scale = float(act_step * max_w2_scale / 16.0)

        # Calibrate ReAct hidden-activation clipping ceiling (90th percentile of positive h1 on X_cal)
        if x_calib.shape[0] > 0:
            x_cal_q_batch, _ = self._quantize_input_adc(x_calib)
            w1_q_cal = unpack_signed_int4(self.w1_packed, self.n_features * self.n_hidden).reshape(
                self.n_features, self.n_hidden
            )
            dot1_cal = np.matmul(x_cal_q_batch.astype(np.int32), w1_q_cal.astype(np.int32))
            acc1_cal = np.clip(
                dot1_cal.astype(np.int64) + self.b1_i32.astype(np.int64)[None, :], INT32_MIN, INT32_MAX
            ).astype(np.int32)
            h1_cal_sat, _ = requantize_saturate_int64(
                acc1_cal,
                self.h1_requant_mult_i16[None, :],
                self.h1_requant_shift,
                0,
                7,
                rounding="half_up",
            )
            pos_h1 = h1_cal_sat[h1_cal_sat > 0]
            self.react_clip_h1 = int(np.clip(np.rint(np.percentile(pos_h1, 90.0)), 1, 7)) if pos_h1.size > 0 else 5
        else:
            self.react_clip_h1 = 5

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
            pred_c, _, h_clips, in_clips, margin, peak_logit = self._forward_integer_core(x_calib[idx])
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

        max_h_clips = min(65535, max(0, int(np.percentile(cal_int4_clips, 99.5)) + 3)) if cal_int4_clips else 8
        max_in_clips = min(65535, max(0, int(np.percentile(cal_in_clips, 99.5)) + 2)) if cal_in_clips else 2
        min_margin = max(8, min(65535, int(np.percentile(cal_margins, 1.0) * 0.55))) if cal_margins else 10
        min_peak = max(12, min(65535, int(np.percentile(cal_peaks, 1.0) * 0.45))) if cal_peaks else 15
        self.scalar_thresholds_u16[:] = np.array(
            [max_h_clips, max_in_clips, min_margin, min_peak], dtype=np.uint16
        )

    def _quantize_input_adc(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simulate `int8` sensor ADC quantization and count +/-127 saturation rail hits."""
        scaled = np.clip(
            np.asarray(x, dtype=np.float64) / max(self.input_scale, 1e-12),
            float(INT32_MIN + 1),
            float(INT32_MAX),
        )
        raw_q = np.rint(scaled).astype(np.int32)
        in_clips = np.sum((raw_q <= -127) | (raw_q >= 127), axis=-1)
        x_q = np.clip(raw_q, -128, 127).astype(np.int8)
        return x_q, in_clips

    def _compute_class_l1(self, x_q_1d: np.ndarray, class_idx: int) -> int:
        diff = np.abs(
            x_q_1d.astype(np.int32) - self.class_centroids_i8[class_idx].astype(np.int32)
        )
        return int(np.sum(diff * self.inv_mad_u8.astype(np.int32)))

    def _forward_integer_core(self, x_1d: np.ndarray) -> tuple[int, int, int, int, int, int]:
        """Pure integer `int8`/`int4`/`int32`/`int64` forward pass + OOD/saturation telemetry."""
        x_arr = np.asarray(x_1d, dtype=np.float64).reshape(-1)
        if x_arr.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} input features, got {x_arr.size}.")
        x_q, in_clips_arr = self._quantize_input_adc(x_arr)
        in_clips = int(in_clips_arr)

        w1_q = unpack_signed_int4(self.w1_packed, self.n_features * self.n_hidden).reshape(
            self.n_features, self.n_hidden
        )
        w2_q = unpack_signed_int4(self.w2_packed, self.n_hidden * self.n_classes).reshape(
            self.n_hidden, self.n_classes
        )

        # Layer 1 integer MAC (`int32` saturating bias add) + widening `int64` round-half-up shift
        dot1_i32 = np.matmul(x_q.astype(np.int32), w1_q.astype(np.int32))
        acc1_i32 = np.clip(
            dot1_i32.astype(np.int64) + self.b1_i32.astype(np.int64), INT32_MIN, INT32_MAX
        ).astype(np.int32)
        h1_sat_i64, scaled1_i64 = requantize_saturate_int64(
            acc1_i32,
            self.h1_requant_mult_i16,
            self.h1_requant_shift,
            0,
            7,
            rounding="half_up",
        )
        # Count hidden units exceeding upper int4 rail (+7) in widened int64 BEFORE clamping
        int4_clips = int(np.sum(scaled1_i64 > 7))
        h1_i4 = h1_sat_i64.astype(np.int8)

        # Layer 2 integer MAC (`int32` saturating bias add) + widening `int64` round-half-up shift
        dot2_i32 = np.matmul(h1_i4.astype(np.int32), w2_q.astype(np.int32))
        acc2_i32 = np.clip(
            dot2_i32.astype(np.int64) + self.b2_i32.astype(np.int64), INT32_MIN, INT32_MAX
        ).astype(np.int32)
        logits_sat_i64, _ = requantize_saturate_int64(
            acc2_i32,
            self.logit_mult_i16,
            4,
            INT32_MIN,
            INT32_MAX,
            rounding="half_up",
        )
        logits_i32 = logits_sat_i64.astype(np.int32)

        pred_class = int(np.argmax(logits_i32))
        sorted_logits = np.sort(logits_i32)
        peak_logit = int(sorted_logits[-1])
        runner_up = int(sorted_logits[-2])
        energy_margin = max(0, peak_logit - runner_up)

        l1_dist = self._compute_class_l1(x_q, pred_class)
        return pred_class, l1_dist, int4_clips, in_clips, energy_margin, peak_logit

    def predict_int4_logits(
        self,
        x: np.ndarray,
        *,
        dequantize: bool = True,
        react_rectify: bool = False,
    ) -> np.ndarray:
        """Return integer (`int32`) or calibrated float32 (`dequantize=True`) logits of the int4 student.

        Supports both 1D single samples `(n_features,)` -> `(n_classes,)` and 2D batches
        `(N, n_features)` -> `(N, n_classes)`. When `react_rectify=True`, clamps penultimate `int4`
        activations at `self.react_clip_h1` (`ReAct`, Sun et al., NeurIPS 2021) prior to Layer 2.
        """
        arr = np.asarray(x, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr2d = arr.reshape(1, -1)
        elif arr.ndim == 2:
            arr2d = arr
        else:
            raise ValueError(f"Expected 1D or 2D input array, got shape {arr.shape}.")
        if arr2d.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} input features, got {arr2d.shape[1]}.")

        x_q, _ = self._quantize_input_adc(arr2d)
        w1_q = unpack_signed_int4(self.w1_packed, self.n_features * self.n_hidden).reshape(
            self.n_features, self.n_hidden
        )
        w2_q = unpack_signed_int4(self.w2_packed, self.n_hidden * self.n_classes).reshape(
            self.n_hidden, self.n_classes
        )

        dot1_i32 = np.matmul(x_q.astype(np.int32), w1_q.astype(np.int32))
        acc1_i32 = np.clip(
            dot1_i32.astype(np.int64) + self.b1_i32.astype(np.int64)[None, :], INT32_MIN, INT32_MAX
        ).astype(np.int32)
        h1_max = int(np.clip(self.react_clip_h1, 1, 7)) if react_rectify else 7
        h1_sat_i64, _ = requantize_saturate_int64(
            acc1_i32,
            self.h1_requant_mult_i16[None, :],
            self.h1_requant_shift,
            0,
            h1_max,
            rounding="half_up",
        )
        h1_i4 = h1_sat_i64.astype(np.int8)

        dot2_i32 = np.matmul(h1_i4.astype(np.int32), w2_q.astype(np.int32))
        acc2_i32 = np.clip(
            dot2_i32.astype(np.int64) + self.b2_i32.astype(np.int64)[None, :], INT32_MIN, INT32_MAX
        ).astype(np.int32)
        logits_sat_i64, _ = requantize_saturate_int64(
            acc2_i32,
            self.logit_mult_i16[None, :],
            4,
            INT32_MIN,
            INT32_MAX,
            rounding="half_up",
        )
        logits_i32 = logits_sat_i64.astype(np.int32)
        if dequantize:
            out = (logits_i32.astype(np.float64) * float(self.logit_scale)).astype(np.float32)
        else:
            out = logits_i32
        return out[0] if squeeze else out

    def predict_naive_int4_argmax(self, x_1d: np.ndarray) -> EdgePrediction:
        """Baseline `int4` classifier emitting `argmax` blindly without drift/clip abstention."""
        pred_c, l1_d, h_clips, in_clips, margin, peak_logit = self._forward_integer_core(x_1d)
        return EdgePrediction(
            class_id=pred_c,
            label=self._class_label(pred_c),
            abstained=False,
            l1_mahalanobis_score=l1_d,
            int4_clip_count=h_clips,
            int8_input_clip_count=in_clips,
            energy_margin_score=margin,
            peak_logit=peak_logit,
        )

    def predict_drift_abstaining_int4(self, x_1d: np.ndarray) -> EdgePrediction:
        """Guarded `int4` classifier triggering deterministic `SAFE_HOLD` on OOD or rail clipping."""
        pred_c, l1_d, h_clips, in_clips, margin, peak_logit = self._forward_integer_core(x_1d)
        l1_thresh = int(self.class_l1_thresh_u16[pred_c])
        max_h_clips = int(self.scalar_thresholds_u16[0])
        max_in_clips = int(self.scalar_thresholds_u16[1])
        min_margin = int(self.scalar_thresholds_u16[2])
        min_peak = int(self.scalar_thresholds_u16[3])

        chosen_class = decide_drift_abstaining_int4(
            pred_c,
            l1_d,
            l1_thresh,
            h_clips,
            max_h_clips,
            in_clips,
            max_in_clips,
            margin,
            min_margin,
            peak_logit,
            min_peak,
        )
        if chosen_class == SAFE_HOLD_CLASS_ID:
            return EdgePrediction(
                class_id=SAFE_HOLD_CLASS_ID,
                label=SAFE_HOLD_LABEL,
                abstained=True,
                l1_mahalanobis_score=l1_d,
                int4_clip_count=h_clips,
                int8_input_clip_count=in_clips,
                energy_margin_score=margin,
                peak_logit=peak_logit,
            )
        return EdgePrediction(
            class_id=pred_c,
            label=self._class_label(pred_c),
            abstained=False,
            l1_mahalanobis_score=l1_d,
            int4_clip_count=h_clips,
            int8_input_clip_count=in_clips,
            energy_margin_score=margin,
            peak_logit=peak_logit,
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

    Compares `naive_int4_argmax` vs `drift_abstaining_int4` on a 24-feature, 6-class synthetic fixture:
      - `silent_ood_error_rate_pct` (fraction of OOD inputs that emit an unhalted class prediction)
      - `in_domain_accuracy_pct`
      - `extra_sram_bytes` (`220 B` for 24 features, 6 classes; `< 512 B` MCU budget).
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
    assert clf.extra_sram_bytes < 512

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
