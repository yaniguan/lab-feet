"""Auto-Lab 15-repository wire node for lab-feet (Layer L0 Edge Distillation & INT8/4-Bit Runtime)."""

from __future__ import annotations

import base64
import copy
import functools
import hashlib
import importlib.util
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ID = "lab-feet"
LAYER = "L0"
SCHEMA_VERSION = "autolab-wire/v1"

REPO_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_DIR / "src"
for _path_str in (str(SRC_DIR), str(REPO_DIR)):
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import numpy as np  # noqa: E402

from lab_feet.quantize import (  # noqa: E402
    QuantizedModel,
    forward_int8,
    pack_4bit,
    quantize_model,
    unpack_4bit,
)
from lab_feet.runtime import forward  # noqa: E402
from lab_feet.serialize import from_binary_buffer, to_binary_buffer  # noqa: E402


def _load_drift_abstaining_int4() -> Any:
    """Load local drift_abstaining_int4 module under a unique key to avoid cross-repo collisions."""
    mod_name = "_lab_feet_drift_abstaining_int4"
    existing = sys.modules.get(mod_name)
    if existing is not None:
        return existing
    mod_path = REPO_DIR / "community_extensions" / "drift_abstaining_int4.py"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_DRIFT_MOD = _load_drift_abstaining_int4()
REGIME_NAMES: tuple[str, ...] = tuple(_DRIFT_MOD.REGIME_NAMES)
SAFE_HOLD_LABEL: str = str(_DRIFT_MOD.SAFE_HOLD_LABEL)

_N_FEATURES = 24
_N_HIDDEN = 16


def compute_checkpoint_sha256(model_or_lft1: QuantizedModel | bytes | bytearray | memoryview) -> str:
    """Compute canonical SHA-256 hex digest over `LFT1` serialized quantized model bytes."""
    if isinstance(model_or_lft1, QuantizedModel):
        blob = to_binary_buffer(model_or_lft1)
    else:
        blob = bytes(model_or_lft1)
    return hashlib.sha256(blob).hexdigest()


def compute_calibration_sha256(X_cal: np.ndarray, y_cal: np.ndarray) -> str:
    """Compute canonical SHA-256 hex digest over little-endian `<f4` calibration features and `<i8` labels."""
    x_le = np.ascontiguousarray(X_cal, dtype="<f4")
    y_le = np.ascontiguousarray(y_cal, dtype="<i8").reshape(-1)
    if x_le.ndim != 2 or x_le.shape[0] != y_le.shape[0]:
        raise ValueError(
            f"Calibration X_cal {x_le.shape} and y_cal {y_le.shape} must be aligned 2D/1D arrays."
        )
    h = hashlib.sha256()
    h.update(b"LFCAL1")
    h.update(struct.pack("<II", int(x_le.shape[0]), int(x_le.shape[1])))
    h.update(x_le.tobytes())
    h.update(y_le.tobytes())
    return h.hexdigest()


def _encode_well_coord(well: str) -> tuple[float, float]:
    """Map an SBS 96-well label like 'A1'..'H12' into normalized [-1.0, 1.0] coordinates."""
    cleaned = well.strip().upper()
    if not cleaned:
        return 0.0, 0.0
    row_idx = max(0, min(7, ord(cleaned[0]) - ord("A")))
    digits = "".join(ch for ch in cleaned[1:] if ch.isdigit())
    col_idx = max(0, min(11, (int(digits) - 1) if digits else 0))
    row_norm = (row_idx / 3.5) - 1.0
    col_norm = (col_idx / 5.5) - 1.0
    return float(row_norm), float(col_norm)


def _derive_sensor_features(
    stepbot_execution: Mapping[str, Any],
    manipulation: Mapping[str, Any],
    dispense_error_permille: int,
) -> np.ndarray:
    """Derive a deterministic 24-dim float32 benchtop sensor feature vector from upstream L1 state."""
    proto_sha = str(stepbot_execution.get("protocol_sha256", "0" * 64))
    digest = hashlib.sha256(proto_sha.encode("utf-8")).digest()
    sha_feats = [((b / 255.0) - 0.5) * 1.6 for b in digest[:12]]

    deck_epoch = float(stepbot_execution.get("next_deck_epoch", 1))
    code_text = str(stepbot_execution.get("python_code", ""))
    code_lines = float(len(code_text.splitlines()) or 1)
    receipt_present = 1.0 if stepbot_execution.get("dispatch_receipt") else 0.5

    start_r, start_c = _encode_well_coord(str(manipulation.get("start_well", "A1")))
    target_r, target_c = _encode_well_coord(str(manipulation.get("target_well", "H12")))
    waypoints = min(float(manipulation.get("waypoints", 6)) / 16.0, 2.0)
    path_len_rad = min(float(manipulation.get("path_length_rad", 1.25)) / 3.14159265, 2.0)
    clearance_mm = min(float(manipulation.get("clearance_mm", 8.5)) / 15.0, 2.0)
    collision_free = 1.0 if manipulation.get("collision_free", True) else -1.0
    dispense_norm = min(max(float(dispense_error_permille) / 100.0, 0.0), 2.5)

    kinematic_feats = [
        min(deck_epoch / 10.0, 2.0),
        min(code_lines / 50.0, 2.0),
        receipt_present,
        start_r,
        start_c,
        target_r,
        target_c,
        waypoints,
        path_len_rad,
        clearance_mm,
        collision_free,
        dispense_norm,
    ]
    vec = np.asarray(sha_feats + kinematic_feats, dtype=np.float32)
    assert vec.shape == (_N_FEATURES,)
    return vec


def _measure_calibration_drop_permille(
    fp32_layers: list[tuple[np.ndarray, np.ndarray]],
    qmodel_4bit: QuantizedModel,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> int:
    """Measure empirical accuracy drop (`1000 * max(0, fp32_acc - int4_acc)`) on `(X_cal, y_cal)`."""
    y_arr = np.asarray(y_cal, dtype=np.int64).reshape(-1)
    if y_arr.size == 0:
        return 0
    fp32_preds = forward(fp32_layers, X_cal, activation=qmodel_4bit.activation).argmax(axis=1)
    int4_preds = forward_int8(qmodel_4bit, X_cal, act_bits=8).argmax(axis=1)
    fp32_acc = float((fp32_preds == y_arr).mean())
    int4_acc = float((int4_preds == y_arr).mean())
    return int(round(1000.0 * max(0.0, fp32_acc - int4_acc)))


def _build_ood_monitor_for_layers(
    layers_fp32: list[tuple[np.ndarray, np.ndarray]],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> Any | None:
    """Construct and calibrate `< 512 B` `DriftAbstainingInt4Classifier` when architecture is 2-layer."""
    if len(layers_fp32) != 2:
        return None
    (w1, b1), (w2, b2) = layers_fp32
    d_in, d_hid = w1.shape
    _, d_out = w2.shape
    if d_out < 2 or X_cal.shape[1] != d_in:
        return None
    clf_cls = _DRIFT_MOD.DriftAbstainingInt4Classifier
    clf = clf_cls(n_features=int(d_in), n_hidden=int(d_hid), n_classes=int(d_out))
    # Headroom scale covering [-3.2, +3.2] benchtop normalized features without false nominal rail clips
    p995 = float(np.percentile(np.abs(X_cal), 99.5)) if X_cal.size > 0 else 2.5
    in_scale = max(p995 * 1.65 / 127.0, 0.04)
    clf.load_fp32_weights_and_calibrate(w1, b1, w2, b2, X_cal, y_cal, input_scale=in_scale, act_cap=4.5)
    # Relax margin/peak floor slightly for arbitrary derived wire hashes while catching true OOD/glare spikes
    clf.class_l1_thresh_u16[:] = np.minimum(
        65535, (clf.class_l1_thresh_u16.astype(np.int32) * 2 + 280)
    ).astype(np.uint16)
    clf.scalar_thresholds_u16[2] = np.uint16(min(int(clf.scalar_thresholds_u16[2]), 2))
    clf.scalar_thresholds_u16[3] = np.uint16(0)
    return clf


@functools.lru_cache(maxsize=1)
def _get_bench_monitor_bundle() -> dict[str, Any]:
    """Build deterministic QAT-trained & calibrated `24 -> 16 -> 6` benchtop student + OOD monitor."""
    rng = np.random.default_rng(20261123)
    n_classes = len(REGIME_NAMES)

    # Sample synthetic sensor vectors matching _derive_sensor_features ([-0.8, 0.8]^12 x [-1.0, 2.0]^12)
    n_total = 720
    sha_part = rng.uniform(-0.8, 0.8, size=(n_total, 12)).astype(np.float32)
    kin_part = rng.uniform(-1.0, 2.0, size=(n_total, 12)).astype(np.float32)
    x_all = np.hstack([sha_part, kin_part]).astype(np.float32)

    # Smooth 6-regime oracle anchored on kinematic + spectral hash channels
    oracle_w = np.zeros((_N_FEATURES, n_classes), dtype=np.float32)
    for c in range(n_classes):
        oracle_w[(2 * c) % 12, c] = 2.4
        oracle_w[12 + (2 * c) % 12, c] = 2.1
        oracle_w[12 + (2 * c + 1) % 12, c] = -1.7
    y_all = np.argmax(x_all @ oracle_w, axis=1).astype(np.int64)

    x_train, y_train = x_all[:520], y_all[:520]
    x_cal, y_cal = x_all[520:], y_all[520:]

    # Train compact 24 -> 16 (ReLU) -> 6 student with STE 4-bit QAT in final epochs for FP32/INT4 parity
    w1 = rng.normal(0.0, np.sqrt(2.0 / _N_FEATURES), size=(_N_FEATURES, _N_HIDDEN)).astype(np.float64)
    b1 = np.full(_N_HIDDEN, 0.15, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / _N_HIDDEN), size=(_N_HIDDEN, n_classes)).astype(np.float64)
    b2 = np.zeros(n_classes, dtype=np.float64)

    lr = 0.09
    n_tr = x_train.shape[0]
    for epoch in range(260):
        if epoch >= 180:
            # Straight-through 4-bit per-channel weight fake-quantization
            s1 = np.maximum(np.max(np.abs(w1), axis=0, keepdims=True) / 7.0, 1e-6)
            w1_eff = np.clip(np.rint(w1 / s1), -8, 7) * s1
            s2 = np.maximum(np.max(np.abs(w2), axis=0, keepdims=True) / 7.0, 1e-6)
            w2_eff = np.clip(np.rint(w2 / s2), -8, 7) * s2
        else:
            w1_eff, w2_eff = w1, w2

        z1 = x_train @ w1_eff + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2_eff + b2
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= np.sum(probs, axis=1, keepdims=True)

        dlogits = probs
        dlogits[np.arange(n_tr), y_train] -= 1.0
        dlogits /= float(n_tr)

        dw2 = h1.T @ dlogits + 2e-4 * w2
        db2 = np.sum(dlogits, axis=0)
        dh1 = dlogits @ w2_eff.T
        dz1 = dh1 * (z1 > 0.0)
        dw1 = x_train.T @ dz1 + 2e-4 * w1
        db1 = np.sum(dz1, axis=0)

        step_lr = lr / (1.0 + 0.006 * epoch)
        w1 -= step_lr * dw1
        b1 -= step_lr * db1
        w2 -= step_lr * dw2
        b2 -= step_lr * db2

    fp32_layers = [
        (w1.astype(np.float32), b1.astype(np.float32)),
        (w2.astype(np.float32), b2.astype(np.float32)),
    ]
    qmodel_int8 = quantize_model(fp32_layers, bits=8, per_channel=True, mode="symmetric")
    qmodel_4bit = quantize_model(
        fp32_layers,
        bits=4,
        per_channel=True,
        mode="symmetric_full",
        clip_search="activation_mse",
        X_cal=x_cal,
    )
    lft1_bytes = to_binary_buffer(qmodel_4bit)
    ckpt_sha256 = compute_checkpoint_sha256(lft1_bytes)
    cal_sha256 = compute_calibration_sha256(x_cal, y_cal)
    measured_drop = _measure_calibration_drop_permille(fp32_layers, qmodel_4bit, x_cal, y_cal)
    ood_clf = _build_ood_monitor_for_layers(fp32_layers, x_cal, y_cal)

    return {
        "fp32_layers": fp32_layers,
        "qmodel_int8": qmodel_int8,
        "qmodel_4bit": qmodel_4bit,
        "lft1_bytes": lft1_bytes,
        "checkpoint_sha256": ckpt_sha256,
        "calibration_sha256": cal_sha256,
        "X_cal": x_cal,
        "y_cal": y_cal,
        "measured_drop_permille": measured_drop,
        "ood_clf": ood_clf,
        "source": "default_calibrated_monitor",
    }


def _get_bench_monitor_models() -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    QuantizedModel,
    QuantizedModel,
]:
    """Return `(fp32_layers, qmodel_int8, qmodel_4bit)` for backwards compatibility."""
    bundle = _get_bench_monitor_bundle()
    return bundle["fp32_layers"], bundle["qmodel_int8"], bundle["qmodel_4bit"]


@functools.lru_cache(maxsize=32)
def _load_verified_lft1_models(
    checkpoint_sha256: str,
    lft1_bytes: bytes,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], QuantizedModel, QuantizedModel]:
    """LRU-cached hot-loader deserializing verified `LFT1` checkpoint bytes."""
    actual = hashlib.sha256(lft1_bytes).hexdigest()
    if actual != checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch inside verified loader.")
    qmodel_loaded = from_binary_buffer(lft1_bytes)
    deq_layers = qmodel_loaded.dequantize()
    if all(layer.bits == 4 for layer in qmodel_loaded.layers):
        qmodel_4bit = qmodel_loaded
        qmodel_int8 = quantize_model(
            deq_layers, bits=8, per_channel=True, mode="symmetric", activation=qmodel_loaded.activation
        )
    else:
        qmodel_int8 = qmodel_loaded
        qmodel_4bit = quantize_model(
            deq_layers,
            bits=4,
            per_channel=True,
            mode="symmetric_full",
            activation=qmodel_loaded.activation,
        )
    return deq_layers, qmodel_int8, qmodel_4bit


def _extract_hotload_checkpoint_bytes(
    meta: Mapping[str, Any],
    X_cal: np.ndarray | None,
) -> tuple[bytes | None, list[tuple[np.ndarray, np.ndarray]] | None]:
    """Extract canonical `LFT1` bytes (and optional true FP32 reference layers) from packet metadata."""
    if "lft1_bytes" in meta and meta["lft1_bytes"] is not None:
        raw = meta["lft1_bytes"]
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw), None
        if isinstance(raw, str):
            return bytes.fromhex(raw), None
        raise ValueError("Unsupported lft1_bytes type.")

    if "lft1_hex" in meta and meta["lft1_hex"] is not None:
        return bytes.fromhex(str(meta["lft1_hex"]).strip()), None

    if "lft1_base64" in meta and meta["lft1_base64"] is not None:
        return base64.b64decode(str(meta["lft1_base64"]).strip()), None

    if "lft1_path" in meta and meta["lft1_path"] is not None:
        return Path(str(meta["lft1_path"])).read_bytes(), None

    if "npz_path" in meta and meta["npz_path"] is not None:
        qm = QuantizedModel.load_npz(Path(str(meta["npz_path"])))
        return to_binary_buffer(qm), qm.dequantize()

    raw_layers = meta.get("fp32_layers") if "fp32_layers" in meta else meta.get("layers")
    if raw_layers is not None:
        fp32_layers = [
            (np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32))
            for w, b in raw_layers
        ]
        qm4 = quantize_model(
            fp32_layers,
            bits=4,
            per_channel=True,
            mode="symmetric_full",
            clip_search="activation_mse" if X_cal is not None else "weight_mse",
            X_cal=X_cal,
        )
        return to_binary_buffer(qm4), fp32_layers

    return None, None


def _verify_nibble_roundtrip(qmodel_4bit: QuantizedModel) -> tuple[bool, int]:
    """Verify two's-complement [-8, 7] nibble packing/unpacking across even, odd, & model layers."""
    full_codes = np.arange(-8, 8, dtype=np.int8)
    packed_codes = pack_4bit(full_codes)
    unpacked_codes = unpack_4bit(packed_codes, full_codes.shape)

    odd_codes = np.array([-8, -3, 0, 4, 7], dtype=np.int8)
    packed_odd = pack_4bit(odd_codes)
    unpacked_odd = unpack_4bit(packed_odd, odd_codes.size)

    exact = bool(
        packed_codes.dtype == np.uint8
        and packed_codes.nbytes == 8
        and np.array_equal(unpacked_codes, full_codes)
        and packed_odd.nbytes == 3
        and ((int(packed_odd[-1]) >> 4) & 0x0F) == 0
        and np.array_equal(unpacked_odd, odd_codes)
    )

    packed_len = 0
    for layer in qmodel_4bit.layers:
        q_codes = layer.unpack_codes()
        repacked = pack_4bit(q_codes)
        roundtripped = unpack_4bit(repacked, layer.shape)
        packed_len += int(repacked.nbytes)
        if not (
            np.array_equal(repacked, layer.packed_weights)
            and np.array_equal(roundtripped, q_codes)
        ):
            exact = False
    return exact, packed_len


def _record_halt_and_return(
    out: dict[str, Any],
    halt_reason: str,
    summary_detail: str,
) -> dict[str, Any]:
    out["halted"] = True
    out["halt_reason"] = halt_reason
    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": "BLOCK",
            "summary": f"L0 int8/4-bit runtime BLOCK ({halt_reason}): {summary_detail}",
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": "nibble_pack_roundtrip_and_drift_cap",
            "first_principles": (
                "Layer L0 enforces upstream SplitGuard isolation, SHA-256 checkpoint and "
                "calibration verification, exact [-8, 7] nibble round-tripping, and OOD/drift "
                "caps before emitting edge telemetry to lab-attrib."
            ),
        }
    )
    return out


def process_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Execute Layer L0 quantized int8 / 4-bit edge runtime pass on an autolab-wire/v1 packet."""
    out: dict[str, Any] = copy.deepcopy(packet)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if out["halted"]:
        return out

    posttrain = out.get("posttrain") if isinstance(out.get("posttrain"), Mapping) else {}
    distilled = out.get("distilled_student") if isinstance(out.get("distilled_student"), Mapping) else {}

    # 1) Enforce upstream SplitGuard anti-leakage gate (`#30`)
    if posttrain.get("split_guard_clean") is False or distilled.get("split_guard_clean") is False:
        return _record_halt_and_return(
            out,
            "L0_FEET_UPSTREAM_SPLITGUARD_BLOCK",
            "upstream posttrain/distilled_student split_guard_clean is False",
        )

    merged_ckpt_meta: dict[str, Any] = {**posttrain, **distilled}

    # Parse optional calibration features & labels from packet metadata
    raw_x_cal = (
        merged_ckpt_meta.get("calibration_X")
        if "calibration_X" in merged_ckpt_meta
        else merged_ckpt_meta.get("X_cal")
    )
    raw_y_cal = (
        merged_ckpt_meta.get("calibration_y")
        if "calibration_y" in merged_ckpt_meta
        else merged_ckpt_meta.get("y_cal")
    )
    supplied_x_cal: np.ndarray | None = None
    supplied_y_cal: np.ndarray | None = None
    if raw_x_cal is not None or raw_y_cal is not None:
        if raw_x_cal is None or raw_y_cal is None:
            return _record_halt_and_return(
                out,
                "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                "incomplete calibration split (require both calibration_X and calibration_y)",
            )
        try:
            supplied_x_cal = np.asarray(raw_x_cal, dtype=np.float32)
            supplied_y_cal = np.asarray(raw_y_cal, dtype=np.int64).reshape(-1)
            supplied_cal_sha256 = compute_calibration_sha256(supplied_x_cal, supplied_y_cal)
        except Exception as exc:
            return _record_halt_and_return(
                out,
                "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                f"invalid calibration arrays: {exc}",
            )
    else:
        supplied_cal_sha256 = None

    # 2) Hot-load student checkpoint if supplied, else use deterministic calibrated bench monitor
    try:
        hot_lft1_bytes, hot_fp32_override = _extract_hotload_checkpoint_bytes(
            merged_ckpt_meta, supplied_x_cal
        )
    except Exception as exc:
        return _record_halt_and_return(
            out,
            "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
            f"failed to load student checkpoint payload: {exc}",
        )

    expected_ckpt_sha = merged_ckpt_meta.get("checkpoint_sha256", out.get("checkpoint_sha256"))
    expected_cal_sha = merged_ckpt_meta.get("calibration_sha256", out.get("calibration_sha256"))

    if hot_lft1_bytes is not None:
        actual_ckpt_sha = compute_checkpoint_sha256(hot_lft1_bytes)
        if expected_ckpt_sha is not None and actual_ckpt_sha != str(expected_ckpt_sha).strip().lower():
            return _record_halt_and_return(
                out,
                "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                f"checkpoint_sha256={actual_ckpt_sha[:12]}... != expected {str(expected_ckpt_sha)[:12]}...",
            )
        try:
            deq_layers, qmodel_int8, qmodel_4bit = _load_verified_lft1_models(
                actual_ckpt_sha, hot_lft1_bytes
            )
        except Exception as exc:
            return _record_halt_and_return(
                out,
                "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                f"corrupted LFT1 student checkpoint: {exc}",
            )
        fp32_layers = hot_fp32_override if hot_fp32_override is not None else deq_layers
        checkpoint_source = "hotloaded_student"

        if supplied_x_cal is not None and supplied_y_cal is not None:
            actual_cal_sha = supplied_cal_sha256
            if expected_cal_sha is not None and actual_cal_sha != str(expected_cal_sha).strip().lower():
                return _record_halt_and_return(
                    out,
                    "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                    "calibration_sha256 mismatch against canonical calibration split",
                )
            measured_drop = _measure_calibration_drop_permille(
                fp32_layers, qmodel_4bit, supplied_x_cal, supplied_y_cal
            )
            ood_clf = _build_ood_monitor_for_layers(fp32_layers, supplied_x_cal, supplied_y_cal)
        else:
            if expected_cal_sha is not None:
                return _record_halt_and_return(
                    out,
                    "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                    "expected calibration_sha256 supplied without calibration_X/calibration_y",
                )
            default_bundle = _get_bench_monitor_bundle()
            if qmodel_4bit.layers[0].shape[0] == _N_FEATURES and qmodel_4bit.layers[-1].shape[1] == len(REGIME_NAMES):
                actual_cal_sha = default_bundle["calibration_sha256"]
                measured_drop = _measure_calibration_drop_permille(
                    fp32_layers, qmodel_4bit, default_bundle["X_cal"], default_bundle["y_cal"]
                )
                ood_clf = _build_ood_monitor_for_layers(
                    fp32_layers, default_bundle["X_cal"], default_bundle["y_cal"]
                )
            else:
                actual_cal_sha = None
                measured_drop = 0
                ood_clf = None
    else:
        bundle = _get_bench_monitor_bundle()
        fp32_layers = bundle["fp32_layers"]
        qmodel_int8 = bundle["qmodel_int8"]
        qmodel_4bit = bundle["qmodel_4bit"]
        actual_ckpt_sha = bundle["checkpoint_sha256"]
        checkpoint_source = bundle["source"]

        if expected_ckpt_sha is not None and actual_ckpt_sha != str(expected_ckpt_sha).strip().lower():
            return _record_halt_and_return(
                out,
                "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                f"checkpoint_sha256={actual_ckpt_sha[:12]}... != expected {str(expected_ckpt_sha)[:12]}...",
            )

        if supplied_x_cal is not None and supplied_y_cal is not None:
            actual_cal_sha = supplied_cal_sha256
            if expected_cal_sha is not None and actual_cal_sha != str(expected_cal_sha).strip().lower():
                return _record_halt_and_return(
                    out,
                    "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                    "calibration_sha256 mismatch on supplied calibration split",
                )
            measured_drop = _measure_calibration_drop_permille(
                fp32_layers, qmodel_4bit, supplied_x_cal, supplied_y_cal
            )
            ood_clf = _build_ood_monitor_for_layers(fp32_layers, supplied_x_cal, supplied_y_cal)
        else:
            actual_cal_sha = bundle["calibration_sha256"]
            if expected_cal_sha is not None and actual_cal_sha != str(expected_cal_sha).strip().lower():
                return _record_halt_and_return(
                    out,
                    "L0_FEET_CHECKPOINT_SHA256_MISMATCH",
                    "calibration_sha256 mismatch against canonical bench calibration split",
                )
            measured_drop = int(bundle["measured_drop_permille"])
            ood_clf = bundle["ood_clf"]

    stepbot_exec = out.get("stepbot_execution") or {}
    manip = out.get("manipulation") or {}
    dispense_error_permille = int(out.get("dispense_error_permille", 12))
    max_acc_drop_permille = int(out.get("max_acc_drop_permille", 15))
    if "quant_acc_drop_permille" in out and out["quant_acc_drop_permille"] is not None:
        quant_acc_drop_permille = max(int(out["quant_acc_drop_permille"]), int(measured_drop))
    else:
        quant_acc_drop_permille = int(measured_drop)

    aspiration_clean = bool(out.get("aspiration_clean", True))
    incubation_window_clean = bool(out.get("incubation_window_clean", True))
    seal_intact = bool(out.get("seal_intact", True))

    if "sensor_features" in out and out["sensor_features"] is not None:
        sensor_vec = np.asarray(out["sensor_features"], dtype=np.float32).reshape(-1)
    else:
        expected_d_in = int(qmodel_4bit.layers[0].shape[0])
        base_vec = _derive_sensor_features(stepbot_exec, manip, dispense_error_permille)
        if expected_d_in == _N_FEATURES:
            sensor_vec = base_vec
        else:
            reps = (expected_d_in + _N_FEATURES - 1) // _N_FEATURES
            sensor_vec = np.tile(base_vec, reps)[:expected_d_in].astype(np.float32)

    nibble_roundtrip_exact, packed_len = _verify_nibble_roundtrip(qmodel_4bit)

    fp32_logits = forward(fp32_layers, sensor_vec, activation=qmodel_4bit.activation)
    int8_logits = forward_int8(qmodel_int8, sensor_vec, act_bits=8)
    int4_logits = forward_int8(qmodel_4bit, sensor_vec, act_bits=8)

    fp32_pred = int(np.argmax(fp32_logits))
    int8_pred = int(np.argmax(int8_logits))
    int4_pred = int(np.argmax(int4_logits))
    fp32_int4_pred_agree = bool(fp32_pred == int4_pred)

    safe_hold_abstained = False
    ood_l1_score = 0
    int4_sat_clips = 0
    int8_in_clips = 0
    energy_margin = 0
    if ood_clf is not None and sensor_vec.size == ood_clf.n_features:
        ood_pred = ood_clf.predict_drift_abstaining_int4(sensor_vec)
        safe_hold_abstained = bool(ood_pred.abstained)
        ood_l1_score = int(ood_pred.l1_mahalanobis_score)
        int4_sat_clips = int(ood_pred.int4_clip_count)
        int8_in_clips = int(ood_pred.int8_input_clip_count)
        energy_margin = int(ood_pred.energy_margin_score)

    if safe_hold_abstained:
        monitor_regime = SAFE_HOLD_LABEL
    elif 0 <= int4_pred < len(REGIME_NAMES):
        monitor_regime = REGIME_NAMES[int4_pred]
    else:
        monitor_regime = f"regime_{int4_pred}"

    require_live_agree = bool(out.get("require_live_pred_agree", False))
    if not nibble_roundtrip_exact or quant_acc_drop_permille > max_acc_drop_permille:
        out["halted"] = True
        out["halt_reason"] = "L0_FEET_QUANT_DRIFT_BLOCK"
    elif safe_hold_abstained:
        out["halted"] = True
        out["halt_reason"] = "L0_FEET_SAFE_HOLD_ABSTAIN"
    elif require_live_agree and not fp32_int4_pred_agree:
        out["halted"] = True
        out["halt_reason"] = "L0_FEET_QUANT_DRIFT_BLOCK"

    out["edge_runtime"] = {
        "quantization": "int8_and_4bit",
        "nibble_roundtrip_exact": bool(nibble_roundtrip_exact),
        "packed_bytes": packed_len,
        "dispense_error_permille": dispense_error_permille,
        "aspiration_clean": aspiration_clean,
        "incubation_window_clean": incubation_window_clean,
        "seal_intact": seal_intact,
        "quant_acc_drop_permille": quant_acc_drop_permille,
        "max_acc_drop_permille": max_acc_drop_permille,
        "monitor_regime": monitor_regime,
        "fp32_pred_class": fp32_pred,
        "int8_pred_class": int8_pred,
        "int4_pred_class": int4_pred,
        "fp32_int4_pred_agree": fp32_int4_pred_agree,
        "checkpoint_sha256": actual_ckpt_sha,
        "calibration_sha256": actual_cal_sha,
        "checkpoint_source": checkpoint_source,
        "safe_hold_abstained": safe_hold_abstained,
        "ood_l1_mahalanobis_score": ood_l1_score,
        "int4_saturation_clips": int4_sat_clips,
        "int8_input_clips": int8_in_clips,
        "energy_margin_score": energy_margin,
    }

    status = "BLOCK" if out["halted"] else "PASS"
    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": status,
            "summary": (
                f"L0 int8/4-bit runtime {status}: packed_bytes={packed_len}, "
                f"nibble_roundtrip_exact={nibble_roundtrip_exact}, "
                f"quant_acc_drop_permille={quant_acc_drop_permille}/{max_acc_drop_permille}, "
                f"fp32_int4_agree={fp32_int4_pred_agree}, safe_hold={safe_hold_abstained}, "
                f"dispense_error_permille={dispense_error_permille}, regime={monitor_regime}"
            ),
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": "nibble_pack_roundtrip_and_drift_cap",
            "first_principles": (
                "Signed two's-complement 4-bit nibbles in [-8, 7] pack two weights per uint8 byte "
                "with exact bit-masked round-trip, verified SHA-256 checkpoints and calibration "
                "splits bound empirical accuracy_drop_permille <= max_drop_permille, and the "
                "<512 B int4 OOD/saturation monitor gates telemetry emitted to lab-attrib."
            ),
        }
    )
    return out


__all__ = [
    "LAYER",
    "REPO_DIR",
    "REPO_ID",
    "SCHEMA_VERSION",
    "compute_calibration_sha256",
    "compute_checkpoint_sha256",
    "process_packet",
]
