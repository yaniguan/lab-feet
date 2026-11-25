"""Freestanding C99 / CMSIS-NN / TFLite-Micro static `.rodata` header exporter (`#31`).

Lowers `QuantizedModel` (`4-bit` nibble-packed or `8-bit` weights) and `DriftAbstainingInt4Classifier`
into zero-`malloc`, FPU-free C99 headers with 16-byte aligned `.rodata` arrays, `int32_t` biases,
fixed-point `(mult_i32, shift)` requantization multipliers, compile-time `LAB_FEET_SCRATCH_BYTES`
ping-pong arena bounds, and 5-gate `SAFE_HOLD` drift-abstention routines.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .quantize import INT32_MAX, INT32_MIN, QuantizedModel
from .runtime import _apply_hidden_activation


def _sanitize_c_ident(prefix: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", prefix.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = "lab_feet_" + cleaned
    return cleaned.lower()


def quantize_fixed_point_multiplier(
    real_multiplier: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert positive float multiplier(s) `M` into exact integer `(mult_i32, right_shift_i32)`.

    Chooses `right_shift >= 1` such that `mult_i32 = round(M * 2^right_shift)` utilizes ~30 bits
    of precision in `[1, 2^30 - 1]` and executes in C99 via widening `int64_t` multiply-shift:
        `((int64_t)acc * (int64_t)mult + (1LL << (shift - 1))) >> shift`.
    """
    arr = np.asarray(real_multiplier, dtype=np.float64).reshape(-1)
    mults = np.empty(arr.size, dtype=np.int32)
    shifts = np.empty(arr.size, dtype=np.int32)
    max_mult = (1 << 30) - 1

    for idx, val in enumerate(arr):
        m = float(val)
        if not math.isfinite(m) or m <= 0.0:
            mults[idx] = 0
            shifts[idx] = 1
            continue
        _, exp = math.frexp(m)
        # m = mantissa * 2^exp with mantissa in [0.5, 1.0) -> m * 2^(30 - exp) in [2^29, 2^30)
        shift = max(1, min(46, 30 - exp))
        scaled = round(m * math.ldexp(1.0, shift))
        while scaled > max_mult and shift > 1:
            shift -= 1
            scaled = round(m * math.ldexp(1.0, shift))
        mults[idx] = int(max(1, min(max_mult, scaled)))
        shifts[idx] = int(shift)

    return mults, shifts


def _format_c_int_array(values: np.ndarray, *, hex_u8: bool = False, per_line: int = 12) -> str:
    flat = np.asarray(values).ravel()
    if flat.size == 0:
        return "0"
    tokens: list[str] = []
    for v in flat:
        iv = int(v)
        if hex_u8:
            tokens.append(f"0x{iv & 0xFF:02X}u")
        elif iv == INT32_MIN:
            tokens.append("(-2147483647 - 1)")
        else:
            tokens.append(str(iv))
    lines = [
        "    " + ", ".join(tokens[i : i + per_line])
        for i in range(0, len(tokens), per_line)
    ]
    return ",\n".join(lines)


def prepare_fixed_point_model(
    model: QuantizedModel,
    *,
    input_scale: float | None = None,
    X_cal: np.ndarray | None = None,
) -> dict[str, Any]:
    """Lower a `QuantizedModel` (`relu` or `identity` hidden layers) to pure-integer C99 parameters."""
    if not model.layers:
        raise ValueError("QuantizedModel must contain at least one layer for C99 export.")
    if model.activation not in ("relu", "identity"):
        raise ValueError(
            f"Freestanding FPU-free C99 export supports 'relu' or 'identity' hidden activations, "
            f"got {model.activation!r}."
        )

    d_in_0 = int(model.layers[0].shape[0])
    if X_cal is not None:
        cal_arr = np.asarray(X_cal, dtype=np.float32)
        if cal_arr.ndim == 1:
            cal_arr = cal_arr.reshape(1, -1)
    else:
        rng = np.random.default_rng(0)
        cal_arr = rng.normal(0.0, 1.0, size=(64, d_in_0)).astype(np.float32)

    if input_scale is None:
        p999 = float(np.percentile(np.abs(cal_arr), 99.9)) if cal_arr.size > 0 else 1.0
        in_scale = max(p999 * 1.15 / 127.0, 1e-5)
    else:
        in_scale = max(float(input_scale), 1e-12)

    h_cal = cal_arr
    cur_in_scale = in_scale
    layer_specs: list[dict[str, Any]] = []
    max_feature_dim = d_in_0

    for idx, layer in enumerate(model.layers):
        d_in, d_out = layer.shape
        max_feature_dim = max(max_feature_dim, d_in, d_out)
        is_hidden = idx < len(model.layers) - 1

        w_deq, b_deq = layer.dequantize()
        z_cal = h_cal @ w_deq + b_deq
        h_next = _apply_hidden_activation(z_cal, model.activation) if is_hidden else z_cal

        w_scales = (
            layer.scale.reshape(d_out).astype(np.float64)
            if layer.per_channel
            else np.full(d_out, float(layer.scale.reshape(-1)[0]), dtype=np.float64)
        )
        zp_vec = (
            layer.zero_point.reshape(d_out).astype(np.int8)
            if (layer.zero_point is not None and layer.per_channel)
            else (
                np.full(d_out, int(layer.zero_point.reshape(-1)[0]), dtype=np.int8)
                if layer.zero_point is not None
                else np.zeros(d_out, dtype=np.int8)
            )
        )

        acc_step = np.maximum(cur_in_scale * w_scales, 1e-15)
        bias_i32 = np.clip(
            np.rint(layer.bias.astype(np.float64) / acc_step),
            INT32_MIN + 1,
            INT32_MAX,
        ).astype(np.int32)

        if is_hidden:
            max_act = float(np.percentile(np.abs(h_next), 99.9)) if h_next.size > 0 else 1.0
            out_scale = max(max_act * 1.15 / 127.0, 1e-5)
        else:
            # Final logit scale preserves 8 extra fractional bits (`1/256` of max accumulator step)
            out_scale = max(float(np.max(acc_step)) / 256.0, 1e-9)

        real_mult = acc_step / out_scale
        mult_i32, shift_i32 = quantize_fixed_point_multiplier(real_mult)

        layer_specs.append(
            {
                "index": idx,
                "d_in": d_in,
                "d_out": d_out,
                "bits": layer.bits,
                "is_hidden": is_hidden,
                "has_zp": layer.zero_point is not None,
                "packed_weights": layer.packed_weights.copy(),
                "unpacked_codes": layer.unpack_codes(),
                "zp_i8": zp_vec,
                "bias_i32": bias_i32,
                "mult_i32": mult_i32,
                "shift_i32": shift_i32,
                "in_scale": cur_in_scale,
                "out_scale": out_scale,
            }
        )
        h_cal = h_next
        cur_in_scale = out_scale

    # Ping-pong int8 arena requires two buffers of `max_feature_dim` rounded up to 16-byte alignment
    aligned_dim = ((max_feature_dim + 15) // 16) * 16
    scratch_bytes = max(32, 2 * aligned_dim)

    return {
        "activation": model.activation,
        "input_dim": d_in_0,
        "output_dim": int(model.layers[-1].shape[1]),
        "input_scale": float(in_scale),
        "logit_scale": float(cur_in_scale),
        "scratch_bytes": int(scratch_bytes),
        "aligned_slot_bytes": int(aligned_dim),
        "layers": layer_specs,
    }


def quantize_input_for_c99(x: np.ndarray, input_scale: float) -> np.ndarray:
    """Quantize float input vector/batch into `int8_t` (`[-127, 127]`) for the C99 fixed-point kernel."""
    arr = np.asarray(x, dtype=np.float64)
    scaled = np.clip(arr / max(float(input_scale), 1e-12), -127.0, 127.0)
    # Match C/ADC round-half-away-from-zero or rint? Use np.rint on float inputs; host passes int8_t directly to C99!
    return np.clip(np.rint(scaled), -127, 127).astype(np.int8)


def simulate_c99_fixed_point(
    model_or_spec: QuantizedModel | dict[str, Any],
    x: np.ndarray,
    *,
    input_scale: float | None = None,
    X_cal: np.ndarray | None = None,
    x_is_quantized_int8: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact Python reference matching the generated C99 `<prefix>_infer_fixed` function.

    Returns `(logits_i32, logits_dequant_f32)`. When `x_is_quantized_int8=True`, `x` is already an
    `int8` tensor and `logits_i32` is 100% bit-identical to the compiled C99 kernel output.
    """
    spec = (
        prepare_fixed_point_model(model_or_spec, input_scale=input_scale, X_cal=X_cal)
        if isinstance(model_or_spec, QuantizedModel)
        else model_or_spec
    )
    if x_is_quantized_int8:
        x_arr = np.asarray(x, dtype=np.int8)
    else:
        x_arr = quantize_input_for_c99(x, spec["input_scale"])

    was_1d = x_arr.ndim == 1
    h_q = x_arr.reshape(1, -1) if was_1d else x_arr
    act_mode = spec["activation"]

    for lspec in spec["layers"]:
        w_q = lspec["unpacked_codes"].astype(np.int32)
        zp_i32 = lspec["zp_i8"].astype(np.int32)[None, :]
        w_eff = w_q - zp_i32
        # Match C99 loop: int64 saturating bias + MAC clamped to signed int32
        dot_i64 = np.matmul(h_q.astype(np.int64), w_eff.astype(np.int64))
        acc_i32 = np.clip(
            dot_i64 + lspec["bias_i32"].astype(np.int64)[None, :],
            INT32_MIN,
            INT32_MAX,
        ).astype(np.int32)

        mult_i64 = lspec["mult_i32"].astype(np.int64)[None, :]
        shifts = lspec["shift_i32"].astype(np.int64)[None, :]
        round_off = np.int64(1) << (shifts - 1)
        scaled_i64 = (acc_i32.astype(np.int64) * mult_i64 + round_off) >> shifts

        if lspec["is_hidden"]:
            qmin = 0 if act_mode == "relu" else -127
            h_q = np.clip(scaled_i64, qmin, 127).astype(np.int8)
        else:
            logits_i32 = np.clip(scaled_i64, INT32_MIN, INT32_MAX).astype(np.int32)

    logits_f32 = (logits_i32.astype(np.float64) * float(spec["logit_scale"])).astype(np.float32)
    if was_1d:
        return logits_i32.reshape(-1), logits_f32.reshape(-1)
    return logits_i32, logits_f32


def _export_drift_classifier_c99(clf: Any, prefix: str, guard: str) -> str:
    """Emit freestanding C99 header for `DriftAbstainingInt4Classifier` (`W4A4` + 5-gate `SAFE_HOLD`)."""
    d_in = int(clf.n_features)
    d_hid = int(clf.n_hidden)
    d_out = int(clf.n_classes)
    scratch_bytes = max(32, ((max(d_in, d_hid, d_out) + 15) // 16) * 16)

    lines = [
        f"/* Auto-generated by lab-feet C99 MCU exporter (DriftAbstainingInt4 SAFE_HOLD monitor). */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)",
        "#include <stdalign.h>",
        "#define LAB_FEET_ALIGN16 alignas(16)",
        "#elif defined(__GNUC__) || defined(__clang__)",
        "#define LAB_FEET_ALIGN16 __attribute__((aligned(16)))",
        "#elif defined(_MSC_VER)",
        "#define LAB_FEET_ALIGN16 __declspec(align(16))",
        "#else",
        "#define LAB_FEET_ALIGN16",
        "#endif",
        "",
        f"#define {prefix.upper()}_INPUT_DIM {d_in}",
        f"#define {prefix.upper()}_HIDDEN_DIM {d_hid}",
        f"#define {prefix.upper()}_OUTPUT_DIM {d_out}",
        f"#define LAB_FEET_SCRATCH_BYTES {scratch_bytes}",
        "#define LAB_FEET_SAFE_HOLD (-1)",
        "",
        f"LAB_FEET_ALIGN16 static const uint8_t {prefix}_w1_packed[{clf.w1_packed.size}] = {{",
        _format_c_int_array(clf.w1_packed, hex_u8=True),
        "};",
        f"LAB_FEET_ALIGN16 static const int32_t {prefix}_b1_i32[{d_hid}] = {{",
        _format_c_int_array(clf.b1_i32),
        "};",
        f"LAB_FEET_ALIGN16 static const int16_t {prefix}_h1_mult_i16[{d_hid}] = {{",
        _format_c_int_array(clf.h1_requant_mult_i16),
        "};",
        f"LAB_FEET_ALIGN16 static const uint8_t {prefix}_w2_packed[{clf.w2_packed.size}] = {{",
        _format_c_int_array(clf.w2_packed, hex_u8=True),
        "};",
        f"LAB_FEET_ALIGN16 static const int32_t {prefix}_b2_i32[{d_out}] = {{",
        _format_c_int_array(clf.b2_i32),
        "};",
        f"LAB_FEET_ALIGN16 static const int16_t {prefix}_logit_mult_i16[{d_out}] = {{",
        _format_c_int_array(clf.logit_mult_i16),
        "};",
        f"LAB_FEET_ALIGN16 static const int8_t {prefix}_centroids_i8[{clf.class_centroids_i8.size}] = {{",
        _format_c_int_array(clf.class_centroids_i8),
        "};",
        f"LAB_FEET_ALIGN16 static const uint8_t {prefix}_inv_mad_u8[{d_in}] = {{",
        _format_c_int_array(clf.inv_mad_u8),
        "};",
        f"LAB_FEET_ALIGN16 static const uint16_t {prefix}_l1_thresh_u16[{d_out}] = {{",
        _format_c_int_array(clf.class_l1_thresh_u16),
        "};",
        f"LAB_FEET_ALIGN16 static const uint16_t {prefix}_scalar_thresh_u16[4] = {{",
        _format_c_int_array(clf.scalar_thresholds_u16),
        "};",
        "",
        "static inline int8_t lab_feet_unpack_int4_nibble(const uint8_t *packed, size_t idx) {",
        "    uint8_t byte_val = packed[idx >> 1u];",
        "    uint8_t nib = (idx & 1u) ? ((uint8_t)(byte_val >> 4u) & 0x0Fu) : (byte_val & 0x0Fu);",
        "    return (nib >= 8u) ? (int8_t)((int)nib - 16) : (int8_t)nib;",
        "}",
        "",
        "static inline int32_t lab_feet_sat_add_i32(int64_t a, int64_t b) {",
        "    int64_t s = a + b;",
        "    if (s > 2147483647LL) return 2147483647;",
        "    if (s < (-2147483647LL - 1LL)) return (-2147483647 - 1);",
        "    return (int32_t)s;",
        "}",
        "",
        f"static inline int {prefix}_infer_drift_abstaining_int4(",
        "    const int8_t *x_q,",
        "    int32_t in_clips,",
        "    int32_t *out_logits_i32",
        ") {",
        f"    int8_t h1[{d_hid}];",
        "    int32_t int4_clips = 0;",
        "    size_t i, j;",
        f"    for (j = 0; j < {d_hid}u; ++j) {{",
        "        int64_t dot = 0;",
        f"        for (i = 0; i < {d_in}u; ++i) {{",
        f"            int8_t w = lab_feet_unpack_int4_nibble({prefix}_w1_packed, i * {d_hid}u + j);",
        "            dot += (int64_t)x_q[i] * (int64_t)w;",
        "        }",
        f"        int32_t acc = lab_feet_sat_add_i32(dot, (int64_t){prefix}_b1_i32[j]);",
        f"        int64_t prod = (int64_t)acc * (int64_t){prefix}_h1_mult_i16[j] + (1LL << {max(0, int(clf.h1_requant_shift) - 1)});",
        f"        int64_t scaled = prod >> {int(clf.h1_requant_shift)};",
        "        if (scaled > 7LL) { ++int4_clips; h1[j] = 7; }",
        "        else if (scaled < 0LL) { h1[j] = 0; }",
        "        else { h1[j] = (int8_t)scaled; }",
        "    }",
        "    int best_c = 0;",
        "    int32_t peak = (-2147483647 - 1);",
        "    int32_t runner_up = (-2147483647 - 1);",
        f"    for (j = 0; j < {d_out}u; ++j) {{",
        "        int64_t dot = 0;",
        f"        for (i = 0; i < {d_hid}u; ++i) {{",
        f"            int8_t w = lab_feet_unpack_int4_nibble({prefix}_w2_packed, i * {d_out}u + j);",
        "            dot += (int64_t)h1[i] * (int64_t)w;",
        "        }",
        f"        int32_t acc = lab_feet_sat_add_i32(dot, (int64_t){prefix}_b2_i32[j]);",
        f"        int64_t scaled = ((int64_t)acc * (int64_t){prefix}_logit_mult_i16[j] + 8LL) >> 4;",
        "        int32_t logit = lab_feet_sat_add_i32(scaled, 0);",
        "        if (out_logits_i32 != NULL) { out_logits_i32[j] = logit; }",
        "        if (j == 0u || logit > peak) { runner_up = peak; peak = logit; best_c = (int)j; }",
        "        else if (j == 1u || logit > runner_up) { runner_up = logit; }",
        "    }",
        "    int32_t margin = (peak > runner_up) ? (peak - runner_up) : 0;",
        "    int32_t l1_dist = 0;",
        f"    for (i = 0; i < {d_in}u; ++i) {{",
        f"        int32_t diff = (int32_t)x_q[i] - (int32_t){prefix}_centroids_i8[(size_t)best_c * {d_in}u + i];",
        "        if (diff < 0) diff = -diff;",
        f"        l1_dist += diff * (int32_t){prefix}_inv_mad_u8[i];",
        "    }",
        f"    if (l1_dist > (int32_t){prefix}_l1_thresh_u16[best_c] ||",
        f"        int4_clips > (int32_t){prefix}_scalar_thresh_u16[0] ||",
        f"        in_clips > (int32_t){prefix}_scalar_thresh_u16[1] ||",
        f"        margin < (int32_t){prefix}_scalar_thresh_u16[2] ||",
        f"        peak < (int32_t){prefix}_scalar_thresh_u16[3]) {{",
        "        return LAB_FEET_SAFE_HOLD;",
        "    }",
        "    return best_c;",
        "}",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


def export_c99_header(
    model: QuantizedModel | Any,
    output_path: str | Path | None = None,
    *,
    prefix: str = "lab_feet_model",
    input_scale: float | None = None,
    X_cal: np.ndarray | None = None,
) -> str:
    """Export `QuantizedModel` (or `DriftAbstainingInt4Classifier`) to a freestanding C99 `.h` file."""
    c_prefix = _sanitize_c_ident(prefix)
    guard = f"{c_prefix.upper()}_WEIGHTS_H_"

    if not isinstance(model, QuantizedModel) and hasattr(model, "w1_packed") and hasattr(model, "class_centroids_i8"):
        header_text = _export_drift_classifier_c99(model, c_prefix, guard)
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(header_text, encoding="utf-8")
        return header_text

    if not isinstance(model, QuantizedModel):
        raise TypeError(
            f"export_c99_header expects a QuantizedModel or DriftAbstainingInt4Classifier, got {type(model).__name__}."
        )

    spec = prepare_fixed_point_model(model, input_scale=input_scale, X_cal=X_cal)
    n_layers = len(spec["layers"])
    slot_bytes = int(spec["aligned_slot_bytes"])
    scratch_bytes = int(spec["scratch_bytes"])
    act_is_relu = 1 if spec["activation"] == "relu" else 0

    lines: list[str] = [
        "/* Auto-generated by lab-feet freestanding C99 / TFLite-Micro static weight exporter. */",
        f"/* Input scale: {spec['input_scale']:.10e} | Output logit scale: {spec['logit_scale']:.10e} */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)",
        "#include <stdalign.h>",
        "#define LAB_FEET_ALIGN16 alignas(16)",
        "#elif defined(__GNUC__) || defined(__clang__)",
        "#define LAB_FEET_ALIGN16 __attribute__((aligned(16)))",
        "#elif defined(_MSC_VER)",
        "#define LAB_FEET_ALIGN16 __declspec(align(16))",
        "#else",
        "#define LAB_FEET_ALIGN16",
        "#endif",
        "",
        f"#define {c_prefix.upper()}_NUM_LAYERS {n_layers}",
        f"#define {c_prefix.upper()}_INPUT_DIM {spec['input_dim']}",
        f"#define {c_prefix.upper()}_OUTPUT_DIM {spec['output_dim']}",
        f"#define {c_prefix.upper()}_ARENA_SLOT_BYTES {slot_bytes}",
        f"#define LAB_FEET_SCRATCH_BYTES {scratch_bytes}",
        "#define LAB_FEET_SAFE_HOLD (-1)",
        "",
    ]

    for lspec in spec["layers"]:
        li = lspec["index"]
        d_in = lspec["d_in"]
        d_out = lspec["d_out"]
        bits = lspec["bits"]
        pw = lspec["packed_weights"]
        if bits == 4:
            lines.extend(
                [
                    f"LAB_FEET_ALIGN16 static const uint8_t {c_prefix}_l{li}_weights[{max(1, pw.size)}] = {{",
                    _format_c_int_array(pw, hex_u8=True),
                    "};",
                ]
            )
        else:
            lines.extend(
                [
                    f"LAB_FEET_ALIGN16 static const int8_t {c_prefix}_l{li}_weights[{max(1, pw.size)}] = {{",
                    _format_c_int_array(pw, hex_u8=False),
                    "};",
                ]
            )
        lines.extend(
            [
                f"LAB_FEET_ALIGN16 static const int8_t {c_prefix}_l{li}_zp[{max(1, d_out)}] = {{",
                _format_c_int_array(lspec["zp_i8"]),
                "};",
                f"LAB_FEET_ALIGN16 static const int32_t {c_prefix}_l{li}_bias[{max(1, d_out)}] = {{",
                _format_c_int_array(lspec["bias_i32"]),
                "};",
                f"LAB_FEET_ALIGN16 static const int32_t {c_prefix}_l{li}_mult[{max(1, d_out)}] = {{",
                _format_c_int_array(lspec["mult_i32"]),
                "};",
                f"LAB_FEET_ALIGN16 static const int32_t {c_prefix}_l{li}_shift[{max(1, d_out)}] = {{",
                _format_c_int_array(lspec["shift_i32"]),
                "};",
                "",
            ]
        )

    lines.extend(
        [
            "static inline int8_t lab_feet_unpack_int4_at(const uint8_t *packed, size_t idx) {",
            "    uint8_t byte_val = packed[idx >> 1u];",
            "    uint8_t nib = (idx & 1u) ? ((uint8_t)(byte_val >> 4u) & 0x0Fu) : (byte_val & 0x0Fu);",
            "    return (nib >= 8u) ? (int8_t)((int)nib - 16) : (int8_t)nib;",
            "}",
            "",
            "static inline int32_t lab_feet_requant_saturate_i64(",
            "    int32_t acc,",
            "    int32_t mult,",
            "    int32_t shift,",
            "    int32_t qmin,",
            "    int32_t qmax",
            ") {",
            "    int64_t round_off = (shift > 0) ? (1LL << (shift - 1)) : 0LL;",
            "    int64_t scaled = ((int64_t)acc * (int64_t)mult + round_off) >> shift;",
            "    if (scaled < (int64_t)qmin) return qmin;",
            "    if (scaled > (int64_t)qmax) return qmax;",
            "    return (int32_t)scaled;",
            "}",
            "",
            "static inline int lab_feet_decide_drift_abstaining_int4(",
            "    int argmax_class,",
            "    int32_t l1_score,",
            "    int32_t l1_thresh,",
            "    int32_t int4_clips,",
            "    int32_t max_int4_clips,",
            "    int32_t int8_input_clips,",
            "    int32_t max_input_clips,",
            "    int32_t energy_margin,",
            "    int32_t min_energy_margin,",
            "    int32_t peak_logit,",
            "    int32_t min_peak_logit",
            ") {",
            "    if (l1_score > l1_thresh ||",
            "        int4_clips > max_int4_clips ||",
            "        int8_input_clips > max_input_clips ||",
            "        energy_margin < min_energy_margin ||",
            "        peak_logit < min_peak_logit) {",
            "        return LAB_FEET_SAFE_HOLD;",
            "    }",
            "    return argmax_class;",
            "}",
            "",
            "static inline void lab_feet_dense_layer_fixed(",
            "    const int8_t *in_q,",
            "    size_t d_in,",
            "    size_t d_out,",
            "    int bits,",
            "    const void *weights,",
            "    const int8_t *zp,",
            "    const int32_t *bias_i32,",
            "    const int32_t *mult_i32,",
            "    const int32_t *shift_i32,",
            "    int is_hidden,",
            "    int act_is_relu,",
            "    int8_t *out_hidden_q,",
            "    int32_t *out_logits_i32",
            ") {",
            "    const uint8_t *w_u8 = (const uint8_t *)weights;",
            "    const int8_t *w_i8 = (const int8_t *)weights;",
            "    size_t i, j;",
            "    for (j = 0; j < d_out; ++j) {",
            "        int64_t dot = 0;",
            "        int32_t zp_j = (int32_t)zp[j];",
            "        for (i = 0; i < d_in; ++i) {",
            "            int32_t w_code = (bits == 4)",
            "                ? (int32_t)lab_feet_unpack_int4_at(w_u8, i * d_out + j)",
            "                : (int32_t)w_i8[i * d_out + j];",
            "            dot += (int64_t)in_q[i] * (int64_t)(w_code - zp_j);",
            "        }",
            "        int64_t sum = dot + (int64_t)bias_i32[j];",
            "        int32_t acc = (sum > 2147483647LL) ? 2147483647",
            "                    : ((sum < (-2147483647LL - 1LL)) ? (-2147483647 - 1) : (int32_t)sum);",
            "        if (is_hidden) {",
            "            int32_t qmin = act_is_relu ? 0 : -127;",
            "            out_hidden_q[j] = (int8_t)lab_feet_requant_saturate_i64(",
            "                acc, mult_i32[j], shift_i32[j], qmin, 127",
            "            );",
            "        } else {",
            "            out_logits_i32[j] = lab_feet_requant_saturate_i64(",
            "                acc, mult_i32[j], shift_i32[j], (-2147483647 - 1), 2147483647",
            "            );",
            "        }",
            "    }",
            "}",
            "",
            f"static inline int {c_prefix}_infer_fixed(",
            "    const int8_t *x_q,",
            "    int32_t *out_logits_i32,",
            "    int8_t *scratch_arena",
            ") {",
            f"    int8_t local_arena[LAB_FEET_SCRATCH_BYTES];",
            "    int8_t *arena = (scratch_arena != NULL) ? scratch_arena : local_arena;",
            "    int8_t *buf_a = arena;",
            f"    int8_t *buf_b = arena + {slot_bytes};",
            "    const int8_t *cur_in = x_q;",
            "    int8_t *cur_out = buf_a;",
            "    size_t j;",
        ]
    )

    for lspec in spec["layers"]:
        li = lspec["index"]
        d_in = lspec["d_in"]
        d_out = lspec["d_out"]
        bits = lspec["bits"]
        if lspec["is_hidden"]:
            lines.extend(
                [
                    "    lab_feet_dense_layer_fixed(",
                    f"        cur_in, {d_in}u, {d_out}u, {bits},",
                    f"        (const void *){c_prefix}_l{li}_weights, {c_prefix}_l{li}_zp,",
                    f"        {c_prefix}_l{li}_bias, {c_prefix}_l{li}_mult, {c_prefix}_l{li}_shift,",
                    f"        1, {act_is_relu}, cur_out, NULL",
                    "    );",
                    "    cur_in = cur_out;",
                    "    cur_out = (cur_out == buf_a) ? buf_b : buf_a;",
                ]
            )
        else:
            lines.extend(
                [
                    "    lab_feet_dense_layer_fixed(",
                    f"        cur_in, {d_in}u, {d_out}u, {bits},",
                    f"        (const void *){c_prefix}_l{li}_weights, {c_prefix}_l{li}_zp,",
                    f"        {c_prefix}_l{li}_bias, {c_prefix}_l{li}_mult, {c_prefix}_l{li}_shift,",
                    "        0, 0, NULL, out_logits_i32",
                    "    );",
                ]
            )

    lines.extend(
        [
            "    int best_idx = 0;",
            "    int32_t best_val = out_logits_i32[0];",
            f"    for (j = 1u; j < {spec['output_dim']}u; ++j) {{",
            "        if (out_logits_i32[j] > best_val) {",
            "            best_val = out_logits_i32[j];",
            "            best_idx = (int)j;",
            "        }",
            "    }",
            "    return best_idx;",
            "}",
            "",
            "static inline int lab_feet_infer_w4a8(",
            "    const int8_t *x_q,",
            "    int32_t *out_logits_i32,",
            "    int8_t *scratch_arena",
            ") {",
            f"    return {c_prefix}_infer_fixed(x_q, out_logits_i32, scratch_arena);",
            "}",
            "",
            f"#endif /* {guard} */",
            "",
        ]
    )

    header_text = "\n".join(lines)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(header_text, encoding="utf-8")
    return header_text
