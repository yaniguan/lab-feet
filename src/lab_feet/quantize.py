"""Symmetric/asymmetric per-tensor & per-channel quantization, 4-bit packing, MSE clipping, and W8A8 matmul."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np

from .runtime import _apply_hidden_activation, forward

QuantMode = Literal["symmetric", "symmetric_full", "asymmetric"]
ClipSearchMode = Literal["none", "weight_mse", "activation_mse", "percentile"]

VALID_MODES = ("symmetric", "symmetric_full", "asymmetric")
VALID_CLIP_SEARCH = ("none", "weight_mse", "activation_mse", "percentile")

INT32_MIN: int = -(1 << 31)
INT32_MAX: int = (1 << 31) - 1
UINT32_MAX: int = (1 << 32) - 1


def _validate_bits(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)):
        raise ValueError(f"bits must be an integer in [2, 8], got {bits!r}.")
    b = int(bits)
    if not (2 <= b <= 8):
        raise ValueError(f"bits must be in [2, 8] for int8 storage, got {b}.")
    return b


def max_safe_int32_fan_in(
    act_bits: int = 8,
    weight_bits: int = 8,
    *,
    asymmetric: bool = False,
    unsigned_act: bool = False,
    symmetric_full: bool = False,
) -> int:
    """Return worst-case inner dimension `K_max` guaranteed not to overflow signed `int32` MACs.

    For standard symmetric `W8A8` (`act_qmax = 127`, `w_qmax = 127`), verifies
    `fan_in * 127 * 127 <= INT32_MAX` (`K_max = 133,144`). For asymmetric `W8A8`
    (`|w_q - zp| <= 255`), `K_max = 2,147,483,647 // (127 * 255) = 66,311`.
    """
    ab = _validate_bits(act_bits)
    wb = _validate_bits(weight_bits)
    act_qmax = ((1 << ab) - 1) if unsigned_act else ((1 << (ab - 1)) - 1)
    if asymmetric:
        w_qmax = (1 << wb) - 1
    elif symmetric_full:
        w_qmax = 1 << (wb - 1)
    else:
        w_qmax = (1 << (wb - 1)) - 1
    worst_prod = max(1, int(act_qmax) * int(w_qmax))
    return int(INT32_MAX // worst_prod)


def verify_int32_mac_bound(
    fan_in: int,
    act_qmax: int = 127,
    weight_qmax: int = 127,
) -> bool:
    """Verify `fan_in * act_qmax * weight_qmax <= INT32_MAX` without 32-bit overflow."""
    if fan_in < 0 or act_qmax < 0 or weight_qmax < 0:
        raise ValueError("fan_in, act_qmax, and weight_qmax must be non-negative.")
    return int(fan_in) * int(act_qmax) * int(weight_qmax) <= INT32_MAX


def pack_4bit(
    q: np.ndarray,
    *,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Pack signed two's-complement 4-bit integer codes [-8, 7] two per uint8 byte.

    For odd-length arrays (`orig_len % 2 == 1`), the final byte stores the last element in its
    low nibble (`bits 0..3`) and canonical zero padding (`0x0`) in its high nibble (`bits 4..7`).
    When `return_metadata=True`, also returns `{"orig_len": int, "orig_shape": tuple[int, ...]}`.
    """
    arr = np.asarray(q)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"pack_4bit requires an integer array, got dtype={arr.dtype}.")
    meta = {"orig_len": int(arr.size), "orig_shape": tuple(int(d) for d in arr.shape)}
    if arr.size == 0:
        empty = np.empty(0, dtype=np.uint8)
        return (empty, meta) if return_metadata else empty

    q_min = int(arr.min())
    q_max = int(arr.max())
    if q_min < -8 or q_max > 7:
        raise ValueError(
            f"pack_4bit requires signed 4-bit codes in [-8, 7], got range [{q_min}, {q_max}]."
        )

    flat_u8 = (arr.ravel().astype(np.int16) & 0x0F).astype(np.uint8)
    n = flat_u8.size
    if n % 2 == 1:
        padded = np.zeros(n + 1, dtype=np.uint8)
        padded[:n] = flat_u8
    else:
        padded = flat_u8
    packed = (padded[0::2] | (padded[1::2] << 4)).astype(np.uint8)
    if return_metadata:
        return packed, meta
    return packed


def unpack_4bit(
    packed: np.ndarray,
    shape: tuple[int, ...] | list[int] | int,
    *,
    strict_padding: bool = True,
) -> np.ndarray:
    """Unpack uint8 byte buffer of two's-complement nibbles back into int8 tensor of `shape`.

    Accepts either an integer `orig_len` (producing a 1D array `(orig_len,)`) or a `shape` tuple,
    enforces exact byte count `packed.size == (n_elems + 1) // 2`, and verifies canonical zero
    high-nibble padding on odd element counts when `strict_padding=True`.
    """
    raw = np.asarray(packed)
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"unpack_4bit requires an integer/byte buffer, got dtype={raw.dtype}.")

    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        shape_tuple: tuple[int, ...] = (int(shape),)
    else:
        shape_tuple = tuple(int(d) for d in shape)  # type: ignore[arg-type]

    if any(d < 0 for d in shape_tuple):
        raise ValueError(f"Negative dimension in shape {shape_tuple}.")
    n_elems = int(np.prod(shape_tuple, dtype=np.int64)) if shape_tuple else 1

    # View/convert raw integer buffer cleanly as uint8 so signed int8 byte buffers never sign-extend
    packed_u8 = raw.astype(np.uint8, copy=False).ravel()
    expected_bytes = (n_elems + 1) // 2 if n_elems > 0 else 0

    if packed_u8.size < expected_bytes:
        raise ValueError(
            f"Packed buffer of {packed_u8.size} bytes ({packed_u8.size * 2} nibbles) "
            f"is too small for shape {shape_tuple} ({n_elems} elements, expected {expected_bytes} bytes)."
        )
    if packed_u8.size > expected_bytes:
        raise ValueError(
            f"Packed buffer of {packed_u8.size} bytes exceeds expected {expected_bytes} bytes "
            f"for shape {shape_tuple} ({n_elems} elements)."
        )
    if n_elems == 0:
        return np.zeros(shape_tuple, dtype=np.int8)

    if strict_padding and (n_elems % 2 == 1):
        pad_nibble = (int(packed_u8[-1]) >> 4) & 0x0F
        if pad_nibble != 0:
            raise ValueError(
                f"Non-zero high padding nibble 0x{pad_nibble:X} in odd-length 4-bit buffer "
                f"(n_elems={n_elems})."
            )

    lo = (packed_u8 & 0x0F).astype(np.int8)
    hi = ((packed_u8 >> 4) & 0x0F).astype(np.int8)
    unpacked = np.empty(packed_u8.size * 2, dtype=np.int8)
    unpacked[0::2] = lo
    unpacked[1::2] = hi
    signed = np.where(unpacked > 7, unpacked - 16, unpacked).astype(np.int8)
    return signed[:n_elems].reshape(shape_tuple)


def smoothquant_migrate(
    X: np.ndarray,
    W: np.ndarray,
    *,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Migrate per-channel activation outlier difficulty into weights via SmoothQuant `diag(s)`.

    Computes `s_j = clamp(max(|X_j|)^alpha / max(|W_j|)^(1 - alpha), eps, 1/eps)` so
    `(X / s) @ (s[:, None] * W) == X @ W` while balancing dynamic range across activations and weights.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"SmoothQuant alpha must lie in [0, 1], got {alpha}.")
    X_f32 = np.asarray(X, dtype=np.float32)
    W_f32 = np.asarray(W, dtype=np.float32)
    if W_f32.ndim != 2 or X_f32.shape[-1] != W_f32.shape[0]:
        raise ValueError(
            f"Incompatible shapes for SmoothQuant migration: X={X_f32.shape}, W={W_f32.shape}."
        )
    x_2d = X_f32.reshape(-1, X_f32.shape[-1])
    act_max = np.maximum(np.max(np.abs(x_2d), axis=0), eps)
    w_max = np.maximum(np.max(np.abs(W_f32), axis=1), eps)
    scales = np.clip((act_max**alpha) / (w_max ** (1.0 - alpha)), eps, 1.0 / eps).astype(np.float32)
    X_smooth = (X_f32 / scales).astype(np.float32)
    W_smooth = (scales[:, None] * W_f32).astype(np.float32)
    return X_smooth, W_smooth, scales


def _quantize_given_alpha(
    W_f32: np.ndarray,
    bits: int,
    *,
    per_channel: bool,
    mode: QuantMode,
    alpha: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (q_int8, scale_f32, zp_int8) for a given clipping multiplier alpha in (0, 1]."""
    reduce_axis = 0 if (per_channel and W_f32.ndim >= 2) else None
    keepdims = reduce_axis is not None

    if mode == "symmetric":
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
        amax = np.max(np.abs(W_f32), axis=reduce_axis, keepdims=keepdims) * alpha
        zero_mask = amax == 0.0
        scale = np.where(zero_mask, 1.0, amax / qmax).astype(np.float32)
        zp = np.zeros_like(scale, dtype=np.int8)
        q = np.where(zero_mask, 0, np.clip(np.round(W_f32 / scale), qmin, qmax)).astype(np.int8)
        return q, scale, zp

    if mode == "symmetric_full":
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
        pos_max = np.max(np.maximum(W_f32, 0.0), axis=reduce_axis, keepdims=keepdims) * alpha
        neg_max = np.max(np.maximum(-W_f32, 0.0), axis=reduce_axis, keepdims=keepdims) * alpha
        raw_scale = np.maximum(pos_max / qmax, neg_max / (-qmin))
        zero_mask = raw_scale == 0.0
        scale = np.where(zero_mask, 1.0, raw_scale).astype(np.float32)
        zp = np.zeros_like(scale, dtype=np.int8)
        q = np.where(zero_mask, 0, np.clip(np.round(W_f32 / scale), qmin, qmax)).astype(np.int8)
        return q, scale, zp

    if mode == "asymmetric":
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
        qspan = float(qmax - qmin)
        w_min = np.minimum(np.min(W_f32, axis=reduce_axis, keepdims=keepdims), 0.0) * alpha
        w_max = np.maximum(np.max(W_f32, axis=reduce_axis, keepdims=keepdims), 0.0) * alpha
        span = w_max - w_min
        zero_mask = span == 0.0
        scale = np.where(zero_mask, 1.0, span / qspan).astype(np.float32)
        raw_zp = np.clip(np.round(qmin - w_min / scale), qmin, qmax)
        zp = np.where(zero_mask, 0, raw_zp).astype(np.int8)
        q = np.where(
            zero_mask,
            0,
            np.clip(np.round(W_f32 / scale + zp.astype(np.float32)), qmin, qmax),
        ).astype(np.int8)
        return q, scale, zp

    raise ValueError(f"Unsupported quantization mode {mode!r}; expected one of {VALID_MODES}.")


def _build_alpha_candidates(
    W_f32: np.ndarray,
    alpha_min: float,
    grid_steps: int,
) -> np.ndarray:
    """Build candidate clipping factors in `[eff_alpha_min, 1.0]`, extending adaptively on heavy tails."""
    base_grid = np.linspace(alpha_min, 1.0, grid_steps, dtype=np.float32)
    abs_w = np.abs(W_f32.astype(np.float64))
    amax = float(np.max(abs_w))
    if amax <= 0.0:
        return base_grid
    rms = float(np.sqrt(np.mean(np.square(abs_w))))
    if rms > 0.0 and (amax / rms) > 6.0:
        adaptive_floor = float(min(alpha_min, max(3.0 * rms / amax, 0.03)))
        if adaptive_floor < alpha_min:
            tail_grid = np.geomspace(adaptive_floor, alpha_min, 24, dtype=np.float32)
            return np.unique(np.concatenate([tail_grid, base_grid])).astype(np.float32)
    return base_grid


def quantize(
    W: np.ndarray,
    bits: int = 8,
    *,
    per_channel: bool = False,
    mode: QuantMode = "symmetric",
    clip_search: ClipSearchMode = "none",
    X_cal: np.ndarray | None = None,
    alpha_min: float = 0.25,
    grid_steps: int = 31,
    percentile: float = 99.5,
    return_zero_point: bool | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize weight tensor `W` into int8 codes with zero-division guard and optional MSE clip search.

    Returns:
      - `(q, scale)` for symmetric / symmetric_full modes (unless `return_zero_point=True`), where
        `scale` is `np.float32` scalar when `per_channel=False` or shape `(1, d_out)` when `per_channel=True`.
      - `(q, scale, zero_point)` when `mode == 'asymmetric'` or `return_zero_point=True`.
    """
    b = _validate_bits(bits)
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {VALID_MODES}.")
    if clip_search not in VALID_CLIP_SEARCH:
        raise ValueError(f"Unsupported clip_search {clip_search!r}; expected one of {VALID_CLIP_SEARCH}.")
    if clip_search == "activation_mse" and X_cal is None:
        raise ValueError("X_cal calibration activations must be provided when clip_search='activation_mse'.")

    W_f32 = np.asarray(W, dtype=np.float32)
    emit_zp = (mode == "asymmetric") if return_zero_point is None else bool(return_zero_point)

    if W_f32.size == 0:
        empty_q = np.zeros(W_f32.shape, dtype=np.int8)
        s_out = (
            np.ones((1, W_f32.shape[1]), dtype=np.float32)
            if (per_channel and W_f32.ndim >= 2)
            else np.float32(1.0)
        )
        if emit_zp:
            zp_out = np.zeros_like(s_out, dtype=np.int8)
            return empty_q, s_out, zp_out
        return empty_q, s_out

    if clip_search == "none":
        q, scale, zp = _quantize_given_alpha(
            W_f32, b, per_channel=per_channel, mode=mode, alpha=1.0
        )
    elif clip_search == "percentile":
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"percentile must lie in (0, 100], got {percentile}.")
        reduce_axis = 0 if (per_channel and W_f32.ndim >= 2) else None
        keepdims = reduce_axis is not None
        amax = np.max(np.abs(W_f32), axis=reduce_axis, keepdims=keepdims)
        pct_val = np.percentile(np.abs(W_f32), percentile, axis=reduce_axis, keepdims=keepdims)
        alpha_pct = np.where(amax > 0.0, np.clip(pct_val / np.maximum(amax, 1e-12), 0.01, 1.0), 1.0)
        q, scale, zp = _quantize_given_alpha(
            W_f32, b, per_channel=per_channel, mode=mode, alpha=alpha_pct.astype(np.float32)
        )
    else:
        if grid_steps < 2 or not (0.0 < alpha_min <= 1.0):
            raise ValueError("Require grid_steps >= 2 and 0 < alpha_min <= 1.0.")
        alphas = _build_alpha_candidates(W_f32, alpha_min, grid_steps)
        X_cal_f32 = np.asarray(X_cal, dtype=np.float32) if X_cal is not None else None
        Y_ref = (X_cal_f32 @ W_f32) if (clip_search == "activation_mse" and X_cal_f32 is not None) else None

        if per_channel and W_f32.ndim >= 2:
            d_out = W_f32.shape[1]
            best_err = np.full((1, d_out), np.inf, dtype=np.float64)
            best_q = np.zeros_like(W_f32, dtype=np.int8)
            best_s = np.ones((1, d_out), dtype=np.float32)
            best_zp = np.zeros((1, d_out), dtype=np.int8)
            for a in alphas:
                q_c, s_c, zp_c = _quantize_given_alpha(
                    W_f32, b, per_channel=True, mode=mode, alpha=float(a)
                )
                W_hat = (q_c.astype(np.float32) - zp_c.astype(np.float32)) * s_c
                if clip_search == "activation_mse" and X_cal_f32 is not None and Y_ref is not None:
                    diff = Y_ref - (X_cal_f32 @ W_hat)
                    err = np.sum(np.square(diff, dtype=np.float64), axis=0, keepdims=True)
                else:
                    diff = W_f32 - W_hat
                    err = np.sum(np.square(diff, dtype=np.float64), axis=0, keepdims=True)
                improved = err < best_err
                best_err = np.where(improved, err, best_err)
                best_s = np.where(improved, s_c, best_s).astype(np.float32)
                best_zp = np.where(improved, zp_c, best_zp).astype(np.int8)
                best_q = np.where(np.broadcast_to(improved, W_f32.shape), q_c, best_q).astype(np.int8)
            q, scale, zp = best_q, best_s, best_zp
        else:
            # Guard against single-element extreme spike crushing >95% of bulk weights to 0
            nonzero_frac = float(np.mean(W_f32 != 0.0))
            abs_w = np.abs(W_f32)
            amax_scalar = float(np.max(abs_w))
            p99_scalar = float(np.percentile(abs_w, 99.0))
            has_extreme_spike = (
                W_f32.size >= 16
                and nonzero_frac >= 0.5
                and p99_scalar > 0.0
                and amax_scalar > 6.0 * p99_scalar
            )

            best_scalar_err = float("inf")
            q, scale, zp = _quantize_given_alpha(
                W_f32, b, per_channel=False, mode=mode, alpha=1.0
            )
            # If extreme isolated spike causes catastrophic zero-collapse (>95% of weights crushed to 0),
            # evaluate MSE against winsorized bulk reference so normal weights retain resolution.
            eval_target = W_f32
            if (
                clip_search == "weight_mse"
                and has_extreme_spike
                and float(np.mean(q != 0)) < 0.05
            ):
                bulk_cap = float(2.5 * p99_scalar)
                eval_target = np.clip(W_f32, -bulk_cap, bulk_cap).astype(np.float32)
                # Ensure candidate grid reaches the bulk dynamic range
                bulk_alpha = float(np.clip(bulk_cap / amax_scalar, 0.01, 1.0))
                if bulk_alpha < float(alphas[0]):
                    extra = np.geomspace(max(bulk_alpha * 0.5, 0.01), float(alphas[0]), 16, dtype=np.float32)
                    alphas = np.unique(np.concatenate([extra, alphas])).astype(np.float32)

            for a in alphas:
                q_c, s_c, zp_c = _quantize_given_alpha(
                    W_f32, b, per_channel=False, mode=mode, alpha=float(a)
                )
                W_hat = (q_c.astype(np.float32) - zp_c.astype(np.float32)) * s_c
                if clip_search == "activation_mse" and X_cal_f32 is not None and Y_ref is not None:
                    scalar_err = float(np.sum(np.square(Y_ref - (X_cal_f32 @ W_hat), dtype=np.float64)))
                else:
                    scalar_err = float(np.sum(np.square(eval_target - W_hat, dtype=np.float64)))
                if scalar_err < best_scalar_err:
                    best_scalar_err = scalar_err
                    q, scale, zp = q_c, s_c, zp_c

    if not (per_channel and W_f32.ndim >= 2):
        scale_out = np.float32(float(np.asarray(scale).reshape(-1)[0]))
        zp_out = np.int8(int(np.asarray(zp).reshape(-1)[0]))
    else:
        scale_out = np.asarray(scale, dtype=np.float32)
        zp_out = np.asarray(zp, dtype=np.int8)

    if emit_zp:
        return q, scale_out, zp_out
    return q, scale_out


@dataclass
class QuantizedLayer:
    """Actual serialized quantized layer storing int8 or nibble-packed uint8 weights."""

    packed_weights: np.ndarray
    scale: np.ndarray
    bias: np.ndarray
    shape: tuple[int, int]
    bits: int
    per_channel: bool = False
    mode: QuantMode = "symmetric"
    zero_point: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_bits(self.bits)
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode {self.mode!r}; expected one of {VALID_MODES}.")
        if len(self.shape) != 2:
            raise ValueError(f"QuantizedLayer shape must be 2D `(d_in, d_out)`, got {self.shape!r}.")
        d_in, d_out = int(self.shape[0]), int(self.shape[1])
        if d_in < 0 or d_out < 0:
            raise ValueError(f"QuantizedLayer shape dimensions must be >= 0, got {(d_in, d_out)}.")
        if d_in > UINT32_MAX or d_out > UINT32_MAX or (d_in * d_out) > UINT32_MAX:
            raise ValueError(
                f"QuantizedLayer dimensions {(d_in, d_out)} exceed 32-bit LFT1 container bounds."
            )
        self.shape = (d_in, d_out)

        expected_dtype = np.uint8 if self.bits == 4 else np.int8
        self.packed_weights = np.ascontiguousarray(self.packed_weights, dtype=expected_dtype).ravel()
        n_elems = d_in * d_out
        expected_packed = (n_elems + 1) // 2 if (self.bits == 4 and n_elems > 0) else (0 if n_elems == 0 else n_elems)
        if self.packed_weights.size != expected_packed:
            raise ValueError(
                f"QuantizedLayer packed_weights size mismatch: got {self.packed_weights.size} elements "
                f"for shape {self.shape} at {self.bits}-bit (expected {expected_packed})."
            )
        if self.bits == 4 and (n_elems % 2 == 1):
            pad_nibble = (int(self.packed_weights[-1]) >> 4) & 0x0F
            if pad_nibble != 0:
                raise ValueError(
                    f"QuantizedLayer odd-element 4-bit buffer has non-zero high padding nibble 0x{pad_nibble:X}."
                )

        self.bias = np.ascontiguousarray(self.bias, dtype=np.float32).reshape(-1)
        if self.bias.size != d_out:
            raise ValueError(
                f"QuantizedLayer bias length {self.bias.size} does not match d_out={d_out}."
            )

        scale_arr = np.asarray(self.scale, dtype=np.float32)
        expected_scale_len = d_out if self.per_channel else 1
        if scale_arr.size != expected_scale_len:
            raise ValueError(
                f"QuantizedLayer scale size {scale_arr.size} does not match expected {expected_scale_len} "
                f"(per_channel={self.per_channel}, d_out={d_out})."
            )
        if self.per_channel:
            self.scale = np.ascontiguousarray(scale_arr.reshape(1, d_out), dtype=np.float32)
        else:
            self.scale = np.ascontiguousarray(scale_arr.reshape(1), dtype=np.float32)

        if self.zero_point is not None:
            zp_arr = np.asarray(self.zero_point, dtype=np.int8)
            if zp_arr.size != expected_scale_len:
                raise ValueError(
                    f"QuantizedLayer zero_point size {zp_arr.size} does not match expected {expected_scale_len}."
                )
            if self.per_channel:
                self.zero_point = np.ascontiguousarray(zp_arr.reshape(1, d_out), dtype=np.int8)
            else:
                self.zero_point = np.ascontiguousarray(zp_arr.reshape(1), dtype=np.int8)

    @property
    def orig_len(self) -> int:
        """Exact logical weight count `d_in * d_out` prior to 4-bit nibble padding."""
        return int(self.shape[0] * self.shape[1])

    @property
    def nbytes(self) -> int:
        """Exact byte footprint of stored packed_weights + fp32 bias + fp32 scale (+ optional zp)."""
        zp_bytes = 0 if self.zero_point is None else int(self.zero_point.nbytes)
        return int(self.packed_weights.nbytes + self.bias.nbytes + self.scale.nbytes + zp_bytes)

    @property
    def n_params(self) -> int:
        return int(self.shape[0] * self.shape[1] + self.bias.size)

    def unpack_codes(self) -> np.ndarray:
        """Return int8 weight code matrix of shape `self.shape`."""
        if self.bits == 4:
            return unpack_4bit(self.packed_weights, self.shape, strict_padding=True)
        return self.packed_weights.reshape(self.shape)

    def dequantize(self) -> tuple[np.ndarray, np.ndarray]:
        """Dequantize weights to float32 `(W_deq, bias)`."""
        q = self.unpack_codes().astype(np.float32)
        if self.zero_point is not None:
            q = q - self.zero_point.astype(np.float32)
        return (q * self.scale).astype(np.float32), self.bias

    def save_npz(self, path: str | Path) -> Path:
        """Persist this single QuantizedLayer to a compressed/uncompressed `.npz` file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "packed_weights": self.packed_weights,
            "scale": self.scale,
            "bias": self.bias,
            "shape": np.array(self.shape, dtype=np.int64),
            "orig_len": np.array([self.orig_len], dtype=np.int64),
            "bits": np.array([self.bits], dtype=np.int32),
            "per_channel": np.array([int(self.per_channel)], dtype=np.uint8),
            "mode": np.array([self.mode]),
        }
        if self.zero_point is not None:
            payload["zero_point"] = self.zero_point
        np.savez(out, **payload)
        return out if out.suffix == ".npz" else out.with_suffix(".npz")

    @classmethod
    def load_npz(cls, path: str | Path) -> "QuantizedLayer":
        with np.load(path, allow_pickle=False) as data:
            zp = data["zero_point"] if "zero_point" in data.files else None
            shape_arr = data["shape"]
            shape_tuple = (int(shape_arr[0]), int(shape_arr[1]))
            if "orig_len" in data.files:
                recorded_len = int(data["orig_len"][0])
                if recorded_len != shape_tuple[0] * shape_tuple[1]:
                    raise ValueError(
                        f"Corrupted QuantizedLayer .npz: orig_len={recorded_len} != shape product {shape_tuple}."
                    )
            return cls(
                packed_weights=data["packed_weights"],
                scale=data["scale"],
                bias=data["bias"],
                shape=shape_tuple,
                bits=int(data["bits"][0]),
                per_channel=bool(data["per_channel"][0]),
                mode=str(data["mode"][0]),  # type: ignore[arg-type]
                zero_point=zp,
            )


@dataclass
class QuantizedModel:
    """Multi-layer quantized MLP retaining genuine packed weight buffers and exact `.nbytes`."""

    layers: list[QuantizedLayer]
    activation: str = "relu"
    calibrated_act_clip_max: tuple[float, ...] | None = None

    @property
    def nbytes(self) -> int:
        return int(sum(layer.nbytes for layer in self.layers))

    @property
    def n_params(self) -> int:
        return int(sum(layer.n_params for layer in self.layers))

    def dequantize(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [layer.dequantize() for layer in self.layers]

    def forward_dequant(self, x: np.ndarray) -> np.ndarray:
        return forward(self.dequantize(), x, activation=self.activation)

    def forward_int8(
        self,
        x: np.ndarray,
        *,
        act_bits: int = 8,
        act_clip_alpha: float = 1.0,
        act_clip_percentile: float | None = None,
        act_clip_max: float | Sequence[float] | None = None,
        unsigned_post_relu: bool = False,
    ) -> np.ndarray:
        eff_clip = self.calibrated_act_clip_max if act_clip_max is None else act_clip_max
        return forward_int8(
            self,
            x,
            act_bits=act_bits,
            act_clip_alpha=act_clip_alpha,
            act_clip_percentile=act_clip_percentile,
            act_clip_max=eff_clip,
            unsigned_post_relu=unsigned_post_relu,
        )

    def __iter__(self) -> Iterator[object]:
        """Support legacy `deq_layers, nbytes = quantize_model(layers, bits)` tuple unpacking."""
        yield self.dequantize()
        yield self.nbytes

    def save_npz(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "n_layers": np.array([len(self.layers)], dtype=np.int32),
            "activation": np.array([self.activation]),
        }
        if self.calibrated_act_clip_max is not None:
            arrays["calibrated_act_clip_max"] = np.array(
                self.calibrated_act_clip_max, dtype=np.float32
            )
        for idx, layer in enumerate(self.layers):
            p = f"l{idx}_"
            arrays[p + "packed"] = layer.packed_weights
            arrays[p + "scale"] = layer.scale
            arrays[p + "bias"] = layer.bias
            arrays[p + "shape"] = np.array(layer.shape, dtype=np.int64)
            arrays[p + "orig_len"] = np.array([layer.orig_len], dtype=np.int64)
            arrays[p + "bits"] = np.array([layer.bits], dtype=np.int32)
            arrays[p + "per_channel"] = np.array([int(layer.per_channel)], dtype=np.uint8)
            arrays[p + "mode"] = np.array([layer.mode])
            if layer.zero_point is not None:
                arrays[p + "zp"] = layer.zero_point
        np.savez(out, **arrays)
        return out if out.suffix == ".npz" else out.with_suffix(".npz")

    @classmethod
    def load_npz(cls, path: str | Path) -> "QuantizedModel":
        with np.load(path, allow_pickle=False) as data:
            n_layers = int(data["n_layers"][0])
            act = str(data["activation"][0]) if "activation" in data.files else "relu"
            cal_clips: tuple[float, ...] | None = None
            if "calibrated_act_clip_max" in data.files:
                cal_clips = tuple(float(v) for v in data["calibrated_act_clip_max"])
            loaded_layers: list[QuantizedLayer] = []
            for idx in range(n_layers):
                p = f"l{idx}_"
                sh = data[p + "shape"]
                shape_tuple = (int(sh[0]), int(sh[1]))
                if (p + "orig_len") in data.files:
                    rec_len = int(data[p + "orig_len"][0])
                    if rec_len != shape_tuple[0] * shape_tuple[1]:
                        raise ValueError(
                            f"Corrupted QuantizedModel .npz at layer {idx}: orig_len={rec_len} != {shape_tuple}."
                        )
                zp = data[p + "zp"] if (p + "zp") in data.files else None
                loaded_layers.append(
                    QuantizedLayer(
                        packed_weights=data[p + "packed"],
                        scale=data[p + "scale"],
                        bias=data[p + "bias"],
                        shape=shape_tuple,
                        bits=int(data[p + "bits"][0]),
                        per_channel=bool(data[p + "per_channel"][0]),
                        mode=str(data[p + "mode"][0]),  # type: ignore[arg-type]
                        zero_point=zp,
                    )
                )
            return cls(
                layers=loaded_layers,
                activation=act,
                calibrated_act_clip_max=cal_clips,
            )


def quantize_model(
    layers: list[tuple[np.ndarray, np.ndarray]],
    bits: int = 8,
    *,
    per_channel: bool = False,
    mode: QuantMode = "symmetric",
    clip_search: ClipSearchMode = "none",
    X_cal: np.ndarray | None = None,
    activation: str = "relu",
    alpha_min: float = 0.25,
    grid_steps: int = 31,
    quantize_cal_activations: bool = True,
    cal_act_bits: int = 8,
    cal_act_clip_percentile: float | None = None,
    calibrate_static_act_clips: bool = False,
) -> QuantizedModel:
    """Quantize an MLP's weight matrices into a `QuantizedModel` with genuine packed buffers.

    When `X_cal` is provided and `quantize_cal_activations=True`, calibration activations `h_cal`
    are quantized consistently with `_quantize_activations_int8` before optimizing subsequent layers
    so multi-layer `activation_mse` search accounts for upstream activation quantization error.
    """
    b = _validate_bits(bits)
    q_layers: list[QuantizedLayer] = []
    h_cal = np.asarray(X_cal, dtype=np.float32) if X_cal is not None else None
    static_act_clips: list[float] = []

    for layer_idx, (W, bias) in enumerate(layers):
        W_f32 = np.asarray(W, dtype=np.float32)
        b_f32 = np.asarray(bias, dtype=np.float32)
        if W_f32.ndim != 2:
            raise ValueError(f"Expected 2D weight matrix at layer {layer_idx}, got shape {W_f32.shape}.")

        x_cal_for_layer = h_cal
        if h_cal is not None:
            if calibrate_static_act_clips:
                pct = 99.5 if cal_act_clip_percentile is None else float(cal_act_clip_percentile)
                static_act_clips.append(float(np.percentile(np.abs(h_cal), pct)))
            if quantize_cal_activations:
                is_nonneg_relu = layer_idx > 0 and activation == "relu" and bool(np.all(h_cal >= 0.0))
                hq_cal, sh_cal = _quantize_activations_int8(
                    h_cal,
                    act_bits=cal_act_bits,
                    act_clip_percentile=cal_act_clip_percentile,
                    unsigned=is_nonneg_relu,
                )
                x_cal_for_layer = (hq_cal.astype(np.float32) * sh_cal).astype(np.float32)

        q_res = quantize(
            W_f32,
            b,
            per_channel=per_channel,
            mode=mode,
            clip_search=clip_search,
            X_cal=x_cal_for_layer,
            alpha_min=alpha_min,
            grid_steps=grid_steps,
            return_zero_point=(mode == "asymmetric"),
        )
        if mode == "asymmetric":
            q_codes, scale, zp = q_res  # type: ignore[misc]
        else:
            q_codes, scale = q_res  # type: ignore[misc]
            zp = None

        packed = pack_4bit(q_codes) if b == 4 else np.ascontiguousarray(q_codes, dtype=np.int8)
        qlayer = QuantizedLayer(
            packed_weights=packed,  # type: ignore[arg-type]
            scale=np.asarray(scale, dtype=np.float32),
            bias=b_f32,
            shape=(W_f32.shape[0], W_f32.shape[1]),
            bits=b,
            per_channel=per_channel,
            mode=mode,
            zero_point=zp,
        )
        q_layers.append(qlayer)

        # Propagate calibration activations through the quantized hidden layer
        if h_cal is not None and layer_idx < len(layers) - 1:
            W_deq, b_deq = qlayer.dequantize()
            in_acts = x_cal_for_layer if (quantize_cal_activations and x_cal_for_layer is not None) else h_cal
            h_cal = _apply_hidden_activation(in_acts @ W_deq + b_deq, activation)

    return QuantizedModel(
        layers=q_layers,
        activation=activation,
        calibrated_act_clip_max=tuple(static_act_clips) if static_act_clips else None,
    )


def _quantize_activations_int8(
    x: np.ndarray,
    act_bits: int = 8,
    *,
    act_clip_alpha: float = 1.0,
    act_clip_percentile: float | None = None,
    act_clip_max: float | np.ndarray | None = None,
    unsigned: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample symmetric (`int8`) or non-negative unsigned (`uint8`) activation quantizer.

    Supports optional `act_clip_alpha` in `(0, 1]`, per-sample `act_clip_percentile` in `(0, 100]`,
    and static calibrated `act_clip_max` so isolated sensor/glare spikes cannot crush normal channels.
    """
    b = _validate_bits(act_bits)
    if not (0.0 < act_clip_alpha <= 1.0):
        raise ValueError(f"act_clip_alpha must lie in (0, 1], got {act_clip_alpha}.")
    if act_clip_percentile is not None and not (0.0 < act_clip_percentile <= 100.0):
        raise ValueError(f"act_clip_percentile must lie in (0, 100], got {act_clip_percentile}.")

    x_f32 = np.asarray(x, dtype=np.float32)
    if x_f32.ndim == 1:
        x_2d = x_f32.reshape(1, -1)
        squeeze = True
    else:
        x_2d = x_f32
        squeeze = False

    if unsigned:
        if x_2d.size > 0 and float(np.min(x_2d)) < -1e-6:
            raise ValueError(
                "unsigned=True activation quantization requires non-negative inputs (e.g., post-ReLU)."
            )
        x_work = np.maximum(x_2d, np.float32(0.0))
        qmin = 0
        qmax = (1 << b) - 1
        out_dtype = np.uint8
    else:
        x_work = x_2d
        qmax = (1 << (b - 1)) - 1
        qmin = -qmax
        out_dtype = np.int8

    abs_x = np.abs(x_work)
    row_max = np.max(abs_x, axis=1, keepdims=True)
    if act_clip_max is not None:
        clip_arr = np.asarray(act_clip_max, dtype=np.float32)
        if clip_arr.ndim == 0:
            amax = np.full((x_work.shape[0], 1), float(clip_arr) * act_clip_alpha, dtype=np.float32)
        else:
            amax = np.broadcast_to(clip_arr.reshape(-1, 1), (x_work.shape[0], 1)).astype(np.float32) * act_clip_alpha
    elif act_clip_percentile is not None and abs_x.shape[1] > 0:
        pct = np.percentile(abs_x, act_clip_percentile, axis=1, keepdims=True).astype(np.float32)
        # Fallback to row_max on sparse rows where the percentile lands on 0 despite nonzero entries
        base_amax = np.where((pct == 0.0) & (row_max > 0.0), row_max, pct)
        amax = (base_amax * act_clip_alpha).astype(np.float32)
    else:
        amax = (row_max * act_clip_alpha).astype(np.float32)

    zero_mask = (amax <= 0.0) | (row_max == 0.0)
    scale = np.where(zero_mask, 1.0, amax / float(qmax)).astype(np.float32)
    x_q = np.where(zero_mask, 0, np.clip(np.round(x_work / scale), qmin, qmax)).astype(out_dtype)
    if squeeze:
        return x_q.reshape(-1), scale.reshape(1)
    return x_q, scale


def _tiled_int32_matmul(
    h_q: np.ndarray,
    W_q: np.ndarray,
    zero_point: np.ndarray | None,
    *,
    k_max: int,
) -> np.ndarray:
    """Accumulate `h_q @ (W_q - zp)` with guaranteed overflow-free `int32` MAC tiles."""
    d_in = int(W_q.shape[0])
    if d_in <= k_max:
        acc_i32 = np.matmul(h_q.astype(np.int32), W_q.astype(np.int32))
        if zero_point is not None:
            row_sums_i32 = np.sum(h_q.astype(np.int32), axis=1, keepdims=True, dtype=np.int32)
            acc_i32 = acc_i32 - row_sums_i32 * zero_point.astype(np.int32)
        return acc_i32

    # Wide reduction (`d_in > K_max`): execute each K_max chunk strictly in `int32` without wraparound
    # and sum tile accumulators into widened `int64` outer accumulator.
    acc_i64 = np.zeros((h_q.shape[0], W_q.shape[1]), dtype=np.int64)
    zp_i32 = zero_point.astype(np.int32) if zero_point is not None else None
    for k0 in range(0, d_in, k_max):
        k1 = min(k0 + k_max, d_in)
        h_tile = h_q[:, k0:k1].astype(np.int32)
        w_tile = W_q[k0:k1, :].astype(np.int32)
        tile_acc_i32 = np.matmul(h_tile, w_tile)
        if zp_i32 is not None:
            tile_row_sums_i32 = np.sum(h_tile, axis=1, keepdims=True, dtype=np.int32)
            tile_acc_i32 = tile_acc_i32 - tile_row_sums_i32 * zp_i32
        acc_i64 += tile_acc_i32.astype(np.int64)
    return acc_i64


def forward_int8(
    qmodel: QuantizedModel | list[QuantizedLayer],
    x: np.ndarray,
    *,
    act_bits: int = 8,
    act_clip_alpha: float = 1.0,
    act_clip_percentile: float | None = None,
    act_clip_max: float | Sequence[float] | None = None,
    unsigned_post_relu: bool = False,
) -> np.ndarray:
    """Reference integer matmul inference (`W8A8` / `WbAa`) with verified `int32` MAC tile bounds."""
    if isinstance(qmodel, QuantizedModel):
        qlayers = qmodel.layers
        activation = qmodel.activation
        if act_clip_max is None and qmodel.calibrated_act_clip_max is not None:
            act_clip_max = qmodel.calibrated_act_clip_max
    else:
        qlayers = list(qmodel)
        activation = "relu"
    if not qlayers:
        raise ValueError("QuantizedModel must contain at least one layer.")

    x_arr = np.asarray(x, dtype=np.float32)
    was_1d = x_arr.ndim == 1
    h = x_arr.reshape(1, -1) if was_1d else x_arr

    per_layer_clips: Sequence[float] | None = None
    scalar_clip: float | None = None
    if act_clip_max is not None:
        if isinstance(act_clip_max, (int, float, np.floating, np.integer)):
            scalar_clip = float(act_clip_max)
        else:
            per_layer_clips = list(act_clip_max)
            if len(per_layer_clips) != len(qlayers):
                raise ValueError(
                    f"act_clip_max sequence length {len(per_layer_clips)} != {len(qlayers)} layers."
                )

    for idx, qlayer in enumerate(qlayers):
        use_unsigned = bool(unsigned_post_relu and idx > 0 and activation == "relu")
        layer_clip = per_layer_clips[idx] if per_layer_clips is not None else scalar_clip
        h_q, s_h = _quantize_activations_int8(
            h,
            act_bits=act_bits,
            act_clip_alpha=act_clip_alpha,
            act_clip_percentile=act_clip_percentile,
            act_clip_max=layer_clip,
            unsigned=use_unsigned,
        )
        W_q = qlayer.unpack_codes()  # int8 of shape (d_in, d_out)

        k_max = max_safe_int32_fan_in(
            act_bits=act_bits,
            weight_bits=qlayer.bits,
            asymmetric=(qlayer.zero_point is not None),
            unsigned_act=use_unsigned,
            symmetric_full=(qlayer.mode == "symmetric_full"),
        )
        acc = _tiled_int32_matmul(h_q, W_q, qlayer.zero_point, k_max=k_max)

        # Rescale accumulator by (activation_scale * weight_scale) + fp32 bias
        if acc.dtype == np.int64:
            out = (
                acc.astype(np.float64) * (s_h.astype(np.float64) * qlayer.scale.astype(np.float64))
                + qlayer.bias.astype(np.float64)
            ).astype(np.float32)
        else:
            out = acc.astype(np.float32) * (s_h * qlayer.scale) + qlayer.bias

        if idx < len(qlayers) - 1:
            h = _apply_hidden_activation(out, activation)
        else:
            h = out

    return h.reshape(-1) if was_1d else h
