"""Symmetric/asymmetric per-tensor & per-channel quantization, 4-bit packing, MSE clipping, and W8A8 matmul."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from .runtime import _apply_hidden_activation, forward

QuantMode = Literal["symmetric", "symmetric_full", "asymmetric"]
ClipSearchMode = Literal["none", "weight_mse", "activation_mse"]

VALID_MODES = ("symmetric", "symmetric_full", "asymmetric")
VALID_CLIP_SEARCH = ("none", "weight_mse", "activation_mse")


def _validate_bits(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)):
        raise ValueError(f"bits must be an integer in [2, 8], got {bits!r}.")
    b = int(bits)
    if not (2 <= b <= 8):
        raise ValueError(f"bits must be in [2, 8] for int8 storage, got {b}.")
    return b


def pack_4bit(q: np.ndarray) -> np.ndarray:
    """Pack signed two's-complement 4-bit integer codes [-8, 7] two per uint8 byte."""
    arr = np.asarray(q)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"pack_4bit requires an integer array, got dtype={arr.dtype}.")
    if arr.size == 0:
        return np.empty(0, dtype=np.uint8)
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
    return (padded[0::2] | (padded[1::2] << 4)).astype(np.uint8)


def unpack_4bit(packed: np.ndarray, shape: tuple[int, ...] | list[int]) -> np.ndarray:
    """Unpack uint8 byte buffer of two's-complement nibbles back into int8 tensor of `shape`."""
    raw = np.asarray(packed)
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"unpack_4bit requires an integer/byte buffer, got dtype={raw.dtype}.")
    shape_tuple = tuple(int(d) for d in shape)
    if any(d < 0 for d in shape_tuple):
        raise ValueError(f"Negative dimension in shape {shape_tuple}.")
    n_elems = int(np.prod(shape_tuple, dtype=np.int64)) if shape_tuple else 1
    if n_elems == 0:
        return np.zeros(shape_tuple, dtype=np.int8)

    # View/convert raw integer buffer cleanly as uint8 so signed int8 byte buffers never sign-extend
    packed_u8 = raw.astype(np.uint8, copy=False).ravel()
    if packed_u8.size * 2 < n_elems:
        raise ValueError(
            f"Packed buffer of {packed_u8.size} bytes ({packed_u8.size * 2} nibbles) "
            f"is too small for shape {shape_tuple} ({n_elems} elements)."
        )

    lo = (packed_u8 & 0x0F).astype(np.int8)
    hi = ((packed_u8 >> 4) & 0x0F).astype(np.int8)
    unpacked = np.empty(packed_u8.size * 2, dtype=np.int8)
    unpacked[0::2] = lo
    unpacked[1::2] = hi
    signed = np.where(unpacked > 7, unpacked - 16, unpacked).astype(np.int8)
    return signed[:n_elems].reshape(shape_tuple)


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


def quantize(
    W: np.ndarray,
    bits: int = 8,
    *,
    per_channel: bool = False,
    mode: QuantMode = "symmetric",
    clip_search: ClipSearchMode = "none",
    X_cal: np.ndarray | None = None,
    alpha_min: float = 0.70,
    grid_steps: int = 31,
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
    else:
        if grid_steps < 2 or not (0.0 < alpha_min <= 1.0):
            raise ValueError("Require grid_steps >= 2 and 0 < alpha_min <= 1.0.")
        alphas = np.linspace(alpha_min, 1.0, grid_steps, dtype=np.float32)
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
            best_scalar_err = float("inf")
            q, scale, zp = _quantize_given_alpha(
                W_f32, b, per_channel=False, mode=mode, alpha=1.0
            )
            for a in alphas:
                q_c, s_c, zp_c = _quantize_given_alpha(
                    W_f32, b, per_channel=False, mode=mode, alpha=float(a)
                )
                W_hat = (q_c.astype(np.float32) - zp_c.astype(np.float32)) * s_c
                if clip_search == "activation_mse" and X_cal_f32 is not None and Y_ref is not None:
                    scalar_err = float(np.sum(np.square(Y_ref - (X_cal_f32 @ W_hat), dtype=np.float64)))
                else:
                    scalar_err = float(np.sum(np.square(W_f32 - W_hat, dtype=np.float64)))
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
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        expected_dtype = np.uint8 if self.bits == 4 else np.int8
        self.packed_weights = np.ascontiguousarray(self.packed_weights, dtype=expected_dtype)
        self.bias = np.ascontiguousarray(self.bias, dtype=np.float32).reshape(-1)
        scale_arr = np.asarray(self.scale, dtype=np.float32)
        if self.per_channel:
            self.scale = np.ascontiguousarray(scale_arr.reshape(1, self.shape[1]), dtype=np.float32)
        else:
            self.scale = np.ascontiguousarray(scale_arr.reshape(1), dtype=np.float32)
        if self.zero_point is not None:
            zp_arr = np.asarray(self.zero_point, dtype=np.int8)
            if self.per_channel:
                self.zero_point = np.ascontiguousarray(zp_arr.reshape(1, self.shape[1]), dtype=np.int8)
            else:
                self.zero_point = np.ascontiguousarray(zp_arr.reshape(1), dtype=np.int8)

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
            return unpack_4bit(self.packed_weights, self.shape)
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
            return cls(
                packed_weights=data["packed_weights"],
                scale=data["scale"],
                bias=data["bias"],
                shape=(int(shape_arr[0]), int(shape_arr[1])),
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

    def forward_int8(self, x: np.ndarray, *, act_bits: int = 8) -> np.ndarray:
        return forward_int8(self, x, act_bits=act_bits)

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
        for idx, layer in enumerate(self.layers):
            p = f"l{idx}_"
            arrays[p + "packed"] = layer.packed_weights
            arrays[p + "scale"] = layer.scale
            arrays[p + "bias"] = layer.bias
            arrays[p + "shape"] = np.array(layer.shape, dtype=np.int64)
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
            loaded_layers: list[QuantizedLayer] = []
            for idx in range(n_layers):
                p = f"l{idx}_"
                sh = data[p + "shape"]
                zp = data[p + "zp"] if (p + "zp") in data.files else None
                loaded_layers.append(
                    QuantizedLayer(
                        packed_weights=data[p + "packed"],
                        scale=data[p + "scale"],
                        bias=data[p + "bias"],
                        shape=(int(sh[0]), int(sh[1])),
                        bits=int(data[p + "bits"][0]),
                        per_channel=bool(data[p + "per_channel"][0]),
                        mode=str(data[p + "mode"][0]),  # type: ignore[arg-type]
                        zero_point=zp,
                    )
                )
            return cls(layers=loaded_layers, activation=act)


def quantize_model(
    layers: list[tuple[np.ndarray, np.ndarray]],
    bits: int = 8,
    *,
    per_channel: bool = False,
    mode: QuantMode = "symmetric",
    clip_search: ClipSearchMode = "none",
    X_cal: np.ndarray | None = None,
    activation: str = "relu",
) -> QuantizedModel:
    """Quantize an MLP's weight matrices into a `QuantizedModel` with genuine packed buffers.

    Note: `QuantizedModel` also implements `__iter__` yielding `(dequantized_layers, nbytes)`,
    preserving drop-in compatibility with `deq_layers, nbytes = quantize_model(layers, bits)`.
    """
    b = _validate_bits(bits)
    q_layers: list[QuantizedLayer] = []
    h_cal = np.asarray(X_cal, dtype=np.float32) if X_cal is not None else None

    for layer_idx, (W, bias) in enumerate(layers):
        W_f32 = np.asarray(W, dtype=np.float32)
        b_f32 = np.asarray(bias, dtype=np.float32)
        if W_f32.ndim != 2:
            raise ValueError(f"Expected 2D weight matrix at layer {layer_idx}, got shape {W_f32.shape}.")

        q_res = quantize(
            W_f32,
            b,
            per_channel=per_channel,
            mode=mode,
            clip_search=clip_search,
            X_cal=h_cal,
            return_zero_point=(mode == "asymmetric"),
        )
        if mode == "asymmetric":
            q_codes, scale, zp = q_res  # type: ignore[misc]
        else:
            q_codes, scale = q_res  # type: ignore[misc]
            zp = None

        packed = pack_4bit(q_codes) if b == 4 else np.ascontiguousarray(q_codes, dtype=np.int8)
        qlayer = QuantizedLayer(
            packed_weights=packed,
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
            h_cal = _apply_hidden_activation(h_cal @ W_deq + b_deq, activation)

    return QuantizedModel(layers=q_layers, activation=activation)


def _quantize_activations_int8(x: np.ndarray, act_bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic per-sample symmetric int8 activation quantizer with zero-row division guard."""
    b = _validate_bits(act_bits)
    qmax = (1 << (b - 1)) - 1
    x_f32 = np.asarray(x, dtype=np.float32)
    if x_f32.ndim == 1:
        x_2d = x_f32.reshape(1, -1)
        squeeze = True
    else:
        x_2d = x_f32
        squeeze = False
    amax = np.max(np.abs(x_2d), axis=1, keepdims=True)
    zero_mask = amax == 0.0
    scale = np.where(zero_mask, 1.0, amax / qmax).astype(np.float32)
    x_q = np.where(zero_mask, 0, np.clip(np.round(x_2d / scale), -qmax, qmax)).astype(np.int8)
    if squeeze:
        return x_q.reshape(-1), scale.reshape(1)
    return x_q, scale


def forward_int8(
    qmodel: QuantizedModel | list[QuantizedLayer],
    x: np.ndarray,
    *,
    act_bits: int = 8,
) -> np.ndarray:
    """Reference integer matmul inference (`W8A8` / `WbA8`) using `int32` accumulators.

    Quantizes each layer's input activations dynamically to `int8`, multiplies `int8` activations
    against unpacked `int8` weight codes in exact `int32` arithmetic (`dtype=np.int32`), applies
    any asymmetric weight zero-point correction in `int32`, and rescales to `float32` for bias + ReLU.
    """
    if isinstance(qmodel, QuantizedModel):
        qlayers = qmodel.layers
        activation = qmodel.activation
    else:
        qlayers = list(qmodel)
        activation = "relu"
    if not qlayers:
        raise ValueError("QuantizedModel must contain at least one layer.")

    x_arr = np.asarray(x, dtype=np.float32)
    was_1d = x_arr.ndim == 1
    h = x_arr.reshape(1, -1) if was_1d else x_arr

    for idx, qlayer in enumerate(qlayers):
        h_q, s_h = _quantize_activations_int8(h, act_bits=act_bits)
        W_q = qlayer.unpack_codes()  # int8 of shape (d_in, d_out)

        # True integer accumulation in int32
        acc_i32 = np.matmul(h_q.astype(np.int32), W_q.astype(np.int32))
        if qlayer.zero_point is not None:
            row_sums_i32 = np.sum(h_q.astype(np.int32), axis=1, keepdims=True)
            acc_i32 = acc_i32 - row_sums_i32 * qlayer.zero_point.astype(np.int32)

        # Rescale int32 accumulator by (activation_scale * weight_scale) + fp32 bias
        out = acc_i32.astype(np.float32) * (s_h * qlayer.scale) + qlayer.bias
        if idx < len(qlayers) - 1:
            h = _apply_hidden_activation(out, activation)
        else:
            h = out

    return h.reshape(-1) if was_1d else h

