"""Flat binary (`LFT1` GGUF-lite) serializer/deserializer verified against exact tensor byte counts."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .quantize import QuantizedLayer, QuantizedModel, QuantMode
from .runtime import SUPPORTED_ACTIVATIONS

BINARY_MAGIC = b"LFT1"
BINARY_VERSION = 1

# Global header: magic (4s), version (B), act_code (B), n_layers (H) -> 8 bytes
_GLOBAL_HDR = struct.Struct("<4sBBH")
# Per-layer header: d_in (I), d_out (I), bits (B), per_channel (B), mode_code (B), has_zp (B),
#                   scale_bytes (I), zp_bytes (I), bias_bytes (I), payload_bytes (I) -> 28 bytes
_LAYER_HDR = struct.Struct("<IIBBBBIIII")

_ACT_TO_CODE = {name: idx for idx, name in enumerate(SUPPORTED_ACTIVATIONS)}
_CODE_TO_ACT = {idx: name for name, idx in _ACT_TO_CODE.items()}

_MODE_TO_CODE: dict[str, int] = {"symmetric": 0, "symmetric_full": 1, "asymmetric": 2}
_CODE_TO_MODE: dict[int, QuantMode] = {
    0: "symmetric",
    1: "symmetric_full",
    2: "asymmetric",
}


def compute_binary_header_bytes(n_layers: int) -> int:
    """Return deterministic byte overhead of the `LFT1` container header (`8 + 28 * n_layers`)."""
    return _GLOBAL_HDR.size + n_layers * _LAYER_HDR.size


def to_binary_buffer(model: QuantizedModel) -> bytes:
    """Serialize `QuantizedModel` into a compact little-endian flat binary buffer (`LFT1`).

    Guarantees `len(blob) == compute_binary_header_bytes(len(model.layers)) + model.nbytes`.
    """
    if model.activation not in _ACT_TO_CODE:
        raise ValueError(f"Unsupported activation {model.activation!r} for binary serialization.")

    chunks: list[bytes] = [
        _GLOBAL_HDR.pack(
            BINARY_MAGIC,
            BINARY_VERSION,
            _ACT_TO_CODE[model.activation],
            len(model.layers),
        )
    ]

    for layer in model.layers:
        d_in, d_out = layer.shape
        bias_le = np.ascontiguousarray(layer.bias, dtype="<f4").tobytes()
        scale_le = np.ascontiguousarray(layer.scale, dtype="<f4").tobytes()
        has_zp = 1 if layer.zero_point is not None else 0
        zp_bytes = (
            np.ascontiguousarray(layer.zero_point, dtype=np.int8).tobytes()
            if layer.zero_point is not None
            else b""
        )
        payload_dtype = np.uint8 if layer.bits == 4 else np.int8
        payload_bytes = np.ascontiguousarray(layer.packed_weights, dtype=payload_dtype).tobytes()

        chunks.append(
            _LAYER_HDR.pack(
                d_in,
                d_out,
                layer.bits,
                int(layer.per_channel),
                _MODE_TO_CODE[layer.mode],
                has_zp,
                len(scale_le),
                len(zp_bytes),
                len(bias_le),
                len(payload_bytes),
            )
        )
        chunks.append(bias_le)
        chunks.append(scale_le)
        if zp_bytes:
            chunks.append(zp_bytes)
        chunks.append(payload_bytes)

    return b"".join(chunks)


def from_binary_buffer(blob: bytes | bytearray | memoryview) -> QuantizedModel:
    """Deserialize a `QuantizedModel` from an `LFT1` flat binary buffer with strict bounds checks."""
    mv = memoryview(blob)
    total = len(mv)
    if total < _GLOBAL_HDR.size:
        raise ValueError(f"Truncated LFT1 buffer: {total} bytes < {_GLOBAL_HDR.size}.")

    magic, version, act_code, n_layers = _GLOBAL_HDR.unpack_from(mv, 0)
    if magic != BINARY_MAGIC:
        raise ValueError(f"Invalid binary magic {magic!r}; expected {BINARY_MAGIC!r}.")
    if version != BINARY_VERSION:
        raise ValueError(f"Unsupported binary format version {version}.")
    if act_code not in _CODE_TO_ACT:
        raise ValueError(f"Unknown activation code {act_code}.")

    offset = _GLOBAL_HDR.size
    layers: list[QuantizedLayer] = []
    for idx in range(n_layers):
        if offset + _LAYER_HDR.size > total:
            raise ValueError(f"Truncated layer header at layer {idx}.")
        (
            d_in,
            d_out,
            bits,
            per_ch_flag,
            mode_code,
            has_zp,
            scale_nbytes,
            zp_nbytes,
            bias_nbytes,
            payload_nbytes,
        ) = _LAYER_HDR.unpack_from(mv, offset)
        offset += _LAYER_HDR.size

        if mode_code not in _CODE_TO_MODE:
            raise ValueError(f"Unknown quantization mode code {mode_code} at layer {idx}.")

        needed = bias_nbytes + scale_nbytes + zp_nbytes + payload_nbytes
        if offset + needed > total:
            raise ValueError(f"Truncated tensor payload at layer {idx}.")

        bias = np.frombuffer(mv[offset : offset + bias_nbytes], dtype="<f4").astype(np.float32, copy=True)
        offset += bias_nbytes

        scale = np.frombuffer(mv[offset : offset + scale_nbytes], dtype="<f4").astype(np.float32, copy=True)
        offset += scale_nbytes

        if has_zp:
            zp = np.frombuffer(mv[offset : offset + zp_nbytes], dtype=np.int8).copy()
            offset += zp_nbytes
        else:
            zp = None

        payload_dtype = np.uint8 if bits == 4 else np.int8
        packed = np.frombuffer(mv[offset : offset + payload_nbytes], dtype=payload_dtype).copy()
        offset += payload_nbytes

        layers.append(
            QuantizedLayer(
                packed_weights=packed,
                scale=scale,
                bias=bias,
                shape=(int(d_in), int(d_out)),
                bits=int(bits),
                per_channel=bool(per_ch_flag),
                mode=_CODE_TO_MODE[mode_code],
                zero_point=zp,
            )
        )

    if offset != total:
        raise ValueError(f"Trailing {total - offset} unparsed byte(s) in LFT1 buffer.")

    return QuantizedModel(layers=layers, activation=_CODE_TO_ACT[act_code])


def save_quantized_binary(model: QuantizedModel, path: str | Path) -> Path:
    """Write `QuantizedModel` in `LFT1` binary format to disk and verify exact file size."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = to_binary_buffer(model)
    out.write_bytes(blob)
    expected = compute_binary_header_bytes(len(model.layers)) + model.nbytes
    actual = out.stat().st_size
    if actual != expected:  # pragma: no cover
        raise RuntimeError(f"On-disk byte mismatch: wrote {actual} bytes, expected {expected}.")
    return out


def load_quantized_binary(path: str | Path) -> QuantizedModel:
    """Load `QuantizedModel` from an on-disk `LFT1` binary file."""
    return from_binary_buffer(Path(path).read_bytes())
