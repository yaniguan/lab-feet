"""Flat binary (`LFT1` GGUF-lite) serializer/deserializer verified against exact tensor byte counts.

Binary layout invariants (`LFT1` version 1):
  - Global header (8 bytes, `<4sBBH`): magic `b"LFT1"`, version `1`, `act_code`, `n_layers`.
  - Per-layer header (28 bytes, `<IIBBBBIIII`): `d_in`, `d_out`, `bits`, `per_channel` (`0|1`),
    `mode_code`, `has_zp` (`0|1`), `scale_nbytes`, `zp_nbytes`, `bias_nbytes`, `payload_nbytes`.
  - Odd-element 4-bit nibble invariant: when `bits == 4`, `payload_nbytes == ceil(d_in * d_out / 2)`,
    and if `(d_in * d_out) % 2 == 1`, the unused high nibble `(payload[-1] >> 4) & 0x0F` MUST be `0`.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .quantize import UINT32_MAX, QuantizedLayer, QuantizedModel, QuantMode, _validate_bits
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
    if len(model.layers) > 0xFFFF:
        raise ValueError(f"Too many layers ({len(model.layers)}) for LFT1 uint16 layer counter.")

    chunks: list[bytes] = [
        _GLOBAL_HDR.pack(
            BINARY_MAGIC,
            BINARY_VERSION,
            _ACT_TO_CODE[model.activation],
            len(model.layers),
        )
    ]

    for idx, layer in enumerate(model.layers):
        d_in, d_out = layer.shape
        if d_in < 0 or d_out < 0 or d_in > UINT32_MAX or d_out > UINT32_MAX or (d_in * d_out) > UINT32_MAX:
            raise ValueError(f"Layer {idx} shape {(d_in, d_out)} exceeds LFT1 32-bit bounds.")

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

        n_elems = d_in * d_out
        expected_payload = (n_elems + 1) // 2 if (layer.bits == 4 and n_elems > 0) else (0 if n_elems == 0 else n_elems)
        if len(payload_bytes) != expected_payload:
            raise ValueError(
                f"Layer {idx} payload_bytes={len(payload_bytes)} != expected {expected_payload} for shape {(d_in, d_out)}."
            )
        if layer.bits == 4 and (n_elems % 2 == 1) and ((payload_bytes[-1] >> 4) & 0x0F) != 0:
            raise ValueError(f"Layer {idx} odd-length 4-bit payload has non-zero high padding nibble.")

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

        _validate_bits(int(bits))
        if per_ch_flag not in (0, 1):
            raise ValueError(f"Invalid per_channel flag {per_ch_flag} at layer {idx}; expected 0 or 1.")
        if has_zp not in (0, 1):
            raise ValueError(f"Invalid has_zp flag {has_zp} at layer {idx}; expected 0 or 1.")
        if mode_code not in _CODE_TO_MODE:
            raise ValueError(f"Unknown quantization mode code {mode_code} at layer {idx}.")

        n_elems = int(d_in) * int(d_out)
        if n_elems > UINT32_MAX:
            raise ValueError(
                f"Layer {idx} element count {d_in} * {d_out} = {n_elems} overflows 32-bit LFT1 limit."
            )

        expected_bias_nbytes = int(d_out) * 4
        if bias_nbytes != expected_bias_nbytes:
            raise ValueError(
                f"Malformed LFT1 bias_nbytes at layer {idx}: got {bias_nbytes}, expected {expected_bias_nbytes} (d_out={d_out})."
            )

        expected_scale_elems = int(d_out) if per_ch_flag else 1
        expected_scale_nbytes = expected_scale_elems * 4
        if scale_nbytes != expected_scale_nbytes:
            raise ValueError(
                f"Malformed LFT1 scale_nbytes at layer {idx}: got {scale_nbytes}, expected {expected_scale_nbytes}."
            )

        expected_zp_nbytes = expected_scale_elems if has_zp else 0
        if zp_nbytes != expected_zp_nbytes:
            raise ValueError(
                f"Malformed LFT1 zp_nbytes at layer {idx}: got {zp_nbytes}, expected {expected_zp_nbytes}."
            )

        expected_payload_nbytes = (
            ((n_elems + 1) // 2 if n_elems > 0 else 0) if bits == 4 else n_elems
        )
        if payload_nbytes != expected_payload_nbytes:
            raise ValueError(
                f"Malformed LFT1 payload_nbytes at layer {idx}: got {payload_nbytes}, "
                f"expected {expected_payload_nbytes} for shape {(d_in, d_out)} at {bits}-bit."
            )

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

        if bits == 4 and (n_elems % 2 == 1) and ((int(packed[-1]) >> 4) & 0x0F) != 0:
            raise ValueError(
                f"Corrupted LFT1 odd-length 4-bit high padding nibble at layer {idx}."
            )

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
