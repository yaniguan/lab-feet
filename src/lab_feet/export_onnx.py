"""ONNX `opset >= 21` quantized graph (JSON + ModelProto `.onnx`) exporter (`#31`).

Exports `QuantizedModel` (`4-bit` packed `INT4` or `8-bit` `INT8`) to:
  1. A JSON-serializable ONNX graph + initializer dictionary (`export_quantized_graph`) with a
     zero-dependency bit-exact Python reference evaluator (`evaluate_quantized_graph`).
  2. Standard binary ONNX `ModelProto` bytes (`export_quantized_onnx`, `ir_version=10`, `opset>=21`)
     loadable directly by `onnxruntime.InferenceSession` (ARM64 NEON `SDOT`, x86 AVX-VNNI, and
     NVIDIA Jetson Orin TensorRT/CUDA ExecutionProviders) without requiring the `onnx` Python
     build package at export time.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .export_c import prepare_fixed_point_model
from .quantize import QuantizedModel, pack_4bit, unpack_4bit

# ONNX TensorProto.DataType enum constants (onnx.proto3)
ONNX_FLOAT = 1
ONNX_UINT8 = 2
ONNX_INT8 = 3
ONNX_INT32 = 6
ONNX_INT64 = 7
ONNX_UINT4 = 21
ONNX_INT4 = 22

_DTYPE_NAME_TO_CODE: dict[str, int] = {
    "FLOAT": ONNX_FLOAT,
    "UINT8": ONNX_UINT8,
    "INT8": ONNX_INT8,
    "INT32": ONNX_INT32,
    "INT64": ONNX_INT64,
    "UINT4": ONNX_UINT4,
    "INT4": ONNX_INT4,
}

_ACT_TO_ONNX_OP: dict[str, str | None] = {
    "relu": "Relu",
    "tanh": "Tanh",
    "logistic": "Sigmoid",
    "identity": None,
}


def _encode_varint(value: int) -> bytes:
    v = int(value)
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _pb_varint(field_num: int, value: int) -> bytes:
    return _encode_varint((field_num << 3) | 0) + _encode_varint(value)


def _pb_bytes(field_num: int, payload: bytes | str) -> bytes:
    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    return _encode_varint((field_num << 3) | 2) + _encode_varint(len(raw)) + raw


def _pb_attr_int(name: str, val: int) -> bytes:
    # AttributeProto: name (1), i (3), type=INT=2 (20)
    return _pb_bytes(1, name) + _pb_varint(3, int(val)) + _pb_varint(20, 2)


def _pb_tensor_proto(name: str, data_type: int, dims: list[int], raw_data: bytes) -> bytes:
    # TensorProto: dims (1), data_type (2), name (8), raw_data (9)
    msg = bytearray()
    for d in dims:
        msg += _pb_varint(1, int(d))
    msg += _pb_varint(2, int(data_type))
    msg += _pb_bytes(8, name)
    msg += _pb_bytes(9, raw_data)
    return bytes(msg)


def _pb_value_info(name: str, elem_type: int, dims: list[int | str]) -> bytes:
    # ValueInfoProto: name (1), type (2) -> TypeProto: tensor_type (1) -> elem_type (1), shape (2)
    shape_msg = bytearray()
    for d in dims:
        dim_entry = _pb_bytes(2, d) if isinstance(d, str) else _pb_varint(1, int(d))
        shape_msg += _pb_bytes(1, bytes(dim_entry))
    tensor_type = _pb_varint(1, int(elem_type)) + _pb_bytes(2, bytes(shape_msg))
    type_proto = _pb_bytes(1, tensor_type)
    return _pb_bytes(1, name) + _pb_bytes(2, type_proto)


def _pb_node_proto(
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    *,
    name: str = "",
    domain: str = "",
    attributes: dict[str, int] | None = None,
) -> bytes:
    msg = bytearray()
    for inp in inputs:
        msg += _pb_bytes(1, inp)
    for out in outputs:
        msg += _pb_bytes(2, out)
    if name:
        msg += _pb_bytes(3, name)
    msg += _pb_bytes(4, op_type)
    if attributes:
        for k, v in attributes.items():
            msg += _pb_bytes(5, _pb_attr_int(k, v))
    if domain:
        msg += _pb_bytes(7, domain)
    return bytes(msg)


def _make_initializer_entry(name: str, dtype_name: str, dims: list[int], raw_bytes: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "data_type": dtype_name,
        "data_type_code": _DTYPE_NAME_TO_CODE[dtype_name],
        "dims": [int(d) for d in dims],
        "nbytes": len(raw_bytes),
        "raw_base64": base64.b64encode(raw_bytes).decode("ascii"),
    }


def export_quantized_graph(
    model: QuantizedModel,
    *,
    opset: int = 21,
    operator_mode: Literal["qdq", "qlinear"] = "qdq",
    graph_name: str = "lab_feet_quantized_mlp",
    input_scale: float | None = None,
    X_cal: np.ndarray | None = None,
) -> dict[str, Any]:
    """Export `QuantizedModel` as a JSON-serializable ONNX `opset >= 21` graph & initializer spec.

    Modes:
      - `'qdq'` (default): Emits packed `INT4` (`bits == 4`) or `INT8` (`bits != 4`) weight
        initializers with `DequantizeLinear` (`opset >= 21`) + `MatMul` + `Add` + activation,
        which ONNX Runtime fuses into `DynamicQuantizeMatMul` / `MatMulNBits` SIMD kernels while
        maintaining exact dequantized graph semantics.
      - `'qlinear'`: Emits static `W8A8` / `WbA8` `QuantizeLinear` + `QLinearMatMul` +
        `DequantizeLinear` + `Add` + activation nodes using calibrated activation scales.
    """
    if opset < 21:
        raise ValueError(f"ONNX INT4 quantized graph export requires opset >= 21, got {opset}.")
    if operator_mode not in ("qdq", "qlinear"):
        raise ValueError(f"Unsupported operator_mode {operator_mode!r}; expected 'qdq' or 'qlinear'.")
    if not model.layers:
        raise ValueError("QuantizedModel must contain at least one layer.")
    if model.activation not in _ACT_TO_ONNX_OP:
        raise ValueError(f"Unsupported model activation {model.activation!r} for ONNX export.")

    d_in_0 = int(model.layers[0].shape[0])
    d_out_last = int(model.layers[-1].shape[1])
    initializers: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    act_op = _ACT_TO_ONNX_OP[model.activation]

    if operator_mode == "qdq":
        cur_tensor = "input"
        for idx, layer in enumerate(model.layers):
            d_in, d_out = layer.shape
            is_hidden = idx < len(model.layers) - 1
            p = f"l{idx}"

            if layer.bits == 4:
                w_dtype = "INT4"
                w_raw = np.ascontiguousarray(layer.packed_weights, dtype=np.uint8).tobytes()
            else:
                w_dtype = "INT8"
                w_raw = np.ascontiguousarray(layer.packed_weights, dtype=np.int8).tobytes()
            initializers.append(_make_initializer_entry(f"{p}_w_q", w_dtype, [d_in, d_out], w_raw))

            if layer.per_channel:
                s_dims = [d_out]
                s_raw = np.ascontiguousarray(layer.scale.reshape(d_out), dtype="<f4").tobytes()
                zp_arr = (
                    layer.zero_point.reshape(d_out).astype(np.int8)
                    if layer.zero_point is not None
                    else np.zeros(d_out, dtype=np.int8)
                )
                dq_attrs: dict[str, int] = {"axis": 1}
            else:
                s_dims = []
                s_raw = np.array(float(layer.scale.reshape(-1)[0]), dtype="<f4").tobytes()
                zp_arr = (
                    np.array([int(layer.zero_point.reshape(-1)[0])], dtype=np.int8)
                    if layer.zero_point is not None
                    else np.array([0], dtype=np.int8)
                )
                dq_attrs = {}

            initializers.append(_make_initializer_entry(f"{p}_w_scale", "FLOAT", s_dims, s_raw))
            if layer.bits == 4:
                zp_raw = pack_4bit(zp_arr).tobytes()  # type: ignore[union-attr]
                initializers.append(_make_initializer_entry(f"{p}_w_zp", "INT4", s_dims, zp_raw))
            else:
                zp_raw = np.ascontiguousarray(zp_arr, dtype=np.int8).tobytes()
                initializers.append(_make_initializer_entry(f"{p}_w_zp", "INT8", s_dims, zp_raw))

            b_raw = np.ascontiguousarray(layer.bias.reshape(d_out), dtype="<f4").tobytes()
            initializers.append(_make_initializer_entry(f"{p}_bias", "FLOAT", [d_out], b_raw))

            w_deq_name = f"{p}_w_deq"
            nodes.append(
                {
                    "name": f"{p}_dequant_w",
                    "op_type": "DequantizeLinear",
                    "inputs": [f"{p}_w_q", f"{p}_w_scale", f"{p}_w_zp"],
                    "outputs": [w_deq_name],
                    "attributes": dq_attrs,
                }
            )
            mm_out = f"{p}_mm"
            nodes.append(
                {
                    "name": f"{p}_matmul",
                    "op_type": "MatMul",
                    "inputs": [cur_tensor, w_deq_name],
                    "outputs": [mm_out],
                    "attributes": {},
                }
            )
            add_out = f"{p}_pre_act" if (is_hidden and act_op is not None) else (f"{p}_out" if is_hidden else "logits")
            nodes.append(
                {
                    "name": f"{p}_add_bias",
                    "op_type": "Add",
                    "inputs": [mm_out, f"{p}_bias"],
                    "outputs": [add_out],
                    "attributes": {},
                }
            )
            if is_hidden and act_op is not None:
                act_out = f"{p}_act"
                nodes.append(
                    {
                        "name": f"{p}_{act_op.lower()}",
                        "op_type": act_op,
                        "inputs": [add_out],
                        "outputs": [act_out],
                        "attributes": {},
                    }
                )
                cur_tensor = act_out
            else:
                cur_tensor = add_out
    else:
        # Static QLinearMatMul mode (`QuantizeLinear` -> `QLinearMatMul` -> `DequantizeLinear` -> `Add` -> act)
        fp_spec = prepare_fixed_point_model(
            QuantizedModel(layers=model.layers, activation="relu" if model.activation != "identity" else "identity"),
            input_scale=input_scale,
            X_cal=X_cal,
        )
        zero_i8_raw = np.array(0, dtype=np.int8).tobytes()
        initializers.append(_make_initializer_entry("zero_zp_i8", "INT8", [], zero_i8_raw))

        in_scale_raw = np.array(float(fp_spec["input_scale"]), dtype="<f4").tobytes()
        initializers.append(_make_initializer_entry("in_scale", "FLOAT", [], in_scale_raw))
        nodes.append(
            {
                "name": "quantize_input",
                "op_type": "QuantizeLinear",
                "inputs": ["input", "in_scale", "zero_zp_i8"],
                "outputs": ["l0_in_q"],
                "attributes": {},
            }
        )
        cur_q = "l0_in_q"
        cur_s = "in_scale"

        for idx, layer in enumerate(model.layers):
            d_in, d_out = layer.shape
            is_hidden = idx < len(model.layers) - 1
            p = f"l{idx}"
            lspec = fp_spec["layers"][idx]

            w_i8 = np.ascontiguousarray(layer.unpack_codes(), dtype=np.int8)
            initializers.append(_make_initializer_entry(f"{p}_w_i8", "INT8", [d_in, d_out], w_i8.tobytes()))

            if layer.per_channel:
                ws_dims = [d_out]
                ws_raw = np.ascontiguousarray(layer.scale.reshape(d_out), dtype="<f4").tobytes()
                wzp_arr = (
                    layer.zero_point.reshape(d_out).astype(np.int8)
                    if layer.zero_point is not None
                    else np.zeros(d_out, dtype=np.int8)
                )
                wzp_dims = [d_out]
                wzp_raw = wzp_arr.tobytes()
            else:
                ws_dims = []
                ws_raw = np.array(float(layer.scale.reshape(-1)[0]), dtype="<f4").tobytes()
                wzp_val = int(layer.zero_point.reshape(-1)[0]) if layer.zero_point is not None else 0
                wzp_dims = []
                wzp_raw = np.array(wzp_val, dtype=np.int8).tobytes()

            initializers.append(_make_initializer_entry(f"{p}_w_scale", "FLOAT", ws_dims, ws_raw))
            initializers.append(_make_initializer_entry(f"{p}_w_zp", "INT8", wzp_dims, wzp_raw))

            # Output scale for QLinearMatMul (`int8` output before bias addition)
            y_scale_val = max(float(lspec["out_scale"]) * (256.0 if not is_hidden else 1.0), 1e-5)
            initializers.append(
                _make_initializer_entry(
                    f"{p}_y_scale", "FLOAT", [], np.array(y_scale_val, dtype="<f4").tobytes()
                )
            )
            b_raw = np.ascontiguousarray(layer.bias.reshape(d_out), dtype="<f4").tobytes()
            initializers.append(_make_initializer_entry(f"{p}_bias", "FLOAT", [d_out], b_raw))

            qmm_out = f"{p}_mm_q"
            nodes.append(
                {
                    "name": f"{p}_qlinear_matmul",
                    "op_type": "QLinearMatMul",
                    "inputs": [
                        cur_q,
                        cur_s,
                        "zero_zp_i8",
                        f"{p}_w_i8",
                        f"{p}_w_scale",
                        f"{p}_w_zp",
                        f"{p}_y_scale",
                        "zero_zp_i8",
                    ],
                    "outputs": [qmm_out],
                    "attributes": {},
                }
            )
            deq_out = f"{p}_mm_f32"
            nodes.append(
                {
                    "name": f"{p}_dequant_out",
                    "op_type": "DequantizeLinear",
                    "inputs": [qmm_out, f"{p}_y_scale", "zero_zp_i8"],
                    "outputs": [deq_out],
                    "attributes": {},
                }
            )
            add_out = f"{p}_pre_act" if (is_hidden and act_op is not None) else (f"{p}_out" if is_hidden else "logits")
            nodes.append(
                {
                    "name": f"{p}_add_bias",
                    "op_type": "Add",
                    "inputs": [deq_out, f"{p}_bias"],
                    "outputs": [add_out],
                    "attributes": {},
                }
            )
            if is_hidden:
                if act_op is not None:
                    act_out = f"{p}_act"
                    nodes.append(
                        {
                            "name": f"{p}_{act_op.lower()}",
                            "op_type": act_op,
                            "inputs": [add_out],
                            "outputs": [act_out],
                            "attributes": {},
                        }
                    )
                else:
                    act_out = add_out
                next_q = f"l{idx + 1}_in_q"
                nodes.append(
                    {
                        "name": f"{p}_requant_act",
                        "op_type": "QuantizeLinear",
                        "inputs": [act_out, f"{p}_y_scale", "zero_zp_i8"],
                        "outputs": [next_q],
                        "attributes": {},
                    }
                )
                cur_q = next_q
                cur_s = f"{p}_y_scale"

    return {
        "format": "onnx-graph-json/v1",
        "ir_version": 10,
        "opset_version": int(opset),
        "producer_name": "lab-feet",
        "producer_version": "0.2.0",
        "graph_name": graph_name,
        "operator_mode": operator_mode,
        "inputs": [{"name": "input", "elem_type": "FLOAT", "dims": ["batch", d_in_0]}],
        "outputs": [{"name": "logits", "elem_type": "FLOAT", "dims": ["batch", d_out_last]}],
        "initializers": initializers,
        "nodes": nodes,
    }


def _decode_initializer_numpy(init_entry: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(init_entry["raw_base64"])
    dtype_name = init_entry["data_type"]
    dims = tuple(int(d) for d in init_entry["dims"])
    if dtype_name == "FLOAT":
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        return arr.reshape(dims)
    if dtype_name == "INT8":
        arr = np.frombuffer(raw, dtype=np.int8).copy()
        return arr.reshape(dims)
    if dtype_name == "UINT8":
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        return arr.reshape(dims)
    if dtype_name == "INT32":
        arr = np.frombuffer(raw, dtype="<i4").astype(np.int32, copy=True)
        return arr.reshape(dims)
    if dtype_name == "INT4":
        packed = np.frombuffer(raw, dtype=np.uint8).copy()
        if dims == ():
            return unpack_4bit(packed, (1,), strict_padding=False).reshape(())
        return unpack_4bit(packed, dims, strict_padding=True)
    raise ValueError(f"Unsupported initializer dtype {dtype_name!r}")


def evaluate_quantized_graph(graph_spec: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Zero-dependency pure-Python ONNX graph interpreter verifying bit-exact node semantics."""
    x_arr = np.asarray(x, dtype=np.float32)
    was_1d = x_arr.ndim == 1
    env: dict[str, np.ndarray] = {
        "input": x_arr.reshape(1, -1) if was_1d else x_arr,
    }
    for init in graph_spec["initializers"]:
        env[init["name"]] = _decode_initializer_numpy(init)

    for node in graph_spec["nodes"]:
        op = node["op_type"]
        ins = [env[name] for name in node["inputs"]]
        attrs = node.get("attributes") or {}

        if op == "DequantizeLinear":
            q_val = ins[0].astype(np.float32)
            scale = ins[1].astype(np.float32)
            zp = ins[2].astype(np.float32) if len(ins) >= 3 else np.float32(0.0)
            axis = attrs.get("axis")
            if axis is not None and scale.ndim == 1:
                shape = [1] * q_val.ndim
                shape[int(axis)] = scale.size
                scale = scale.reshape(shape)
                zp = zp.reshape(shape)
            res = ((q_val - zp) * scale).astype(np.float32)
        elif op == "QuantizeLinear":
            val = ins[0].astype(np.float32)
            scale = ins[1].astype(np.float32)
            zp = ins[2]
            raw_q = np.rint(val / scale) + zp.astype(np.float32)
            res = np.clip(raw_q, -128, 127).astype(zp.dtype)
        elif op == "QLinearMatMul":
            a, a_s, a_zp, b, b_s, b_zp, y_s, y_zp = ins
            acc_i32 = np.matmul(
                a.astype(np.int32) - a_zp.astype(np.int32),
                b.astype(np.int32) - b_zp.astype(np.int32),
            )
            eff_scale = (a_s.astype(np.float32) * b_s.astype(np.float32)) / y_s.astype(np.float32)
            res = np.clip(np.rint(acc_i32.astype(np.float32) * eff_scale) + y_zp.astype(np.float32), -128, 127).astype(
                y_zp.dtype
            )
        elif op == "MatMul":
            res = np.matmul(ins[0].astype(np.float32), ins[1].astype(np.float32)).astype(np.float32)
        elif op == "Add":
            res = (ins[0].astype(np.float32) + ins[1].astype(np.float32)).astype(np.float32)
        elif op == "Relu":
            res = np.maximum(ins[0], np.float32(0.0)).astype(np.float32)
        elif op == "Tanh":
            res = np.tanh(ins[0]).astype(np.float32)
        elif op == "Sigmoid":
            res = (np.float32(1.0) / (np.float32(1.0) + np.exp(-np.clip(ins[0], -80.0, 80.0)))).astype(np.float32)
        else:
            raise ValueError(f"Unsupported ONNX operator {op!r} in evaluate_quantized_graph.")

        env[node["outputs"][0]] = res

    out_name = graph_spec["outputs"][0]["name"]
    out_arr = env[out_name]
    return out_arr.reshape(-1) if was_1d else out_arr


def serialize_onnx_proto(graph_spec: dict[str, Any]) -> bytes:
    """Encode an `export_quantized_graph` spec into standard binary ONNX `ModelProto` bytes."""
    g_msg = bytearray()
    for node in graph_spec["nodes"]:
        g_msg += _pb_bytes(
            1,
            _pb_node_proto(
                node["op_type"],
                list(node["inputs"]),
                list(node["outputs"]),
                name=str(node.get("name", "")),
                domain=str(node.get("domain", "")),
                attributes=node.get("attributes") or {},
            ),
        )
    g_msg += _pb_bytes(2, str(graph_spec["graph_name"]))
    for init in graph_spec["initializers"]:
        raw = base64.b64decode(init["raw_base64"])
        g_msg += _pb_bytes(
            5,
            _pb_tensor_proto(
                str(init["name"]),
                int(init["data_type_code"]),
                list(init["dims"]),
                raw,
            ),
        )
    for inp in graph_spec["inputs"]:
        g_msg += _pb_bytes(
            11,
            _pb_value_info(
                str(inp["name"]),
                _DTYPE_NAME_TO_CODE[str(inp["elem_type"])],
                list(inp["dims"]),
            ),
        )
    for out in graph_spec["outputs"]:
        g_msg += _pb_bytes(
            12,
            _pb_value_info(
                str(out["name"]),
                _DTYPE_NAME_TO_CODE[str(out["elem_type"])],
                list(out["dims"]),
            ),
        )

    opset_msg = _pb_bytes(1, "") + _pb_varint(2, int(graph_spec["opset_version"]))
    model_msg = (
        _pb_varint(1, int(graph_spec["ir_version"]))
        + _pb_bytes(2, str(graph_spec["producer_name"]))
        + _pb_bytes(3, str(graph_spec["producer_version"]))
        + _pb_bytes(7, bytes(g_msg))
        + _pb_bytes(8, opset_msg)
    )
    return bytes(model_msg)


def export_quantized_onnx(
    model: QuantizedModel,
    output_path: str | Path | None = None,
    *,
    opset: int = 21,
    operator_mode: Literal["qdq", "qlinear"] = "qdq",
    graph_name: str = "lab_feet_quantized_mlp",
    input_scale: float | None = None,
    X_cal: np.ndarray | None = None,
    json_out: str | Path | None = None,
) -> bytes:
    """Export `QuantizedModel` to binary `.onnx` `ModelProto` bytes (and optional `.json` graph)."""
    graph_spec = export_quantized_graph(
        model,
        opset=opset,
        operator_mode=operator_mode,
        graph_name=graph_name,
        input_scale=input_scale,
        X_cal=X_cal,
    )
    proto_bytes = serialize_onnx_proto(graph_spec)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(proto_bytes)
    if json_out is not None:
        jpath = Path(json_out)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(graph_spec, indent=2) + "\n", encoding="utf-8")
    return proto_bytes
