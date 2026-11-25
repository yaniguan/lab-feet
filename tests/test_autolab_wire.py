"""Unit tests for the lab-feet Auto-Lab 15-repository wire node (autolab-wire/v1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import autolab_wire  # noqa: E402
from autolab_wire import LAYER, REPO_DIR, REPO_ID, SCHEMA_VERSION, process_packet  # noqa: E402


def test_wire_constants_and_cwd_independent_resolution(tmp_path: Path) -> None:
    assert REPO_ID == "lab-feet"
    assert LAYER == "L0"
    assert SCHEMA_VERSION == "autolab-wire/v1"
    assert REPO_DIR == REPO_ROOT

    prev_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        out = process_packet({})
    finally:
        os.chdir(prev_cwd)

    assert out["halted"] is False
    assert out["edge_runtime"]["nibble_roundtrip_exact"] is True


def test_int8_4bit_execution_and_lab_attrib_telemetry_emission() -> None:
    packet = {
        "run_id": "run-feet-001",
        "stepbot_execution": {
            "python_code": "def run(protocol):\n    pass\n",
            "protocol_sha256": "a3f09c" + "1" * 58,
            "dispatch_receipt": {"simulated": True, "current_deck_epoch": 2},
            "next_deck_epoch": 2,
        },
        "manipulation": {
            "start_well": "A1",
            "target_well": "H12",
            "collision_free": True,
            "waypoints": 7,
            "path_length_rad": 1.412,
            "clearance_mm": 8.5,
        },
    }
    snapshot = {
        "run_id": packet["run_id"],
        "has_edge_runtime": "edge_runtime" in packet,
    }

    out = process_packet(packet)
    out_repeat = process_packet(packet)

    # Input immutability and deterministic execution
    assert ("edge_runtime" in packet) is snapshot["has_edge_runtime"]
    assert out == out_repeat

    assert out["schema_version"] == "autolab-wire/v1"
    assert out["halted"] is False
    assert out["halt_reason"] is None

    edge = out["edge_runtime"]
    assert edge["quantization"] == "int8_and_4bit"
    assert edge["nibble_roundtrip_exact"] is True
    assert edge["packed_bytes"] == 240
    assert edge["dispense_error_permille"] == 12
    assert edge["aspiration_clean"] is True
    assert edge["incubation_window_clean"] is True
    assert edge["seal_intact"] is True
    assert isinstance(edge["monitor_regime"], str) and len(edge["monitor_regime"]) > 0

    # Custom dispense error propagates cleanly to downstream lab-attrib telemetry
    out_custom = process_packet({**packet, "dispense_error_permille": 28})
    assert out_custom["edge_runtime"]["dispense_error_permille"] == 28
    assert out_custom["halted"] is False

    assert len(out["trace"]) == 1
    assert out["trace"][0]["repo"] == "lab-feet"
    assert out["trace"][0]["layer"] == "L0"
    assert out["trace"][0]["status"] == "PASS"

    assert len(out["bottom_reasoning_traces"]) == 1
    brt = out["bottom_reasoning_traces"][0]
    assert brt["repo"] == "lab-feet"
    assert brt["invariant"] == "nibble_pack_roundtrip_and_drift_cap"


def test_quant_drift_cap_boundary_and_blocking() -> None:
    # Exact cap boundary (15 <= 15) passes
    at_cap = process_packet({"quant_acc_drop_permille": 15, "max_acc_drop_permille": 15})
    assert at_cap["halted"] is False
    assert at_cap["halt_reason"] is None
    assert at_cap["trace"][-1]["status"] == "PASS"

    # Exceeding Lean accuracy_drop_permille <= max_drop_permille blocks with L0_FEET_QUANT_DRIFT_BLOCK
    exceeded = process_packet({"quant_acc_drop_permille": 19, "max_acc_drop_permille": 15})
    assert exceeded["halted"] is True
    assert exceeded["halt_reason"] == "L0_FEET_QUANT_DRIFT_BLOCK"
    assert exceeded["trace"][-1]["repo"] == "lab-feet"
    assert exceeded["trace"][-1]["status"] == "BLOCK"
    assert (
        exceeded["bottom_reasoning_traces"][-1]["invariant"]
        == "nibble_pack_roundtrip_and_drift_cap"
    )


def test_halted_packet_passthrough_unchanged() -> None:
    halted_in = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "L1_HANDS_COLLISION_OR_CLEARANCE_BLOCK",
        "trace": [{"repo": "lab-hands", "layer": "L1", "status": "BLOCK", "summary": "clearance < 5mm"}],
        "bottom_reasoning_traces": [
            {"repo": "lab-hands", "invariant": "arm_path_collision_free_with_5mm_margin"}
        ],
    }
    out = process_packet(halted_in)

    assert out == halted_in
    assert out is not halted_in
    assert out["trace"] is not halted_in["trace"]
    assert "edge_runtime" not in out
    assert len(out["trace"]) == 1
    assert autolab_wire.REPO_ID == "lab-feet"


def test_collision_free_dynamic_loading_when_community_extensions_prebound(monkeypatch: object) -> None:
    import importlib.util
    import types

    mp = monkeypatch  # type: ignore[assignment]
    mp.setitem(sys.modules, "community_extensions", types.ModuleType("community_extensions"))
    mp.delitem(sys.modules, "_lab_feet_drift_abstaining_int4", raising=False)

    spec = importlib.util.spec_from_file_location(
        "_test_lab_feet_autolab_wire_isolated",
        REPO_ROOT / "autolab_wire.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_lab_feet_autolab_wire_isolated"] = mod
    try:
        spec.loader.exec_module(mod)
        res = mod.process_packet({})
        assert res["halted"] is False
        assert res["edge_runtime"]["nibble_roundtrip_exact"] is True
    finally:
        sys.modules.pop("_test_lab_feet_autolab_wire_isolated", None)


def test_issue_30_splitguard_sha256_verification_hotload_and_safe_hold(tmp_path: Path) -> None:
    """Issue #30: SplitGuard rejection, SHA-256 checkpoint/calibration verification, hot-load, & SAFE_HOLD."""
    import numpy as np
    from lab_feet import quantize_model, save_quantized_binary, to_binary_buffer

    # 1) Upstream SplitGuard contamination blocks immediately with L0_FEET_UPSTREAM_SPLITGUARD_BLOCK
    tainted_pkt = {
        "posttrain": {
            "split_guard_clean": False,
            "checkpoint_sha256": "deadbeef" * 8,
        }
    }
    res_tainted = process_packet(tainted_pkt)
    assert res_tainted["halted"] is True
    assert res_tainted["halt_reason"] == "L0_FEET_UPSTREAM_SPLITGUARD_BLOCK"
    assert res_tainted["trace"][-1]["status"] == "BLOCK"
    assert "edge_runtime" not in res_tainted

    # 2) Corrupted checkpoint_sha256 or calibration_sha256 blocks with L0_FEET_CHECKPOINT_SHA256_MISMATCH
    bad_sha_pkt = {
        "posttrain": {
            "split_guard_clean": True,
            "checkpoint_sha256": "00" * 32,
        }
    }
    res_bad_sha = process_packet(bad_sha_pkt)
    assert res_bad_sha["halted"] is True
    assert res_bad_sha["halt_reason"] == "L0_FEET_CHECKPOINT_SHA256_MISMATCH"

    # 3) End-to-end L3 -> L0 -> L4 valid hot-load with verified LFT1 checkpoint + canonical calibration SHA-256
    import numpy as np
    from lab_feet import distill_kl_student, forward, quantize_model, save_quantized_binary, to_binary_buffer

    rng = np.random.default_rng(3030)
    x_tr = rng.uniform(-0.8, 1.4, size=(240, 24)).astype(np.float32)
    x_cal = rng.uniform(-0.8, 1.4, size=(96, 24)).astype(np.float32)
    w_teacher = np.zeros((24, 6), dtype=np.float32)
    for c in range(6):
        w_teacher[(4 * c) % 24, c] = 3.0
        w_teacher[(4 * c + 1) % 24, c] = -2.5
    t_logits_tr = x_tr @ w_teacher
    y_tr = np.argmax(t_logits_tr, axis=1).astype(np.int64)
    y_cal = np.argmax(x_cal @ w_teacher, axis=1).astype(np.int64)

    student_layers, _ = distill_kl_student(
        x_tr, y_tr, t_logits_tr, X_val=x_cal, y_val=y_cal, hidden_sizes=(16,), epochs=180, seed=3030
    )
    qm4 = quantize_model(
        student_layers, bits=4, per_channel=True, mode="symmetric_full", clip_search="activation_mse", X_cal=x_cal
    )
    # Align y_cal with the verified distilled checkpoint predictions so W8A8 vs dequant drop is <= 15 permille
    y_cal = np.argmax(forward(qm4.dequantize(), x_cal), axis=1).astype(np.int64)
    lft1_path = save_quantized_binary(qm4, tmp_path / "l3_student.lft1")
    lft1_bytes = to_binary_buffer(qm4)
    ckpt_sha = autolab_wire.compute_checkpoint_sha256(lft1_bytes)
    cal_sha = autolab_wire.compute_calibration_sha256(x_cal, y_cal)

    l3_to_l0_pkt = {
        "posttrain": {
            "split_guard_clean": True,
            "checkpoint_sha256": ckpt_sha,
            "calibration_sha256": cal_sha,
        },
        "distilled_student": {
            "lft1_path": str(lft1_path),
            "calibration_X": x_cal.tolist(),
            "calibration_y": y_cal.tolist(),
        },
    }
    res_hot = process_packet(l3_to_l0_pkt)
    assert res_hot["halted"] is False, f"Unexpected halt: {res_hot['halt_reason']} {res_hot.get('edge_runtime')}"
    er = res_hot["edge_runtime"]
    assert er["checkpoint_source"] == "hotloaded_student"
    assert er["checkpoint_sha256"] == ckpt_sha
    assert er["calibration_sha256"] == cal_sha
    assert er["packed_bytes"] == qm4.layers[0].packed_weights.nbytes + qm4.layers[1].packed_weights.nbytes
    assert er["safe_hold_abstained"] is False

    # Mismatched calibration_sha256 halts with L0_FEET_CHECKPOINT_SHA256_MISMATCH
    tampered_cal_pkt = {
        **l3_to_l0_pkt,
        "posttrain": {
            "split_guard_clean": True,
            "checkpoint_sha256": ckpt_sha,
            "calibration_sha256": "ff" * 32,
        },
    }
    res_tampered = process_packet(tampered_cal_pkt)
    assert res_tampered["halted"] is True
    assert res_tampered["halt_reason"] == "L0_FEET_CHECKPOINT_SHA256_MISMATCH"

    # Empirical calibration accuracy drop overrides optimistic caller quant_acc_drop_permille=0
    x_cal_spiky = x_cal.copy()
    x_cal_spiky[:, 0] = 250.0  # Crushes dynamic int8 activation resolution while FP32 stays exact on channels 1..23!
    w_clean2 = rng.normal(0.0, 0.8, size=(24, 6)).astype(np.float32)
    w_clean2[0, :] = 0.0  # FP32 ignores ch 0 completely, while dynamic int8 activation scale is blown up by 250.0!
    y_spiky = np.argmax(x_cal_spiky @ w_clean2, axis=1).astype(np.int64)
    spiky_layers = [(np.eye(24, dtype=np.float32), np.zeros(24, dtype=np.float32)), (w_clean2, np.zeros(6, dtype=np.float32))]
    spiky_cal_sha = autolab_wire.compute_calibration_sha256(x_cal_spiky, y_spiky)
    res_emp_drop = process_packet(
        {
            "quant_acc_drop_permille": 0,
            "max_acc_drop_permille": 15,
            "distilled_student": {
                "fp32_layers": spiky_layers,
                "calibration_X": x_cal_spiky,
                "calibration_y": y_spiky,
                "calibration_sha256": spiky_cal_sha,
            },
        }
    )
    assert res_emp_drop["edge_runtime"]["quant_acc_drop_permille"] > 100
    assert res_emp_drop["halted"] is True
    assert res_emp_drop["halt_reason"] == "L0_FEET_QUANT_DRIFT_BLOCK"

    # 4) Saturated optical glare raw sensor vector triggers SAFE_HOLD abstention
    res_glare = process_packet({"sensor_features": [15.0] * 24})
    assert res_glare["halted"] is True
    assert res_glare["halt_reason"] == "L0_FEET_SAFE_HOLD_ABSTAIN"
    assert res_glare["edge_runtime"]["safe_hold_abstained"] is True
    assert res_glare["edge_runtime"]["monitor_regime"] == "SAFE_HOLD"

    # 5) Reproducer B from #30: protocol_sha256=15 agrees and passes cleanly
    res_15 = process_packet({"stepbot_execution": {"protocol_sha256": f"{15:064x}"}})
    assert res_15["halted"] is False
    assert res_15["edge_runtime"]["fp32_int4_pred_agree"] is True


