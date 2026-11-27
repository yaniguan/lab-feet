"""Unit, formal parity, and benchmark tests for lab-feet community_extensions (Idea #20, #27..#29)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from community_extensions import (  # noqa: E402
    SAFE_HOLD_CLASS_ID,
    SAFE_HOLD_LABEL,
    DriftAbstainingInt4Classifier,
    decide_drift_abstaining_int4,
    pack_signed_int4,
    requantize_saturate_int64,
    run_benchmark,
    unpack_signed_int4,
)
from lab_feet import export_c99_header  # noqa: E402


def test_signed_int4_pack_unpack_roundtrip() -> None:
    vals = np.array([-8, -7, -1, 0, 1, 6, 7, -4, 3], dtype=np.int8)
    packed = pack_signed_int4(vals)
    assert packed.dtype == np.uint8
    assert packed.size == 5
    unpacked = unpack_signed_int4(packed, total_elements=vals.size)
    assert np.array_equal(vals, unpacked)

    with pytest.raises(ValueError, match="\\[-8, \\+7\\]"):
        pack_signed_int4(np.array([8], dtype=np.int8))


def test_issue_27_odd_layer_dimensions_and_unpack_bounds() -> None:
    clf = DriftAbstainingInt4Classifier(n_features=25, n_hidden=15, n_classes=5)
    assert clf.w1_packed.size == (25 * 15 + 1) // 2 == 188
    assert clf.w2_packed.size == (15 * 5 + 1) // 2 == 38
    assert len(unpack_signed_int4(clf.w2_packed, 75)) == 75

    pred = clf.predict_naive_int4_argmax(np.zeros(25, dtype=np.float32))
    assert 0 <= pred.class_id < 5

    # Under-allocated or over-allocated byte buffers and corrupted odd padding raise ValueError
    with pytest.raises(ValueError, match="expected 38 bytes"):
        unpack_signed_int4(np.zeros(37, dtype=np.uint8), 75)
    with pytest.raises(ValueError, match="expected 38 bytes"):
        unpack_signed_int4(np.zeros(39, dtype=np.uint8), 75)

    bad_pad = np.zeros(38, dtype=np.uint8)
    bad_pad[-1] = 0x30
    with pytest.raises(ValueError, match="padding nibble"):
        unpack_signed_int4(bad_pad, 75)


def test_issue_28_int64_requantization_no_wraparound_and_small_scale_bias_clamp() -> None:
    # 128-feature sensor vector under +127 rail saturation must NOT wrap int32 * int16 negative
    clf = DriftAbstainingInt4Classifier(n_features=128, n_hidden=4, n_classes=2)
    clf.input_scale = 1.0
    clf.w1_packed = np.full((128 * 4) // 2, 0x77, dtype=np.uint8)  # all weights = +7
    clf.h1_requant_mult_i16 = np.full(4, 32767, dtype=np.int16)
    clf.h1_requant_shift = 10

    pred = clf.predict_drift_abstaining_int4(np.full(128, 127.0, dtype=np.float64))
    assert pred.int4_clip_count == 4
    assert pred.abstained is True
    assert pred.class_id == SAFE_HOLD_CLASS_ID

    # Small effective scale (~1e-11) clamps biases cleanly into int32 without overflow crash
    clf_tiny = DriftAbstainingInt4Classifier(n_features=8, n_hidden=6, n_classes=3)
    w1_zero = np.zeros((8, 6), dtype=np.float64)
    b1_huge = np.array([25.0, -25.0, 0.0, 1.0, -1.0, 5.0], dtype=np.float64)
    w2_zero = np.zeros((6, 3), dtype=np.float64)
    b2_huge = np.array([10.0, -10.0, 0.0], dtype=np.float64)
    x_cal = np.zeros((12, 8), dtype=np.float64)
    y_cal = np.arange(12, dtype=np.int64) % 3
    clf_tiny.load_fp32_weights_and_calibrate(
        w1_zero, b1_huge, w2_zero, b2_huge, x_cal, y_cal, input_scale=1e-5
    )
    assert clf_tiny.b1_i32.dtype == np.int32
    assert int(clf_tiny.b1_i32[0]) == (1 << 31) - 1
    assert int(clf_tiny.b1_i32[1]) == -(1 << 31) + 1


def test_issue_29_lean_formal_theorems_rounding_parity_and_5_gate_equivalence_on_synthetic_fixture(
    tmp_path: Path,
) -> None:
    # 1) Nearest-integer rounding offset eliminates -0.5 LSB truncation bias and stays within 1 LSB of ties-to-even
    sat_hu, raw_hu = requantize_saturate_int64(1, 1023, 10, -8, 7, rounding="half_up")
    assert int(raw_hu) == 1 and int(sat_hu) == 1

    accs = np.arange(-2048, 2049, dtype=np.int64)
    hu, _ = requantize_saturate_int64(accs, 512, 10, -8, 7, rounding="half_up")
    te, _ = requantize_saturate_int64(accs, 512, 10, -8, 7, rounding="ties_to_even")
    assert int(np.max(np.abs(hu - te))) <= 1

    # 2) Signed peak_logit < min_peak_logit and int8_input_clips > max_input_clips trigger SAFE_HOLD
    assert decide_drift_abstaining_int4(2, 200, 780, 0, 8, 24, 2, 20, 12, 30, 12) == SAFE_HOLD_CLASS_ID
    assert decide_drift_abstaining_int4(2, 200, 780, 0, 8, 0, 2, 20, 12, -18, 12) == SAFE_HOLD_CLASS_ID
    assert decide_drift_abstaining_int4(2, 200, 780, 0, 8, 0, 2, 20, 12, 30, 12) == 2

    # 3) 100% decision equivalence across all 400 benchmark inputs (240 ID + 160 OOD) + C99 header export
    from community_extensions.drift_abstaining_int4 import (
        _generate_benchtop_sensor_dataset,
        _generate_ood_and_saturated_inputs,
    )

    rng = np.random.default_rng(20261123)
    x_all, y_all, _ = _generate_benchtop_sensor_dataset(rng, n_per_class=160)
    x_ood = _generate_ood_and_saturated_inputs(rng, n_ood=160)
    clf = DriftAbstainingInt4Classifier()
    clf.fit_and_calibrate(x_all[:540], y_all[:540], x_all[540:720], y_all[540:720], seed=20261123)

    eval_inputs = np.vstack([x_all[720:960], x_ood])
    assert eval_inputs.shape == (400, 24)
    for row in eval_inputs:
        pred = clf.predict_drift_abstaining_int4(row)
        naive = clf.predict_naive_int4_argmax(row)
        lean_decision = decide_drift_abstaining_int4(
            naive.class_id,
            pred.l1_mahalanobis_score,
            int(clf.class_l1_thresh_u16[naive.class_id]),
            pred.int4_clip_count,
            int(clf.scalar_thresholds_u16[0]),
            pred.int8_input_clip_count,
            int(clf.scalar_thresholds_u16[1]),
            pred.energy_margin_score,
            int(clf.scalar_thresholds_u16[2]),
            pred.peak_logit,
            int(clf.scalar_thresholds_u16[3]),
        )
        assert pred.class_id == lean_decision

    c99_hdr = export_c99_header(clf, prefix="bench_drift")
    assert "bench_drift_infer_drift_abstaining_int4" in c99_hdr
    assert "LAB_FEET_SAFE_HOLD" in c99_hdr

    # 4) Verify LabFeetLean.lean has zero sorry/admit tokens outside comments and compiles cleanly when lake is present
    import re

    lean_src = (_ROOT / "lean" / "LabFeetLean.lean").read_text(encoding="utf-8")
    lean_no_block = re.sub(r"/-.*?-/", "", lean_src, flags=re.DOTALL)
    lean_code_only = "\n".join(ln.split("--")[0] for ln in lean_no_block.splitlines())
    assert re.search(r"\b(sorry|admit)\b", lean_code_only) is None
    for required_sym in (
        "unpackInt4_packInt4_id",
        "packInt4Single_zero_high_nibble",
        "requantSaturate_in_bounds",
        "requantSaturate_rounding_diff_le_one",
        "driftMonitor_safe_hold_on_int8_input_clips",
        "driftMonitor_safe_hold_on_low_peak_logit",
        "decideDriftAbstainingInt4_sound",
    ):
        assert required_sym in lean_src

    lake_bin = shutil.which("lake")
    if lake_bin is None:
        elan_lake = Path.home() / ".elan" / "bin" / ("lake.exe" if sys.platform == "win32" else "lake")
        if elan_lake.is_file():
            lake_bin = str(elan_lake)
    if lake_bin is not None:
        lean_copy = tmp_path / "lean_build_check"
        shutil.copytree(_ROOT / "lean", lean_copy)
        proc = subprocess.run([lake_bin, "build"], cwd=str(lean_copy), capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"lake build failed:\n{proc.stdout}\n{proc.stderr}"
        assert "warning:" not in proc.stdout.lower() and "warning:" not in proc.stderr.lower()


def test_monitor_extra_sram_footprints_24d_32d_128d() -> None:
    clf_24 = DriftAbstainingInt4Classifier(n_features=24, n_hidden=32, n_classes=6)
    clf_32 = DriftAbstainingInt4Classifier(n_features=32, n_hidden=24, n_classes=6)
    clf_128 = DriftAbstainingInt4Classifier(n_features=128, n_hidden=32, n_classes=6)
    assert clf_24.extra_sram_bytes == 220 and clf_24.extra_sram_bytes < 512
    assert clf_32.extra_sram_bytes == 276 and clf_32.extra_sram_bytes < 512
    assert clf_128.extra_sram_bytes == 948 and clf_128.extra_sram_bytes > 512


def test_saturated_and_drifted_inputs_trigger_safe_hold_on_synthetic_fixture() -> None:
    rng = np.random.default_rng(42)
    from community_extensions.drift_abstaining_int4 import _generate_benchtop_sensor_dataset

    x_all, y_all, _ = _generate_benchtop_sensor_dataset(rng, n_per_class=80)
    clf = DriftAbstainingInt4Classifier()
    clf.fit_and_calibrate(x_all[:360], y_all[:360], x_all[360:], y_all[360:], seed=42)

    # Verify predict_int4_logits matches naive argmax 100% across calibration samples
    logits_batch = clf.predict_int4_logits(x_all[360:400], dequantize=True)
    assert logits_batch.shape == (40, 6)
    for i in range(40):
        assert int(np.argmax(logits_batch[i])) == clf.predict_naive_int4_argmax(x_all[360 + i]).class_id
        assert np.allclose(clf.predict_int4_logits(x_all[360 + i]), logits_batch[i], atol=1e-6)

    react_logits = clf.predict_int4_logits(x_all[360:400], dequantize=True, react_rectify=True)
    assert react_logits.shape == (40, 6) and np.all(np.isfinite(react_logits))

    # Saturated optical glare input (+15 across all channels)
    saturated = np.full(24, 15.0, dtype=np.float64)
    naive_pred = clf.predict_naive_int4_argmax(saturated)
    safe_pred = clf.predict_drift_abstaining_int4(saturated)

    assert naive_pred.abstained is False
    assert safe_pred.abstained is True
    assert safe_pred.label == SAFE_HOLD_LABEL
    assert safe_pred.class_id == -1
    assert safe_pred.int8_input_clip_count == 24
    assert safe_pred.int4_clip_count > 0


def test_benchmark_400_synthetic_fixture_edge_inputs_accuracy_and_ood_abstention() -> None:
    report = run_benchmark()
    assert report["n_total_inputs"] == 400
    assert report["n_in_domain_inputs"] == 240
    assert report["n_ood_saturated_inputs"] == 160
    assert report["extra_sram_bytes"] < 512

    naive = report["naive_int4_argmax"]
    guarded = report["drift_abstaining_int4"]

    assert naive["silent_ood_error_rate_pct"] == 100.0
    assert naive["in_domain_accuracy_pct"] >= 92.0

    assert guarded["silent_ood_error_rate_pct"] <= 2.0
    assert guarded["ood_safe_hold_rate_pct"] >= 98.0
    assert guarded["in_domain_accuracy_pct"] >= 90.0
    assert guarded["extra_sram_bytes"] < 512
