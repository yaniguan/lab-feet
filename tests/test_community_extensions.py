"""Unit and benchmark tests for lab-feet community_extensions (Idea #20)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from community_extensions import (  # noqa: E402
    SAFE_HOLD_LABEL,
    DriftAbstainingInt4Classifier,
    pack_signed_int4,
    run_benchmark,
    unpack_signed_int4,
)


def test_signed_int4_pack_unpack_roundtrip() -> None:
    vals = np.array([-8, -7, -1, 0, 1, 6, 7, -4, 3], dtype=np.int8)
    packed = pack_signed_int4(vals)
    assert packed.dtype == np.uint8
    assert packed.size == 5
    unpacked = unpack_signed_int4(packed, total_elements=vals.size)
    assert np.array_equal(vals, unpacked)

    with pytest.raises(ValueError, match="\\[-8, \\+7\\]"):
        pack_signed_int4(np.array([8], dtype=np.int8))


def test_monitor_extra_sram_strictly_below_512_bytes() -> None:
    clf = DriftAbstainingInt4Classifier(n_features=24, n_hidden=32, n_classes=6)
    assert clf.extra_sram_bytes < 512
    assert clf.extra_sram_bytes == 220


def test_saturated_and_drifted_inputs_trigger_safe_hold() -> None:
    rng = np.random.default_rng(42)
    from community_extensions.drift_abstaining_int4 import _generate_benchtop_sensor_dataset

    x_all, y_all, _ = _generate_benchtop_sensor_dataset(rng, n_per_class=80)
    clf = DriftAbstainingInt4Classifier()
    clf.fit_and_calibrate(x_all[:360], y_all[:360], x_all[360:], y_all[360:], seed=42)

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


def test_benchmark_400_edge_inputs_accuracy_and_ood_abstention() -> None:
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
