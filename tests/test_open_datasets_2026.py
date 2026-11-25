"""Offline unit, SPDX license gate, distillation/quantization, and multi-batch drift tests for open datasets."""

from __future__ import annotations

import csv
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.fetch_open_datasets as fetch_cli  # noqa: E402
from lab_feet import (  # noqa: E402
    COMMERCIAL_SAFE_SPDX_LICENSES,
    OPEN_EDGE_DATASETS,
    DatasetLicenseError,
    DeterministicFeatureNormalizer,
    OpenEdgeDataset,
    evaluate_temporal_drift_abstaining_int4,
    get_open_edge_dataset,
    list_open_edge_datasets,
    load_open_edge_dataset,
    run_open_dataset_distill_quantize,
    verify_dataset_license,
)


def _write_mnist1d_fixture(path: Path, *, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    n_classes = 10
    n_features = 40
    protos = np.zeros((n_classes, n_features), dtype=np.float32)
    t_grid = np.linspace(0.0, 2.0 * np.pi, n_features, dtype=np.float32)
    for c in range(n_classes):
        protos[c] = 1.8 * np.sin((c + 1) * 0.5 * t_grid + 0.4 * c)
        protos[c, (4 * c) % n_features : ((4 * c) % n_features) + 4] += 2.6

    x_tr_list, y_tr_list, x_te_list, y_te_list = [], [], [], []
    for c in range(n_classes):
        x_tr_list.append(protos[c][None, :] + rng.normal(0.0, 0.28, size=(24, n_features)).astype(np.float32))
        y_tr_list.extend([c] * 24)
        x_te_list.append(protos[c][None, :] + rng.normal(0.0, 0.28, size=(8, n_features)).astype(np.float32))
        y_te_list.extend([c] * 8)

    payload = {
        "x": np.vstack(x_tr_list),
        "y": np.asarray(y_tr_list, dtype=np.int64),
        "x_test": np.vstack(x_te_list),
        "y_test": np.asarray(y_te_list, dtype=np.int64),
    }
    path.write_bytes(pickle.dumps(payload, protocol=4))
    return path


def _format_libsvm_row(label_1based: int, vec: np.ndarray) -> str:
    feats = " ".join(f"{idx + 1}:{float(val):.6f}" for idx, val in enumerate(vec))
    return f"{label_1based} {feats}"


def _write_gas_sensor_drift_zip(
    path: Path,
    *,
    n_batches: int = 5,
    n_per_class_batch1: int = 32,
    n_per_class_later: int = 10,
    seed: int = 17,
) -> Path:
    rng = np.random.default_rng(seed)
    n_classes = 6
    n_features = 128
    protos = np.zeros((n_classes, n_features), dtype=np.float64)
    for c in range(n_classes):
        # Mimic MOX steady-state resistance delta_R (high magnitude) + EMA transient ratios
        steady_idx = [8 * s for s in range(16)]
        protos[c, steady_idx] = 4000.0 + 2800.0 * np.sin(np.arange(16) * 0.7 + c * 1.05)
        active = [(21 * c + k * 3) % n_features for k in range(12)]
        protos[c, active] += np.linspace(9.0, -7.5, len(active))

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write batch10 / batch2 out of lexicographic order when n_batches >= 10 to test numeric sort
        batch_order = list(range(1, n_batches + 1))[::-1]
        for b in batch_order:
            n_pc = n_per_class_batch1 if b == 1 else n_per_class_later
            lines: list[str] = []
            for c in range(n_classes):
                base = protos[c][None, :] + rng.normal(0.0, 0.22, size=(n_pc, n_features))
                base[:, 0::8] *= 1.0 + rng.normal(0.0, 0.04, size=(n_pc, 16))
                if b == 2:
                    # Mild calibration drift
                    base += 0.15
                elif b == 3:
                    # Moderate sensitivity drift
                    base *= 1.35
                    base += 0.85
                elif b == 4:
                    # Severe MOX catalyst poisoning / cross-channel inversion
                    base = -1.8 * base[:, ::-1] + 25.0
                elif b >= 5:
                    # Extreme MOX heater rail saturation & baseline runaway
                    base = np.full((n_pc, n_features), 1.0e7, dtype=np.float64) * (
                        1.0 if c % 2 == 0 else -1.0
                    )
                for row in base:
                    lines.append(_format_libsvm_row(c + 1, row))
            zf.writestr(f"Dataset/batch{b}.dat", "\n".join(lines) + "\n")
    return path


def _write_uci_har_nested_zip(path: Path, *, seed: int = 23) -> Path:
    rng = np.random.default_rng(seed)
    x_tr = rng.normal(0.0, 0.5, size=(18, 561)).astype(np.float32)
    y_tr = (np.arange(18, dtype=np.int64) % 6) + 1
    x_te = rng.normal(0.0, 0.5, size=(12, 561)).astype(np.float32)
    y_te = (np.arange(12, dtype=np.int64) % 6) + 1

    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w", compression=zipfile.ZIP_DEFLATED) as inner:
        # Include macOS resource-fork decoy to verify parser ignores __MACOSX/._*
        inner.writestr("__MACOSX/UCI HAR Dataset/train/._X_train.txt", b"\x00\x05\x16\x07corrupt")
        s_xtr, s_xte = io.StringIO(), io.StringIO()
        np.savetxt(s_xtr, x_tr, fmt="%.5f")
        np.savetxt(s_xte, x_te, fmt="%.5f")
        inner.writestr("UCI HAR Dataset/train/X_train.txt", s_xtr.getvalue())
        inner.writestr("UCI HAR Dataset/train/y_train.txt", "\n".join(str(v) for v in y_tr) + "\n")
        inner.writestr("UCI HAR Dataset/test/X_test.txt", s_xte.getvalue())
        inner.writestr("UCI HAR Dataset/test/y_test.txt", "\n".join(str(v) for v in y_te) + "\n")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("UCI HAR Dataset.names", "HAR metadata\n")
        outer.writestr("UCI HAR Dataset.zip", inner_buf.getvalue())
    return path


def _write_cwru_csv_fixture(path: Path, *, seed: int = 29) -> Path:
    rng = np.random.default_rng(seed)
    n_classes = 4
    n_features = 17
    protos = np.zeros((n_classes, n_features), dtype=np.float64)
    for c in range(n_classes):
        protos[c, (4 * c) % n_features : ((4 * c) % n_features) + 4] = [3.2, -2.4, 2.7, 1.9]

    fault_sizes = [0.0, 0.007, 0.014, 0.021]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["fault_size", "segment_idx", "all_features", "label"])
    seg = 0
    for fs in fault_sizes:
        for c in range(n_classes):
            for _ in range(12):
                vec = protos[c] * (1.0 + 8.0 * fs) + rng.normal(0.0, 0.20, size=n_features)
                writer.writerow([fs, seg, json.dumps([round(float(v), 6) for v in vec]), c])
                seg += 1
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


def _write_uci_hydraulic_zip(path: Path, *, seed: int = 31) -> Path:
    rng = np.random.default_rng(seed)
    valves = [100, 90, 80, 73]
    coolers = [100, 20, 3]
    vs_rows: list[np.ndarray] = []
    prof_rows: list[list[int]] = []
    for cooler in coolers:
        for v_idx, valve in enumerate(valves):
            for _ in range(3):
                row = rng.normal(loc=float(v_idx), scale=0.25, size=60)
                vs_rows.append(row)
                prof_rows.append([cooler, valve, 0, 130, 0])

    s_vs, s_pr = io.StringIO(), io.StringIO()
    np.savetxt(s_vs, np.vstack(vs_rows), fmt="%.4f", delimiter="\t")
    np.savetxt(s_pr, np.asarray(prof_rows, dtype=np.int64), fmt="%d", delimiter="\t")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("VS1.txt", s_vs.getvalue())
        zf.writestr("profile.txt", s_pr.getvalue())
    return path


def _write_sers_bacteria_csv(path: Path, *, seed: int = 37) -> Path:
    rng = np.random.default_rng(seed)
    strains = [
        ("E. coli", "29922"),
        ("E. coli", "BAA-2523"),
        ("E. coli", "K4"),
        ("P. aeruginosa", "10145"),
        ("P. aeruginosa", "27853"),
        ("P. aeruginosa", "9027"),
        ("S. aureus", "25923"),
        ("S. aureus", "43300"),
        ("S. aureus", "6538"),
    ]
    wavenumbers = [f"{248.0 + 1.44 * i:.3f}" for i in range(1215)]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["species", "strain", "strain_label_in_raw_file", "raw_file_number", "acquisition_order", "export_index"]
        + wavenumbers
    )
    idx = 1
    for c_idx, (sp, st) in enumerate(strains):
        for acq in (1, 2, 3):
            spec = np.zeros(1215, dtype=np.float32)
            spec[(120 * c_idx) % 1200 : ((120 * c_idx) % 1200) + 15] = 1.5
            spec += rng.normal(0.0, 0.05, size=1215).astype(np.float32)
            writer.writerow([sp, st, f"{sp}_{st}", acq, acq, idx] + [f"{float(v):.5f}" for v in spec])
            idx += 1
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


def _write_ethanol_nir_zip(path: Path, *, seed: int = 41) -> Path:
    rng = np.random.default_rng(seed)
    tr_rows, te_rows = [], []
    for cls_1based in (1, 2, 3, 4):
        for _ in range(4):
            tr_spec = rng.normal(loc=0.2 * cls_1based, scale=0.05, size=1751)
            te_spec = rng.normal(loc=0.2 * cls_1based + 0.3, scale=0.05, size=1751)
            tr_rows.append(np.concatenate([[float(cls_1based)], tr_spec]))
            te_rows.append(np.concatenate([[float(cls_1based)], te_spec]))
    s_tr, s_te = io.StringIO(), io.StringIO()
    np.savetxt(s_tr, np.vstack(tr_rows), fmt="%.6e")
    np.savetxt(s_te, np.vstack(te_rows), fmt="%.6e")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("EthanolLevel_TRAIN.txt", s_tr.getvalue())
        zf.writestr("EthanolLevel_TEST.txt", s_te.getvalue())
    return path


def test_open_edge_dataset_registry_metadata_and_modalities() -> None:
    specs = list_open_edge_datasets()
    assert len(specs) >= 7
    assert set(OPEN_EDGE_DATASETS.keys()) == {
        "mnist1d",
        "uci_gas_sensor_drift",
        "uci_har_inertial",
        "cwru_bearing_vibration",
        "uci_hydraulic_vibration",
        "sers_bacteria_raman_2026",
        "ethanol_nir_spectroscopy",
    }

    modalities = set()
    for spec in specs:
        assert spec.n_features >= 1
        assert spec.n_classes >= 2
        assert isinstance(spec.has_temporal_drift_batches, bool)
        assert spec.url.startswith("https://")
        assert spec.landing_url.startswith("https://")
        assert spec.spdx_license in COMMERCIAL_SAFE_SPDX_LICENSES
        assert spec.commercial_use_ok is True
        assert len(spec.title) > 5 and len(spec.citation) > 10
        verify_dataset_license(spec, require_commercial_use=True)
        modalities.add(spec.modality)

    assert {
        "1d_signal",
        "chemical_gas_sensor_drift",
        "inertial_imu",
        "vibration_fault",
        "raman_spectroscopy",
        "nir_spectroscopy",
    }.issubset(modalities)

    drift_specs = list_open_edge_datasets(commercial_only=True, drift_capable_only=True)
    drift_ids = {s.dataset_id for s in drift_specs}
    assert "uci_gas_sensor_drift" in drift_ids
    assert "cwru_bearing_vibration" in drift_ids
    assert "mnist1d" not in drift_ids

    assert get_open_edge_dataset("uci_gas_sensor_drift").n_features == 128
    with pytest.raises(KeyError, match="Unknown dataset_id"):
        get_open_edge_dataset("nonexistent_sensor_dataset")


def test_spdx_license_gate_blocks_noncommercial_and_invalid_specs() -> None:
    with pytest.raises(DatasetLicenseError, match="non-commercial"):
        OpenEdgeDataset(
            dataset_id="contradictory_nc",
            n_features=16,
            n_classes=3,
            has_temporal_drift_batches=False,
            url="https://example.org/data.zip",
            spdx_license="CC-BY-NC-4.0",
            commercial_use_ok=True,
        )

    eval_only = OpenEdgeDataset(
        dataset_id="academic_eval_bench",
        n_features=32,
        n_classes=4,
        has_temporal_drift_batches=False,
        url="https://example.org/eval.zip",
        spdx_license="CC-BY-NC-4.0",
        commercial_use_ok=False,
    )
    with pytest.raises(DatasetLicenseError, match="prohibits commercial"):
        verify_dataset_license(eval_only, require_commercial_use=True)
    # Allowed when commercial requirement is explicitly relaxed for academic evaluation
    verify_dataset_license(eval_only, require_commercial_use=False)

    unverified = OpenEdgeDataset(
        dataset_id="unknown_lic",
        n_features=16,
        n_classes=2,
        has_temporal_drift_batches=False,
        url="https://example.org/raw.csv",
        spdx_license="CUSTOM",
        commercial_use_ok=False,
    )
    with pytest.raises(DatasetLicenseError, match="unverified SPDX"):
        verify_dataset_license(unverified, require_commercial_use=False)

    with pytest.raises(ValueError, match="HTTPS"):
        OpenEdgeDataset(
            dataset_id="insecure_http",
            n_features=8,
            n_classes=2,
            has_temporal_drift_batches=False,
            url="http://example.org/plain.csv",
            spdx_license="MIT",
            commercial_use_ok=True,
        )


def test_deterministic_feature_normalizer_modes_and_leakage_free_sha256() -> None:
    rng = np.random.default_rng(7)
    X_tr = rng.normal(loc=10.0, scale=4.0, size=(48, 12)).astype(np.float32)
    X_tr[:, 0] = 42.0  # Constant zero-variance channel guard
    X_tr[:, 1] *= 2500.0  # Wide MOX-like dynamic range channel

    for kind in ("zscore", "robust_iqr", "minmax_symmetric", "signed_log1p_zscore", "snv_zscore"):
        norm = DeterministicFeatureNormalizer(kind=kind).fit(X_tr)  # type: ignore[arg-type]
        X_out = norm.transform(X_tr)
        assert X_out.shape == X_tr.shape
        assert X_out.dtype == np.float32
        assert np.all(np.isfinite(X_out))
        # Single-sample 1D transform matches row 0 of batch 2D transform
        assert np.allclose(norm.transform(X_tr[0]), X_out[0], atol=1e-6)

        sha_before = norm.state_sha256()
        assert len(sha_before) == 64
        # Transforming extreme OOD test inputs never mutates training normalizer state
        _ = norm.transform(np.full((5, 12), 1e6, dtype=np.float32))
        assert norm.state_sha256() == sha_before

    # Verify SNV removes multiplicative laser power scaling & additive baseline offset per spectrum
    base_spectra = rng.normal(0.0, 1.0, size=(30, 64)).astype(np.float64)
    snv = DeterministicFeatureNormalizer(kind="snv_zscore").fit(base_spectra)
    shifted_scaled = 3.7 * base_spectra + 125.0
    assert np.allclose(snv.transform(base_spectra), snv.transform(shifted_scaled), atol=1e-5)


def test_all_seven_dataset_archive_parsers_and_restricted_pickle_guard(tmp_path: Path) -> None:
    fixtures = {
        "mnist1d": _write_mnist1d_fixture(tmp_path / "mnist1d_data.pkl"),
        "uci_gas_sensor_drift": _write_gas_sensor_drift_zip(
            tmp_path / "gas_sensor_array_drift_dataset.zip", n_batches=10, n_per_class_batch1=4, n_per_class_later=2
        ),
        "uci_har_inertial": _write_uci_har_nested_zip(tmp_path / "har.zip"),
        "cwru_bearing_vibration": _write_cwru_csv_fixture(tmp_path / "CWRU.csv"),
        "uci_hydraulic_vibration": _write_uci_hydraulic_zip(tmp_path / "hydraulic.zip"),
        "sers_bacteria_raman_2026": _write_sers_bacteria_csv(tmp_path / "sers.csv"),
        "ethanol_nir_spectroscopy": _write_ethanol_nir_zip(tmp_path / "EthanolLevel.zip"),
    }

    for ds_id, fpath in fixtures.items():
        loaded = load_open_edge_dataset(ds_id, archive_path=fpath)
        assert loaded.X.dtype == np.float32
        assert loaded.y.dtype == np.int64
        assert loaded.X.shape[1] == loaded.spec.n_features
        assert int(loaded.y.min()) == 0
        assert int(loaded.y.max()) == loaded.spec.n_classes - 1
        assert len(loaded.raw_sha256) == 64
        if loaded.spec.has_temporal_drift_batches:
            assert loaded.batch_ids is not None
            assert len(np.unique(loaded.batch_ids)) >= 2

    # Specifically verify numeric batch ordering (1, 2, ..., 10) for UCI Gas Sensor Drift
    gas = load_open_edge_dataset("uci_gas_sensor_drift", archive_path=fixtures["uci_gas_sensor_drift"])
    assert gas.batch_ids is not None
    # First sample comes from batch 1 and last sample comes from batch 10 (not batch 9!)
    assert int(gas.batch_ids[0]) == 1
    assert int(gas.batch_ids[-1]) == 10
    assert list(np.unique(gas.batch_ids)) == list(range(1, 11))

    # Verify restricted unpickler blocks non-ndarray code execution gadgets
    class _EvilGadget:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            import os

            return (os.system, ("echo blocked",))

    bad_pkl = tmp_path / "evil_mnist1d.pkl"
    bad_pkl.write_bytes(pickle.dumps(_EvilGadget()))
    with pytest.raises(pickle.UnpicklingError, match="Disallowed pickle global"):
        load_open_edge_dataset("mnist1d", archive_path=bad_pkl)


def test_open_dataset_distill_quantize_int8_w8a8_and_int4_lft1(tmp_path: Path) -> None:
    m1d_path = _write_mnist1d_fixture(tmp_path / "mnist1d_data.pkl")
    m1d = load_open_edge_dataset("mnist1d", archive_path=m1d_path)
    report_m1d = run_open_dataset_distill_quantize(
        m1d,
        teacher_hidden=(96, 64),
        student_hidden=(32,),
        student_epochs=160,
        seed=0,
    )
    assert report_m1d["dataset_id"] == "mnist1d"
    assert report_m1d["spdx_license"] == "Apache-2.0"
    assert report_m1d["student_kl_fp32"]["test_acc"] >= 0.90
    assert report_m1d["student_int8_w8a8"]["test_acc"] >= 0.90
    assert report_m1d["student_int4_pc_mse"]["test_acc"] >= 0.88
    assert report_m1d["student_int4_pc_mse"]["agreement_with_kl_fp32"] >= 0.90
    assert (
        report_m1d["student_int4_pc_mse"]["weight_bytes"]
        < report_m1d["student_int8_w8a8"]["weight_bytes"]
        < report_m1d["student_kl_fp32"]["weight_bytes"]
        < report_m1d["teacher_fp32"]["weight_bytes"]
    )

    cwru_path = _write_cwru_csv_fixture(tmp_path / "CWRU.csv")
    cwru = load_open_edge_dataset("cwru_bearing_vibration", archive_path=cwru_path)
    report_cwru = run_open_dataset_distill_quantize(
        cwru,
        teacher_hidden=(64, 32),
        student_hidden=(16,),
        student_epochs=140,
        seed=0,
    )
    assert report_cwru["normalizer"]["kind"] == "robust_iqr"
    assert report_cwru["student_int4_pc_mse"]["test_acc"] >= 0.88
    assert report_cwru["student_int4_pc_mse"]["w4a8_test_acc"] >= 0.85


def test_uci_gas_sensor_multi_batch_drift_abstaining_int4_safe_hold(tmp_path: Path) -> None:
    gas_zip = _write_gas_sensor_drift_zip(
        tmp_path / "gas_drift_5batch.zip",
        n_batches=5,
        n_per_class_batch1=36,
        n_per_class_later=12,
        seed=2026,
    )
    gas = load_open_edge_dataset("uci_gas_sensor_drift", archive_path=gas_zip)

    # 1) Full 128-channel MOX drift evaluation trained strictly on Batch 1
    full_eval = evaluate_temporal_drift_abstaining_int4(
        gas,
        source_batches=[1],
        n_hidden=32,
        seed=2026,
    )
    assert full_eval["dataset_id"] == "uci_gas_sensor_drift"
    assert full_eval["source_batches"] == [1]
    assert full_eval["evaluated_drift_batches"] == [2, 3, 4, 5]
    assert full_eval["n_selected_features"] == 128

    src = full_eval["source_in_domain"]
    assert src["naive_int4_accuracy_pct"] >= 90.0
    assert src["guarded_accepted_accuracy_pct"] >= 90.0

    by_batch = {row["batch_id"]: row for row in full_eval["drift_batches"]}
    # Severe/saturated later drift batches (4 & 5) trigger deterministic SAFE_HOLD abstention
    assert by_batch[2]["guarded_safe_hold_rate_pct"] < by_batch[4]["guarded_safe_hold_rate_pct"]
    assert by_batch[5]["guarded_safe_hold_rate_pct"] >= 90.0
    assert by_batch[5]["guarded_silent_error_rate_pct"] <= 10.0
    assert by_batch[5]["naive_silent_error_rate_pct"] >= 60.0

    # 2) MCU SRAM-budgeted (< 512 B) 32-channel top-F-ratio MOX projection
    mcu_eval = evaluate_temporal_drift_abstaining_int4(
        gas,
        source_batches=[1],
        n_hidden=24,
        max_monitor_features=32,
        seed=2026,
    )
    assert mcu_eval["n_selected_features"] == 32
    assert mcu_eval["extra_sram_bytes"] < 512
    mcu_b5 = {r["batch_id"]: r for r in mcu_eval["drift_batches"]}[5]
    assert mcu_b5["guarded_safe_hold_rate_pct"] >= 90.0


def test_fetch_open_datasets_cli_list_and_mocked_ondemand_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc_list = fetch_cli.main(["--list", "--json"])
    assert rc_list == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) >= 7
    assert any(item["dataset_id"] == "uci_gas_sensor_drift" for item in listed)

    fixture_src = _write_cwru_csv_fixture(tmp_path / "mock_cwru.csv")
    raw_bytes = fixture_src.read_bytes()

    class _DummyHTTPResp:
        status = 200

        def __init__(self, payload: bytes) -> None:
            self._buf = io.BytesIO(payload)

        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        def __enter__(self) -> "_DummyHTTPResp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=60.0: _DummyHTTPResp(raw_bytes),
    )

    out_data_dir = tmp_path / "data"
    rc_fetch = fetch_cli.main(
        [
            "--dataset",
            "cwru_bearing_vibration",
            "--data-dir",
            str(out_data_dir),
            "--smoke-distill",
            "--smoke-drift",
            "--json",
        ]
    )
    assert rc_fetch == 0
    payload = json.loads(capsys.readouterr().out)
    fetched_row = payload["fetched"][0]
    assert fetched_row["dataset_id"] == "cwru_bearing_vibration"
    assert fetched_row["n_features"] == 17
    assert "distill_quantize" in fetched_row
    assert "drift_abstaining_int4" in fetched_row
    assert (out_data_dir / "cwru_bearing_vibration" / "manifest.json").is_file()
