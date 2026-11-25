"""Comprehensive unit and regression tests for lab-feet resolving audit issues #1..#24."""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor

import demo.distill_quantize as demo_mod
from lab_feet import (
    BINARY_MAGIC,
    QuantizedLayer,
    QuantizedModel,
    compute_binary_header_bytes,
    cpu_model,
    distill_kl_student,
    equivariant_irrep_linear,
    evaluate_quantized_graph,
    export_c99_header,
    export_quantized_graph,
    export_quantized_onnx,
    fit_sklearn_mlp,
    forward,
    forward_int8,
    from_binary_buffer,
    get_layers,
    guard_equivariant_spatial_quantization,
    kl_divergence_soft_targets,
    latency_stats,
    load_quantized_binary,
    make_stratified_splits,
    max_safe_int32_fan_in,
    max_so3_equivariance_error,
    pack_4bit,
    prepare_fixed_point_model,
    quantize,
    quantize_equivariant_irrep_weights,
    quantize_input_for_c99,
    quantize_model,
    random_so3_rotation,
    run_autoencoder_compression_benchmark,
    run_digits_benchmark,
    run_wetlab_acoustic_benchmark,
    save_quantized_binary,
    simulate_c99_fixed_point,
    smoothquant_migrate,
    softmax,
    to_binary_buffer,
    unpack_4bit,
    verify_int32_mac_bound,
    zero_mean_logits,
)
from lab_feet.cli import main as cli_main
from lab_feet.quantize import _quantize_activations_int8

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_issue_1_cpu_model_cross_platform_and_arm_fallback(tmp_path: Path) -> None:
    """Issue #1: cpu_model() never crashes on missing /proc/cpuinfo and parses x86 + ARM64."""
    missing = str(tmp_path / "no_such_cpuinfo")
    fallback = cpu_model(cpuinfo_path=missing)
    assert isinstance(fallback, str) and len(fallback) > 0

    x86_info = tmp_path / "cpuinfo_x86"
    x86_info.write_text("processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Platinum\n", encoding="utf-8")
    assert cpu_model(cpuinfo_path=str(x86_info)) == "Intel(R) Xeon(R) Platinum"

    arm_info = tmp_path / "cpuinfo_arm"
    arm_info.write_text("Serial\t\t: 10000000\nModel\t\t: Raspberry Pi 5 Model B Rev 1.0\n", encoding="utf-8")
    assert cpu_model(cpuinfo_path=str(arm_info)) == "Raspberry Pi 5 Model B Rev 1.0"


def test_issue_2_quantize_all_zero_and_zero_channel_guard() -> None:
    """Issue #2: All-zero tensors and zero channels never divide by zero or produce NaN."""
    zeros = np.zeros((16, 8), dtype=np.float32)
    for bits in (8, 4):
        for per_ch in (False, True):
            for mode in ("symmetric", "symmetric_full"):
                q, s = quantize(zeros, bits, per_channel=per_ch, mode=mode)  # type: ignore[misc]
                assert q.dtype == np.int8
                assert np.all(q == 0)
                assert np.all(np.isfinite(s)) and np.all(s == 1.0)

            qa, sa, zpa = quantize(zeros, bits, per_channel=per_ch, mode="asymmetric")  # type: ignore[misc]
            assert qa.dtype == np.int8 and np.all(qa == 0)
            assert np.all(sa == 1.0) and np.all(zpa == 0)

    # Partial zero output channel in per_channel=True
    mixed = np.zeros((8, 4), dtype=np.float32)
    mixed[:, 1] = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    q_mix, s_mix = quantize(mixed, 4, per_channel=True)  # type: ignore[misc]
    assert np.all(q_mix[:, 0] == 0)
    assert float(s_mix[0, 0]) == 1.0
    assert np.all(np.isfinite(q_mix.astype(np.float32) * s_mix))


def test_issue_3_zero_mean_logits_and_kl_t2_distillation() -> None:
    """Issue #3: Zero-mean logits invariant and KL(T^2) + CE soft-target distillation."""
    rng = np.random.default_rng(7)
    logits = rng.normal(loc=5.0, scale=2.0, size=(20, 6)).astype(np.float32)
    centered = zero_mean_logits(logits)
    assert np.allclose(centered.mean(axis=1), 0.0, atol=1e-6)
    # Softmax is invariant to per-sample row offsets
    assert np.allclose(softmax(logits), softmax(centered), atol=1e-6)

    # KL divergence is 0 on identical logits and strictly positive otherwise
    assert np.isclose(kl_divergence_soft_targets(logits, logits, temperature=4.0), 0.0, atol=1e-9)
    assert kl_divergence_soft_targets(logits, logits[:, ::-1], temperature=4.0) > 0.1


def test_issue_4_blas_single_thread_pinning() -> None:
    """Issue #4: BLAS thread environment variables are pinned to 1."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ.get(var) == "1"


def test_issue_5_quantize_bit_width_validation() -> None:
    """Issue #5: Reject invalid bit-widths outside [2, 8] or boolean/float values."""
    W = np.ones((4, 4), dtype=np.float32)
    for bad_bits in (0, 1, -4, 9, 16, True, False):
        with pytest.raises(ValueError):
            quantize(W, bad_bits)  # type: ignore[arg-type]


def test_issue_6_and_7_pack_unpack_4bit_invariants_and_int8_buffer_views() -> None:
    """Issues #6, #7, #15: Exhaustive -8..7 round-trips, odd shapes, int8 views, and bounds check."""
    all_codes = np.arange(-8, 8, dtype=np.int8)
    packed_all = pack_4bit(all_codes)
    assert packed_all.dtype == np.uint8
    assert packed_all.nbytes == 8  # 16 weights -> 8 bytes (exact 2 weights/byte)
    assert np.array_equal(unpack_4bit(packed_all, all_codes.shape), all_codes)

    # Regression test for #6: passing a signed int8 view of packed bytes must NOT sign-extend high nibbles!
    signed_view = packed_all.view(np.int8)
    assert np.array_equal(unpack_4bit(signed_view, all_codes.shape), all_codes)

    # Odd shapes, 1-element, empty, and 3D tensors
    rng = np.random.default_rng(42)
    for shape in [(0, 4), (1,), (7,), (3, 5), (4, 4), (3, 5, 7)]:
        codes = rng.integers(-8, 8, size=shape, dtype=np.int8)
        p = pack_4bit(codes)
        assert p.nbytes == (codes.size + 1) // 2
        u = unpack_4bit(p, shape)
        assert u.shape == shape and u.dtype == np.int8
        assert np.array_equal(u, codes)

    # Regression test for #7: out-of-range codes must raise ValueError
    with pytest.raises(ValueError, match="range"):
        pack_4bit(np.array([12], dtype=np.int8))
    with pytest.raises(ValueError, match="range"):
        pack_4bit(np.array([-9], dtype=np.int16))
    with pytest.raises(TypeError):
        pack_4bit(np.array([1.0, 2.0], dtype=np.float32))


def test_issue_8_and_9_per_channel_and_full_range_asymmetric_modes() -> None:
    """Issues #8 & #9: Per-channel scaling reduces outlier error; symmetric_full/asymmetric populate -8."""
    rng = np.random.default_rng(11)
    W = rng.normal(0.0, 0.1, size=(32, 8)).astype(np.float32)
    W[:, 0] *= 50.0  # Huge outlier in channel 0 only

    qm_pt = quantize_model([(W, np.zeros(8, dtype=np.float32))], 4, per_channel=False)
    qm_pc = quantize_model([(W, np.zeros(8, dtype=np.float32))], 4, per_channel=True)

    # Per-channel stores 8 float32 scales (32 B) instead of 1 float32 scale (4 B) -> +28 B
    assert qm_pc.nbytes - qm_pt.nbytes == (8 - 1) * 4

    mse_pt = float(np.mean(np.square(W[:, 1:] - qm_pt.dequantize()[0][0][:, 1:])))
    mse_pc = float(np.mean(np.square(W[:, 1:] - qm_pc.dequantize()[0][0][:, 1:])))
    assert mse_pc < mse_pt * 0.05  # >20x lower MSE on non-outlier channels!

    # Issue #9: symmetric clamps strictly to [-7, 7] (leaving -8 unused), while symmetric_full & asymmetric populate -8
    neg_heavy = np.array([[-8.0, 7.0, -4.0, 0.0]], dtype=np.float32)
    q_sym, _ = quantize(neg_heavy, 4, mode="symmetric")  # type: ignore[misc]
    assert int(q_sym.min()) == -7 and int(q_sym.max()) <= 7

    q_full, s_full = quantize(neg_heavy, 4, mode="symmetric_full")  # type: ignore[misc]
    assert int(q_full.min()) == -8 and int(q_full.max()) == 7
    assert np.allclose(q_full.astype(np.float32) * s_full, neg_heavy)

    q_asym, s_asym, zp_asym = quantize(np.array([[-15.0, 0.0]], dtype=np.float32), 4, mode="asymmetric")  # type: ignore[misc]
    assert int(q_asym.min()) == -8 and int(q_asym.max()) == 7
    deq_asym = (q_asym.astype(np.float32) - zp_asym.astype(np.float32)) * s_asym
    assert np.allclose(deq_asym, np.array([[-15.0, 0.0]], dtype=np.float32), atol=1e-5)


def test_issue_10_and_15_activation_validation_softmax_and_sklearn_equivalence() -> None:
    """Issues #10 & #15: Reject non-ReLU in get_layers() by default; verify exact sklearn parity."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 12)).astype(np.float32)
    y = rng.integers(0, 4, size=40)

    # Softmax shift invariance (exact in float64; bounded by float32 input addition ULP in float32)
    z = rng.normal(size=(10, 5)).astype(np.float32)
    assert np.allclose(softmax(z + 1000.0), softmax(z), atol=1e-5)
    z64 = z.astype(np.float64)
    assert np.allclose(softmax(z64 + 1000.0), softmax(z64), atol=1e-12)
    assert np.allclose(softmax(z).sum(axis=1), 1.0, atol=1e-6)

    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", max_iter=50, random_state=0)
    fit_sklearn_mlp(clf, X, y)
    layers = get_layers(clf)
    assert np.allclose(softmax(forward(layers, X)), clf.predict_proba(X), atol=1e-5)

    tanh_clf = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", max_iter=20, random_state=0)
    fit_sklearn_mlp(tanh_clf, X, y)
    with pytest.raises(ValueError, match="Unsupported MLP hidden activation 'tanh'"):
        get_layers(tanh_clf)
    tanh_layers = get_layers(tanh_clf, require_relu=False)
    assert np.allclose(softmax(forward(tanh_layers, X, activation="tanh")), tanh_clf.predict_proba(X), atol=1e-5)


def test_issue_11_and_18_scoped_warnings_and_scoped_rng_state() -> None:
    """Issues #11 & #18: No global ConvergenceWarning filter and no global np.random.seed mutation."""
    # Verify module imports did not inject a global ConvergenceWarning ignore filter
    for flt in warnings.filters:
        action, _msg, category, _mod, _ln = flt
        assert not (action == "ignore" and category.__name__ == "ConvergenceWarning")

    X = np.ones((20, 4), dtype=np.float32)
    y = np.arange(20) % 3
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1, random_state=0)
    filters_before = list(warnings.filters)
    diag = fit_sklearn_mlp(clf, X, y)
    assert list(warnings.filters) == filters_before
    assert diag.n_iter == 1
    assert diag.converged is False

    # Verify global legacy NumPy RNG state is completely untouched by distill_kl_student
    np.random.seed(12345)
    state_before = np.random.get_state()[1].copy()
    t_logits = np.zeros((20, 3), dtype=np.float32)
    distill_kl_student(X, y, t_logits, hidden_sizes=(8,), epochs=5, seed=99)
    state_after = np.random.get_state()[1].copy()
    assert np.array_equal(state_before, state_after)


def test_issue_12_forward_int8_w8a8_integer_matmul() -> None:
    """Issue #12: True W8A8 integer matmul matches fp32 forward closely and handles zero inputs."""
    rng = np.random.default_rng(19)
    X = rng.normal(size=(32, 16)).astype(np.float32)
    X[0] = 0.0  # Test zero-row activation guard
    layers = [
        (rng.normal(0.0, 0.3, size=(16, 24)).astype(np.float32), rng.normal(0.0, 0.1, size=24).astype(np.float32)),
        (rng.normal(0.0, 0.3, size=(24, 6)).astype(np.float32), rng.normal(0.0, 0.1, size=6).astype(np.float32)),
    ]
    q8 = quantize_model(layers, 8, per_channel=True, mode="symmetric")
    out_fp32 = forward(layers, X)
    out_w8a8 = forward_int8(q8, X)
    assert out_w8a8.shape == out_fp32.shape
    assert np.all(np.isfinite(out_w8a8))
    assert float((out_w8a8.argmax(axis=1) == out_fp32.argmax(axis=1)).mean()) >= 0.93


def test_issue_13_and_23_quantized_layer_npz_and_lft1_binary_serialization(tmp_path: Path) -> None:
    """Issues #13 & #23: Genuine packed buffers, .npz round-trip, and byte-exact LFT1 binary format."""
    rng = np.random.default_rng(23)
    layers = [
        (rng.normal(0.0, 0.5, size=(64, 32)).astype(np.float32), rng.normal(0.0, 0.1, size=32).astype(np.float32)),
        (rng.normal(0.0, 0.5, size=(32, 10)).astype(np.float32), rng.normal(0.0, 0.1, size=10).astype(np.float32)),
    ]
    X_test = rng.normal(size=(10, 64)).astype(np.float32)

    # Check exact student byte sizes: int8 per-tensor = 2544 B, 4-bit per-tensor = 1360 B, 4-bit pc = 1520 B
    q8_pt = quantize_model(layers, 8, per_channel=False)
    q4_pt = quantize_model(layers, 4, per_channel=False)
    q4_pc = quantize_model(layers, 4, per_channel=True)
    q4_asym = quantize_model(layers, 4, per_channel=True, mode="asymmetric")

    assert q8_pt.nbytes == 2544
    assert q4_pt.nbytes == 1360
    assert q4_pc.nbytes == 1520
    assert q4_pt.layers[0].packed_weights.dtype == np.uint8
    assert q4_pt.layers[0].packed_weights.nbytes == (64 * 32) // 2

    # Legacy tuple unpacking works seamlessly
    deq_layers, nb = q4_pt
    assert nb == 1360 and len(deq_layers) == 2

    # Single QuantizedLayer .npz round-trip
    layer_npz = q4_pc.layers[0].save_npz(tmp_path / "layer0.npz")
    reloaded_layer = QuantizedLayer.load_npz(layer_npz)
    assert np.allclose(reloaded_layer.dequantize()[0], q4_pc.layers[0].dequantize()[0])

    # Full QuantizedModel .npz round-trip
    model_npz = q4_asym.save_npz(tmp_path / "model_asym.npz")
    reloaded_npz = QuantizedModel.load_npz(model_npz)
    assert np.allclose(reloaded_npz.forward_dequant(X_test), q4_asym.forward_dequant(X_test))

    # Flat LFT1 binary container round-trip and exact on-disk byte verification
    for tag, qm in [("q8", q8_pt), ("q4_pt", q4_pt), ("q4_pc", q4_pc), ("q4_asym", q4_asym)]:
        bin_file = save_quantized_binary(qm, tmp_path / f"{tag}.lft1")
        expected_size = compute_binary_header_bytes(len(qm.layers)) + qm.nbytes
        assert bin_file.stat().st_size == expected_size
        assert bin_file.read_bytes()[:4] == BINARY_MAGIC
        reloaded_bin = load_quantized_binary(bin_file)
        assert reloaded_bin.nbytes == qm.nbytes
        assert np.allclose(reloaded_bin.forward_dequant(X_test), qm.forward_dequant(X_test))

    with pytest.raises(ValueError, match="magic"):
        from_binary_buffer(b"BAD!" + b"\x00" * 32)


def test_issue_14_latency_stats_percentiles() -> None:
    """Issue #14: latency_stats reports median_us, p95_us, p99_us, iqr_us, and mean_us."""
    layers = [(np.ones((8, 4), dtype=np.float32), np.zeros(4, dtype=np.float32))]
    X = np.ones((16, 8), dtype=np.float32)
    stats = latency_stats(layers, X, n_warmup=10, n_timing=80)
    assert 0.0 < stats.median_us <= stats.p95_us <= stats.p99_us
    assert stats.iqr_us >= 0.0


def test_issue_20_mse_optimal_clipping_scale_search() -> None:
    """Issue #20: Grid-searched clipping ratio reduces 4-bit MSE on heavy-tailed weights."""
    rng = np.random.default_rng(2026)
    W = rng.normal(0.0, 0.2, size=(128, 16)).astype(np.float32)
    W[0, :] = 3.5  # Single extreme row outlier in every column
    X_cal = rng.normal(0.0, 1.0, size=(64, 128)).astype(np.float32)

    q_abs, s_abs = quantize(W, 4, per_channel=True, clip_search="none")  # type: ignore[misc]
    q_wmse, s_wmse = quantize(W, 4, per_channel=True, clip_search="weight_mse")  # type: ignore[misc]
    q_amse, s_amse = quantize(W, 4, per_channel=True, clip_search="activation_mse", X_cal=X_cal)  # type: ignore[misc]

    err_abs = float(np.mean(np.square(W - q_abs.astype(np.float32) * s_abs)))
    err_wmse = float(np.mean(np.square(W - q_wmse.astype(np.float32) * s_wmse)))
    assert err_wmse < err_abs

    act_err_abs = float(np.mean(np.square(X_cal @ W - X_cal @ (q_abs.astype(np.float32) * s_abs))))
    act_err_amse = float(np.mean(np.square(X_cal @ W - X_cal @ (q_amse.astype(np.float32) * s_amse))))
    assert act_err_amse < act_err_abs


def test_issue_24_so3_equivariant_irrep_quantization() -> None:
    """Issue #24: Scalar channel weight quantization commutes with 3D rotations R in SO(3)."""
    rng = np.random.default_rng(77)
    X_vec = rng.normal(size=(12, 8, 3)).astype(np.float32)  # 12 atoms, 8 channels, l=1 vectors
    W_inv = rng.normal(0.0, 0.6, size=(8, 6)).astype(np.float32)

    for bits in (8, 4):
        qlayer = quantize_equivariant_irrep_weights(
            W_inv, bits=bits, irrep_l=1, expected_channel_shape=(8, 6)
        )
        for rot_seed in range(5):
            R = random_so3_rotation(rot_seed)
            err = max_so3_equivariance_error(X_vec, qlayer, R)
            assert err < 1e-5, f"SO(3) equivariance broken at {bits}-bit: err={err}"

    # Guard rejects flattened (8*3, 6*3) = (24, 18) Cartesian matrix
    W_bad_spatial = rng.normal(size=(24, 18)).astype(np.float32)
    with pytest.raises(ValueError, match="breaks SO\\(3\\) rotation equivariance"):
        guard_equivariant_spatial_quantization(W_bad_spatial, irrep_l=1, expected_channel_shape=(8, 6))


def test_issues_21_22_and_cli_schema_end_to_end(tmp_path: Path) -> None:
    """Issues #16, #21, #22: End-to-end digits, wet-lab domain, autoencoder compression, and JSON schema."""
    wetlab = run_wetlab_acoustic_benchmark(seed=0, n_warmup=5, n_timing=20)
    by_w = {r["name"]: r for r in wetlab["models"]}
    assert by_w["wetlab student (16,) KL+CE fp32"]["test_acc"] >= by_w["wetlab student (16,) hard fp32"]["test_acc"]

    ae = run_autoencoder_compression_benchmark(seed=0)
    assert ae["compression_ratio_4bit"] > 7.0
    assert ae["int4_anomaly_auroc"] >= 0.98

    json_report = tmp_path / "report.json"
    bin_file = tmp_path / "student.lft1"
    rc = demo_mod.main(
        [
            "--digits-only",
            "--n-warmup",
            "10",
            "--n-timing",
            "50",
            "--assert-thresholds",
            "--json-out",
            str(json_report),
            "--export-bin",
            str(bin_file),
        ]
    )
    assert rc == 0
    assert json_report.is_file() and bin_file.is_file()

    schema = json.loads((REPO_ROOT / "schema" / "benchmark_report.schema.json").read_text(encoding="utf-8"))
    report_data = json.loads(json_report.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(report_data)

    assert cli_main(["inspect-binary", str(bin_file)]) == 0
    schema_export = tmp_path / "exported_schema.json"
    assert cli_main(["export-schema", "--output", str(schema_export)]) == 0
    assert schema_export.is_file()


def test_issue_26_activation_clipping_unsigned_post_relu_smoothquant_and_outlier_weights() -> None:
    """Issue #26: Calibrated percentile/static activation clipping, unsigned ReLU, SmoothQuant, and alpha_min=0.25."""
    # (A) Single glare/EMI spike on channel 0 no longer crushes normal post-ReLU channels at 4-bit
    x_relu = np.array([[40.0, 1.5, 2.0, 2.8, 0.9, 1.2, 2.5, 1.8]], dtype=np.float32)
    xq_unclipped, sq_unclipped = _quantize_activations_int8(x_relu, act_bits=4)
    assert np.all(xq_unclipped[0, 1:] == 0)

    xq_pct, sq_pct = _quantize_activations_int8(
        x_relu, act_bits=4, act_clip_percentile=85.0, unsigned=True
    )
    assert xq_pct.dtype == np.uint8
    assert int(xq_pct.max()) == 15
    assert np.all(xq_pct[0, 1:] > 0)
    mse_unclipped_normal = float(np.mean((x_relu[0, 1:] - (xq_unclipped.astype(np.float32) * sq_unclipped)[0, 1:]) ** 2))
    mse_clipped_normal = float(np.mean((x_relu[0, 1:] - (xq_pct.astype(np.float32) * sq_pct)[0, 1:]) ** 2))
    assert mse_clipped_normal * 5.0 < mse_unclipped_normal

    # Negative inputs rejected when unsigned=True
    with pytest.raises(ValueError, match="unsigned=True"):
        _quantize_activations_int8(np.array([-0.5, 1.0], dtype=np.float32), act_bits=4, unsigned=True)

    # SmoothQuant migration preserves exact FP32 matmul X @ W while reducing 4-bit activation quantization MSE
    rng = np.random.default_rng(26)
    X_spiky = rng.normal(0.0, 0.4, size=(32, 16)).astype(np.float32)
    X_spiky[:, 0] *= 35.0
    W_clean = rng.normal(0.0, 0.3, size=(16, 8)).astype(np.float32)
    X_sm, W_sm, sm_scales = smoothquant_migrate(X_spiky, W_clean, alpha=0.5)
    assert sm_scales.shape == (16,)
    assert np.allclose(X_sm @ W_sm, X_spiky @ W_clean, atol=1e-4)

    q_raw = quantize_model([(W_clean, np.zeros(8, dtype=np.float32))], bits=4, per_channel=True)
    q_sm = quantize_model([(W_sm, np.zeros(8, dtype=np.float32))], bits=4, per_channel=True)
    y_true = X_spiky @ W_clean
    mse_raw_w4a4 = float(np.mean((y_true - forward_int8(q_raw, X_spiky, act_bits=4)) ** 2))
    mse_sm_w4a4 = float(np.mean((y_true - forward_int8(q_sm, X_sm, act_bits=4)) ** 2))
    assert mse_sm_w4a4 * 5.0 < mse_raw_w4a4

    # (B) Per-tensor 4-bit weight_mse search retains non-outlier weights instead of collapsing 100% to 0
    rng_b = np.random.default_rng(0)
    W_out = rng_b.normal(0.0, 0.2, size=(32, 8)).astype(np.float32)
    W_out[0, 0] = 12.0
    qm_none = quantize_model([(W_out, np.zeros(8, dtype=np.float32))], bits=4, per_channel=False, clip_search="none")
    qm_wmse = quantize_model(
        [(W_out, np.zeros(8, dtype=np.float32))],
        bits=4,
        per_channel=False,
        clip_search="weight_mse",
        alpha_min=0.25,
        grid_steps=41,
    )
    w_deq_none = qm_none.dequantize()[0][0]
    w_deq_wmse = qm_wmse.dequantize()[0][0]
    assert float(np.mean(w_deq_wmse[1:, 1:] != 0.0)) > 0.70
    bulk_mse_none = float(np.mean((W_out[:, 1:] - w_deq_none[:, 1:]) ** 2))
    bulk_mse_wmse = float(np.mean((W_out[:, 1:] - w_deq_wmse[:, 1:]) ** 2))
    assert bulk_mse_wmse * 5.0 < bulk_mse_none

    # (C) Consistent quantized h_cal propagation & static activation clip calibration
    layers_2 = [
        (rng.normal(0.0, 0.4, size=(16, 12)).astype(np.float32), np.zeros(12, dtype=np.float32)),
        (rng.normal(0.0, 0.4, size=(12, 4)).astype(np.float32), np.zeros(4, dtype=np.float32)),
    ]
    X_cal = rng.normal(0.0, 1.0, size=(48, 16)).astype(np.float32)
    qm_qcal = quantize_model(
        layers_2,
        bits=4,
        per_channel=True,
        clip_search="activation_mse",
        X_cal=X_cal,
        quantize_cal_activations=True,
        cal_act_bits=4,
        calibrate_static_act_clips=True,
    )
    assert qm_qcal.calibrated_act_clip_max is not None and len(qm_qcal.calibrated_act_clip_max) == 2
    out_u = qm_qcal.forward_int8(X_cal[:8], act_bits=4, unsigned_post_relu=True)
    assert out_u.shape == (8, 4) and np.all(np.isfinite(out_u))


def test_issue_27_odd_length_nibble_metadata_and_strict_payload_validation(tmp_path: Path) -> None:
    """Issue #27: Exact orig_len/shape metadata, odd padding checks, and LFT1 payload validation."""
    odd_vec = np.array([-8, -5, -1, 0, 3, 6, 7], dtype=np.int8)
    packed, meta = pack_4bit(odd_vec, return_metadata=True)  # type: ignore[misc]
    assert meta == {"orig_len": 7, "orig_shape": (7,)}
    assert packed.size == 4
    assert ((int(packed[-1]) >> 4) & 0x0F) == 0
    assert np.array_equal(unpack_4bit(packed, meta["orig_len"]), odd_vec)

    # Oversized buffers and non-zero high padding nibbles are rejected
    with pytest.raises(ValueError, match="exceeds expected"):
        unpack_4bit(np.zeros(100, dtype=np.uint8), (3, 5))

    corrupted_pad = packed.copy()
    corrupted_pad[-1] |= np.uint8(0x50)
    with pytest.raises(ValueError, match="padding nibble"):
        unpack_4bit(corrupted_pad, (7,))

    # Invalid QuantizedLayer payload / bias shapes are rejected immediately at construction
    with pytest.raises(ValueError, match="packed_weights size mismatch"):
        QuantizedLayer(
            packed_weights=np.array([0xFF, 0xFF], dtype=np.uint8),
            scale=np.ones(1, dtype=np.float32),
            bias=np.zeros(5, dtype=np.float32),
            shape=(3, 5),
            bits=4,
        )
    with pytest.raises(ValueError, match="bias length"):
        QuantizedLayer(
            packed_weights=np.zeros(8, dtype=np.uint8),
            scale=np.ones(1, dtype=np.float32),
            bias=np.zeros(99, dtype=np.float32),
            shape=(3, 5),
            bits=4,
        )

    # Odd-shaped multi-layer model `(25, 15) -> (15, 5)` round-trips cleanly across LFT1 and .npz
    rng = np.random.default_rng(27)
    odd_layers = [
        (rng.normal(0.0, 0.3, size=(25, 15)).astype(np.float32), rng.normal(0.0, 0.05, size=15).astype(np.float32)),
        (rng.normal(0.0, 0.3, size=(15, 5)).astype(np.float32), rng.normal(0.0, 0.05, size=5).astype(np.float32)),
    ]
    qm_odd = quantize_model(odd_layers, bits=4, per_channel=True, mode="symmetric_full")
    assert qm_odd.layers[0].orig_len == 375 and qm_odd.layers[0].packed_weights.size == 188
    assert qm_odd.layers[1].orig_len == 75 and qm_odd.layers[1].packed_weights.size == 38

    blob = bytearray(to_binary_buffer(qm_odd))
    reloaded = from_binary_buffer(blob)
    x_probe = rng.normal(size=(6, 25)).astype(np.float32)
    assert np.allclose(reloaded.forward_dequant(x_probe), qm_odd.forward_dequant(x_probe))

    npz_path = qm_odd.save_npz(tmp_path / "odd_model.npz")
    assert np.allclose(QuantizedModel.load_npz(npz_path).forward_dequant(x_probe), qm_odd.forward_dequant(x_probe))

    # Corrupting LFT1 odd high-nibble padding or header payload_nbytes raises ValueError
    corrupted_lft1 = bytearray(blob)
    # Last byte of LFT1 is the final byte of layer 1's odd (75-element) packed_weights
    corrupted_lft1[-1] |= 0xF0
    with pytest.raises(ValueError, match="padding nibble"):
        from_binary_buffer(corrupted_lft1)

    bad_hdr_lft1 = bytearray(blob)
    # Overwrite d_in of layer 0 from 25 to 26 while keeping payload_nbytes=188
    bad_hdr_lft1[8:12] = (26).to_bytes(4, "little")
    with pytest.raises(ValueError, match="Malformed LFT1 payload_nbytes"):
        from_binary_buffer(bad_hdr_lft1)


def test_issue_28_int32_mac_bounds_and_wide_layer_tiled_forward_int8() -> None:
    """Issue #28: Verify worst-case int32 MAC bounds and exact tiled reduction for d_in > 70,000."""
    assert max_safe_int32_fan_in(8, 8, asymmetric=False) == 133144
    assert max_safe_int32_fan_in(8, 8, asymmetric=True) == 66311
    assert verify_int32_mac_bound(133144, 127, 127) is True
    assert verify_int32_mac_bound(133145, 127, 127) is False

    # Asymmetric W8A8 with d_in = 72,000 (> 66,311) saturating every product to 127 * (127 - (-128)) = 32,385:
    # Untiled int32 would wrap around 72,000 * 32,385 = 2,331,720,000 > INT32_MAX -> negative!
    d_in = 72000
    qlayer_wide = QuantizedLayer(
        packed_weights=np.full((d_in, 2), 127, dtype=np.int8),
        scale=np.array([1.0, 0.5], dtype=np.float32),
        bias=np.zeros(2, dtype=np.float32),
        shape=(d_in, 2),
        bits=8,
        per_channel=True,
        mode="asymmetric",
        zero_point=np.array([-128, -128], dtype=np.int8),
    )
    qm_wide = QuantizedModel(layers=[qlayer_wide], activation="relu")
    x_ones = np.ones((1, d_in), dtype=np.float32)  # quantizes to +127 across all 72,000 channels, scale = 1/127
    out_wide = forward_int8(qm_wide, x_ones, act_bits=8)

    # Exact mathematical result: (72,000 * 127 * 255) * ((1/127) * [1.0, 0.5]) = 72,000 * 255 * [1.0, 0.5]
    expected = np.array([[72000.0 * 255.0, 72000.0 * 127.5]], dtype=np.float32)
    assert np.all(out_wide > 0.0)
    assert np.allclose(out_wide, expected, rtol=1e-5)


def test_issue_31_c99_header_and_onnx_exporters_bit_exact_parity(tmp_path: Path) -> None:
    """Issue #31: Freestanding C99 header (gcc -std=c99 -Wall -Wextra -Werror) and ONNX opset>=21 parity."""
    import importlib.util
    import shutil

    rng = np.random.default_rng(31)
    layers = [
        (rng.normal(0.0, 0.4, size=(25, 15)).astype(np.float32), rng.normal(0.0, 0.08, size=15).astype(np.float32)),
        (rng.normal(0.0, 0.4, size=(15, 5)).astype(np.float32), rng.normal(0.0, 0.08, size=5).astype(np.float32)),
    ]
    X_cal = rng.normal(0.0, 1.0, size=(64, 25)).astype(np.float32)
    X_test = rng.normal(0.0, 1.0, size=(24, 25)).astype(np.float32)

    q4 = quantize_model(layers, bits=4, per_channel=True, mode="symmetric_full", clip_search="activation_mse", X_cal=X_cal)
    lft1_file = save_quantized_binary(q4, tmp_path / "student_odd.lft1")

    # 1) CLI + Python freestanding C99 export verification
    hdr_cli = tmp_path / "cli_weights.h"
    assert cli_main(["export-c-header", str(lft1_file), "-o", str(hdr_cli), "--prefix", "bench_m0"]) == 0
    hdr_text_cli = hdr_cli.read_text(encoding="utf-8")
    assert "alignas(16)" in hdr_text_cli
    assert "#define LAB_FEET_SCRATCH_BYTES" in hdr_text_cli
    assert "lab_feet_infer_w4a8" in hdr_text_cli
    assert "lab_feet_decide_drift_abstaining_int4" in hdr_text_cli
    assert "malloc" not in hdr_text_cli

    spec = prepare_fixed_point_model(q4, X_cal=X_cal)
    x_i8 = quantize_input_for_c99(X_test, spec["input_scale"])
    ref_logits_i32, ref_logits_f32 = simulate_c99_fixed_point(spec, x_i8, x_is_quantized_int8=True)
    deq_preds = q4.forward_dequant(X_test).argmax(axis=1)
    assert float((ref_logits_f32.argmax(axis=1) == deq_preds).mean()) >= 0.91

    # Compile and execute C99 header if a C compiler is available on the host
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if cc is None and Path(r"C:\msys64\ucrt64\bin\gcc.exe").is_file():
        cc = r"C:\msys64\ucrt64\bin\gcc.exe"
    if cc is not None:
        hdr_cal = tmp_path / "cal_weights.h"
        export_c99_header(q4, hdr_cal, prefix="cal_m0", X_cal=X_cal)
        rows_c = ",\n".join("{" + ",".join(str(int(v)) for v in row) + "}" for row in x_i8)
        c_driver = tmp_path / "driver.c"
        c_driver.write_text(
            '#include <stdio.h>\n'
            '#include "cal_weights.h"\n'
            "int main(void) {\n"
            f"    static const int8_t xs[{len(x_i8)}][25] = {{ {rows_c} }};\n"
            "    int32_t logits[5];\n"
            f"    for (int s = 0; s < {len(x_i8)}; ++s) {{\n"
            "        int cls = lab_feet_infer_w4a8(xs[s], logits, NULL);\n"
            '        printf("%d:%d,%d,%d,%d,%d\\n", cls, logits[0], logits[1], logits[2], logits[3], logits[4]);\n'
            "    }\n"
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        exe_out = tmp_path / ("driver.exe" if os.name == "nt" else "driver")
        subprocess.run(
            [cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", str(c_driver), "-o", str(exe_out)],
            check=True,
        )
        lines = subprocess.check_output([str(exe_out)], text=True).strip().splitlines()
        assert len(lines) == len(x_i8)
        for idx_s, line in enumerate(lines):
            cls_str, vec_str = line.split(":")
            c_logits = [int(tok) for tok in vec_str.split(",")]
            assert c_logits == ref_logits_i32[idx_s].tolist()
            assert int(cls_str) == int(np.argmax(ref_logits_i32[idx_s]))

    # 2) ONNX opset >= 21 graph JSON + ModelProto export & reference/ORT parity
    graph_qdq = export_quantized_graph(q4, opset=21, operator_mode="qdq")
    assert graph_qdq["opset_version"] == 21
    assert any(init["data_type"] == "INT4" for init in graph_qdq["initializers"])
    assert np.allclose(evaluate_quantized_graph(graph_qdq, X_test), q4.forward_dequant(X_test), atol=1e-5)

    graph_qlin = export_quantized_graph(q4, opset=21, operator_mode="qlinear", X_cal=X_cal)
    assert any(node["op_type"] == "QLinearMatMul" for node in graph_qlin["nodes"])
    py_qlin = evaluate_quantized_graph(graph_qlin, X_test)
    assert py_qlin.shape == (24, 5)

    onnx_file = tmp_path / "student_odd.onnx"
    json_graph_file = tmp_path / "student_odd.onnx.json"
    assert (
        cli_main(
            [
                "export-onnx",
                str(lft1_file),
                "-o",
                str(onnx_file),
                "--opset",
                "21",
                "--json-out",
                str(json_graph_file),
            ]
        )
        == 0
    )
    assert onnx_file.is_file() and onnx_file.stat().st_size > 64
    assert json_graph_file.is_file()

    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort

        so_exact = ort.SessionOptions()
        so_exact.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_exact = ort.InferenceSession(onnx_file.read_bytes(), sess_options=so_exact, providers=["CPUExecutionProvider"])
        ort_exact = sess_exact.run(None, {"input": X_test})[0]
        assert np.allclose(ort_exact, q4.forward_dequant(X_test), atol=1e-5)

        sess_opt = ort.InferenceSession(onnx_file.read_bytes(), providers=["CPUExecutionProvider"])
        ort_opt = sess_opt.run(None, {"input": X_test})[0]
        assert float((ort_opt.argmax(axis=1) == q4.forward_dequant(X_test).argmax(axis=1)).mean()) >= 0.995

        onnx_qlin_bytes = export_quantized_onnx(q4, opset=21, operator_mode="qlinear", X_cal=X_cal)
        sess_qlin = ort.InferenceSession(onnx_qlin_bytes, providers=["CPUExecutionProvider"])
        ort_qlin = sess_qlin.run(None, {"input": X_test})[0]
        assert np.allclose(ort_qlin, py_qlin, atol=1e-5)


def test_issues_16_17_19_ci_requirements_readme_and_english_only() -> None:
    """Issues #16, #17, #19: Verify requirements pins, multi-OS CI matrix, README grammar, and English-only."""
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "numpy>=1.26" in req
    assert "scikit-learn>=1.4" in req

    wf = (REPO_ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
    for os_name in ("ubuntu-latest", "windows-latest", "macos-latest"):
        assert os_name in wf
    assert "pytest" in wf and "--assert-thresholds" in wf

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "runtimes, The" not in readme

    out = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-c", "-o", "--exclude-standard"],
        text=True,
        encoding="utf-8",
    )
    for rel in out.splitlines():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        for ch in text:
            code = ord(ch)
            assert not (
                0x4E00 <= code <= 0x9FFF
                or 0x3400 <= code <= 0x4DBF
                or 0x3000 <= code <= 0x303F
                or 0xFF00 <= code <= 0xFFEF
            ), f"Non-English CJK character {ch!r} in {rel}"
