"""End-to-end benchmarks: sklearn digits (single-seed & N-seed), synthetic wet-lab regimes, and synthetic autoencoder."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .distill import distill_kl_student, fit_sklearn_mlp, zero_mean_logits
from .export_c import prepare_fixed_point_model, simulate_c99_fixed_point
from .frontier import quantize_model_wht_gptq
from .quantize import QuantizedModel, forward_int8, quantize_model
from .runtime import (
    accuracy,
    cpu_model,
    forward,
    fp32_bytes,
    get_layers,
    latency_stats,
    n_params,
    softmax,
)
from .serialize import compute_binary_header_bytes, from_binary_buffer, to_binary_buffer


def make_stratified_splits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.30,
    val_fraction_of_dev: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create leakage-free stratified train / val / test splits with StandardScaler fit strictly on train."""
    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=val_fraction_of_dev, random_state=seed, stratify=y_dev
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    return X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te


def _agreement(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    return float((preds_a == preds_b).mean())


def run_digits_benchmark(
    *,
    seed: int = 0,
    temperature: float = 4.0,
    alpha: float = 0.25,
    n_warmup: int = 200,
    n_timing: int = 2000,
) -> dict[str, Any]:
    """Run full teacher-student distillation & quantization benchmark on 8x8 digits."""
    X, y = load_digits(return_X_y=True)
    X_tr, y_tr, X_val, y_val, X_te, y_te = make_stratified_splits(X, y, seed=seed)

    teacher_clf = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=350, random_state=seed)
    teacher_diag = fit_sklearn_mlp(teacher_clf, X_tr, y_tr)
    t_layers = get_layers(teacher_clf)

    hard_clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=350, random_state=seed)
    hard_diag = fit_sklearn_mlp(hard_clf, X_tr, y_tr)
    hard_layers = get_layers(hard_clf)

    t_logits_tr = forward(t_layers, X_tr)
    mean_max_soft_prob = float(softmax(t_logits_tr / temperature).max(axis=1).mean())

    # Same-trainer plain CE control (`alpha=1.0`, identical Adam trainer, epochs, & val selection, #37)
    ce_layers, ce_diag = distill_kl_student(
        X_tr,
        y_tr,
        t_logits_tr,
        X_val=X_val,
        y_val=y_val,
        hidden_sizes=(32,),
        temperature=temperature,
        alpha=1.0,
        epochs=240,
        seed=seed,
    )

    # 1) Legacy uncentered logit MSE student (demonstrates sample-offset capacity waste)
    uncentered_reg = MLPRegressor(
        hidden_layer_sizes=(32,), max_iter=600, tol=1e-5, random_state=seed
    )
    uncentered_diag = fit_sklearn_mlp(uncentered_reg, X_tr, t_logits_tr / temperature)
    uncentered_layers = get_layers(uncentered_reg)

    # 2) Zero-mean centered logit MSE student (Hinton et al. 2015 high-T equivalence)
    centered_targets = zero_mean_logits(t_logits_tr) / temperature
    centered_reg = MLPRegressor(
        hidden_layer_sizes=(32,), max_iter=600, tol=1e-5, random_state=seed
    )
    centered_diag = fit_sklearn_mlp(centered_reg, X_tr, centered_targets)
    centered_layers = get_layers(centered_reg)

    # 3) Hybrid KL(T^2) + hard-label CE student selected on X_val
    kl_layers, kl_diag = distill_kl_student(
        X_tr,
        y_tr,
        t_logits_tr,
        X_val=X_val,
        y_val=y_val,
        hidden_sizes=(32,),
        temperature=temperature,
        alpha=alpha,
        epochs=240,
        seed=seed,
    )

    # Post-training quantization of the KL+CE distilled student (calibrated on X_val only)
    q_int8_pt: QuantizedModel = quantize_model(kl_layers, 8, per_channel=False, mode="symmetric")
    q_int4_pt: QuantizedModel = quantize_model(kl_layers, 4, per_channel=False, mode="symmetric")
    q_int4_pc: QuantizedModel = quantize_model(kl_layers, 4, per_channel=True, mode="symmetric")
    q_int4_asym: QuantizedModel = quantize_model(kl_layers, 4, per_channel=True, mode="asymmetric")
    q_int4_mse: QuantizedModel = quantize_model(
        kl_layers,
        4,
        per_channel=True,
        mode="symmetric",
        clip_search="activation_mse",
        X_cal=X_val,
    )
    wht_q4 = quantize_model_wht_gptq(kl_layers, bits=4, X_cal=X_val, seed=seed)

    # Verify lossless flat binary round-trip on the 4-bit per-channel MSE model
    bin_blob = to_binary_buffer(q_int4_mse)
    expected_bin_len = compute_binary_header_bytes(len(q_int4_mse.layers)) + q_int4_mse.nbytes
    assert len(bin_blob) == expected_bin_len
    reloaded_q4 = from_binary_buffer(bin_blob)
    assert reloaded_q4.nbytes == q_int4_mse.nbytes

    ref_kl_preds_te = forward(kl_layers, X_te).argmax(axis=1)

    def _eval_fp32_variant(
        name: str,
        layers: list[tuple[np.ndarray, np.ndarray]],
        nbytes_val: int,
        diag_dict: dict[str, Any] | None = None,
        agree_ref: np.ndarray | None = None,
        *,
        timing_mode: str = "fp32",
    ) -> dict[str, Any]:
        val_acc = accuracy(layers, X_val, y_val)
        te_preds = forward(layers, X_te).argmax(axis=1)
        te_acc = float((te_preds == y_te).mean())
        lat = latency_stats(layers, X_te, n_warmup=n_warmup, n_timing=n_timing)
        return {
            "name": name,
            "val_acc": round(val_acc, 4),
            "test_acc": round(te_acc, 4),
            "params": n_params(layers),
            "weight_bytes": int(nbytes_val),
            "median_us": round(lat.median_us, 2),
            "p95_us": round(lat.p95_us, 2),
            "p99_us": round(lat.p99_us, 2),
            "iqr_us": round(lat.iqr_us, 2),
            "timing_mode": timing_mode,
            "agreement_with_kl_fp32": (
                round(_agreement(te_preds, agree_ref), 4) if agree_ref is not None else None
            ),
            "training": diag_dict,
        }

    def _eval_integer_variant(
        name: str,
        runner: Any,
        params_count: int,
        nbytes_val: int,
        *,
        timing_mode: str,
    ) -> dict[str, Any]:
        val_preds = runner(X_val).argmax(axis=1)
        te_preds = runner(X_te).argmax(axis=1)
        lat = latency_stats(runner, X_te, n_warmup=n_warmup, n_timing=n_timing)
        return {
            "name": name,
            "val_acc": round(float((val_preds == y_val).mean()), 4),
            "test_acc": round(float((te_preds == y_te).mean()), 4),
            "params": int(params_count),
            "weight_bytes": int(nbytes_val),
            "median_us": round(lat.median_us, 2),
            "p95_us": round(lat.p95_us, 2),
            "p99_us": round(lat.p99_us, 2),
            "iqr_us": round(lat.iqr_us, 2),
            "timing_mode": timing_mode,
            "agreement_with_kl_fp32": round(_agreement(te_preds, ref_kl_preds_te), 4),
            "training": None,
        }

    c99_spec = prepare_fixed_point_model(reloaded_q4, X_cal=X_val)

    rows: list[dict[str, Any]] = [
        _eval_fp32_variant(
            "teacher (256,256) fp32", t_layers, fp32_bytes(t_layers), teacher_diag.to_dict()
        ),
        _eval_fp32_variant(
            "sklearn MLPClassifier hard-label",
            hard_layers,
            fp32_bytes(hard_layers),
            hard_diag.to_dict(),
        ),
        _eval_fp32_variant(
            "student same-trainer CE fp32",
            ce_layers,
            fp32_bytes(ce_layers),
            ce_diag.to_dict(),
        ),
        _eval_fp32_variant(
            "student uncentered-MSE fp32",
            uncentered_layers,
            fp32_bytes(uncentered_layers),
            uncentered_diag.to_dict(),
        ),
        _eval_fp32_variant(
            "student zero-mean-MSE fp32",
            centered_layers,
            fp32_bytes(centered_layers),
            centered_diag.to_dict(),
        ),
        _eval_fp32_variant(
            "student hybrid KL(T^2)+CE fp32",
            kl_layers,
            fp32_bytes(kl_layers),
            kl_diag.to_dict(),
            ref_kl_preds_te,
        ),
        _eval_fp32_variant(
            "student KL int8 pt (dequantized fp32 timing)",
            q_int8_pt.dequantize(),
            q_int8_pt.nbytes,
            None,
            ref_kl_preds_te,
            timing_mode="dequantized_fp32",
        ),
        _eval_integer_variant(
            "student KL int8 W8A8 (int32)",
            lambda sample: forward_int8(q_int8_pt, sample),
            q_int8_pt.n_params,
            q_int8_pt.nbytes,
            timing_mode="integer_w8a8",
        ),
        _eval_fp32_variant(
            "student KL 4-bit pt (dequantized fp32 timing)",
            q_int4_pt.dequantize(),
            q_int4_pt.nbytes,
            None,
            ref_kl_preds_te,
            timing_mode="dequantized_fp32",
        ),
        _eval_fp32_variant(
            "student KL 4-bit pc (dequantized fp32 timing)",
            q_int4_pc.dequantize(),
            q_int4_pc.nbytes,
            None,
            ref_kl_preds_te,
            timing_mode="dequantized_fp32",
        ),
        _eval_fp32_variant(
            "student KL 4-bit asym-pc (dequantized fp32 timing)",
            q_int4_asym.dequantize(),
            q_int4_asym.nbytes,
            None,
            ref_kl_preds_te,
            timing_mode="dequantized_fp32",
        ),
        _eval_fp32_variant(
            "student KL 4-bit pc+MSE (dequantized fp32 timing)",
            reloaded_q4.dequantize(),
            reloaded_q4.nbytes,
            None,
            ref_kl_preds_te,
            timing_mode="dequantized_fp32",
        ),
        _eval_integer_variant(
            "student KL 4-bit W4A8 (int32)",
            lambda sample: forward_int8(reloaded_q4, sample, act_bits=8),
            reloaded_q4.n_params,
            reloaded_q4.nbytes,
            timing_mode="integer_w4a8",
        ),
        _eval_integer_variant(
            "student KL 4-bit WHT+GPTQ W4A8 (int32)",
            lambda sample: wht_q4.forward_int8(sample, act_bits=8),
            reloaded_q4.n_params,
            wht_q4.packed_weight_bytes,
            timing_mode="integer_w4a8",
        ),
        _eval_integer_variant(
            "student KL 4-bit C99 fixed-point (int32)",
            lambda sample: simulate_c99_fixed_point(c99_spec, sample)[1],
            reloaded_q4.n_params,
            reloaded_q4.nbytes,
            timing_mode="integer_c99_sim",
        ),
    ]

    return {
        "benchmark": "sklearn_digits",
        "cpu": cpu_model(),
        "seed": seed,
        "temperature": temperature,
        "alpha": alpha,
        "splits": {
            "train": int(len(X_tr)),
            "val": int(len(X_val)),
            "test": int(len(X_te)),
            "features": int(X_tr.shape[1]),
        },
        "teacher_mean_max_soft_prob": round(mean_max_soft_prob, 4),
        "timing_config": {"n_warmup": n_warmup, "n_timing": n_timing},
        "binary_container_verification": {
            "format": "LFT1",
            "payload_weight_bytes": q_int4_mse.nbytes,
            "header_bytes": compute_binary_header_bytes(len(q_int4_mse.layers)),
            "total_blob_bytes": len(bin_blob),
        },
        "models": rows,
        "_artifacts": {
            "q_int4_mse": q_int4_mse,
            "q_int8_pt": q_int8_pt,
        },
    }


def run_digits_multiseed_benchmark(
    *,
    n_seeds: int = 10,
    temperature: float = 4.0,
    alpha: float = 0.25,
) -> dict[str, Any]:
    """Run N-seed split-and-init digits distillation & quantization comparison (#37)."""
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}.")

    X, y = load_digits(return_X_y=True)
    tracked_names = [
        "teacher (256,256) fp32",
        "sklearn MLPClassifier hard-label",
        "student same-trainer CE fp32",
        "student hybrid KL(T^2)+CE fp32",
        "student KL int8 W8A8 (int32)",
        "student KL 4-bit W4A8 (int32)",
        "student KL 4-bit WHT+GPTQ W4A8 (int32)",
    ]
    per_model_val: dict[str, list[float]] = {k: [] for k in tracked_names}
    per_model_test: dict[str, list[float]] = {k: [] for k in tracked_names}
    per_model_bytes: dict[str, int] = {}
    per_model_params: dict[str, int] = {}

    for seed in range(n_seeds):
        X_tr, y_tr, X_val, y_val, X_te, y_te = make_stratified_splits(X, y, seed=seed)

        teacher_clf = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=350, random_state=seed)
        fit_sklearn_mlp(teacher_clf, X_tr, y_tr)
        t_layers = get_layers(teacher_clf)
        t_logits_tr = forward(t_layers, X_tr)

        hard_clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=350, random_state=seed)
        fit_sklearn_mlp(hard_clf, X_tr, y_tr)
        hard_layers = get_layers(hard_clf)

        ce_layers, _ = distill_kl_student(
            X_tr,
            y_tr,
            t_logits_tr,
            X_val=X_val,
            y_val=y_val,
            hidden_sizes=(32,),
            temperature=temperature,
            alpha=1.0,
            epochs=240,
            seed=seed,
        )
        kl_layers, _ = distill_kl_student(
            X_tr,
            y_tr,
            t_logits_tr,
            X_val=X_val,
            y_val=y_val,
            hidden_sizes=(32,),
            temperature=temperature,
            alpha=alpha,
            epochs=240,
            seed=seed,
        )

        q8 = quantize_model(kl_layers, 8, per_channel=False, mode="symmetric")
        q4 = quantize_model(
            kl_layers, 4, per_channel=True, mode="symmetric", clip_search="activation_mse", X_cal=X_val
        )
        wht_q4 = quantize_model_wht_gptq(kl_layers, bits=4, X_cal=X_val, seed=seed)

        def _rec(name: str, v_acc: float, te_acc: float, p_cnt: int, w_bytes: int) -> None:
            per_model_val[name].append(v_acc)
            per_model_test[name].append(te_acc)
            per_model_params[name] = int(p_cnt)
            per_model_bytes[name] = int(w_bytes)

        _rec(
            "teacher (256,256) fp32",
            accuracy(t_layers, X_val, y_val),
            accuracy(t_layers, X_te, y_te),
            n_params(t_layers),
            fp32_bytes(t_layers),
        )
        _rec(
            "sklearn MLPClassifier hard-label",
            accuracy(hard_layers, X_val, y_val),
            accuracy(hard_layers, X_te, y_te),
            n_params(hard_layers),
            fp32_bytes(hard_layers),
        )
        _rec(
            "student same-trainer CE fp32",
            accuracy(ce_layers, X_val, y_val),
            accuracy(ce_layers, X_te, y_te),
            n_params(ce_layers),
            fp32_bytes(ce_layers),
        )
        _rec(
            "student hybrid KL(T^2)+CE fp32",
            accuracy(kl_layers, X_val, y_val),
            accuracy(kl_layers, X_te, y_te),
            n_params(kl_layers),
            fp32_bytes(kl_layers),
        )
        _rec(
            "student KL int8 W8A8 (int32)",
            float((forward_int8(q8, X_val).argmax(axis=1) == y_val).mean()),
            float((forward_int8(q8, X_te).argmax(axis=1) == y_te).mean()),
            q8.n_params,
            q8.nbytes,
        )
        _rec(
            "student KL 4-bit W4A8 (int32)",
            float((forward_int8(q4, X_val, act_bits=8).argmax(axis=1) == y_val).mean()),
            float((forward_int8(q4, X_te, act_bits=8).argmax(axis=1) == y_te).mean()),
            q4.n_params,
            q4.nbytes,
        )
        _rec(
            "student KL 4-bit WHT+GPTQ W4A8 (int32)",
            float((wht_q4.forward_int8(X_val, act_bits=8).argmax(axis=1) == y_val).mean()),
            float((wht_q4.forward_int8(X_te, act_bits=8).argmax(axis=1) == y_te).mean()),
            q4.n_params,
            wht_q4.packed_weight_bytes,
        )

    kl_accs = per_model_test["student hybrid KL(T^2)+CE fp32"]
    ce_accs = per_model_test["student same-trainer CE fp32"]
    wins = sum(1 for k, c in zip(kl_accs, ce_accs) if k > c + 1e-12)
    losses = sum(1 for k, c in zip(kl_accs, ce_accs) if k < c - 1e-12)
    ties = int(n_seeds) - wins - losses

    rows_summary: list[dict[str, Any]] = []
    for name in tracked_names:
        v_arr = np.asarray(per_model_val[name], dtype=np.float64)
        t_arr = np.asarray(per_model_test[name], dtype=np.float64)
        rows_summary.append(
            {
                "name": name,
                "mean_val_acc": round(float(np.mean(v_arr)), 4),
                "std_val_acc": round(float(np.std(v_arr)), 4),
                "mean_test_acc": round(float(np.mean(t_arr)), 4),
                "std_test_acc": round(float(np.std(t_arr)), 4),
                "min_test_acc": round(float(np.min(t_arr)), 4),
                "max_test_acc": round(float(np.max(t_arr)), 4),
                "params": per_model_params[name],
                "weight_bytes": per_model_bytes[name],
                "per_seed_test_acc": [round(float(x), 4) for x in t_arr],
            }
        )

    return {
        "benchmark": "sklearn_digits_multiseed",
        "split_protocol": "split_and_init_seeds",
        "n_seeds": int(n_seeds),
        "seeds": list(range(n_seeds)),
        "temperature": temperature,
        "alpha": alpha,
        "kl_vs_same_trainer_ce": {
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "mean_test_acc_diff": round(float(np.mean(np.asarray(kl_accs) - np.asarray(ce_accs))), 4),
        },
        "models": rows_summary,
    }


def generate_wetlab_spectral_dataset(
    n_samples: int = 1800,
    n_features: int = 48,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 6-class toy benchmark (`i.i.d.` Gaussian inputs + random 2-layer oracle)."""
    rng = np.random.default_rng(seed)
    n_classes = 6
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    W1 = rng.normal(0.0, 0.35, size=(n_features, 64)).astype(np.float32)
    W2 = rng.normal(0.0, 0.45, size=(64, n_classes)).astype(np.float32)
    h = np.maximum(X @ W1, 0.0) + 0.25 * np.square(X[:, :16] @ W1[:16, :])
    logits = h @ W2 + rng.normal(0.0, 0.15, size=(n_samples, n_classes)).astype(np.float32)
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


def run_wetlab_acoustic_benchmark(
    *,
    seed: int = 0,
    temperature: float = 3.5,
    alpha: float = 0.20,
    n_warmup: int = 100,
    n_timing: int = 500,
) -> dict[str, Any]:
    """Benchmark distillation & 4-bit quantization on the synthetic 6-class 48-feature toy task."""
    X, y = generate_wetlab_spectral_dataset(n_samples=1800, n_features=48, seed=seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te = make_stratified_splits(X, y, seed=seed)

    teacher = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=400, random_state=seed)
    t_diag = fit_sklearn_mlp(teacher, X_tr, y_tr)
    t_layers = get_layers(teacher)

    student_hidden = (16,)
    hard = MLPClassifier(hidden_layer_sizes=student_hidden, max_iter=400, random_state=seed)
    h_diag = fit_sklearn_mlp(hard, X_tr, y_tr)
    h_layers = get_layers(hard)

    t_logits_tr = forward(t_layers, X_tr)
    ce_layers, ce_diag = distill_kl_student(
        X_tr,
        y_tr,
        t_logits_tr,
        X_val=X_val,
        y_val=y_val,
        hidden_sizes=student_hidden,
        temperature=temperature,
        alpha=1.0,
        epochs=260,
        seed=seed,
    )
    kl_layers, kl_diag = distill_kl_student(
        X_tr,
        y_tr,
        t_logits_tr,
        X_val=X_val,
        y_val=y_val,
        hidden_sizes=student_hidden,
        temperature=temperature,
        alpha=alpha,
        epochs=260,
        seed=seed,
    )

    q8 = quantize_model(kl_layers, 8, per_channel=True, mode="symmetric")
    q4 = quantize_model(
        kl_layers,
        4,
        per_channel=True,
        mode="symmetric",
        clip_search="activation_mse",
        X_cal=X_val,
    )
    ref_preds = forward(kl_layers, X_te).argmax(axis=1)

    models_summary: list[dict[str, Any]] = []
    for label, layers, nb, ref, t_mode in [
        ("synthetic wetlab teacher (256,256) fp32", t_layers, fp32_bytes(t_layers), None, "fp32"),
        ("synthetic wetlab sklearn MLPClassifier hard-label", h_layers, fp32_bytes(h_layers), None, "fp32"),
        ("synthetic wetlab same-trainer CE (16,) fp32", ce_layers, fp32_bytes(ce_layers), None, "fp32"),
        ("synthetic wetlab student (16,) KL+CE fp32", kl_layers, fp32_bytes(kl_layers), ref_preds, "fp32"),
        (
            "synthetic wetlab int8 pc (dequantized fp32 timing)",
            q8.dequantize(),
            q8.nbytes,
            ref_preds,
            "dequantized_fp32",
        ),
        (
            "synthetic wetlab 4-bit pc+MSE (dequantized fp32 timing)",
            q4.dequantize(),
            q4.nbytes,
            ref_preds,
            "dequantized_fp32",
        ),
    ]:
        lat = latency_stats(layers, X_te, n_warmup=n_warmup, n_timing=n_timing)
        te_preds = forward(layers, X_te).argmax(axis=1)
        models_summary.append(
            {
                "name": label,
                "val_acc": round(accuracy(layers, X_val, y_val), 4),
                "test_acc": round(float((te_preds == y_te).mean()), 4),
                "params": n_params(layers),
                "weight_bytes": int(nb),
                "median_us": round(lat.median_us, 2),
                "p95_us": round(lat.p95_us, 2),
                "p99_us": round(lat.p99_us, 2),
                "iqr_us": round(lat.iqr_us, 2),
                "timing_mode": t_mode,
                "agreement_with_kl_fp32": (
                    round(_agreement(te_preds, ref), 4) if ref is not None else None
                ),
            }
        )

    w4a8_lat = latency_stats(
        lambda sample: forward_int8(q4, sample, act_bits=8),
        X_te,
        n_warmup=n_warmup,
        n_timing=n_timing,
    )
    w4a8_val_preds = forward_int8(q4, X_val, act_bits=8).argmax(axis=1)
    w4a8_te_preds = forward_int8(q4, X_te, act_bits=8).argmax(axis=1)
    models_summary.append(
        {
            "name": "synthetic wetlab 4-bit W4A8 (int32)",
            "val_acc": round(float((w4a8_val_preds == y_val).mean()), 4),
            "test_acc": round(float((w4a8_te_preds == y_te).mean()), 4),
            "params": q4.n_params,
            "weight_bytes": int(q4.nbytes),
            "median_us": round(w4a8_lat.median_us, 2),
            "p95_us": round(w4a8_lat.p95_us, 2),
            "p99_us": round(w4a8_lat.p99_us, 2),
            "iqr_us": round(w4a8_lat.iqr_us, 2),
            "timing_mode": "integer_w4a8",
            "agreement_with_kl_fp32": round(_agreement(w4a8_te_preds, ref_preds), 4),
        }
    )

    return {
        "benchmark": "synthetic_wetlab_acoustic_regimes",
        "is_synthetic": True,
        "splits": {
            "train": int(len(X_tr)),
            "val": int(len(X_val)),
            "test": int(len(X_te)),
            "features": int(X_tr.shape[1]),
        },
        "training": {
            "teacher": t_diag.to_dict(),
            "sklearn_hard_student": h_diag.to_dict(),
            "same_trainer_ce_student": ce_diag.to_dict(),
            "kl_student": kl_diag.to_dict(),
        },
        "models": models_summary,
    }


def run_autoencoder_compression_benchmark(*, seed: int = 0) -> dict[str, Any]:
    """Compress a synthetic 8-D linear subspace `320 -> 128 -> 128 -> 8 -> 128 -> 128 -> 320` autoencoder."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(0.0, 0.4, size=(8, 320)).astype(np.float32)
    z_tr = rng.normal(0.0, 1.0, size=(320, 8)).astype(np.float32)
    z_te_norm = rng.normal(0.0, 1.0, size=(80, 8)).astype(np.float32)
    X_tr = (z_tr @ basis + rng.normal(0.0, 0.04, size=(320, 320))).astype(np.float32)
    X_norm = (z_te_norm @ basis + rng.normal(0.0, 0.04, size=(80, 320))).astype(np.float32)
    X_anom = (X_norm + rng.normal(0.0, 1.15, size=(80, 320))).astype(np.float32)

    ae = MLPRegressor(
        hidden_layer_sizes=(128, 128, 8, 128, 128),
        activation="relu",
        max_iter=250,
        random_state=seed,
    )
    diag = fit_sklearn_mlp(ae, X_tr, X_tr)
    ae_layers = get_layers(ae)

    q8 = quantize_model(ae_layers, 8, per_channel=True, mode="symmetric")
    q4 = quantize_model(
        ae_layers,
        4,
        per_channel=True,
        mode="symmetric",
        clip_search="weight_mse",
    )

    def _recon_mse(layers_eval: list[tuple[np.ndarray, np.ndarray]], data: np.ndarray) -> np.ndarray:
        recon = forward(layers_eval, data)
        return np.mean(np.square(data - recon), axis=1)

    fp32_norm_err = _recon_mse(ae_layers, X_norm)
    fp32_anom_err = _recon_mse(ae_layers, X_anom)
    q4_norm_err = _recon_mse(q4.dequantize(), X_norm)
    q4_anom_err = _recon_mse(q4.dequantize(), X_anom)

    fp32_auroc = float((fp32_anom_err[:, None] > fp32_norm_err[None, :]).mean())
    q4_auroc = float((q4_anom_err[:, None] > q4_norm_err[None, :]).mean())
    fp32_total = fp32_bytes(ae_layers)

    return {
        "benchmark": "synthetic_lab_ears_autoencoder_compression",
        "is_synthetic": True,
        "architecture": "320 -> 128 -> 128 -> 8 -> 128 -> 128 -> 320",
        "params": n_params(ae_layers),
        "fp32_bytes": fp32_total,
        "int8_bytes": q8.nbytes,
        "int4_bytes": q4.nbytes,
        "compression_ratio_4bit": round(fp32_total / q4.nbytes, 2),
        "fp32_normal_mse": round(float(fp32_norm_err.mean()), 6),
        "fp32_anomaly_mse": round(float(fp32_anom_err.mean()), 6),
        "int4_normal_mse": round(float(q4_norm_err.mean()), 6),
        "int4_anomaly_mse": round(float(q4_anom_err.mean()), 6),
        "fp32_anomaly_auroc": round(fp32_auroc, 4),
        "int4_anomaly_auroc": round(q4_auroc, 4),
        "training": diag.to_dict(),
    }
