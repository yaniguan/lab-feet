"""End-to-end benchmarks: sklearn digits, wet-lab instrument acoustic regimes, and lab-ears autoencoder."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .distill import distill_kl_student, fit_sklearn_mlp, zero_mean_logits
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

    # 3) True hybrid KL(T^2) + hard-label CE student selected on X_val
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
            "agreement_with_kl_fp32": (
                round(_agreement(te_preds, agree_ref), 4) if agree_ref is not None else None
            ),
            "training": diag_dict,
        }

    rows: list[dict[str, Any]] = [
        _eval_fp32_variant(
            "teacher (256,256) fp32", t_layers, fp32_bytes(t_layers), teacher_diag.to_dict()
        ),
        _eval_fp32_variant(
            "student hard-label fp32", hard_layers, fp32_bytes(hard_layers), hard_diag.to_dict()
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
            "student KL int8 (per-tensor)",
            q_int8_pt.dequantize(),
            q_int8_pt.nbytes,
            None,
            ref_kl_preds_te,
        ),
    ]

    # True W8A8 integer matmul evaluation (int8 activations x int8 weights -> int32 accumulator)
    w8a8_val_preds = forward_int8(q_int8_pt, X_val).argmax(axis=1)
    w8a8_te_preds = forward_int8(q_int8_pt, X_te).argmax(axis=1)
    w8a8_lat = latency_stats(
        lambda sample: forward_int8(q_int8_pt, sample),
        X_te,
        n_warmup=n_warmup,
        n_timing=n_timing,
    )
    rows.append(
        {
            "name": "student KL int8 W8A8 (int32)",
            "val_acc": round(float((w8a8_val_preds == y_val).mean()), 4),
            "test_acc": round(float((w8a8_te_preds == y_te).mean()), 4),
            "params": q_int8_pt.n_params,
            "weight_bytes": q_int8_pt.nbytes,
            "median_us": round(w8a8_lat.median_us, 2),
            "p95_us": round(w8a8_lat.p95_us, 2),
            "p99_us": round(w8a8_lat.p99_us, 2),
            "iqr_us": round(w8a8_lat.iqr_us, 2),
            "agreement_with_kl_fp32": round(_agreement(w8a8_te_preds, ref_kl_preds_te), 4),
            "training": None,
        }
    )

    rows.extend(
        [
            _eval_fp32_variant(
                "student KL 4-bit (per-tensor)",
                q_int4_pt.dequantize(),
                q_int4_pt.nbytes,
                None,
                ref_kl_preds_te,
            ),
            _eval_fp32_variant(
                "student KL 4-bit (per-channel)",
                q_int4_pc.dequantize(),
                q_int4_pc.nbytes,
                None,
                ref_kl_preds_te,
            ),
            _eval_fp32_variant(
                "student KL 4-bit (asym-pc)",
                q_int4_asym.dequantize(),
                q_int4_asym.nbytes,
                None,
                ref_kl_preds_te,
            ),
            _eval_fp32_variant(
                "student KL 4-bit (pc+MSE-clip)",
                reloaded_q4.dequantize(),
                reloaded_q4.nbytes,
                None,
                ref_kl_preds_te,
            ),
        ]
    )

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


def generate_wetlab_spectral_dataset(
    n_samples: int = 1800,
    n_features: int = 48,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic 6-regime wet-lab benchtop acoustic/vibration mel-band spectra.

    Regimes model non-linear harmonic & cross-band couplings observed in `lab-ears` benchtop sensors:
      0: normal_orbital_shaker, 1: ultrasonic_cavitation, 2: pipette_air_aspiration,
      3: centrifuge_imbalance, 4: peltier_thermal_oscillation, 5: dry_pump_stall.
    """
    rng = np.random.default_rng(seed)
    n_classes = 6
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    # Smooth two-layer ReLU + quadratic cross-harmonic ground-truth acoustic response over X
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
    """Benchmark distillation & 4-bit quantization on 6-class wet-lab instrument spectral regimes."""
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
    for label, layers, nb, ref in [
        ("wetlab teacher (256,256) fp32", t_layers, fp32_bytes(t_layers), None),
        ("wetlab student (16,) hard fp32", h_layers, fp32_bytes(h_layers), None),
        ("wetlab student (16,) KL+CE fp32", kl_layers, fp32_bytes(kl_layers), ref_preds),
        ("wetlab student (16,) int8 pc", q8.dequantize(), q8.nbytes, ref_preds),
        ("wetlab student (16,) 4-bit pc+MSE", q4.dequantize(), q4.nbytes, ref_preds),
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
                "agreement_with_kl_fp32": (
                    round(_agreement(te_preds, ref), 4) if ref is not None else None
                ),
            }
        )

    return {
        "benchmark": "wetlab_acoustic_regimes",
        "splits": {
            "train": int(len(X_tr)),
            "val": int(len(X_val)),
            "test": int(len(X_te)),
            "features": int(X_tr.shape[1]),
        },
        "training": {
            "teacher": t_diag.to_dict(),
            "hard_student": h_diag.to_dict(),
            "kl_student": kl_diag.to_dict(),
        },
        "models": models_summary,
    }


def run_autoencoder_compression_benchmark(*, seed: int = 0) -> dict[str, Any]:
    """Compress a `lab-ears` `320 -> 128 -> 128 -> 8 -> 128 -> 128 -> 320` MLPRegressor autoencoder."""
    rng = np.random.default_rng(seed)
    # Normal benchtop mel-frame patches live near an 8D harmonic subspace in R^320
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

    # Pairwise AUROC (probability that an anomalous frame scores higher MSE than a normal frame)
    fp32_auroc = float((fp32_anom_err[:, None] > fp32_norm_err[None, :]).mean())
    q4_auroc = float((q4_anom_err[:, None] > q4_norm_err[None, :]).mean())
    fp32_total = fp32_bytes(ae_layers)

    return {
        "benchmark": "lab_ears_autoencoder_compression",
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
