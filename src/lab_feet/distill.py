"""Knowledge distillation: zero-mean logit MSE, hybrid KL(T^2) + CE Adam trainer, and scoped sklearn fit."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .runtime import accuracy, forward, softmax


@dataclass(frozen=True)
class TrainingDiagnostics:
    """Convergence diagnostics captured without mutating global warning filters."""

    n_iter: int
    final_loss: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_sklearn_mlp(model: Any, X: np.ndarray, y: np.ndarray) -> TrainingDiagnostics:
    """Fit a scikit-learn MLP inside a scoped `catch_warnings` block and record convergence."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    hit_cap = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return TrainingDiagnostics(
        n_iter=int(getattr(model, "n_iter_", 0)),
        final_loss=float(getattr(model, "loss_", float("nan"))),
        converged=not hit_cap,
    )


def zero_mean_logits(logits: np.ndarray) -> np.ndarray:
    """Subtract sample-wise mean across classes (`sum_k z_{i,k} = 0`), required for Hinton MSE <-> KL equivalence."""
    arr = np.asarray(logits, dtype=np.float32)
    return arr - np.mean(arr, axis=1, keepdims=True)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z64 = np.asarray(z, dtype=np.float64)
    shifted = z64 - np.max(z64, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_sum_exp


def kl_divergence_soft_targets(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    temperature: float,
) -> float:
    """Compute batch-mean `T^2 * KL(softmax(z_t / T) || softmax(z_s / T))` in float64."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    log_p = _log_softmax(teacher_logits / temperature)
    log_q = _log_softmax(student_logits / temperature)
    p = np.exp(log_p)
    kl_per_sample = np.sum(p * (log_p - log_q), axis=1)
    return float((temperature**2) * np.mean(kl_per_sample))


def distill_kl_student(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    teacher_logits_tr: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    hidden_sizes: tuple[int, ...] = (32,),
    temperature: float = 4.0,
    alpha: float = 0.25,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 220,
    seed: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], TrainingDiagnostics]:
    """Train a ReLU MLP student with exact hybrid KL-divergence (`T^2` scaled) + hard-label CE loss.

    Objective:
        L(theta) = (1 - alpha) * T^2 * KL(p_T || q_T) + alpha * CE(y, z_s) + 0.5 * wd * ||W||_F^2

    Uses a scoped `np.random.default_rng(seed)` (never mutating global RNG state) and selects
    the best checkpoint on `(X_val, y_val)` when validation data is supplied.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if len(hidden_sizes) < 1:
        raise ValueError("hidden_sizes must have at least one hidden layer.")

    rng = np.random.default_rng(seed)
    X_f = np.asarray(X_tr, dtype=np.float32)
    y_i = np.asarray(y_tr, dtype=np.int64)
    n_samples, d_in = X_f.shape
    n_classes = int(teacher_logits_tr.shape[1])

    # Precompute softened teacher probability distribution p_T = softmax(z_t / T)
    p_teacher = softmax((np.asarray(teacher_logits_tr, dtype=np.float64) / temperature)).astype(np.float32)

    # He / Glorot initialization using scoped Generator
    dims = [d_in, *hidden_sizes, n_classes]
    Ws: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    for idx in range(len(dims) - 1):
        fan_in, fan_out = dims[idx], dims[idx + 1]
        scale = np.sqrt(2.0 / fan_in) if idx < len(dims) - 2 else np.sqrt(2.0 / (fan_in + fan_out))
        Ws.append(rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32))
        bs.append(np.zeros(fan_out, dtype=np.float32))

    # Adam first and second moments
    mW = [np.zeros_like(W) for W in Ws]
    vW = [np.zeros_like(W) for W in Ws]
    mb = [np.zeros_like(b) for b in bs]
    vb = [np.zeros_like(b) for b in bs]
    beta1, beta2, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)

    step = 0
    best_val_score = -float("inf")
    best_layers = [(W.copy(), b.copy()) for W, b in zip(Ws, bs)]
    final_loss = float("nan")
    L = len(Ws)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_samples)
        epoch_loss_sum = 0.0

        for start in range(0, n_samples, batch_size):
            idx_b = perm[start : start + batch_size]
            xb = X_f[idx_b]
            yb = y_i[idx_b]
            pt_b = p_teacher[idx_b]
            bsz = float(len(idx_b))

            # Forward pass caching pre-activations and activations
            acts: list[np.ndarray] = [xb]
            pre_acts: list[np.ndarray] = []
            h = xb
            for l_idx in range(L - 1):
                z_hid = h @ Ws[l_idx] + bs[l_idx]
                pre_acts.append(z_hid)
                h = np.maximum(z_hid, np.float32(0.0))
                acts.append(h)
            z_out = h @ Ws[-1] + bs[-1]

            # Soft-target gradient: d/dz_s [T^2 * KL(p_T || q_T)] = T * (softmax(z_s / T) - p_T)
            q_soft = softmax(z_out / np.float32(temperature)).astype(np.float32)
            grad_kl = np.float32(temperature) * (q_soft - pt_b)

            # Hard-label gradient: d/dz_s [CE(y, z_s)] = softmax(z_s) - one_hot(y)
            q_hard = softmax(z_out).astype(np.float32)
            grad_ce = q_hard.copy()
            grad_ce[np.arange(len(idx_b)), yb] -= np.float32(1.0)

            dz = ((np.float32(1.0 - alpha) * grad_kl) + (np.float32(alpha) * grad_ce)) / np.float32(bsz)

            # Track batch loss
            kl_val = kl_divergence_soft_targets(
                teacher_logits_tr[idx_b], z_out, temperature
            )
            log_q1 = _log_softmax(z_out)
            ce_val = float(-np.mean(log_q1[np.arange(len(idx_b)), yb]))
            epoch_loss_sum += ((1.0 - alpha) * kl_val + alpha * ce_val) * bsz

            # Backward pass across all layers
            gWs: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * L
            gbs: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * L
            grad_cur = dz
            for l_idx in range(L - 1, -1, -1):
                gWs[l_idx] = acts[l_idx].T @ grad_cur + np.float32(weight_decay) * Ws[l_idx]
                gbs[l_idx] = np.sum(grad_cur, axis=0)
                if l_idx > 0:
                    dh = grad_cur @ Ws[l_idx].T
                    grad_cur = dh * (pre_acts[l_idx - 1] > 0.0).astype(np.float32)

            # Adam parameter update
            step += 1
            bc1 = np.float32(1.0 - float(beta1) ** step)
            bc2 = np.float32(1.0 - float(beta2) ** step)
            lr_t = np.float32(lr) * np.sqrt(bc2) / bc1

            for l_idx in range(L):
                mW[l_idx] = beta1 * mW[l_idx] + (np.float32(1.0) - beta1) * gWs[l_idx]
                vW[l_idx] = beta2 * vW[l_idx] + (np.float32(1.0) - beta2) * (gWs[l_idx] * gWs[l_idx])
                Ws[l_idx] -= lr_t * mW[l_idx] / (np.sqrt(vW[l_idx]) + eps)

                mb[l_idx] = beta1 * mb[l_idx] + (np.float32(1.0) - beta1) * gbs[l_idx]
                vb[l_idx] = beta2 * vb[l_idx] + (np.float32(1.0) - beta2) * (gbs[l_idx] * gbs[l_idx])
                bs[l_idx] -= lr_t * mb[l_idx] / (np.sqrt(vb[l_idx]) + eps)

        final_loss = epoch_loss_sum / float(n_samples)
        cur_layers = [(W.copy(), b.copy()) for W, b in zip(Ws, bs)]
        if X_val is not None and y_val is not None:
            val_acc = accuracy(cur_layers, X_val, y_val)
            # Tie-break equal validation accuracies in favor of lower validation CE loss
            val_logits = forward(cur_layers, X_val)
            val_ce = float(-np.mean(_log_softmax(val_logits)[np.arange(len(y_val)), y_val]))
            score = val_acc - 1e-4 * val_ce
            if score >= best_val_score:
                best_val_score = score
                best_layers = cur_layers
        else:
            best_layers = cur_layers

    return best_layers, TrainingDiagnostics(
        n_iter=epochs,
        final_loss=float(final_loss),
        converged=True,
    )
