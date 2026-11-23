"""Cross-platform CPU detection, BLAS single-thread pinning, forward pass, and percentile latency."""

from __future__ import annotations

import contextlib
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

# Pin single-threaded BLAS defaults before NumPy initializes worker pools if imported first
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import numpy as np  # noqa: E402

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover
    _threadpool_limits = None

SUPPORTED_ACTIVATIONS = ("relu", "tanh", "logistic", "identity")


@contextlib.contextmanager
def single_thread_blas() -> Iterator[None]:
    """Pin native OpenBLAS/MKL/OpenMP threadpools to 1 thread for deterministic batch-1 timing."""
    if _threadpool_limits is not None:
        with _threadpool_limits(limits=1):
            yield
    else:  # pragma: no cover
        yield


def cpu_model(cpuinfo_path: str = "/proc/cpuinfo") -> str:
    """Return human-readable CPU identifier across Linux (x86/ARM64), macOS, and Windows."""
    try:
        with open(cpuinfo_path, encoding="utf-8", errors="replace") as f:
            arm_model: str | None = None
            for line in f:
                if line.startswith("model name"):
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
                elif line.startswith(("Model", "Hardware", "Processor")) and ":" in line:
                    candidate = line.split(":", 1)[1].strip()
                    if candidate and arm_model is None:
                        arm_model = candidate
            if arm_model:
                return arm_model
    except OSError:
        pass

    proc = platform.processor().strip()
    if proc:
        return proc
    mach = platform.machine().strip()
    if mach:
        return mach
    return "unknown"


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise (or 1D) softmax computed in float64."""
    in_arr = np.asarray(z)
    out_dtype = np.float32 if in_arr.dtype == np.float32 else np.float64
    arr64 = in_arr.astype(np.float64, copy=False)
    shifted = arr64 - np.max(arr64, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    return probs.astype(out_dtype, copy=False)


def get_layers(model: Any, *, require_relu: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract (W, b) float32 tuples from a fitted scikit-learn MLPClassifier or MLPRegressor.

    Raises ValueError if hidden `model.activation != 'relu'` when `require_relu=True`
    so non-ReLU estimators never silently run through a hardcoded ReLU forward pass.
    """
    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise TypeError("Expected a fitted scikit-learn MLP with 'coefs_' and 'intercepts_'.")

    activation = getattr(model, "activation", "relu")
    if require_relu and activation != "relu":
        raise ValueError(
            f"Unsupported MLP hidden activation {activation!r}; get_layers() requires 'relu' "
            "(or pass activation explicitly to forward())."
        )

    out_act = getattr(model, "out_activation_", "identity")
    layers = [
        (np.asarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32))
        for W, b in zip(model.coefs_, model.intercepts_)
    ]
    # Binary MLPClassifier uses 1 logistic output unit; expand [-z, +z] if needed for argmax parity
    if out_act == "logistic" and layers[-1][0].shape[1] == 1 and hasattr(model, "classes_"):
        W_last, b_last = layers[-1]
        W_two = np.concatenate([-0.5 * W_last, 0.5 * W_last], axis=1).astype(np.float32)
        b_two = np.concatenate([-0.5 * b_last, 0.5 * b_last], axis=0).astype(np.float32)
        layers[-1] = (W_two, b_two)
    return layers


def _apply_hidden_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(x, np.float32(0.0))
    if activation == "tanh":
        return np.tanh(x)
    if activation == "logistic":
        return np.float32(1.0) / (np.float32(1.0) + np.exp(-np.clip(x, -80.0, 80.0)))
    if activation == "identity":
        return x
    raise ValueError(
        f"Unsupported activation {activation!r}; expected one of {SUPPORTED_ACTIVATIONS}."
    )


def forward(
    layers: list[tuple[np.ndarray, np.ndarray]],
    x: np.ndarray,
    *,
    activation: str = "relu",
) -> np.ndarray:
    """Shared NumPy forward pass: configurable hidden activation (default ReLU), linear output."""
    if not layers:
        raise ValueError("layers must be non-empty.")
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation {activation!r}; expected one of {SUPPORTED_ACTIVATIONS}."
        )
    h = np.asarray(x, dtype=np.float32)
    for W, b in layers[:-1]:
        h = _apply_hidden_activation(h @ W + b, activation)
    W_last, b_last = layers[-1]
    return h @ W_last + b_last


def accuracy(layers: list[tuple[np.ndarray, np.ndarray]], X: np.ndarray, y: np.ndarray) -> float:
    """Top-1 classification accuracy."""
    preds = forward(layers, X).argmax(axis=1)
    return float((preds == y).mean())


def n_params(layers: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """Total scalar parameter count across all weights and biases."""
    return int(sum(W.size + b.size for W, b in layers))


def fp32_bytes(layers: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """Total uncompressed float32 storage footprint in bytes."""
    return int(sum(np.asarray(W, dtype=np.float32).nbytes + np.asarray(b, dtype=np.float32).nbytes for W, b in layers))


@dataclass(frozen=True)
class LatencyStats:
    median_us: float
    p95_us: float
    p99_us: float
    iqr_us: float
    mean_us: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def latency_stats(
    target: list[tuple[np.ndarray, np.ndarray]] | Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    n_warmup: int = 200,
    n_timing: int = 2000,
) -> LatencyStats:
    """Measure warmup-aware batch-1 CPU latency percentiles (median, p95, p99, IQR) with pinned BLAS."""
    if len(X) == 0:
        raise ValueError("X must contain at least one sample for latency measurement.")
    if n_timing < 1 or n_warmup < 0:
        raise ValueError("n_timing must be >= 1 and n_warmup >= 0.")

    runner: Callable[[np.ndarray], Any]
    if callable(target):
        runner = target
    else:
        layers = target
        runner = lambda sample: forward(layers, sample)  # noqa: E731

    X_f32 = np.asarray(X, dtype=np.float32)
    rows = [X_f32[i : i + 1] for i in range(len(X_f32))]
    n_rows = len(rows)

    with single_thread_blas():
        for i in range(n_warmup):
            runner(rows[i % n_rows])
        times_ns = np.empty(n_timing, dtype=np.float64)
        for i in range(n_timing):
            sample = rows[i % n_rows]
            t0 = time.perf_counter_ns()
            runner(sample)
            times_ns[i] = time.perf_counter_ns() - t0

    times_us = times_ns * 1e-3
    p25, p50, p75, p95, p99 = np.percentile(times_us, [25.0, 50.0, 75.0, 95.0, 99.0])
    return LatencyStats(
        median_us=float(p50),
        p95_us=float(p95),
        p99_us=float(p99),
        iqr_us=float(p75 - p25),
        mean_us=float(np.mean(times_us)),
    )


def latency_us(
    layers: list[tuple[np.ndarray, np.ndarray]] | Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    n_warmup: int = 200,
    n_timing: int = 2000,
) -> float:
    """Median batch-1 CPU latency in microseconds (convenience wrapper over `latency_stats`)."""
    return latency_stats(layers, X, n_warmup=n_warmup, n_timing=n_timing).median_us
