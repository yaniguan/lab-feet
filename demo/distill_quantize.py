"""Knowledge distillation and post-training weight quantization on sklearn digits.

A large MLP teacher (256, 256) is distilled into a small MLP student (32,).
The distilled student is then quantized per-tensor (symmetric int8 and 4-bit)
in numpy. For every model we report test accuracy, parameter count, weight
storage size and batch-1 CPU latency measured with one shared numpy forward pass.
"""
import time
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED = 0
T = 4.0
N_TIMING = 2000
N_WARMUP = 200
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def cpu_model():
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def get_layers(model):
    """Return a list of (W, b) float32 arrays from a fitted sklearn MLP."""
    return [(W.astype(np.float32), b.astype(np.float32))
            for W, b in zip(model.coefs_, model.intercepts_)]


def forward(layers, x):
    """Shared numpy forward pass: ReLU hidden layers, linear output (scores)."""
    h = x
    for W, b in layers[:-1]:
        h = np.maximum(h @ W + b, 0.0)
    W, b = layers[-1]
    return h @ W + b


def accuracy(layers, X, y):
    return float((forward(layers, X).argmax(axis=1) == y).mean())


def n_params(layers):
    return sum(W.size + b.size for W, b in layers)


def quantize(W, bits):
    """Per-tensor symmetric quantization; returns integer codes and scale."""
    qmax = 2 ** (bits - 1) - 1          # 127 for int8, 7 for 4-bit
    scale = np.abs(W).max() / qmax
    q = np.clip(np.round(W / scale), -qmax - 1, qmax).astype(np.int8)
    return q, np.float32(scale)


def pack_4bit(q):
    """Pack signed 4-bit codes (-8..7) two per byte."""
    u = (q.ravel().astype(np.int16) & 0x0F).astype(np.uint8)
    if u.size % 2:
        u = np.append(u, np.uint8(0))
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8)


def unpack_4bit(packed, shape):
    lo = (packed & 0x0F).astype(np.int8)
    hi = (packed >> 4).astype(np.int8)
    u = np.empty(packed.size * 2, dtype=np.int8)
    u[0::2], u[1::2] = lo, hi
    u = np.where(u > 7, u - 16, u).astype(np.int8)
    return u[:int(np.prod(shape))].reshape(shape)


def quantize_model(layers, bits):
    """Quantize weight matrices (biases stay fp32). Returns dequantized layers
    for inference and the number of bytes needed to store the model."""
    deq, nbytes = [], 0
    for W, b in layers:
        q, s = quantize(W, bits)
        if bits == 4:
            packed = pack_4bit(q)
            q = unpack_4bit(packed, W.shape)   # round-trip through the packed form
            nbytes += packed.nbytes
        else:
            nbytes += q.nbytes
        nbytes += b.nbytes + 4                 # fp32 bias + fp32 scale
        deq.append((q.astype(np.float32) * s, b))
    return deq, nbytes


def latency_us(layers, X):
    """Median batch-1 latency in microseconds over N_TIMING forward passes."""
    rows = [X[i:i + 1] for i in range(len(X))]
    for i in range(N_WARMUP):
        forward(layers, rows[i % len(rows)])
    times = []
    for i in range(N_TIMING):
        x = rows[i % len(rows)]
        t0 = time.perf_counter()
        forward(layers, x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e6)


def main():
    np.random.seed(SEED)
    print(f"CPU: {cpu_model()}")
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    print(f"train={len(X_tr)} test={len(X_te)} features={X_tr.shape[1]}")

    teacher = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=300,
                            random_state=SEED).fit(X_tr, y_tr)
    hard = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300,
                         random_state=SEED).fit(X_tr, y_tr)

    # Distillation with plain sklearn: MLPClassifier only accepts hard labels,
    # so the student is an MLPRegressor fit to the teacher's soft outputs.
    # We take the teacher's pre-softmax logits from our numpy forward pass and
    # regress on logits / T (MSE on logits is the classic high-T limit of the
    # distillation loss). We also tried regressing on softmax(logits / T), but
    # those targets are small and flat, so sklearn's tolerance-based stopping
    # ends training early (0.904 test accuracy). Logits / T keep the relative
    # class similarities with a usable scale. The student predicts argmax.
    t_logits = forward(get_layers(teacher), X_tr)
    soft_targets = t_logits / T
    print(f"teacher soft probs softmax(logits/T): mean max prob = "
          f"{softmax(soft_targets).max(axis=1).mean():.4f}")
    distilled = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500,
                             random_state=SEED).fit(X_tr, soft_targets)

    fp32_bytes = lambda L: sum(W.nbytes + b.nbytes for W, b in L)
    d_layers = get_layers(distilled)
    int8_layers, int8_bytes = quantize_model(d_layers, 8)
    int4_layers, int4_bytes = quantize_model(d_layers, 4)

    models = [
        ("teacher (256,256) fp32", get_layers(teacher), None),
        ("student hard-label fp32", get_layers(hard), None),
        ("student distilled fp32", d_layers, None),
        ("student distilled int8", int8_layers, int8_bytes),
        ("student distilled 4-bit", int4_layers, int4_bytes),
    ]

    print(f"distillation temperature T={T}")
    print(f"latency: median of {N_TIMING} batch-1 numpy forward passes "
          f"after {N_WARMUP} warmup (quantized models run dequantized fp32 weights)")
    print()
    header = f"{'model':<26}{'test_acc':>9}{'params':>9}{'weight_bytes':>14}{'latency_us':>12}"
    print(header)
    print("-" * len(header))
    for name, layers, qbytes in models:
        nbytes = fp32_bytes(layers) if qbytes is None else qbytes
        acc = accuracy(layers, X_te, y_te)
        lat = latency_us(layers, X_te)
        print(f"{name:<26}{acc:>9.4f}{n_params(layers):>9d}{nbytes:>14d}{lat:>12.2f}")

    print()
    agree8 = (forward(int8_layers, X_te).argmax(1) == forward(d_layers, X_te).argmax(1)).mean()
    agree4 = (forward(int4_layers, X_te).argmax(1) == forward(d_layers, X_te).argmax(1)).mean()
    print(f"prediction agreement with fp32 distilled student: int8={agree8:.4f} 4-bit={agree4:.4f}")
    print("weight_bytes for quantized models = packed weights + fp32 biases + one fp32 scale per tensor")


if __name__ == "__main__":
    main()
