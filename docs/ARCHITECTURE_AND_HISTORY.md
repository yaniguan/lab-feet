# `lab-feet` — Architecture and History

> **Auto-Lab Ecosystem**: [`auto-lab`](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)  
> **Layer**: `L0 · Edge Distillation & Quantized Runtime`  
> **License**: MIT (repository code; external datasets and repositories retain their respective licenses)

---

## 1. Object

`lab-feet` is the **Edge Distillation & Quantized Runtime (`L0`)** foundation of the [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab). Within the `L0`–`L6` Brain/Body stack, `L0` enables the lab agent to operate autonomously on low-cost, low-power boards mounted directly beside wet-lab instruments: cloud GPUs are invoked only in short bursts for teacher training or heavy simulation, while compact student models are distilled, quantized to `int8` and `4-bit`, and run locally with predictable single-sample (`batch=1`) latency.

### Inputs and Outputs (`demo/distill_quantize.py`)

- **Inputs**:
  - `scikit-learn` bundled `8 × 8` handwritten digits dataset (`1,797` grayscale images, `64` features, `10` classes `0..9`), stratified `70% / 30%` into `1,257` training and `540` test examples (`SEED = 0`).
  - Distillation temperature `T = 4.0`, bit-widths `bits` $\in \{8, 4\}$, and latency benchmark parameters (`N_WARMUP = 200`, `N_TIMING = 2000`).
- **Outputs**:
  - Five evaluated model variants (fp32 `(256, 256)` teacher, fp32 `(32,)` hard-label student, fp32 `(32,)` distilled student, `int8` distilled student, and packed `4-bit` distilled student) scored with a single shared `numpy` forward pass (`forward(layers, x)`), reporting test accuracy, parameter count, serialized weight footprint in bytes, `batch=1` median CPU latency in microseconds, and top-1 prediction agreement against the fp32 distilled student.

### Scope Boundaries

- **Implemented today**: Self-contained CPU reference pipeline for high-temperature logit regression distillation, symmetric per-tensor `int8` and two's-complement nibble-packed `4-bit` post-training weight quantization (`pack_4bit` / `unpack_4bit`), storage accounting, and single-sample `numpy` forward-pass timing, paired with a curated survey of LLM/VLM/VLA quantization methods and C++ edge runtimes.
- **Out of scope / next stages**: True integer or low-bit SIMD/NPU matrix multiplication kernels (`llama.cpp`, `ExecuTorch`, `mlc-llm`, `Embodied.cpp` — in the current `numpy` baseline, quantized weights are dequantized back to `float32` prior to `h @ W + b`); quantization-aware training (QAT) and activation quantization; distilling wet-lab domain students (protocol slot taggers, small Whisper encoders, equivariant force heads, or VLA manipulation policies) onto physical ARM/NPU edge boards.

---

## 2. Data

### 2.1 Active Benchmark Dataset

- **Dataset**: `sklearn.datasets.load_digits` (UCI Optical Recognition of Handwritten Digits, bundled with `scikit-learn` under the **BSD-3-Clause License**).
- **Dimensions & Split**: `1,797` samples $\times$ `64` pixel intensities (`0..16`), `10` balanced digit classes (`0..9`). Split via `train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)` into `X_tr` (`1,257 × 64`) and `X_te` (`540 × 64`).
- **Preprocessing**: Standardized per feature using `StandardScaler().fit(X_tr)` (preventing test-set leakage) and cast to `np.float32`.

### 2.2 Weight Tensor & Bit-Packing Representation

Every trained `scikit-learn` MLP is exported via `get_layers(model)` into a canonical list of `(W, b)` `float32` tuples:

| Variant | Layer Shapes (`W`, `b`) | Total Params | Weight Storage Formula | Total Bytes | Compression vs. Teacher |
|---|---|---:|---|---:|---:|
| Teacher `(256, 256)` fp32 | `(64,256)+(256,256)+(256,10)` | 85,002 | $4 \times 85{,}002$ | 340,008 B | `1.00x` |
| Student `(32,)` fp32 | `(64,32)+(32,10)` | 2,410 | $4 \times 2{,}410$ | 9,640 B | `35.27x` |
| Student `(32,)` `int8` | `2,368` `int8` weights + `42` fp32 biases | 2,410 | $2{,}368 + 4(42) + 2(4)$ | 2,544 B | `133.65x` (`3.79x` vs. student fp32) |
| Student `(32,)` `4-bit` | `2,368` `int4` nibbles + `42` fp32 biases | 2,410 | $2{,}368/2 + 4(42) + 2(4)$ | 1,360 B | `250.01x` (`7.09x` vs. student fp32) |

- **Two's-complement `4-bit` byte packing (`pack_4bit` / `unpack_4bit`)**:
  - Signed integer codes $q \in [-8, 7]$ are masked to 4-bit unsigned nibbles `(q & 0x0F)` (with zero-padding if `q.size` is odd) and packed two per `uint8` byte: `packed = u[0::2] | (u[1::2] << 4)`.
  - `unpack_4bit(packed, W.shape)` extracts low and high nibbles, restores sign via `np.where(u > 7, u - 16, u)`, slices away any odd-length pad nibble, and reshapes back to `W.shape`—guaranteeing that measured `4-bit` accuracy reflects exact byte-packed storage.

---

## 3. Model

### 3.1 Teacher and Student MLP Architectures

All models share the same inference graph in `forward(layers, x)`: ReLU hidden layers $h^{(\ell)} = \max(h^{(\ell-1)} W^{(\ell)} + b^{(\ell)}, 0)$ followed by a linear score head $z = h^{(L-1)} W^{(L)} + b^{(L)}$:

1. **Teacher (`MLPClassifier`)**: `64 -> 256 (ReLU) -> 256 (ReLU) -> 10 (Linear)` (`85,002` parameters).
2. **Hard-Label Student (`MLPClassifier`)**: `64 -> 32 (ReLU) -> 10 (Linear)` (`2,410` parameters).
3. **Distilled Student (`MLPRegressor`)**: Identical `64 -> 32 (ReLU) -> 10 (Linear)` architecture (`2,410` parameters), predicting continuous scaled logits $\hat{z} \approx z_{\text{teacher}} / T$ and classifying test inputs via $\arg\max_k \hat{z}_k$.

### 3.2 Symmetric Per-Tensor Weight Quantizer (`quantize`, `quantize_model`)

For each linear weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ and target bit-width $b \in \{8, 4\}$:
$$q_{\max} = 2^{b-1} - 1 \quad (127 \text{ for int8},\; 7 \text{ for 4-bit}), \qquad s = \frac{\max_{i,j} |W_{i,j}|}{q_{\max}}$$
$$q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{W}{s}\right),\, -q_{\max} - 1,\, q_{\max}\right) \in \mathbb{Z}^{d_{\text{in}} \times d_{\text{out}}}, \qquad \tilde{W} = \operatorname{float32}(q) \cdot \operatorname{float32}(s)$$
Biases $b^{(\ell)}$ remain in `float32` (`42` parameters = `168` bytes across both student layers), and one `float32` scale scalar (`4` bytes) is stored per weight matrix.

---

## 4. Training

### 4.1 Distillation via Scaled Logit Regression ($T = 4.0$)

- **Teacher & hard-label student**: Fit with `scikit-learn`'s Adam optimizer (`lr = 1e-3`, `max_iter = 300`, `random_state = 0`) on discrete class labels `y_tr`.
- **High-temperature logit matching**: Because `sklearn.neural_network.MLPClassifier` supports only discrete class targets rather than soft probability distributions, `distill_quantize.py` extracts pre-softmax teacher logits $z_t = \text{forward}(\theta_{\text{teacher}}, X_{\text{tr}})$ and trains `MLPRegressor(..., max_iter=500, random_state=0)` to minimize squared error against temperature-scaled logits $y_{\text{soft}} = z_t / T$ with $T = 4.0$:
  $$\mathcal{L}_{\text{distill}}(\theta) = \frac{1}{2N} \sum_{i=1}^{N} \left\| f_\theta(x_i) - \frac{z_t(x_i)}{T} \right\|_2^2$$
  As shown by Hinton et al. (2015), matching zero-mean logits under MSE is the high-temperature ($T \to \infty$) limit of KL-divergence distillation. At $T = 4.0$, the teacher's softened distribution `softmax(logits / T)` has mean maximum probability `0.7447`.
- **Documented design trade-offs & honest negative finding**:
  1. Regressing directly on `softmax(logits / T)` yielded `0.904` test accuracy because soft probabilities in $\mathbb{R}^{10}$ are small and flat, triggering `MLPRegressor`'s `tol = 1e-4` early-convergence check prematurely; regressing on `logits / T` preserves inter-class dark knowledge at a healthy gradient scale (`0.9685` test accuracy).
  2. On this low-noise 64-feature task, **plain hard-label student training (`0.9741`) slightly outperforms pure logit distillation (`0.9685`)** by `0.56` percentage points because pure `MLPRegressor` distillation omits the standard $\alpha \mathcal{L}_{\text{CE}}(y_{\text{true}}, \hat{y})$ ground-truth anchor. Both numbers are reported side by side.

### 4.2 Reference Benchmark Results & Honest Runtime Accounting

Measured on an Intel Xeon CPU (`demo/output.txt`, `200` warmup + `2,000` `batch=1` single-sample `numpy` forward passes, 2026-11-18):

| Model | Test Accuracy | Params | Weight Bytes | Batch-1 Latency ($\mu\text{s}$) | Agreement w/ Distilled fp32 |
|---|---:|---:|---:|---:|---:|
| `teacher (256,256) fp32` | `0.9778` | 85,002 | 340,008 | `9.51` | — |
| `student hard-label fp32` | `0.9741` | 2,410 | 9,640 | `3.82` | — |
| `student distilled fp32` | `0.9685` | 2,410 | 9,640 | `3.83` | `1.0000` |
| `student distilled int8` | `0.9685` | 2,410 | 2,544 | `3.86` | `1.0000` |
| `student distilled 4-bit` | `0.9648` | 2,410 | 1,360 | `3.86` | `0.9889` |

- **Honest `numpy` dequantized matmul note**: `quantize_model()` converts `q` back to `float32` (`q.astype(np.float32) * s`) so all five rows execute the exact same `numpy` `float32` `h @ W + b` kernel. Consequently, **architecture compression (`(256, 256) -> (32,)`) delivers a `2.5x` latency speedup (`9.51 -> 3.82 us`), whereas `numpy` fake-quantization (`int8` / `4-bit`) demonstrates `3.8x–7.1x` memory compression with near-zero accuracy loss (`0.00` / `-0.37` pp) but zero wall-clock speedup (`3.86 us` vs. `3.83 us`)**. Achieving wall-clock quantization speedups requires an integer/low-bit C++/NPU execution engine (`llama.cpp`, `ExecuTorch`, `Embodied.cpp`).

---

## 5. History

| Date (ISO-8601) | Commit | Branch / Ref | Summary of Changes |
|---|---|---|---|
| `2026-11-18 01:36:44 -0800` | `a7b1532` (`a7b153235a61c6689714bb5bf494966792ada265`) | `main` | **Initial baseline (`Add curated resources and first demo`)**: Added teacher-student logit distillation + symmetric `int8` and packed `4-bit` quantization script (`demo/distill_quantize.py`), recorded Xeon CPU accuracy/storage/latency benchmark (`demo/output.txt`), configured GitHub Actions CI (`.github/workflows/demo.yml`), and curated quantization algorithms (`GPTQ`, `AWQ`, `SmoothQuant`, `QLoRA`, `BitVLA`) and edge runtimes (`llama.cpp`, `mlc-llm`, `ExecuTorch`, `Embodied.cpp`) in `README.md`. |

---

## 6. Unique Properties

### Distinctive Invariants & Guarantees

1. **Exact Nibble-Level Bit-Packing Verification**: Rather than simulating 4-bit quantization via float clamping alone, `quantize_model(layers, 4)` packs signed codes into `uint8` bytes (`pack_4bit`) and unpacks them (`unpack_4bit`) before scoring—proving exact byte-size accounting (`1,360` bytes) and `100%` `int8` / `98.89%` `4-bit` top-1 prediction agreement.
2. **Framework-Neutral Latency Comparison**: Exporting both `MLPClassifier` and `MLPRegressor` weights into raw `float32` `(W, b)` tuples and running one shared `forward()` function eliminates `scikit-learn` estimator wrapper overhead from `batch=1` latency measurements.
3. **Minimal Dependency Stack**: Requires only `numpy` and `scikit-learn` (`requirements.txt`), running end-to-end training, distillation, bit-packing, and 10,000+ timing passes in seconds on any CPU.

### Known Limitations

- **Linux-only `/proc/cpuinfo` read**: `cpu_model()` opens `"/proc/cpuinfo"` unconditionally without `try/except` or `platform.processor()` fallback, raising `FileNotFoundError` on Windows and macOS.
- **Zero-tensor scale division**: In `quantize(W, bits)`, if an entire weight tensor `W` is zero, `scale = np.abs(W).max() / qmax` evaluates to `0.0`, producing `NaN` in `W / scale`.
- **Per-tensor vs. per-channel scaling**: Uses a single scalar `scale` per matrix rather than per-output-channel scales, which degrades more sharply at `4-bit` when outlier channels appear in larger Transformer/VLA layers.
- **Dequantized `fp32` execution**: Measures storage footprint and numerical fidelity, not hardware `int8`/`int4` throughput or activation quantization.

### Ecosystem Interfaces Across Auto-Lab

- **Serving `L2` (`lab-ears`)**: Compresses small Whisper ASR models (`distil-whisper` / `whisper.cpp`) and `acoustic/` anomaly autoencoders so contact-microphone boards can score instrument audio locally without streaming raw audio off-bench.
- **Serving `L1` (`lab-hands`)**: Compresses learned visuomotor and visuo-tactile policies (`SmolVLA`, `BitVLA`, `ACT`) into `int8`/`4-bit` weights for deterministic sub-millisecond robot control loops next to the arm.
- **Serving `L3` (`YLM`) & `L5` (`lab-decide`)**: Enables local benchtop inference of distilled molecular property/force predictors and calibrated uncertainty heads when cloud GPU links are unavailable or latency-bound.
