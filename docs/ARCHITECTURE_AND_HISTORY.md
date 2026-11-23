# `lab-feet` — Architecture and History

> **Auto-Lab Ecosystem**: [`auto-lab`](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)  
> **Layer**: `L0 · Edge Distillation & Quantized Runtime`  
> **License**: MIT (repository code; external datasets and repositories retain their respective licenses)

---

## 1. Object

`lab-feet` is the **Edge Distillation & Quantized Runtime (`L0`)** substrate of the [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab). Within the `L0`–`L6` Brain/Body architecture, `L0` compiles and compresses trained models into low-footprint, deterministic edge artifacts that execute directly next to wet-lab instruments with single-thread `batch=1` predictability.

### Package & CLI Architecture (`src/lab_feet/`)
- `lab_feet.distill`: Scoped `fit_sklearn_mlp` convergence tracker, sample-wise `zero_mean_logits`, and pure-NumPy mini-batch Adam trainer optimizing $(1 - \alpha)\, T^2\, \mathrm{KL}(p_T \| q_T) + \alpha\, \mathrm{CE}(y, z_s)$.
- `lab_feet.quantize`: Bit-validated (`bits` in `[2, 8]`), zero-division-guarded per-tensor and per-output-channel quantizer (`symmetric`, `symmetric_full`, `asymmetric`), MSE-optimal clipping grid search (`weight_mse`, `activation_mse`), two's-complement nibble packer (`pack_4bit` / `unpack_4bit`), `QuantizedLayer` / `QuantizedModel` containers, and `forward_int8` (`W8A8` `int32` integer matmul).
- `lab_feet.serialize`: Flat little-endian `LFT1` binary container (`8 + 28 * L` header bytes + exact `model.nbytes` payload) and `.npz` persistence.
- `lab_feet.equivariant`: $SO(3)$-equivariant scalar multiplicity weight quantizer and spatial-axis rounding guard for [`YLM`](https://github.com/yaniguan/YLM) spherical-harmonic force heads.
- `lab_feet.runtime`: Cross-platform `cpu_model()` (Linux x86/ARM64, macOS, Windows), single-thread BLAS pinning (`OMP`/`OPENBLAS`/`MKL` + `threadpoolctl`), activation-validated `get_layers()`, `float64`-stabilized `softmax()`, and `latency_stats()` (`median`, `p95`, `p99`, `IQR`).

---

## 2. Data & Splits

1. **`scikit-learn` Digits (`run_digits_benchmark`)**: `1,797 x 64` images split stratified without leakage into `train=1,068`, `val=189`, and `test=540` (`StandardScaler` fit strictly on `train`). `val` drives KL distillation checkpoint selection and post-training `activation_mse` clipping search; `test` is touched once for final reporting.
2. **Wet-Lab Spectral Regimes (`run_wetlab_acoustic_benchmark`)**: `1,800 x 48` deterministic 6-regime benchtop harmonic/cavitation dataset modeling capacity-constrained edge classification (`(256, 256)` teacher -> `(16,)` student).
3. **`lab-ears` Acoustic Autoencoder (`run_autoencoder_compression_benchmark`)**: `320 -> 128 -> 128 -> 8 -> 128 -> 128 -> 320` (`117,576` parameters, `470,304 B` fp32) compressed `7.23x` to `65,088 B` in per-channel `4-bit` while preserving `> 0.98` anomaly detection AUROC.

---

## 3. Model & Numerical Invariants

### 3.1 Distillation Objective
Given teacher logits $z_t$ and student logits $z_s$ at temperature $T > 0$ with $p_T = \operatorname{softmax}(z_t / T)$ and $q_T = \operatorname{softmax}(z_s / T)$:
$$\mathcal{L}(\theta) = (1 - \alpha)\, T^2 \, \mathrm{KL}(p_T \| q_T) + \alpha \, \mathrm{CE}(y, z_s) + \frac{\lambda}{2}\sum_{\ell}\|W^{(\ell)}\|_F^2$$
The $T^2$ factor ensures $\nabla_{z_s}\!\left[T^2\,\mathrm{KL}(p_T \| q_T)\right] = T\,(q_T - p_T) = \mathcal{O}(1)$ as $T$ increases, preventing soft-target gradient vanishing relative to hard-label cross-entropy.

### 3.2 Quantization Modes, Bit-Packing & `LFT1` Layout
- **Zero-scale safety**: Any tensor or output channel with zero dynamic range sets `scale = 1.0` and `q = 0` (`int8`), preventing `0 / 0 -> NaN`.
- **Two's-complement `4-bit` storage**: `pack_4bit` validates codes in `[-8, 7]` and packs two nibbles per `uint8` byte (`lo | (hi << 4)`). `unpack_4bit` casts raw buffers to `uint8` and masks both `(packed & 0x0F)` and `((packed >> 4) & 0x0F)` to prevent signed arithmetic right-shift corruption.
- **Student storage breakdown (`(64, 32) -> (32, 10)`, `2,410` params)**:
  - `fp32`: `9,640 B`
  - `int8` per-tensor: `2,368 + 168 + 8 = 2,544 B`
  - `4-bit` per-tensor: `1,184` (`uint8` packed) `+ 168` (`fp32` bias) `+ 8` (`fp32` scale) `= 1,360 B`
  - `4-bit` per-channel: `1,184 + 168 + 168` (`42` channel scales) `= 1,520 B` (`LFT1` file = `1,520 + 64 = 1,584 B`).
- **$SO(3)$ Equivariance**: For spherical-harmonic features $X \in \mathbb{R}^{N \times C_{\text{in}} \times (2\ell+1)}$, quantizing scalar channel weights $W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$ with zero bias commutes with 3D rotations: $(X D^{(\ell)}(R)^\top)\tilde{W} = (X \tilde{W}) D^{(\ell)}(R)^\top$.

---

## 4. History

| Date | Ref | Summary |
|---|---|---|
| `2026-11-18` | `a7b1532` (`main`) | Initial CPU logit-regression distillation + per-tensor `int8`/`4-bit` demo and resource catalog. |
| `2026-11-23` | `0.2.0` (`fix/resolve-audit-issues`) | Resolved audit issues `#1..#24`: packaged `src/lab_feet/` + `lab-feet` CLI, cross-platform `cpu_model`, single-thread BLAS percentile timing, leakage-free `train/val/test` splits, hybrid `KL(T^2)+CE` & zero-mean MSE distillation, per-channel/asymmetric/MSE-clipped `int8` & `4-bit` `QuantizedModel`, `W8A8` `int32` matmul, `LFT1` binary serializer, wet-lab & `lab-ears` benchmarks, $SO(3)$-equivariant quantization, multi-OS CI, and `pytest` suite. |
