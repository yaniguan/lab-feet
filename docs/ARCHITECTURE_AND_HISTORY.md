# `lab-feet` — Architecture and History

> **Auto-Lab Ecosystem**: [`auto-lab`](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)  
> **Layer**: `L0 · Edge Distillation & Quantized Runtime`  
> **License**: MIT (repository code; external datasets and repositories retain their respective licenses)

---

## 1. Object

`lab-feet` is the **Edge Distillation & Quantized Runtime (`L0`)** substrate of the [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab). Within the `L0`–`L6` Brain/Body architecture, `L0` compiles and compresses trained models into low-footprint, deterministic edge artifacts that execute directly next to wet-lab instruments with single-thread `batch=1` predictability and fail-closed drift monitoring.

### Package & CLI Architecture (`src/lab_feet/`)
- `lab_feet.distill`: Scoped `fit_sklearn_mlp` convergence tracker, sample-wise `zero_mean_logits`, and pure-NumPy mini-batch Adam trainer optimizing $(1 - \alpha)\, T^2\, \mathrm{KL}(p_T \| q_T) + \alpha\, \mathrm{CE}(y, z_s)$ with validation-stagnation `converged` detection and same-trainer plain `CE` (`alpha = 1.0`) controls (`#37`).
- `lab_feet.quantize`: Bit-validated (`bits` in `[2, 8]`), zero-division-guarded per-tensor and per-output-channel quantizer (`symmetric`, `symmetric_full`, `asymmetric`), MSE-optimal clipping grid search (`weight_mse`, `activation_mse`), two's-complement nibble packer (`pack_4bit` / `unpack_4bit`), `QuantizedLayer` / `QuantizedModel` containers, SmoothQuant migration, and `forward_int8` (`W8A8` / `W4A8` `int32` integer matmul).
- `lab_feet.frontier`:
  - **Walsh-Hadamard Transform (`WHT` / SpinQuant-style) Outlier Rotation + Cholesky `GPTQ-lite`**: Exact orthogonal dyadic-block Hadamard rotation $Q = \Pi^\top \operatorname{diag}(s) H_{\text{blk}}$ ($Q^\top Q = I_d$) paired with Schur-complement inverse-Hessian error compensation (`gptq_quantize_weights`, `quantize_model_wht_gptq`).
  - **1D Residual Spectral Conv/MLP Backbone (`SpectralResNet1D`)**: `15k–50k` parameter (`39.1k` default) multi-scale 1D convolutional residual network with adaptive spectral pooling and end-to-end `WHT + GPTQ` `W4A8` integer execution (`QuantizedSpectralResNet1D`).
  - **OOD Abstention & Test-Time Drift Adaptation**: `MSP` (`1 - max_c softmax(z)_c`), Helmholtz Free Energy (`-logsumexp(logits)`), `ReAct` penultimate activation rectification, Ledoit-Wolf `ShrinkageMahalanobisDetector`, `compute_aurc`, and unlabeled Bayesian-shrunk Test-Time Batch Normalization (`TTN`) + Split Conformal Risk Control (`CRC`).
- `lab_feet.export_c` & `lab_feet.export_onnx`: Freestanding zero-`malloc` C99 `.rodata` header generator (`simulate_c99_fixed_point`) and ONNX `opset >= 21` `qdq` & `qlinear` exporter (`MatMulInteger` `int32` output layer + pre-bias calibrated hidden `QLinearMatMul` scales, verified against `onnxruntime` in CI, `#34`).
- `lab_feet.serialize`: Flat little-endian `LFT1` binary container (`8 + 28 * L` header bytes + exact `model.nbytes` payload) and `.npz` persistence.
- `lab_feet.equivariant`: $SO(3)$-equivariant scalar multiplicity weight quantizer and spatial-axis rounding guard for [`YLM`](https://github.com/yaniguan/YLM) spherical-harmonic force heads.
- `lab_feet.datasets`: Typed, SPDX-license-gated `OpenEdgeDataset` registry (`OPEN_EDGE_DATASETS`, 10 real public datasets with pinned SHA-256 digests and exact archive byte sizes, failing on mismatch unless `allow_unverified_sha256=True`), leakage-free `DeterministicFeatureNormalizer`, `split_mode="random" | "temporal"`, and multi-batch `evaluate_temporal_drift_abstaining_int4` (`#36`).

---

## 2. Data & Splits

1. **`scikit-learn` Digits (`run_digits_benchmark` & `run_digits_multiseed_benchmark`)**: `1,797 x 64` images split stratified without leakage into `train=1,068`, `val=189`, and `test=540` (`StandardScaler` fit strictly on `train`). Evaluated both on single `seed 0` and across `10` split-and-init seeds (`0..9`, `--seeds 10`), reporting `mean +- std` and `KL+CE vs same-trainer CE` win/tie/loss counts (`5 / 0 / 5`, `0.9748 +- 0.0049` vs `0.9756 +- 0.0057`).
2. **Synthetic Toy Benchmarks (`run_wetlab_acoustic_benchmark` & `run_autoencoder_compression_benchmark`)**: Explicitly labeled synthetic tasks (`is_synthetic = True`): a 6-class 48-D `i.i.d.` Gaussian 2-layer oracle task and an 8-D linear subspace in $\mathbb{R}^{320}$ `MLPRegressor` autoencoder (`470,304 B -> 65,088 B`, `7.23x` compression).
3. **10 Pinned Open Edge Sensor, Vibration, Gas-Drift & 1D Spectral Datasets (`lab_feet.datasets`)**: Ten permissive-SPDX (`Apache-2.0`, `CC-BY-4.0`, `MIT`) public datasets with pinned SHA-256 and byte-size checks (`mnist1d`, `uci_gas_sensor_drift` 10-batch 36-month MOX drift, `uci_har_inertial`, `cwru_bearing_vibration`, `uci_hydraulic_vibration`, `sers_bacteria_raman_2026`, `ethanol_nir_spectroscopy`, `olive_oil_ftir`, `coffee_ftir`, `meat_ftir`).

---

## 3. Model, Quantization & Monitor SRAM Footprints

- **ONNX `qlinear` & C99 Fixed-Point Parity (`#34`)**: Hidden layers calibrate `QLinearMatMul` output scale and zero-point from the **pre-bias** `X @ W` range on `X_cal` with safety headroom, followed by a dedicated post-activation `QuantizeLinear` scale. The final logit layer uses `MatMulInteger` (`int8 x int8 -> int32`) -> `Cast(FLOAT)` -> `Mul(x_scale * w_scale)` -> `Add(bias)`, preventing `int8` logit saturation and achieving `>= 0.99` top-1 agreement with `simulate_c99_fixed_point` in both `evaluate_quantized_graph` and `onnxruntime`.
- **OOD Monitor SRAM Formula (`#36`)**: For `C` classes and `d` monitored features, `DriftAbstainingInt4Classifier.extra_sram_bytes = (C + 1) * d + 2 * C + 40` bytes (`32 B` MCU accumulator scratch + `(C + 1)*d + 2*C + 8` B tables):
  - `d = 24, C = 6`: **`220 B`** (`< 512 B` MCU SRAM budget)
  - `d = 32, C = 6`: **`276 B`** (`< 512 B` MCU SRAM budget)
  - `d = 128, C = 6`: **`948 B`** (`> 512 B`; `evaluate_temporal_drift_abstaining_int4` emits an explicit `UserWarning` and sets `sram_budget_exceeded = True`).

---

## 4. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`, Fail-Closed `#35`)

`autolab_wire.py` (`REPO_ID = "lab-feet"`, `LAYER = "L0"`) enforces fail-closed L0 safety gating before downstream causal attribution (`lab-attrib`):

- **Fail-Closed Halt Reasons**:
  - `L0_FEET_UPSTREAM_SPLITGUARD_BLOCK`: Upstream `posttrain` or `distilled_student` reports `split_guard_clean is False`.
  - `L0_FEET_CHECKPOINT_SHA256_MISMATCH`: Corrupted `LFT1` payload or mismatched `checkpoint_sha256` / `calibration_sha256`.
  - `L0_FEET_CALIBRATION_REQUIRED`: Hot-loaded student checkpoint lacks a non-empty finite 2D calibration split `(calibration_X, calibration_y)` matching its input/output dimensions, or packet `sensor_features` mismatches `d_in`.
  - `L0_FEET_QUANT_DRIFT_BLOCK`: Nibble round-trip failure or `quant_acc_drop_permille > max_acc_drop_permille` (clamped to `<= 15`; packets may tighten the cap below `15` but can never raise it).
  - `L0_FEET_ACCURACY_FLOOR_BLOCK`: Calibration `cal_fp32_acc_permille` or `cal_int4_acc_permille` falls below `min_cal_acc_permille` (default `700`, clamped `>= 700`), blocking constant/degenerate students whose FP32-vs-INT4 drop is trivially `0`.
  - `L0_FEET_MONITOR_UNAVAILABLE`: The 2-layer `DriftAbstainingInt4Classifier` OOD monitor cannot be built or run (`ood_monitor_active = False`), unless the packet explicitly sets `allow_unmonitored_student = True` (logged into `trace`).
  - `L0_FEET_SAFE_HOLD_ABSTAIN`: Unrelaxed calibrated 5-gate `SAFE_HOLD` monitor abstains on out-of-distribution or rail-clipped input features.
- **Provenance Telemetry (`edge_runtime`)**:
  - Reports `sensor_source` (`"packet"` when real `sensor_features` are supplied vs `"derived_from_protocol_hash"` when deterministically synthesized from `protocol_sha256` / `manipulation`, which is not a physical sensor measurement) and `interlocks_source = "packet_passthrough"` (indicating `dispense_error_permille`, `aspiration_clean`, `incubation_window_clean`, and `seal_intact` are passed through from packet inputs rather than measured by `lab-feet`).

---

## 5. History

| Date | Ref | Summary |
|---|---|---|
| `2026-11-18` | `a7b1532` (`main`) | Initial CPU logit-regression distillation + per-tensor `int8`/`4-bit` demo and resource catalog. |
| `2026-11-23` | `0.2.0` (`fix/resolve-audit-issues`) | Resolved audit issues `#1..#24`: packaged `src/lab_feet/` + `lab-feet` CLI, single-thread BLAS percentile timing, hybrid `KL(T^2)+CE` & zero-mean MSE distillation, packed `4-bit` `QuantizedModel`, `W8A8` `int32` matmul, `LFT1` serializer, and $SO(3)$-equivariant quantization. |
| `2026-11-25` | `fix/phase2-audit-and-wire` | Resolved Phase-2 issues `#26..#31`: calibrated activation clipping & SmoothQuant, odd-length nibble checks, `int32` MAC tiling, `LabFeetLean` formal theorems, SHA-256 wire hot-loading, and C99 + ONNX `opset >= 21` exporters. |
| `2026-11-25` | `feat/open-datasets-2026` | Added SPDX-gated `OpenEdgeDataset` registry, `DeterministicFeatureNormalizer`, and multi-batch `DriftAbstainingInt4Classifier` evaluation. |
| `2026-11-27` | `feat/frontier-onnx-drift-and-quantization` | Resolved issues `#34..#37` + frontier L0 upgrades: `MatMulInteger` `int32` ONNX `qlinear` logits & Ubuntu `onnxruntime` CI (`#34`); fail-closed `autolab_wire.py` calibration/floor/monitor/cap gates & provenance fields (`#35`); pinned SHA-256 across 10 open datasets (adding 3 FT-IR archives), `MSP`/`Energy`/`ReAct`/`Mahalanobis`/`TTN+CRC` drift evaluation on full 13,910-sample UCI Gas Drift, `WHT+GPTQ` `W4A8` & `39.1k`-param `SpectralResNet1D` (`#36`); 10-seed same-trainer `CE` control, validation-stagnation `converged`, and honest integer vs dequantized latency labels (`#37`). |
