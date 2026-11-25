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
- `lab_feet.datasets`: Typed, SPDX-license-gated `OpenEdgeDataset` registry (`OPEN_EDGE_DATASETS`), leakage-free `DeterministicFeatureNormalizer` (`zscore`, `robust_iqr`, `minmax_symmetric`, `signed_log1p_zscore`, `snv_zscore`) with SHA-256 calibration state digests, on-demand archive fetchers (`scripts/fetch_open_datasets.py`), and direct pipelines feeding `distill_kl_student` / `quantize_model` and multi-batch `DriftAbstainingInt4Classifier`.
- `lab_feet.runtime`: Cross-platform `cpu_model()` (Linux x86/ARM64, macOS, Windows), single-thread BLAS pinning (`OMP`/`OPENBLAS`/`MKL` + `threadpoolctl`), activation-validated `get_layers()`, `float64`-stabilized `softmax()`, and `latency_stats()` (`median`, `p95`, `p99`, `IQR`).

---

## 2. Data & Splits

1. **`scikit-learn` Digits (`run_digits_benchmark`)**: `1,797 x 64` images split stratified without leakage into `train=1,068`, `val=189`, and `test=540` (`StandardScaler` fit strictly on `train`). `val` drives KL distillation checkpoint selection and post-training `activation_mse` clipping search; `test` is touched once for final reporting.
2. **Wet-Lab Spectral Regimes (`run_wetlab_acoustic_benchmark`)**: `1,800 x 48` deterministic 6-regime benchtop harmonic/cavitation dataset modeling capacity-constrained edge classification (`(256, 256)` teacher -> `(16,)` student).
3. **`lab-ears` Acoustic Autoencoder (`run_autoencoder_compression_benchmark`)**: `320 -> 128 -> 128 -> 8 -> 128 -> 128 -> 320` (`117,576` parameters, `470,304 B` fp32) compressed `7.23x` to `65,088 B` in per-channel `4-bit` while preserving `> 0.98` anomaly detection AUROC.
4. **Open Edge Sensor, Vibration, Gas-Drift & 1D Spectral Registry (`lab_feet.datasets`)**: Seven permissive-SPDX (`Apache-2.0`, `CC-BY-4.0`, `MIT`) real-world datasets fetched on-demand into gitignored `data/` (`mnist1d` `40`-dim signals, `uci_gas_sensor_drift` `128`-dim 16-MOX array across `10` chronological 36-month aging batches, `uci_har_inertial` `561`-dim 50 Hz IMU windows, `cwru_bearing_vibration` `17`-dim bearing spall severity features, `uci_hydraulic_vibration` `60`-dim `VS1` mechanical vibration across valve degradation & cooler thermal drift, `sers_bacteria_raman_2026` `1,215`-channel gold-nanorod bacterial SERS spectra, and `ethanol_nir_spectroscopy` `1,751`-channel through-glass NIR spectra). Multi-batch drift evaluation (`evaluate_temporal_drift_abstaining_int4`) fits `DeterministicFeatureNormalizer` and optional `< 512 B` MCU F-ratio channel selection strictly on early calibration batches (e.g. `Batch 1` of `uci_gas_sensor_drift`) and tracks `SAFE_HOLD` abstention vs silent classification errors across downstream aging batches (`Batches 2..10`).

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

## 4. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-feet"`, `LAYER = "L0"`) connects `lab-feet` to the 15-repository `auto-lab` execution and learning loop (`lab-ears`, `lab-ask-why`, `lab-decide`, `lab-automation-atlas`, `buy-or-make`, `labshift`, `lab-rehearsal`, `lab-verify`, `stepbot`, `lab-hands`, `lab-feet`, `lab-attrib`, `twinstep`, `labfill`, `labbench-posttrain`):

- **Upstream Interfaces**:
  - `stepbot` (`L1`): Provides `stepbot_execution` (`protocol_sha256`, `lean-mcp/v1` execution receipt, `next_deck_epoch`).
  - `lab-hands` (`L1`): Provides `manipulation` (`start_well`, `target_well`, `collision_free`, `waypoints`, `path_length_rad`, `clearance_mm`).
  - `labbench-posttrain` (`L3`): Provides `SplitGuard`-verified, anti-hack RLVR post-trained policies for distillation into compact `L0` edge models.
- **`L0` Wire Execution & Guardrail (`process_packet`)**:
  - Deep-copies the incoming `autolab-wire/v1` packet, returns halted packets unchanged (`halted == True`), and blocks tainted upstream checkpoints (`split_guard_clean is False`) with `L0_FEET_UPSTREAM_SPLITGUARD_BLOCK`.
  - Supports hot-loading `LFT1`/`.npz`/`(W, b)` student checkpoints verified against `checkpoint_sha256` and canonical `calibration_sha256` (`L0_FEET_CHECKPOINT_SHA256_MISMATCH` on mismatch).
  - Maps `stepbot_execution` and `manipulation` deterministically into a 24-channel benchtop sensor feature vector, executes `pack_4bit` / `unpack_4bit` two's-complement nibble round-trip verification over `[-8, 7]` and model weight layers (`packed_bytes = 240`), runs `W8A8` / `W4A8` `int32`-accumulator inference via `forward_int8`, and evaluates the `< 512 B` 5-gate `DriftAbstainingInt4Classifier` OOD/saturation monitor (`L0_FEET_SAFE_HOLD_ABSTAIN`).
  - Enforces the Lean drift ceiling `quant_acc_drop_permille <= max_acc_drop_permille`; violations halt the packet with `halt_reason = "L0_FEET_QUANT_DRIFT_BLOCK"`.
  - Records `trace` and `bottom_reasoning_traces` (`invariant = "nibble_pack_roundtrip_and_drift_cap"`).
- **Downstream Interface**:
  - `lab-attrib` (`L4`): Reads `out["edge_runtime"]` (`quantization = "int8_and_4bit"`, `nibble_roundtrip_exact`, `packed_bytes`, `dispense_error_permille`, `aspiration_clean`, `incubation_window_clean`, `seal_intact`, `checkpoint_sha256`, `calibration_sha256`, `fp32_int4_pred_agree`, `safe_hold_abstained`) to separate robotic execution artifacts (`EXECUTION_ARTIFACT`) and precursor ghost peaks (`AMBIGUOUS_CHARACTERIZATION`) from certified results admitted into the SHA-256 `CertifiedLedger`.

---

## 5. History

| Date | Ref | Summary |
|---|---|---|
| `2026-11-18` | `a7b1532` (`main`) | Initial CPU logit-regression distillation + per-tensor `int8`/`4-bit` demo and resource catalog. |
| `2026-11-23` | `0.2.0` (`fix/resolve-audit-issues`) | Resolved audit issues `#1..#24`: packaged `src/lab_feet/` + `lab-feet` CLI, cross-platform `cpu_model`, single-thread BLAS percentile timing, leakage-free `train/val/test` splits, hybrid `KL(T^2)+CE` & zero-mean MSE distillation, per-channel/asymmetric/MSE-clipped `int8` & `4-bit` `QuantizedModel`, `W8A8` `int32` matmul, `LFT1` binary serializer, wet-lab & `lab-ears` benchmarks, $SO(3)$-equivariant quantization, multi-OS CI, and `pytest` suite. |
| `2026-11-23` | `autolab-wire/v1` | Added `autolab_wire.py` (`L0` edge runtime wire node) and `tests/test_autolab_wire.py` connecting upstream `stepbot`, `lab-hands`, and `labbench-posttrain` to downstream causal blame gate `lab-attrib`. |
| `2026-11-25` | `fix/phase2-audit-and-wire` | Resolved Phase-2 audit issues `#26..#31`: calibrated activation clipping, unsigned post-ReLU & SmoothQuant (`#26`), odd-length 4-bit `orig_len` & `LFT1` payload validation (`#27`), `int64` saturating requantization & `int32` MAC tiling (`#28`), `LabFeetLean` nibble/requant/5-gate theorems (`#29`), SHA-256 checkpoint hot-loading in `autolab_wire.py` (`#30`), and freestanding C99 `.h` + ONNX `opset >= 21` exporters (`#31`). |
| `2026-11-25` | `feat/open-datasets-2026` | Added `src/lab_feet/datasets.py`, `scripts/fetch_open_datasets.py`, and `tests/test_open_datasets_2026.py`: SPDX-gated `OpenEdgeDataset` registry of 7 real edge sensor, MOX gas-drift (`Batches 1..10`), IMU, bearing/hydraulic vibration, 2026 SERS Raman, and NIR spectroscopy datasets, `DeterministicFeatureNormalizer`, and loaders feeding `distill_quantize` and multi-batch `DriftAbstainingInt4Classifier`. |
