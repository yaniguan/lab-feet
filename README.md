# lab-feet

Let the lab agent *walk on its own feet*: big models teach small ones, small ones get quantized, and they run on a cheap board next to the bench.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/auto-lab/), the feet are Layer `L0 · Edge Distillation & Quantized Runtime`: cloud GPUs run only in short bursts, while small quantized models execute locally beside wet-lab instruments. This repo collects distillation and quantization tools and edge runtimes. The reference package (`src/lab_feet/`) and CPU script (`demo/distill_quantize.py`) distill large teachers into compact students using hybrid $T^2$-scaled KL divergence + cross-entropy, quantize weights to `int8` and packed two's-complement `4-bit` (`uint8` 2-weights-per-byte), run `W8A8` integer `int32`-accumulator matmuls, serialize flat `LFT1` binary models, and guard $SO(3)$-equivariant channel quantization.

## Quickstart & CLI

```bash
pip install -e ".[dev]"

# Run digits + wet-lab spectral regime distillation + lab-ears autoencoder compression
python demo/distill_quantize.py --assert-thresholds --export-bin reports/student_4bit.lft1 --json-out reports/benchmark.json

# Inspect serialized LFT1 binary container or export benchmark JSON Schema
lab-feet inspect-binary reports/student_4bit.lft1
lab-feet export-schema

# List, live-probe, or fetch on-demand open edge sensor & 1D spectral datasets into gitignored data/
python scripts/fetch_open_datasets.py --list
python scripts/fetch_open_datasets.py --verify-urls
python scripts/fetch_open_datasets.py --dataset uci_gas_sensor_drift --smoke-distill --smoke-drift

# Run unit & regression test suite
pytest
```

## Distillation, Quantization & Edge Runtime Benchmark

- **Leakage-free split**: `scikit-learn` digits (`1,797` `8x8` images, `64` features) split stratified into `train=1,068`, `val=189`, and `test=540` (`StandardScaler` fit strictly on `train`; `val` used for checkpoint selection and MSE-clipping calibration).
- **Distillation comparison**: Compares hard-label cross-entropy, uncentered logit MSE, Hinton zero-mean logit MSE, and hybrid $(1 - \alpha)\, T^2\, \mathrm{KL}(p_T \| q_T) + \alpha\, \mathrm{CE}(y, z_s)$ (`T = 4.0`, `alpha = 0.25`). Hybrid `KL(T^2)+CE` lifts the 32-unit student (`2,410` params) to **`0.9852` test accuracy**, outperforming both hard-label training (`0.9759`) and the `85,002`-parameter teacher (`0.9778`).
- **Quantization & `W8A8` integer matmul**: Evaluates `int8` per-tensor (`2,544 B`), true `W8A8` dynamic `int8` activation x `int8` weight `int32` matmul (`100%` prediction agreement), and packed `uint8` `4-bit` (`1,360 B` per-tensor; `1,520 B` per-channel + MSE-clipping search reaching `99.63%` agreement with fp32), verified on disk via the `LFT1` flat binary serializer (`1,520 B` payload + `64 B` header = `1,584 B`).
- **Wet-lab domain & `lab-ears` compression**: Includes a 6-class wet-lab benchtop spectral regime distillation benchmark (`48` mel-band features) and `7.23x` `4-bit` compression (`470,304 B -> 65,088 B`) of the `lab-ears` `320 -> 128 -> 128 -> 8 -> 128 -> 128 -> 320` acoustic anomaly `MLPRegressor` autoencoder.
- **$SO(3)$-equivariant quantization (`lab_feet.equivariant`)**: Quantizes scalar invariant channel-mixing weights `(C_in, C_out)` while blocking naive spatial `(2l + 1)` rounding that would break 3D rotational equivariance $f(X R^\top) = f(X) R^\top$ in [`YLM`](https://github.com/yaniguan/YLM) force heads.

See [`demo/output.txt`](demo/output.txt) for captured reference output (`median`, `p95`, `p99`, and `IQR` latencies in microseconds with single-thread BLAS pinning).

## Datasets

`lab_feet.datasets` (`OPEN_EDGE_DATASETS`) provides a typed, SPDX-license-gated `OpenEdgeDataset` registry, leakage-free `DeterministicFeatureNormalizer` (`zscore`, `robust_iqr`, `minmax_symmetric`, `signed_log1p_zscore`, `snv_zscore`), on-demand fetchers (`scripts/fetch_open_datasets.py` writing into gitignored `data/`), and direct loaders feeding `run_open_dataset_distill_quantize` and multi-batch `DriftAbstainingInt4Classifier` (`evaluate_temporal_drift_abstaining_int4`).

| Dataset | ID (`lab_feet.datasets`) | Shape / Drift | What | License |
|---|---|---|---|---|
| [scikit-learn digits](https://scikit-learn.org/stable/datasets/toy_dataset.html) | *(bundled)* | `1,797 x 64` (10 classes) | `8x8` handwritten digit images bundled with `scikit-learn` | BSD-3-Clause |
| Wet-Lab Spectral Regimes | *(synthetic)* | `1,800 x 48` (6 classes) | Deterministic benchtop harmonic/cavitation mel spectra (`lab_feet.benchmarks`) | MIT |
| [MNIST-1D](https://github.com/greydanus/mnist1d) | `mnist1d` | `5,000 x 40` (10 classes) | Compact 40-point template-warped 1D signal classification (`greydanus/mnist1d`) | Apache-2.0 |
| [UCI Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) | `uci_gas_sensor_drift` | `13,910 x 128` (6 classes, 10 drift batches) | 16-sensor MOX chemical array over 36 months (`batch1..batch10` aging drift, `DOI:10.24432/C5RP6W`) | CC-BY-4.0 |
| [UCI HAR Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | `uci_har_inertial` | `10,299 x 561` (6 classes) | 50 Hz 3-axis accelerometer & 3-axis gyroscope inertial window features (`DOI:10.24432/C54S4K`) | CC-BY-4.0 |
| [CWRU Bearing Vibration](https://huggingface.co/datasets/odysseywt/CWRU) | `cwru_bearing_vibration` | `21,786 x 17` (4 classes, 5 spall batches) | Rolling-element bearing vibration fault features across `0.000"–0.028"` spall severities | MIT |
| [UCI Hydraulic Vibration](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) | `uci_hydraulic_vibration` | `2,205 x 60` (4 classes, 3 thermal batches) | Benchtop hydraulic rig 1-Hz mechanical vibration (`VS1`) across valve wear & cooler drift (`DOI:10.24432/C5CW21`) | CC-BY-4.0 |
| [SERS Gold-Nanorod Bacteria (2026)](https://github.com/erkankaratas-eng/sers-gna-bacteria) | `sers_bacteria_raman_2026` | `117 x 1,215` (9 classes, 3 acq. batches) | SERS Raman micro-spectroscopy (`248–1998 cm^-1`) of 9 clinical bacterial strains (Gulmus et al., 2026) | MIT |
| [EthanolLevel NIR Spectroscopy](https://zenodo.org/records/11190985) | `ethanol_nir_spectroscopy` | `1,004 x 1,751` (4 classes, bottle shift) | Through-glass non-invasive NIR/Raman spectra (`35%–45%` ethanol across 20 glass bottle types) | CC-BY-4.0 |

## Open-source repos

Stars as of 2026-11-18.

| Repo | What | Stars | License |
|---|---|---|---|
| [ollama/ollama](https://github.com/ollama/ollama) | Get up and running with Kimi, GLM, MiniMax, DeepSeek, gpt-oss, Qwen, Gemma and other... | 187,492 | MIT |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | LLM inference in C/C++ | 135,599 | MIT |
| [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | Universal LLM Deployment Engine with ML Compilation | 23,402 | Apache-2.0 |
| [artidoro/qlora](https://github.com/artidoro/qlora) | QLoRA: Efficient Finetuning of Quantized LLMs | 11,072 | MIT |
| [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | Accessible large language models via k-bit quantization for PyTorch. | 8,544 | MIT |
| [pytorch/executorch](https://github.com/pytorch/executorch) | On-device AI across mobile, embedded and edge for PyTorch | 5,180 | custom (see repo) |
| [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) | Machine Learning Containers for NVIDIA Jetson and JetPack-L4T | 4,902 | custom (see repo) |
| [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper) | Distilled variant of Whisper for speech recognition. 6x faster, 50% smaller, within 1%... | 4,117 | MIT |
| [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | [MLSys 2024 Best Paper Award] AWQ: Activation-aware Weight Quantization for LLM... | 3,667 | MIT |
| [huggingface/optimum](https://github.com/huggingface/optimum) | Accelerate inference and training of Transformers, Diffusers, TIMM and Sentence... | 3,524 | Apache-2.0 |
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) | Code for the ICLR 2023 paper "GPTQ: Accurate Post-training Quantization of Generative... | 2,403 | Apache-2.0 |
| [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant) | [ICML 2023] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large... | 1,705 | MIT |
| [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) | LLM model quantization (compression) toolkit with HW acceleration support for Nvidia... | 1,293 | custom (see repo) |

## Papers

| Paper | Venue | Link |
|---|---|---|
| Distilling the Knowledge in a Neural Network | NIPS 2014 workshop | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
| DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | arXiv 2019 | [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) |
| MiniLLM: On-Policy Distillation of Large Language Models | ICLR 2024 | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | NeurIPS 2022 | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | ICLR 2023 | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | ICML 2023 | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | MLSys 2024 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| QLoRA: Efficient Finetuning of Quantized LLMs | NeurIPS 2023 | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation | arXiv 2025 | [arXiv:2506.07530](https://arxiv.org/abs/2506.07530) |

## Conferences and workshops

- [MLSys](https://mlsys.org/)
- [Embedded Vision Summit](https://www.embedded-vision.com/summit)
- [ICML](https://icml.cc/)
- [CoRL - Conference on Robot Learning](https://www.corl.org/)

## Progress

Recent results, newest first. Date = first arXiv version; each line is a short summary of the abstract.

| Date | Paper | In one line |
|---|---|---|
| 2026-10-14 | [Seeing at the Edge: Energy Characterization of Vision-Language Model Inference on Edge Platforms](https://arxiv.org/abs/2610.20136) | Energy of a 2B vision-language model on a Raspberry Pi 5 with NPU and on a small GPU board; NPU offload cuts energy per token by about two thirds. |
| 2026-07-09 | [Rethinking Small VLM Quantization: From Component-Wise Analysis to Hardware-Aware Edge Deployment](https://arxiv.org/abs/2607.08029) | Component-wise quantization of small VLMs on edge boards: MoE backbones tolerate INT4 better than dense ones. |
| 2026-07-02 | [Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots](https://arxiv.org/abs/2607.02501) | Embodied.cpp: a portable C++ runtime for running robot policy models on mixed edge hardware. |
| 2025-06-09 | [BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation](https://arxiv.org/abs/2506.07530) | BitVLA: a vision-language-action model with ternary weights {-1, 0, 1} for manipulation. |

## System Profile: Object · Data · Model · Training · History · Unique Properties

`lab-feet` is part of the [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ecosystem ([Project Studio](https://yaniguan.github.io/auto-lab/)) at **Layer `L0 · Edge Distillation & Quantized Runtime`**.

- **Object**: Edge compute substrate that distills large teacher networks into compact students, quantizes weights to `int8` and packed `4-bit` (`uint8` nibbles), runs `W8A8` integer `int32` matmul verification, and serializes flat `LFT1` binary models for benchtop microcontrollers and SBCs.
- **Data**: Stratified `train=1,068 / val=189 / test=540` split on `scikit-learn` digits (`StandardScaler` fit strictly on `train`), 6-class wet-lab spectral regime data (`1,800 x 48`), `lab-ears` 320-dim mel-patch autoencoder benchmarks, and 7 SPDX-gated open-source edge datasets (`mnist1d`, `uci_gas_sensor_drift` 10-batch 36-month MOX drift, `uci_har_inertial`, `cwru_bearing_vibration`, `uci_hydraulic_vibration`, `sers_bacteria_raman_2026`, `ethanol_nir_spectroscopy`) fetched on-demand into gitignored `data/`.
- **Model**: `(256, 256)` ReLU teacher (`85,002` params, `340,008 B`) -> `(32,)` students (`2,410` params, `9,640 B` fp32 -> `2,544 B` `int8` -> `1,360 B` per-tensor `4-bit` / `1,520 B` per-channel `4-bit`), plus `lab-ears` `320->128->128->8->128->128->320` autoencoder (`470,304 B -> 65,088 B`, `7.23x` smaller) and $SO(3)$-equivariant scalar irrep layers.
- **Training**: Hybrid $T^2$-scaled KL divergence + cross-entropy Adam distillation (`0.9852` test accuracy on digits) compared side-by-side against hard-label CE (`0.9759`), zero-mean logit MSE, and uncentered logit MSE (`0.9685`), with scoped `ConvergenceWarning` tracking and scoped `np.random.default_rng` seeds.
- **History**: Initialized in `a7b1532` (`2026-11-18`) and expanded in `0.2.0` (`fix/resolve-audit-issues`) resolving all 24 audit issues (`#1..#24`).
- **Unique Properties**: Genuine packed `uint8` 2-weights-per-byte `QuantizedLayer` storage verified on disk (`LFT1` binary + `.npz`), `W8A8` `int32` reference execution, MSE-optimal clipping search, single-thread BLAS percentile timing (`median`/`p95`/`p99`/`IQR`), and $SO(3)$ rotation-equivariant channel quantization.

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

[`autolab_wire.py`](autolab_wire.py) implements the Layer `L0` edge runtime node (`REPO_ID = "lab-feet"`, `LAYER = "L0"`) in the 15-repository `auto-lab` packet pipeline (`autolab-wire/v1`):

- **Upstream inputs**:
  - [`stepbot`](https://github.com/yaniguan/stepbot) (`L1`): Supplies `stepbot_execution` (`protocol_sha256`, `lean-mcp/v1` dispatch receipt, and `next_deck_epoch`).
  - [`lab-hands`](https://github.com/yaniguan/lab-hands) (`L1`): Supplies `manipulation` (`start_well`, `target_well`, `collision_free`, RRT `waypoints`, `path_length_rad`, and `clearance_mm`).
  - [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3`): Supplies `SplitGuard`-isolated, anti-hack RLVR post-trained checkpoints distilled into compact `L0` edge students.
- **`L0` verification & execution (`process_packet`)**:
  - Short-circuits unchanged when `packet["halted"]` is `True`, and blocks tainted upstream checkpoints (`split_guard_clean is False`) with `L0_FEET_UPSTREAM_SPLITGUARD_BLOCK`.
  - Supports hot-loading distilled student checkpoints (`LFT1` bytes/hex/path, `.npz`, or `(W, b)` layers) keyed in an LRU cache by SHA-256 (`checkpoint_sha256`) alongside canonical calibration verification (`calibration_sha256`), halting on mismatch with `L0_FEET_CHECKPOINT_SHA256_MISMATCH`.
  - Derives a deterministic 24-feature benchtop sensor vector from `stepbot_execution` and `manipulation`, verifies exact two's-complement `[-8, 7]` nibble round-tripping (`pack_4bit` / `unpack_4bit`), runs quantized `int8` (`W8A8`) and `4-bit` (`W4A8`) `int32`-accumulator monitor inference (`forward_int8`), and checks `< 512 B` OOD/saturation `SAFE_HOLD` (`L0_FEET_SAFE_HOLD_ABSTAIN`).
  - Enforces the Lean fidelity cap `quant_acc_drop_permille <= max_acc_drop_permille` (`invariant = "nibble_pack_roundtrip_and_drift_cap"`), setting `halted = True` and `halt_reason = "L0_FEET_QUANT_DRIFT_BLOCK"` on drift excess.
- **Downstream output**:
  - [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`): Consumes `out["edge_runtime"]` (`quantization`, `nibble_roundtrip_exact`, `packed_bytes`, `dispense_error_permille`, `aspiration_clean`, `incubation_window_clean`, `seal_intact`, `checkpoint_sha256`, `calibration_sha256`, `fp32_int4_pred_agree`, `safe_hold_abstained`) for post-execution causal blame attribution, NNLS precursor deconvolution, and SHA-256 `CertifiedLedger` gating.

## Repository Layout

```text
lab-feet/
├── .github/workflows/
│   └── demo.yml                      # Linux / Windows / macOS CI running pytest & threshold gates
├── community_extensions/
│   └── drift_abstaining_int4.py      # <512 B SRAM int4/int8 OOD & clipping SAFE_HOLD monitor
├── demo/
│   ├── distill_quantize.py           # Benchmark runner (digits, wet-lab regimes, lab-ears AE)
│   └── output.txt                    # Captured accuracy, byte-size, and latency percentile report
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical specification & mathematical invariants
├── lean/
│   └── LabFeetLean.lean              # Lean 4 nibble, saturating requant, SRAM & 5-gate OOD proofs
├── schema/
│   └── benchmark_report.schema.json  # Draft 2020-12 JSON Schema for benchmark JSON reports
├── scripts/
│   └── fetch_open_datasets.py        # On-demand dataset downloader, SPDX verifier & URL prober
├── src/lab_feet/
│   ├── __init__.py                   # Public package API
│   ├── benchmarks.py                 # Digits, wet-lab spectral, and lab-ears AE benchmarks
│   ├── cli.py                        # `lab-feet` CLI (benchmark, inspect-binary, export-c-header, export-onnx)
│   ├── datasets.py                   # OpenEdgeDataset registry, normalizers & multi-batch drift bridges
│   ├── distill.py                    # Zero-mean MSE & hybrid T^2 KL + CE Adam distillation
│   ├── equivariant.py                # SO(3)-equivariant scalar irrep weight quantization & guards
│   ├── export_c.py                   # Freestanding zero-malloc C99 / CMSIS-NN `.rodata` header exporter
│   ├── export_onnx.py                # ONNX opset >= 21 INT4/INT8 QDQ & QLinearMatMul graph/proto exporter
│   ├── quantize.py                   # Symmetric/asymmetric per-tensor/channel int8/4-bit, SmoothQuant & W8A8
│   ├── runtime.py                    # Cross-platform cpu_model, BLAS pinning, forward & percentiles
│   └── serialize.py                  # Flat little-endian LFT1 binary serializer & verifier
├── tests/
│   ├── test_autolab_wire.py          # autolab-wire/v1 L0 packet, SHA-256 hot-load & telemetry tests
│   ├── test_community_extensions.py  # <512 B SRAM OOD/clipping monitor & Lean 5-gate parity tests
│   ├── test_distill_quantize.py      # Offline pytest suite covering issues #1..#31
│   └── test_open_datasets_2026.py    # Registry, SPDX gate, normalizer, distill & gas-drift tests
├── autolab_wire.py                   # L0 node for the 15-repo autolab-wire/v1 pipeline
├── LICENSE                           # MIT License
├── pyproject.toml                    # Package build metadata & `lab-feet` console script
├── README.md                         # Overview, benchmark summary, and edge-runtime catalog
└── requirements.txt                  # Pinned minimum numpy>=1.26 and scikit-learn>=1.4
```

## Contributing

Found a broken link or a missing dataset, repo, or paper? Open an issue or a pull request.

## License

Code in this repo: MIT ([`LICENSE`](LICENSE)). Listed datasets, repos, and papers keep their own licenses.
