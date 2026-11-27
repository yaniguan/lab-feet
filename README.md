# lab-feet

Let the lab agent *walk on its own feet*: big models teach small ones, small ones get quantized, and they run on a cheap board next to the bench.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/auto-lab/), the feet are Layer `L0 · Edge Distillation & Quantized Runtime`: cloud GPUs run only in short bursts, while compact quantized models execute locally beside wet-lab instruments. This repo provides distillation, post-training quantization, and fail-closed edge runtimes (`src/lab_feet/` and `demo/distill_quantize.py`): hybrid $T^2$-scaled KL divergence + cross-entropy vs same-trainer CE controls, `int8` (`W8A8`) and packed two's-complement `4-bit` (`W4A8`) quantization, Walsh-Hadamard (`WHT`) outlier rotation + Cholesky `GPTQ-lite` error compensation, a `39.1k`-parameter 1D Residual Spectral Conv/MLP backbone (`SpectralResNet1D`), freestanding C99 `.h` and ONNX `opset >= 21` (`qdq` & `qlinear`) exporters, `LFT1` flat binary serialization, and $SO(3)$-equivariant scalar channel quantization.

## Quickstart & CLI

```bash
pip install -e ".[dev,edge]"

# Run 10-seed digits comparison + seed-0 latency report + synthetic toy benchmarks
python demo/distill_quantize.py --seeds 10 --assert-thresholds --export-bin reports/student_4bit.lft1 --json-out reports/benchmark.json

# Inspect serialized LFT1 binary container, export C99 header, or export ONNX qdq/qlinear graphs
lab-feet inspect-binary reports/student_4bit.lft1
lab-feet export-c-header reports/student_4bit.lft1 -o reports/student_4bit.h
lab-feet export-onnx reports/student_4bit.lft1 -o reports/student_4bit_qlinear.onnx --mode qlinear
lab-feet export-schema

# List, live-probe, or fetch SHA-256-pinned open edge sensor & 1D spectral datasets into gitignored data/
python scripts/fetch_open_datasets.py --list
python scripts/fetch_open_datasets.py --verify-urls
python scripts/fetch_open_datasets.py --dataset uci_gas_sensor_drift --smoke-distill --smoke-drift

# Run offline unit & regression test suite
pytest
```

## Distillation, Quantization & Edge Runtime Benchmark

### 1. 10-Seed `scikit-learn` Digits Comparison (`--seeds 10`, Split-and-Init Seeds `0..9`)

Each seed `s in 0..9` independently splits `scikit-learn` digits (`1,797` `8x8` images, `64` features) stratified into `train=1,068`, `val=189`, and `test=540` (`StandardScaler` fit strictly on `train`; `val` used for checkpoint selection and quantization calibration) and initializes all networks (`T = 4.0`, `alpha = 0.25`, `240` Adam epochs):

| Model | Params | Weight Bytes | Val Acc (`mean +- std`) | Test Acc (`mean +- std`) | Test Range (`seeds 0..9`) |
|---|---:|---:|---:|---:|---:|
| `teacher (256,256) fp32` | 85,002 | 340,008 B | `0.9825 +- 0.0095` | `0.9774 +- 0.0046` | `[0.9704, 0.9870]` |
| `sklearn MLPClassifier hard-label` | 2,410 | 9,640 B | `0.9741 +- 0.0135` | `0.9700 +- 0.0076` | `[0.9611, 0.9852]` |
| `student same-trainer CE fp32` (`alpha=1.0`) | 2,410 | 9,640 B | `0.9831 +- 0.0094` | `0.9756 +- 0.0057` | `[0.9630, 0.9833]` |
| `student hybrid KL(T^2)+CE fp32` (`alpha=0.25`) | 2,410 | 9,640 B | `0.9852 +- 0.0074` | `0.9748 +- 0.0049` | `[0.9667, 0.9852]` |
| `student KL int8 W8A8 (int32)` | 2,410 | 2,544 B | `0.9847 +- 0.0076` | `0.9750 +- 0.0048` | `[0.9667, 0.9852]` |
| `student KL 4-bit W4A8 (int32)` | 2,410 | 1,520 B | `0.9799 +- 0.0088` | `0.9744 +- 0.0045` | `[0.9685, 0.9833]` |
| `student KL 4-bit WHT+GPTQ W4A8 (int32)` | 2,410 | 1,520 B | `0.9825 +- 0.0089` | `0.9739 +- 0.0048` | `[0.9667, 0.9833]` |

- **Same-trainer `CE` control (`#37`)**: Most of the gap over `sklearn MLPClassifier hard-label` (`0.9700`) comes from mini-batch Adam + validation checkpoint selection (`student same-trainer CE fp32` reaches `0.9756 +- 0.0057`). Against that matched same-trainer `CE` control, hybrid `KL(T^2)+CE` (`0.9748 +- 0.0049`) records **`5 / 0 / 5` (`win / tie / loss`)** over 10 split-and-init seeds (`mean diff = -0.0007`).
- **Seed-0 reference & timing honesty (`demo/output.txt`)**: On `seed 0` alone, `student hybrid KL(T^2)+CE fp32` reaches `0.9852` test accuracy (vs `0.9815` same-trainer `CE` and `0.9778` teacher), `student KL 4-bit W4A8 (int32)` and `WHT+GPTQ W4A8 (int32)` reach `0.9833`, and `student KL 4-bit C99 fixed-point (int32)` reaches `0.9852` (`1,520 B` payload + `64 B` `LFT1` header = `1,584 B`). Dequantized float32 accuracy checks are explicitly labeled `(dequantized fp32 timing)` (`~9–12 us` in BLAS fp32), whereas Python/NumPy integer `W8A8` / `W4A8` / `C99` simulation rows report true unvectorized Python integer simulation times (`~109–476 us`; freestanding compiled C99 / ONNX Runtime `MatMulInteger` kernels are exported via `export-c-header` and `export-onnx --mode qlinear`, verified to `>= 0.99` top-1 C99 parity in Ubuntu CI `#34`).
- **Synthetic toy benchmarks**: `demo/distill_quantize.py` also includes two explicitly **synthetic** sanity tasks: a 6-class 48-feature `i.i.d.` Gaussian 2-layer oracle classification task (`run_wetlab_acoustic_benchmark`) and a synthetic 8-D linear subspace in $\mathbb{R}^{320}$ `MLPRegressor` autoencoder (`run_autoencoder_compression_benchmark`, `470,304 B -> 65,088 B`, `7.23x` `4-bit` compression, `0.9936` anomaly AUROC).
- **$SO(3)$-equivariant quantization (`lab_feet.equivariant`)**: Quantizes scalar invariant channel-mixing weights `(C_in, C_out)` while rejecting spatial `(2l + 1)` elementwise rounding that would break 3D rotational equivariance $f(X R^\top) = f(X) R^\top$ in [`YLM`](https://github.com/yaniguan/YLM) force heads.

## Real Public Datasets & 3-Seed Benchmark (`benchmarks/results/real_datasets_2026.json`)

`lab_feet.datasets` (`OPEN_EDGE_DATASETS`) registers **10 commercial-safe SPDX-gated public sensor, vibration, gas-drift, Raman, NIR, and FT-IR datasets** with **pinned SHA-256 digests and exact archive byte sizes** (raising `DatasetIntegrityError` on mismatch unless `allow_unverified_sha256=True`). Results below are `mean` (`+- std`) test accuracies over 3 seeds (`0, 1, 2`):

| Dataset (`dataset_id`) | Shape (`N x d`, `C`) | License | Teacher FP32 (Random) | Same-Trainer CE (Random) | KL+CE FP32 (Random) | `4-bit` WHT+GPTQ `W4A8` (Random) | Temporal Split (`KL+CE` / `4b-WHT`) | `SpectralResNet1D` `39.1k` `4-bit W4A8` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| [`mnist1d`](https://github.com/greydanus/mnist1d) | `5,000 x 40` (10) | Apache-2.0 | `0.5891 +- 0.0120` | `0.5587 +- 0.0118` | `0.5484 +- 0.0077` | `0.5464` | *(no batches)* | **`0.9352 +- 0.0057`** |
| [`uci_gas_sensor_drift`](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) | `13,910 x 128` (6, 10b) | CC-BY-4.0 | `0.9929 +- 0.0015` | `0.9929 +- 0.0005` | `0.9927 +- 0.0011` | `0.9920` | `0.7286` / `0.7224` | — |
| [`uci_har_inertial`](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | `10,299 x 561` (6) | CC-BY-4.0 | `0.9690 +- 0.0189` | `0.9689 +- 0.0193` | `0.9685 +- 0.0209` | `0.9671` | *(no batches)* | — |
| [`cwru_bearing_vibration`](https://huggingface.co/datasets/odysseywt/CWRU) | `21,786 x 17` (4, 5b) | MIT | `0.9772 +- 0.0005` | `0.9728 +- 0.0035` | `0.9717 +- 0.0004` | `0.9424` | `0.5000` / `0.6663` | — |
| [`uci_hydraulic_vibration`](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) | `2,205 x 60` (4, 3b) | CC-BY-4.0 | `0.4976 +- 0.0162` | `0.5024 +- 0.0148` | `0.5109 +- 0.0039` | `0.5199` | `0.5082` / `0.5082` | — |
| [`sers_bacteria_raman_2026`](https://github.com/erkankaratas-eng/sers-gna-bacteria) | `117 x 1,215` (9, 3b) | MIT | `0.9111 +- 0.0416` | `0.8889 +- 0.0567` | `0.8778 +- 0.0157` | `0.8778` | `0.3796` / `0.3704` | `0.7444 +- 0.0685` |
| [`ethanol_nir_spectroscopy`](https://zenodo.org/records/11190985) | `1,004 x 1,751` (4, 2b) | CC-BY-4.0 | `0.4827 +- 0.0391` | `0.7738 +- 0.0955` | `0.5584 +- 0.0548` | `0.5558` | `0.4873` / `0.4880` | `0.4116 +- 0.0132` |
| [`olive_oil_ftir`](https://www.timeseriesclassification.com/description.php?Dataset=OliveOil) | `60 x 570` (4, 2b) | CC-BY-4.0 | `0.9333 +- 0.0000` | `0.8556 +- 0.0416` | `0.8667 +- 0.0544` | `0.8667` | `0.8889` / `0.8889` | `0.8000 +- 0.1088` |
| [`coffee_ftir`](https://www.timeseriesclassification.com/description.php?Dataset=Coffee) | `56 x 286` (2, 2b) | CC-BY-4.0 | `0.9405 +- 0.0168` | `0.9405 +- 0.0168` | `0.9762 +- 0.0337` | `0.9762` | `0.9762` / `0.9643` | `0.7262 +- 0.1606` |
| [`meat_ftir`](https://www.timeseriesclassification.com/description.php?Dataset=Meat) | `120 x 448` (3, 2b) | CC-BY-4.0 | `0.9667 +- 0.0471` | `0.9556 +- 0.0628` | `0.9667 +- 0.0471` | `0.9722` | `0.9167` / `0.9278` | `0.8944 +- 0.1057` |

- **1D Residual Spectral Conv/MLP (`SpectralResNet1D`)**: On translation-structured 40-point `mnist1d` waveforms, moving from a flat MLP (`0.5484`) to `SpectralResNet1D` (`39,146` params, `156.6 KB` fp32 -> `21.7 KB` packed `4-bit` `WHT+GPTQ W4A8`) lifts test accuracy to **`0.9352 +- 0.0057`**.
- **Random vs Temporal Split (`--smoke-distill`, `#36`)**: On 10-batch `uci_gas_sensor_drift` (`seed 0`), random splitting across all 36 months gives `0.9925` (`KL+CE` / `4-bit MSE`), hiding chronological drift; `split_mode="temporal"` (early batches `train`/`val`, later aging batches `test`) exposes the real drift drop to `0.7285` (`KL+CE`) / `0.7219` (`4-bit MSE`).

## Real 36-Month Gas-Sensor Drift Abstention (`--smoke-drift`, `#36`)

Evaluated on the full `uci_gas_sensor_drift` archive (`9,989,262 B`, SHA-256 `91e8f466f202...`; `13,910` samples, `128` MOX features, `6` classes, trained & calibrated strictly on **Batch 1** and tested across chronological **Batches 2..10**). All score baselines (`MSP` = `1 - max softmax`, Hendrycks & Gimpel 2017; `Energy` = `-logsumexp(logits)`, Liu et al. NeurIPS 2020) are calibrated to match the 5-gate monitor's **`4.49%`** Batch-1 calibration abstain rate, alongside unlabeled Test-Time Batch Normalization + Conformal Risk Control (`TTN+CRC`):

- **Monitor SRAM Table Footprint by Feature Count (`6` classes)**: `(C + 1) * d + 2 * C + 40` bytes = **`220 B` for `24` features**, **`276 B` for `32` features** (`max_monitor_features=32`), and **`948 B` for all `128` features** (exceeds `512 B` MCU SRAM target and emits an explicit runtime `UserWarning`).

| Batch | `n` | Naive `int4` Acc % | Naive Silent Err % | 5-Gate Hold % | 5-Gate Acc % | 5-Gate Silent Err % | MSP Hold % | MSP Acc % | MSP Silent Err % | MSP AURC | Energy Hold % | Energy Acc % | Energy Silent Err % | Energy AURC | `TTN+CRC` Acc % | `TTN+CRC` Silent Err % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `src_id` (B1) | 89 | 94.38 | 5.62 | 2.25 | 96.55 | 3.37 | 1.12 | 95.45 | 4.49 | 0.0052 | 4.49 | 96.47 | 3.37 | 0.0090 | 95.45 | 4.49 |
| `Batch 2` | 1,244 | 87.62 | 12.38 | 2.73 | 88.18 | 11.50 | 1.37 | 88.43 | 11.41 | 0.0615 | 0.00 | 87.62 | 12.38 | 0.0477 | 84.91 | 14.79 |
| `Batch 3` | 1,586 | 83.54 | 16.46 | 3.97 | 83.19 | 16.14 | 0.32 | 83.55 | 16.39 | 0.0669 | 0.13 | 83.65 | 16.33 | 0.0478 | 82.87 | 16.20 |
| `Batch 4` | 161 | 69.57 | 30.43 | 29.19 | 73.68 | 18.63 | 18.63 | 77.86 | 18.01 | 0.1186 | 0.00 | 69.57 | 30.43 | 0.1099 | 80.30 | 16.15 |
| `Batch 5` | 197 | 45.69 | 54.31 | 12.69 | **38.37** | 53.81 | 3.05 | 46.07 | 52.28 | 0.2773 | 0.00 | 45.69 | 54.31 | 0.2679 | **72.88** | **24.37** |
| `Batch 6` | 2,300 | 50.96 | 49.04 | 1.43 | 50.68 | 48.61 | 14.39 | 56.53 | 37.22 | 0.2143 | 7.83 | 55.24 | 41.26 | 0.1875 | **88.08** | **11.61** |
| `Batch 7` | 3,613 | 43.40 | 56.60 | 5.01 | 45.37 | 51.90 | 25.93 | 56.20 | 32.44 | 0.2562 | 1.83 | 44.07 | 54.91 | 0.2464 | **67.78** | **28.70** |
| `Batch 8` | 294 | 24.49 | 75.51 | 12.93 | 27.34 | 63.27 | 53.74 | 44.85 | 25.51 | 0.4949 | 4.76 | 25.71 | 70.75 | 0.4849 | **66.80** | **27.55** |
| `Batch 9` | 470 | 46.81 | 53.19 | 3.62 | 48.34 | 49.79 | 42.98 | 79.10 | 11.91 | 0.2218 | 2.13 | 47.83 | 51.06 | 0.2362 | **66.00** | **28.72** |
| `Batch 10` | 3,600 | 48.78 | 51.22 | 14.61 | 49.35 | 43.25 | 9.31 | 50.47 | 44.92 | 0.3665 | 0.72 | 48.94 | 50.69 | 0.3234 | 46.36 | 46.00 |

- **Empirical takeaway**: Under gradual multi-year MOX chemical baseline shift without rail saturation, the static 5-gate monitor abstains on only `1.43%–29.19%` of samples and does not improve accepted accuracy on `Batch 5` (`38.37%` vs `45.69%` naive) or `Batch 6` (`50.68%` vs `50.96%`). `MSP` cuts silent errors substantially further on `Batches 6–9` (`11.91%–37.22%`), while unlabeled Test-Time Batch Normalization + Conformal Risk Control (`TTN+CRC`) compensates baseline/gain drift across `Batches 4–9` (`66.00%–88.08%` accepted accuracy).

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
| A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks | ICLR 2017 | [arXiv:1610.02136](https://arxiv.org/abs/1610.02136) |
| Energy-based Out-of-distribution Detection | NeurIPS 2020 | [arXiv:2010.03759](https://arxiv.org/abs/2010.03759) |
| ReAct: Out-of-distribution Detection With Rectified Activations | NeurIPS 2021 | [arXiv:2111.12797](https://arxiv.org/abs/2111.12797) |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | ICLR 2023 | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | ICML 2023 | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| Conformal Risk Control | ICLR 2024 | [arXiv:2208.02814](https://arxiv.org/abs/2208.02814) |
| SpinQuant: LLM Quantization with Learned Rotations | ICLR 2025 | [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) |
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

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`, Fail-Closed `#35`)

[`autolab_wire.py`](autolab_wire.py) implements the fail-closed Layer `L0` edge runtime node (`REPO_ID = "lab-feet"`, `LAYER = "L0"`):

- **Fail-closed verification (`process_packet`)**:
  - Short-circuits unchanged when `packet["halted"]` is `True`, and blocks tainted upstream checkpoints (`split_guard_clean is False`) with `L0_FEET_UPSTREAM_SPLITGUARD_BLOCK`.
  - Verifies SHA-256 digests (`checkpoint_sha256` and `calibration_sha256`), halting on mismatch with `L0_FEET_CHECKPOINT_SHA256_MISMATCH`.
  - Halts with **`L0_FEET_CALIBRATION_REQUIRED`** whenever a hot-loaded student checkpoint lacks a non-empty finite 2D calibration split `(calibration_X, calibration_y)` matching its input/output dimensions.
  - Clamps packet `max_acc_drop_permille` to `<= 15` (a packet may tighten the cap, never raise it) and halts with `L0_FEET_QUANT_DRIFT_BLOCK` if `quant_acc_drop_permille > max_acc_drop_permille` or nibble round-tripping fails.
  - Enforces a calibration accuracy floor `min_cal_acc_permille` (default `700`, clamped `>= 700`), halting constant or degenerate students with **`L0_FEET_ACCURACY_FLOOR_BLOCK`**.
  - Builds the unrelaxed calibrated 2-layer `DriftAbstainingInt4Classifier` (`220 B` monitor tables + scratch for `24` features, `6` classes) and halts with **`L0_FEET_MONITOR_UNAVAILABLE`** when the monitor cannot be constructed (`ood_monitor_active = False`, e.g., on 3+ layer students) unless `allow_unmonitored_student = True` is explicitly set and logged in `trace`. Drifted inputs halt with `L0_FEET_SAFE_HOLD_ABSTAIN`.
- **Emitted `edge_runtime` fields & provenance**:
  - Emits `quantization`, `nibble_roundtrip_exact`, `packed_bytes`, `quant_acc_drop_permille`, `max_acc_drop_permille`, `cal_fp32_acc_permille`, `cal_int4_acc_permille`, `min_cal_acc_permille`, `ood_monitor_active`, `allow_unmonitored_student`, `sensor_source` (`"packet"` vs `"derived_from_protocol_hash"`), `interlocks_source` (`"packet_passthrough"`), `monitor_regime`, `fp32_pred_class`, `int8_pred_class`, `int4_pred_class`, `fp32_int4_pred_agree`, `checkpoint_sha256`, `calibration_sha256`, `checkpoint_source`, and `safe_hold_abstained`.
  - **Provenance note**: When `sensor_features` is omitted from the packet, `sensor_source = "derived_from_protocol_hash"` deterministically hashes `protocol_sha256` and `manipulation` fields; these derived vectors and the `interlocks_source = "packet_passthrough"` flags (`dispense_error_permille`, `aspiration_clean`, `incubation_window_clean`, `seal_intact`) are **not physical benchtop measurements**.

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## Repository Layout

```text
lab-feet/
├── .github/workflows/
│   └── demo.yml                      # Linux (+ onnxruntime) / Windows / macOS CI
├── benchmarks/results/
│   └── real_datasets_2026.json       # Committed 10-seed digits + 10-dataset real benchmarks
├── community_extensions/
│   └── drift_abstaining_int4.py      # Int4/int8 OOD & clipping SAFE_HOLD monitor (220 B @ 24d)
├── demo/
│   ├── distill_quantize.py           # Benchmark runner (--seeds N, threshold gates, LFT1 export)
│   └── output.txt                    # Captured seed-0 + 10-seed + synthetic toy benchmark report
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical specification & mathematical invariants
├── lean/
│   └── LabFeetLean.lean              # Lean 4 nibble, saturating requant, SRAM & 5-gate OOD proofs
├── schema/
│   └── benchmark_report.schema.json  # Draft 2020-12 JSON Schema for benchmark JSON reports
├── scripts/
│   └── fetch_open_datasets.py        # SHA-256-verified dataset fetcher, --smoke-distill & --smoke-drift
├── src/lab_feet/
│   ├── __init__.py                   # Public package API
│   ├── benchmarks.py                 # Single-seed & N-seed digits + synthetic toy benchmarks
│   ├── cli.py                        # `lab-feet` CLI (benchmark, inspect-binary, export-c/onnx)
│   ├── datasets.py                   # 10-dataset SHA-256 registry, temporal splits & drift baselines
│   ├── distill.py                    # Zero-mean MSE & hybrid T^2 KL + CE / plain CE Adam trainer
│   ├── equivariant.py                # SO(3)-equivariant scalar irrep weight quantization & guards
│   ├── export_c.py                   # Freestanding zero-malloc C99 / CMSIS-NN `.rodata` exporter
│   ├── export_onnx.py                # ONNX opset >= 21 INT4/INT8 QDQ & MatMulInteger qlinear exporter
│   ├── frontier.py                   # WHT+GPTQ W4A8, 39.1k SpectralResNet1D, MSP/Energy/ReAct/TTN+CRC
│   ├── quantize.py                   # Symmetric/asymmetric per-tensor/channel int8/4-bit & W8A8/W4A8
│   ├── runtime.py                    # Cross-platform cpu_model, BLAS pinning, forward & percentiles
│   └── serialize.py                  # Flat little-endian LFT1 binary serializer & verifier
├── tests/
│   ├── test_autolab_wire.py          # Fail-closed autolab-wire/v1 L0 packet & provenance tests (#35)
│   ├── test_community_extensions.py  # Int4 OOD monitor SRAM, logits & Lean 5-gate parity tests
│   ├── test_distill_quantize.py      # Offline pytest suite covering issues #1..#37 & ONNX Runtime
│   └── test_open_datasets_2026.py    # 10-dataset parsers, SHA-256 gate, temporal split & drift tests
├── autolab_wire.py                   # Fail-closed L0 node for the 15-repo autolab-wire/v1 pipeline
├── LICENSE                           # MIT License
├── pyproject.toml                    # Package build metadata (`dev` & `edge` extras)
├── README.md                         # Overview, 10-seed & real-dataset benchmarks, and wire contract
└── requirements.txt                  # Pinned minimum numpy>=1.26 and scikit-learn>=1.4
```

## Contributing

Found a broken link or a missing dataset, repo, or paper? Open an issue or a pull request.

## License

Code in this repo: MIT ([`LICENSE`](LICENSE)). Listed datasets, repos, and papers keep their own licenses.
