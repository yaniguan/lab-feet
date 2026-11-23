# lab-feet

Let the lab agent *walk on its own feet*: big models teach small ones, small ones get quantized, and they run on a cheap board next to the bench.

## Where it fits

In the lab agent on [yaniguan.github.io](https://yaniguan.github.io/), the feet are planned: cloud GPUs only in short bursts, and small quantized models on the edge. This repo collects distillation and quantization tools and edge runtimes, The first attempt is one small CPU-only script: a big model teaches a small one, then the small one is quantized to int8 and 4-bit.

## First attempt

scikit-learn's bundled digits (8x8 images, 10 classes). Teacher MLP (256,256); Student MLP (32) trained two ways: on hard labels, or distilled by regressing the teacher's logits divided by T=4. The distilled student is then quantized per tensor to int8 and 4-bit. Batch-1 latency is the median of 2000 single-sample numpy forward passes on one CPU.

```bash
pip install numpy scikit-learn
python demo/distill_quantize.py
```

Output (Linux, CPU only, 2026-11-18):

```
CPU: Intel(R) Xeon(R) Processor
train=1257 test=540 features=64
teacher soft probs softmax(logits/T): mean max prob = 0.7447
distillation temperature T=4.0
latency: median of 2000 batch-1 numpy forward passes after 200 warmup (quantized models run dequantized fp32 weights)

model                      test_acc   params  weight_bytes  latency_us
----------------------------------------------------------------------
teacher (256,256) fp32       0.9778    85002        340008        9.51
student hard-label fp32      0.9741     2410          9640        3.82
student distilled fp32       0.9685     2410          9640        3.83
student distilled int8       0.9685     2410          2544        3.86
student distilled 4-bit      0.9648     2410          1360        3.86

prediction agreement with fp32 distilled student: int8=1.0000 4-bit=0.9889
weight_bytes for quantized models = packed weights + fp32 biases + one fp32 scale per tensor
```

The 32-unit hard-label student keeps almost all of the teacher's accuracy (0.974 vs 0.978) with 35x fewer parameters and runs about 2.5x faster. Distillation was slightly *worse* than plain hard-label training on this easy task (0.969 vs 0.974). int8 cost no accuracy and 4-bit cost 0.4 points, but in numpy the weights are dequantized before the matmul, so quantization here saves storage, not time. Latency changes a little from run to run. Next: a real int8 runtime, a small language or force-field student, and timing on a real edge board.

## Datasets

| Dataset | What | License |
|---|---|---|
| [scikit-learn digits](https://scikit-learn.org/stable/datasets/toy_dataset.html) | 1797 8x8 digit images (bundled) | BSD-3 (bundled with scikit-learn) |

## Open-source repos

Stars as of 2026-11-18.

| Repo | What | Stars | License |
|---|---|---:|---|
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

- **Object**: Edge compute substrate that distills large teacher networks into compact student models and quantizes weights to `int8` and `4-bit` for low-cost, low-latency benchtop inference.
- **Data**: `scikit-learn` digits (`1,797` `8x8` grayscale images, `64` features, stratified `1,257` train / `540` test split, `StandardScaler` fit on train), serialized into `fp32`, `int8`, and nibble-packed signed `4-bit` weight buffers.
- **Model**: `(256, 256)` ReLU MLP teacher (`85,002` params, `340,008 B`) -> `(32,)` ReLU MLP students (`2,410` params, `9,640 B` fp32 -> `2,544 B` `int8` -> `1,360 B` packed `4-bit`, `250x` smaller than teacher).
- **Training**: High-temperature scaled logit regression (`logits / T` at `T = 4.0`) vs. hard-label cross-entropy (`0.9685` vs. `0.9741` test accuracy, honestly documenting both); `int8` retains `100%` prediction agreement (`0.9685`) and `4-bit` retains `98.89%` (`0.9648`), running dequantized `fp32` `numpy` matmuls (`3.82–3.86 us` vs. `9.51 us` teacher).
- **History**: `a7b1532` (2026-11-18, `main` teacher-student distillation + `int8`/`4-bit` bit-packing pipeline, CPU latency benchmark, CI workflow, and edge-runtime catalog).
- **Unique Properties**: Verified two's-complement nibble round-trip (`pack_4bit`/`unpack_4bit`), shared `numpy` `batch=1` latency harness, and foundational `L0` runtime role for [`lab-ears`](https://github.com/yaniguan/lab-ears), [`lab-hands`](https://github.com/yaniguan/lab-hands), and [`YLM`](https://github.com/yaniguan/YLM).

Full technical specification: [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

## Repository Layout

```text
lab-feet/
├── .github/workflows/
│   └── demo.yml                  # CI workflow running teacher-student distillation & quantization
├── demo/
│   ├── distill_quantize.py       # Logit distillation (T=4), int8/4-bit packing, and latency timer
│   └── output.txt                # Reference Xeon CPU accuracy, byte-size, and latency table
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
├── LICENSE                       # MIT License
├── README.md                     # Overview, distillation results, and edge-runtime resources
└── requirements.txt              # numpy and scikit-learn
```

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
