"""Knowledge distillation and post-training weight quantization on sklearn digits & wet-lab tasks.

Pins single-thread BLAS prior to importing NumPy so batch-1 microsecond latencies are free
of OpenMP worker-thread wakeup overhead across Linux, macOS, and Windows.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

for _blas_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lab_feet import (  # noqa: E402
    LatencyStats,
    QuantizedLayer,
    QuantizedModel,
    accuracy,
    cpu_model,
    distill_kl_student,
    forward,
    forward_int8,
    get_layers,
    latency_stats,
    latency_us,
    n_params,
    pack_4bit,
    quantize,
    quantize_equivariant_irrep_weights,
    quantize_model,
    save_quantized_binary,
    softmax,
    unpack_4bit,
    zero_mean_logits,
)
from lab_feet.cli import main as cli_main  # noqa: E402

SEED = 0
T = 4.0
N_TIMING = 2000
N_WARMUP = 200


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run teacher-student distillation (KL T^2 + CE vs MSE) and int8/4-bit quantization."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-timing", type=int, default=N_TIMING)
    parser.add_argument("--digits-only", action="store_true", help="Run only the 8x8 digits benchmark.")
    parser.add_argument("--json-out", default=None, help="Optional JSON output report path.")
    parser.add_argument("--export-bin", default=None, help="Optional LFT1 binary model output path.")
    parser.add_argument(
        "--assert-thresholds",
        action="store_true",
        help="Assert int8/W8A8 test_acc >= 0.95 and 4-bit agreement >= 0.97.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cli_argv = [
        "benchmark",
        "--seed",
        str(args.seed),
        "--n-warmup",
        str(args.n_warmup),
        "--n-timing",
        str(args.n_timing),
    ]
    if args.digits_only:
        cli_argv.append("--digits-only")
    if args.json_out:
        cli_argv.extend(["--json-out", str(args.json_out)])
    if args.export_bin:
        cli_argv.extend(["--export-bin", str(args.export_bin)])
    if args.assert_thresholds:
        cli_argv.append("--assert-thresholds")
    return cli_main(cli_argv)


__all__ = [
    "N_TIMING",
    "N_WARMUP",
    "SEED",
    "T",
    "LatencyStats",
    "QuantizedLayer",
    "QuantizedModel",
    "accuracy",
    "cpu_model",
    "distill_kl_student",
    "forward",
    "forward_int8",
    "get_layers",
    "latency_stats",
    "latency_us",
    "main",
    "n_params",
    "pack_4bit",
    "quantize",
    "quantize_equivariant_irrep_weights",
    "quantize_model",
    "save_quantized_binary",
    "softmax",
    "unpack_4bit",
    "zero_mean_logits",
]


if __name__ == "__main__":
    sys.exit(main())
