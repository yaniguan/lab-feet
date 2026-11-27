"""Command-line interface (`lab-feet`) for distillation benchmarks, binary inspection, and schema export."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from .benchmarks import (  # noqa: E402
    run_autoencoder_compression_benchmark,
    run_digits_benchmark,
    run_digits_multiseed_benchmark,
    run_wetlab_acoustic_benchmark,
)
from .export_c import export_c99_header  # noqa: E402
from .export_onnx import export_quantized_onnx  # noqa: E402
from .serialize import compute_binary_header_bytes, load_quantized_binary, save_quantized_binary  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def format_digits_report(digits: dict[str, Any]) -> str:
    sp = digits["splits"]
    tc = digits["timing_config"]
    bver = digits["binary_container_verification"]
    lines = [
        f"CPU: {digits['cpu']} (seed {digits['seed']})",
        f"train={sp['train']} val={sp['val']} test={sp['test']} features={sp['features']}",
        f"teacher soft probs softmax(logits/T): mean max prob = {digits['teacher_mean_max_soft_prob']:.4f}",
        f"distillation temperature T={digits['temperature']} (hybrid KL T^2 weight={1.0 - digits['alpha']:.2f}, CE alpha={digits['alpha']:.2f})",
        f"latency: single-thread BLAS batch-1 passes (warmup={tc['n_warmup']}, timed={tc['n_timing']})",
        "",
    ]
    hdr = (
        f"{'model':<47}{'val_acc':>8}{'test_acc':>9}{'params':>8}"
        f"{'weight_bytes':>13}{'med_us':>8}{'p95_us':>8}{'p99_us':>8}{'iqr_us':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for row in digits["models"]:
        lines.append(
            f"{row['name']:<47}{row['val_acc']:>8.4f}{row['test_acc']:>9.4f}"
            f"{row['params']:>8d}{row['weight_bytes']:>13d}"
            f"{row['median_us']:>8.2f}{row['p95_us']:>8.2f}"
            f"{row['p99_us']:>8.2f}{row['iqr_us']:>8.2f}"
        )

    lines.append("")
    by_name = {r["name"]: r for r in digits["models"]}
    ag8 = by_name["student KL int8 pt (dequantized fp32 timing)"]["agreement_with_kl_fp32"]
    ag_w8a8 = by_name["student KL int8 W8A8 (int32)"]["agreement_with_kl_fp32"]
    ag4_w4a8 = by_name["student KL 4-bit W4A8 (int32)"]["agreement_with_kl_fp32"]
    ag4_wht = by_name["student KL 4-bit WHT+GPTQ W4A8 (int32)"]["agreement_with_kl_fp32"]
    lines.append(
        f"prediction agreement with KL fp32 student: int8-deq={ag8:.4f} W8A8={ag_w8a8:.4f} "
        f"4bit-W4A8={ag4_w4a8:.4f} 4bit-WHT+GPTQ={ag4_wht:.4f}"
    )
    lines.append(
        f"LFT1 binary verification: payload={bver['payload_weight_bytes']} B + "
        f"header={bver['header_bytes']} B = {bver['total_blob_bytes']} B total"
    )
    t_tr = by_name["teacher (256,256) fp32"]["training"]
    h_tr = by_name["sklearn MLPClassifier hard-label"]["training"]
    c_tr = by_name["student same-trainer CE fp32"]["training"]
    k_tr = by_name["student hybrid KL(T^2)+CE fp32"]["training"]
    lines.append(
        f"convergence: teacher(iter={t_tr['n_iter']}, loss={t_tr['final_loss']:.4f}, ok={t_tr['converged']}) | "
        f"sklearn-hard(iter={h_tr['n_iter']}, loss={h_tr['final_loss']:.4f}, ok={h_tr['converged']}) | "
        f"same-trainer-CE(epochs={c_tr['n_iter']}, loss={c_tr['final_loss']:.4f}, ok={c_tr['converged']}) | "
        f"KL+CE(epochs={k_tr['n_iter']}, loss={k_tr['final_loss']:.4f}, ok={k_tr['converged']})"
    )
    return "\n".join(lines)


def format_multiseed_digits_report(ms: dict[str, Any]) -> str:
    cmp = ms["kl_vs_same_trainer_ce"]
    lines = [
        "",
        f"=== Multi-Seed Digits Comparison ({ms['n_seeds']} split-and-init seeds 0..{ms['n_seeds'] - 1}, "
        f"T={ms['temperature']}, alpha={ms['alpha']}) ===",
        f"{'model':<42}{'val_acc (mean +- std)':>24}{'test_acc (mean +- std)':>25}{'range':>17}",
        "-" * 108,
    ]
    for row in ms["models"]:
        val_str = f"{row['mean_val_acc']:.4f} +- {row['std_val_acc']:.4f}"
        te_str = f"{row['mean_test_acc']:.4f} +- {row['std_test_acc']:.4f}"
        rng_str = f"[{row['min_test_acc']:.4f}, {row['max_test_acc']:.4f}]"
        lines.append(f"{row['name']:<42}{val_str:>24}{te_str:>25}{rng_str:>17}")
    lines.append(
        f"KL+CE vs same-trainer CE (win / tie / loss over {ms['n_seeds']} seeds): "
        f"{cmp['wins']} / {cmp['ties']} / {cmp['losses']} "
        f"(mean test_acc diff = {cmp['mean_test_acc_diff']:+.4f})"
    )
    return "\n".join(lines)


def format_wetlab_and_ae_report(wetlab: dict[str, Any], ae: dict[str, Any]) -> str:
    lines = [
        "",
        "=== Synthetic Wet-Lab Toy Regime Distillation (6 classes, 48 i.i.d. Gaussian features) ===",
    ]
    for row in wetlab["models"]:
        lines.append(
            f"  {row['name']:<52} val={row['val_acc']:.4f} test={row['test_acc']:.4f} "
            f"bytes={row['weight_bytes']:>6d} med_us={row['median_us']:>6.2f}"
        )
    lines.extend(
        [
            "",
            f"=== Synthetic lab-ears 8-D Subspace Autoencoder Compression ({ae['architecture']}) ===",
            f"  params={ae['params']} fp32={ae['fp32_bytes']} B -> int8={ae['int8_bytes']} B -> "
            f"4-bit={ae['int4_bytes']} B ({ae['compression_ratio_4bit']:.2f}x compression) "
            f"4-bit anomaly AUROC={ae['int4_anomaly_auroc']:.4f}",
        ]
    )
    return "\n".join(lines)


def check_thresholds(digits: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    by_name = {r["name"]: r for r in digits["models"]}
    int8_acc = by_name["student KL int8 pt (dequantized fp32 timing)"]["test_acc"]
    w8a8_acc = by_name["student KL int8 W8A8 (int32)"]["test_acc"]
    int4_agree = by_name["student KL 4-bit pc+MSE (dequantized fp32 timing)"]["agreement_with_kl_fp32"]
    if int8_acc < 0.95:
        failures.append(f"int8 test_acc {int8_acc:.4f} < 0.95 threshold")
    if w8a8_acc < 0.95:
        failures.append(f"W8A8 int32 test_acc {w8a8_acc:.4f} < 0.95 threshold")
    if (int4_agree or 0.0) < 0.97:
        failures.append(f"4-bit agreement {int4_agree} < 0.97 threshold")
    return failures


def _cmd_benchmark(args: argparse.Namespace) -> int:
    digits_raw = run_digits_benchmark(
        seed=args.seed,
        n_warmup=args.n_warmup,
        n_timing=args.n_timing,
    )
    artifacts = digits_raw.pop("_artifacts")
    print(format_digits_report(digits_raw))

    multiseed_report = None
    if args.seeds is not None:
        multiseed_report = run_digits_multiseed_benchmark(n_seeds=args.seeds)
        print(format_multiseed_digits_report(multiseed_report))

    wetlab = None
    ae = None
    if not args.digits_only:
        wetlab = run_wetlab_acoustic_benchmark(seed=args.seed)
        ae = run_autoencoder_compression_benchmark(seed=args.seed)
        print(format_wetlab_and_ae_report(wetlab, ae))

    if args.export_bin:
        bin_path = save_quantized_binary(artifacts["q_int4_mse"], args.export_bin)
        print(f"\nExported LFT1 4-bit model to {bin_path} ({bin_path.stat().st_size} bytes)")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "digits": digits_raw,
            "wetlab": wetlab,
            "autoencoder": ae,
        }
        if multiseed_report is not None:
            payload["multiseed_digits"] = multiseed_report
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON benchmark report to {out}")

    if args.assert_thresholds:
        errs = check_thresholds(digits_raw)
        if errs:
            for e in errs:
                print(f"THRESHOLD FAILURE: {e}", file=sys.stderr)
            return 1
    return 0


def _cmd_inspect_binary(args: argparse.Namespace) -> int:
    p = Path(args.path)
    model = load_quantized_binary(p)
    hdr = compute_binary_header_bytes(len(model.layers))
    summary = {
        "path": str(p),
        "file_bytes": p.stat().st_size,
        "header_bytes": hdr,
        "weight_bytes": model.nbytes,
        "n_params": model.n_params,
        "activation": model.activation,
        "layers": [
            {
                "shape": list(l.shape),
                "bits": l.bits,
                "per_channel": l.per_channel,
                "mode": l.mode,
                "packed_dtype": str(l.packed_weights.dtype),
                "packed_bytes": int(l.packed_weights.nbytes),
                "scale_bytes": int(l.scale.nbytes),
                "bias_bytes": int(l.bias.nbytes),
                "layer_nbytes": l.nbytes,
            }
            for l in model.layers
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_export_schema(args: argparse.Namespace) -> int:
    schema_file = REPO_ROOT / "schema" / "benchmark_report.schema.json"
    text = schema_file.read_text(encoding="utf-8")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_export_c_header(args: argparse.Namespace) -> int:
    model = load_quantized_binary(args.input)
    export_c99_header(
        model,
        args.output,
        prefix=args.prefix,
        input_scale=args.input_scale,
    )
    return 0


def _cmd_export_onnx(args: argparse.Namespace) -> int:
    model = load_quantized_binary(args.input)
    export_quantized_onnx(
        model,
        args.output,
        opset=args.opset,
        operator_mode=args.mode,
        input_scale=args.input_scale,
        json_out=args.json_out,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab-feet",
        description="Edge distillation, int8/4-bit quantization, and LFT1/C99/ONNX runtime CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="Run distillation and quantization benchmarks")
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Optional number of split-and-init seeds N (e.g. 10) for multi-seed digits evaluation.",
    )
    p_bench.add_argument("--n-warmup", type=int, default=200)
    p_bench.add_argument("--n-timing", type=int, default=2000)
    p_bench.add_argument("--digits-only", action="store_true")
    p_bench.add_argument("--json-out", default=None)
    p_bench.add_argument("--export-bin", default=None)
    p_bench.add_argument("--assert-thresholds", action="store_true")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_ins = sub.add_parser("inspect-binary", help="Inspect an LFT1 quantized binary file")
    p_ins.add_argument("path", help="Path to .lft1 / .bin file")
    p_ins.set_defaults(func=_cmd_inspect_binary)

    p_sch = sub.add_parser("export-schema", help="Print JSON Schema for benchmark reports")
    p_sch.add_argument("--output", "-o", default=None)
    p_sch.set_defaults(func=_cmd_export_schema)

    p_chdr = sub.add_parser("export-c-header", help="Export LFT1 model to freestanding C99 .h header")
    p_chdr.add_argument("input", help="Input .lft1 binary model path")
    p_chdr.add_argument("--output", "-o", required=True, help="Output .h header path")
    p_chdr.add_argument("--prefix", default="lab_feet_model", help="C symbol prefix")
    p_chdr.add_argument("--input-scale", type=float, default=None, help="Optional int8 ADC input scale")
    p_chdr.set_defaults(func=_cmd_export_c_header)

    p_onnx = sub.add_parser("export-onnx", help="Export LFT1 model to ONNX opset >= 21 .onnx graph")
    p_onnx.add_argument("input", help="Input .lft1 binary model path")
    p_onnx.add_argument("--output", "-o", required=True, help="Output .onnx file path")
    p_onnx.add_argument("--opset", type=int, default=21, help="ONNX opset version (>= 21)")
    p_onnx.add_argument("--mode", choices=["qdq", "qlinear"], default="qdq", help="ONNX quantization operator mode")
    p_onnx.add_argument("--input-scale", type=float, default=None, help="Optional static input scale for qlinear")
    p_onnx.add_argument("--json-out", default=None, help="Optional JSON graph dump path")
    p_onnx.set_defaults(func=_cmd_export_onnx)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
