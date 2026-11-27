"""On-demand fetcher, SPDX license verifier, and live URL checker for lab-feet open edge datasets.

Downloads external archives strictly on-demand into gitignored `data/<dataset_id>/`, computes
SHA-256 digests, writes deterministic `manifest.json` receipts, and optionally runs smoke
`distill_quantize` or multi-batch `DriftAbstainingInt4Classifier` evaluations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

for _blas_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(SRC_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lab_feet.datasets import (  # noqa: E402
    DEFAULT_DATA_DIR,
    OPEN_EDGE_DATASETS,
    OpenEdgeDataset,
    evaluate_temporal_drift_abstaining_int4,
    fetch_open_edge_dataset_archive,
    get_open_edge_dataset,
    list_open_edge_datasets,
    load_open_edge_dataset,
    run_open_dataset_distill_quantize,
    verify_dataset_license,
)


def probe_dataset_url(spec: OpenEdgeDataset, *, timeout_sec: float = 20.0) -> dict[str, Any]:
    """Verify live HTTPS reachability of `spec.url` via HEAD with bounded Range GET fallback."""
    headers = {"User-Agent": "lab-feet-open-datasets/0.2.0"}
    try:
        req = urllib.request.Request(spec.url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return {
                "dataset_id": spec.dataset_id,
                "url": spec.url,
                "ok": 200 <= resp.status < 400,
                "status": int(resp.status),
            }
    except Exception:
        req_get = urllib.request.Request(
            spec.url,
            method="GET",
            headers={**headers, "Range": "bytes=0-63"},
        )
        with urllib.request.urlopen(req_get, timeout=timeout_sec) as resp:
            _ = resp.read(64)
            return {
                "dataset_id": spec.dataset_id,
                "url": spec.url,
                "ok": 200 <= resp.status < 400,
                "status": int(resp.status),
            }


def format_registry_table(specs: Sequence[OpenEdgeDataset]) -> str:
    hdr = (
        f"{'dataset_id':<27}{'feats':>7}{'classes':>9}{'drift':>7}"
        f"{'spdx_license':>14}{'comm_ok':>9}  {'modality'}"
    )
    lines = [hdr, "-" * len(hdr)]
    for s in specs:
        lines.append(
            f"{s.dataset_id:<27}{s.n_features:>7d}{s.n_classes:>9d}"
            f"{str(s.has_temporal_drift_batches):>7}{s.spdx_license:>14}"
            f"{str(s.commercial_use_ok):>9}  {s.modality}"
        )
    return "\n".join(lines)


def format_distill_comparison(
    random_rep: dict[str, Any],
    temporal_rep: dict[str, Any] | None = None,
) -> str:
    lines = ["  [Distill & Quantize]"]
    reps: list[tuple[str, dict[str, Any]]] = [("random", random_rep)]
    if temporal_rep is not None:
        reps.append(("temporal", temporal_rep))
    for mode_label, r in reps:
        ce_acc = r["student_ce_fp32"]["test_acc"] if r.get("student_ce_fp32") else float("nan")
        wht_acc = (
            r["student_int4_wht_gptq"]["w4a8_test_acc"]
            if r.get("student_int4_wht_gptq")
            else float("nan")
        )
        lines.append(
            f"    split={mode_label:<8} (tr={r['splits']['train']}/val={r['splits']['val']}/te={r['splits']['test']}): "
            f"teacher={r['teacher_fp32']['test_acc']:.4f} | "
            f"CE={ce_acc:.4f} | "
            f"KL+CE={r['student_kl_fp32']['test_acc']:.4f} | "
            f"W8A8={r['student_int8_w8a8']['test_acc']:.4f} | "
            f"4bit-MSE(W4A8)={r['student_int4_pc_mse']['w4a8_test_acc']:.4f} | "
            f"4bit-WHT+GPTQ(W4A8)={wht_acc:.4f}"
        )
    return "\n".join(lines)


def format_drift_report(drift: dict[str, Any]) -> str:
    cal = drift["calibration_abstain_rates_pct"]
    lines = [
        f"  [Temporal Drift Abstention: source_batches={drift['source_batches']}, "
        f"features={drift['n_selected_features']}, monitor_sram={drift['extra_sram_bytes']} B, "
        f"cal_abstain={cal['five_gate_safe_hold']:.2f}% (MSP={cal['msp_baseline']:.2f}%, Energy={cal['energy_baseline']:.2f}%)]",
    ]
    if drift.get("sram_warning"):
        lines.append(f"  [WARNING] {drift['sram_warning']}")

    hdr = (
        f"    {'Batch':<8}{'n':>6}  {'NaiveAcc%':>9}{'NaiveErr%':>10}  "
        f"{'5G-Hold%':>9}{'5G-Acc%':>8}{'5G-Err%':>8}  "
        f"{'MSP-Hold%':>10}{'MSP-Acc%':>9}{'MSP-Err%':>9}{'MSP-AURC':>9}  "
        f"{'En-Hold%':>9}{'En-Acc%':>8}{'En-Err%':>8}{'En-AURC':>8}  "
        f"{'TTN+CRC-Acc%':>13}{'TTN+CRC-Err%':>13}"
    )
    lines.append(hdr)
    lines.append("    " + "-" * (len(hdr) - 4))
    all_rows = [drift["source_in_domain"], *drift["drift_batches"]]
    for row in all_rows:
        tag = "src_id" if row["batch_tag"] == "source_in_domain_holdout" else f"b{row['batch_id']}"
        msp = row["msp_baseline"]
        eng = row["energy_baseline"]
        ttn = row["ttn_crc_frontier"]
        lines.append(
            f"    {tag:<8}{row['n_samples']:>6d}  "
            f"{row['naive_int4_accuracy_pct']:>9.2f}{row['naive_silent_error_rate_pct']:>10.2f}  "
            f"{row['guarded_safe_hold_rate_pct']:>9.2f}{row['guarded_accepted_accuracy_pct']:>8.2f}{row['guarded_silent_error_rate_pct']:>8.2f}  "
            f"{msp['safe_hold_rate_pct']:>10.2f}{msp['accepted_accuracy_pct']:>9.2f}{msp['silent_error_rate_pct']:>9.2f}{msp['aurc']:>9.4f}  "
            f"{eng['safe_hold_rate_pct']:>9.2f}{eng['accepted_accuracy_pct']:>8.2f}{eng['silent_error_rate_pct']:>8.2f}{eng['aurc']:>8.4f}  "
            f"{ttn['accepted_accuracy_pct']:>13.2f}{ttn['silent_error_rate_pct']:>13.2f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and verify open-source edge sensor, vibration, gas-drift, and spectral datasets."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered OpenEdgeDataset entries and exit.",
    )
    parser.add_argument(
        "--verify-urls",
        action="store_true",
        help="Verify live HTTPS reachability of registered dataset URLs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help="Dataset ID to fetch/evaluate (repeatable).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all registered datasets into --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Target gitignored directory for downloaded archives (default: data/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives even if already cached.",
    )
    parser.add_argument(
        "--allow-non-commercial",
        action="store_true",
        help="Disable strict commercial-use SPDX license gate.",
    )
    parser.add_argument(
        "--allow-unverified-sha256",
        action="store_true",
        help="Allow loading archives whose SHA-256 or byte size differs from pinned registry values.",
    )
    parser.add_argument(
        "--max-monitor-features",
        type=int,
        default=None,
        help="Optional F-ratio feature cap for DriftAbstainingInt4Classifier (e.g. 32 -> 276 B < 512 B).",
    )
    parser.add_argument(
        "--smoke-distill",
        action="store_true",
        help="Run KL(T^2)+CE distillation + int8/4-bit quantization on fetched dataset(s).",
    )
    parser.add_argument(
        "--smoke-drift",
        action="store_true",
        help="Run multi-batch DriftAbstainingInt4Classifier evaluation on drift-capable dataset(s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_comm = not args.allow_non_commercial

    selected_ids: list[str]
    if args.all:
        selected_ids = list(OPEN_EDGE_DATASETS.keys())
    elif args.datasets:
        selected_ids = list(args.datasets)
    else:
        selected_ids = []

    if args.list and not args.verify_urls and not selected_ids:
        specs = list_open_edge_datasets(commercial_only=False)
        if args.json:
            print(json.dumps([s.to_dict() for s in specs], indent=2))
        else:
            print(format_registry_table(specs))
        return 0

    if args.verify_urls:
        probe_ids = selected_ids or list(OPEN_EDGE_DATASETS.keys())
        results = []
        for ds_id in probe_ids:
            spec = get_open_edge_dataset(ds_id)
            verify_dataset_license(spec, require_commercial_use=require_comm)
            results.append(probe_dataset_url(spec))
        if args.json:
            print(json.dumps({"url_checks": results}, indent=2))
        else:
            for r in results:
                print(f"[{'OK' if r['ok'] else 'FAIL'} {r['status']}] {r['dataset_id']}: {r['url']}")
        if not all(r["ok"] for r in results):
            return 1
        if not selected_ids or (not args.all and not args.datasets):
            return 0

    if not selected_ids:
        print(format_registry_table(list_open_edge_datasets()))
        return 0

    manifests: list[dict[str, Any]] = []
    data_dir = Path(args.data_dir)
    for ds_id in selected_ids:
        archive_path = fetch_open_edge_dataset_archive(
            ds_id,
            data_dir=data_dir,
            force=args.force,
            require_commercial_use=require_comm,
            allow_unverified_sha256=args.allow_unverified_sha256,
        )
        loaded = load_open_edge_dataset(
            ds_id,
            data_dir=data_dir,
            archive_path=archive_path,
            require_commercial_use=require_comm,
            allow_unverified_sha256=args.allow_unverified_sha256,
        )
        manifest: dict[str, Any] = {
            "dataset_id": loaded.spec.dataset_id,
            "title": loaded.spec.title,
            "url": loaded.spec.url,
            "spdx_license": loaded.spec.spdx_license,
            "commercial_use_ok": loaded.spec.commercial_use_ok,
            "archive_path": str(archive_path),
            "archive_bytes": int(archive_path.stat().st_size),
            "raw_sha256": loaded.raw_sha256,
            "n_samples": int(loaded.X.shape[0]),
            "n_features": int(loaded.X.shape[1]),
            "n_classes": int(loaded.spec.n_classes),
            "has_temporal_drift_batches": loaded.spec.has_temporal_drift_batches,
            "distinct_batches": (
                sorted(int(b) for b in set(loaded.batch_ids.tolist()))
                if loaded.batch_ids is not None
                else []
            ),
        }
        if args.smoke_distill:
            manifest["distill_quantize"] = run_open_dataset_distill_quantize(
                loaded,
                split_mode="random",
                require_commercial_use=require_comm,
            )
            if loaded.spec.has_temporal_drift_batches and loaded.batch_ids is not None:
                manifest["distill_quantize_temporal"] = run_open_dataset_distill_quantize(
                    loaded,
                    split_mode="temporal",
                    require_commercial_use=require_comm,
                )
        if args.smoke_drift and loaded.spec.has_temporal_drift_batches:
            manifest["drift_abstaining_int4"] = evaluate_temporal_drift_abstaining_int4(
                loaded,
                max_monitor_features=args.max_monitor_features,
                require_commercial_use=require_comm,
            )

        manifest_path = data_dir / loaded.spec.dataset_id / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests.append(manifest)

    if args.json:
        print(json.dumps({"fetched": manifests}, indent=2))
    else:
        for m in manifests:
            print(
                f"Fetched {m['dataset_id']} ({m['spdx_license']}): "
                f"samples={m['n_samples']} features={m['n_features']} "
                f"classes={m['n_classes']} bytes={m['archive_bytes']} sha256={m['raw_sha256'][:16]}..."
            )
            if "distill_quantize" in m:
                print(
                    format_distill_comparison(
                        m["distill_quantize"],
                        m.get("distill_quantize_temporal"),
                    )
                )
            if "drift_abstaining_int4" in m:
                print(format_drift_report(m["drift_abstaining_int4"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
