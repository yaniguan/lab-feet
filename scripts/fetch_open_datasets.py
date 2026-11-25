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
        )
        loaded = load_open_edge_dataset(
            ds_id,
            data_dir=data_dir,
            archive_path=archive_path,
            require_commercial_use=require_comm,
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
                require_commercial_use=require_comm,
            )
        if args.smoke_drift and loaded.spec.has_temporal_drift_batches:
            manifest["drift_abstaining_int4"] = evaluate_temporal_drift_abstaining_int4(
                loaded,
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
                f"classes={m['n_classes']} sha256={m['raw_sha256'][:16]}..."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
