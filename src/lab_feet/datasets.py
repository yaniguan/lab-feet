"""Open-source edge sensor, vibration, gas-sensor drift, and 1D spectral datasets for L0 benchmarking.

Provides:
- `OpenEdgeDataset`: Typed, SPDX-license-gated dataset specification registry covering compact 1D
  signals (`mnist1d`), 36-month chemical MOX sensor drift (`uci_gas_sensor_drift`), 6-axis inertial
  HAR (`uci_har_inertial`), rolling-element bearing vibration faults (`cwru_bearing_vibration`),
  hydraulic test-rig vibration (`uci_hydraulic_vibration`), 2026 bacterial SERS Raman spectra
  (`sers_bacteria_raman_2026`), and through-glass NIR spectroscopy (`ethanol_nir_spectroscopy`).
- `DeterministicFeatureNormalizer`: Leakage-free feature normalizer (`zscore`, `robust_iqr`,
  `minmax_symmetric`, `signed_log1p_zscore`, `snv_zscore`) fit strictly on training data with a
  canonical SHA-256 calibration state digest.
- On-demand downloaders into gitignored `data/`, format parsers, and direct evaluation bridges
  feeding `distill_kl_student` / `quantize_model` (`distill_quantize`) and multi-batch
  `DriftAbstainingInt4Classifier` (`SAFE_HOLD` OOD & saturation abstention).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import pickle
import struct
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .distill import distill_kl_student, fit_sklearn_mlp
from .quantize import QuantizedModel, forward_int8, quantize_model
from .runtime import accuracy, forward, fp32_bytes, get_layers, n_params
from .serialize import compute_binary_header_bytes, from_binary_buffer, to_binary_buffer

NormalizerKind = Literal[
    "zscore",
    "robust_iqr",
    "minmax_symmetric",
    "signed_log1p_zscore",
    "snv_zscore",
]

VALID_NORMALIZERS: tuple[str, ...] = (
    "zscore",
    "robust_iqr",
    "minmax_symmetric",
    "signed_log1p_zscore",
    "snv_zscore",
)

COMMERCIAL_SAFE_SPDX_LICENSES: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC0-1.0",
        "ISC",
        "MIT",
    }
)

_NON_COMMERCIAL_TOKENS: tuple[str, ...] = (
    "-NC",
    "NONCOMMERCIAL",
    "NON-COMMERCIAL",
    "ACADEMIC-ONLY",
    "EVAL-ONLY",
    "RESEARCH-ONLY",
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class DatasetLicenseError(ValueError):
    """Raised when an `OpenEdgeDataset` fails SPDX license validation or commercial-use gating."""


@dataclass(frozen=True)
class OpenEdgeDataset:
    """Typed registry descriptor for an open-source L0 edge sensor or 1D spectral dataset."""

    dataset_id: str
    n_features: int
    n_classes: int
    has_temporal_drift_batches: bool
    url: str
    spdx_license: str
    commercial_use_ok: bool
    title: str = ""
    modality: str = "edge_sensor"
    default_normalizer: NormalizerKind = "zscore"
    landing_url: str = ""
    archive_filename: str = ""
    citation: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty identifier.")
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}.")
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {self.n_classes}.")
        if not self.url.startswith("https://"):
            raise ValueError(f"Dataset url must use HTTPS, got {self.url!r}.")
        if not self.spdx_license or not self.spdx_license.strip():
            raise DatasetLicenseError(f"Dataset {self.dataset_id!r} has an empty spdx_license.")
        if self.default_normalizer not in VALID_NORMALIZERS:
            raise ValueError(
                f"Unsupported default_normalizer {self.default_normalizer!r}; "
                f"expected one of {VALID_NORMALIZERS}."
            )
        upper_lic = self.spdx_license.upper()
        if self.commercial_use_ok and any(tok in upper_lic for tok in _NON_COMMERCIAL_TOKENS):
            raise DatasetLicenseError(
                f"Dataset {self.dataset_id!r} claims commercial_use_ok=True under "
                f"non-commercial license {self.spdx_license!r}."
            )

    def resolved_archive_filename(self) -> str:
        if self.archive_filename:
            return self.archive_filename
        tail = self.url.rstrip("/").split("/")[-1]
        return tail.replace("+", "_") or f"{self.dataset_id}.dat"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_dataset_license(
    spec: OpenEdgeDataset,
    *,
    require_commercial_use: bool = True,
) -> None:
    """Enforce SPDX license hygiene before training or exporting L0 edge firmware artifacts."""
    lic = spec.spdx_license.strip()
    if not lic or lic.upper() in {"UNKNOWN", "NONE", "CUSTOM", "PROPRIETARY"}:
        raise DatasetLicenseError(
            f"Dataset {spec.dataset_id!r} carries unverified SPDX license {spec.spdx_license!r}."
        )
    upper_lic = lic.upper()
    if spec.commercial_use_ok and any(tok in upper_lic for tok in _NON_COMMERCIAL_TOKENS):
        raise DatasetLicenseError(
            f"Dataset {spec.dataset_id!r} has contradictory commercial_use_ok=True for {lic!r}."
        )
    if require_commercial_use:
        if not spec.commercial_use_ok:
            raise DatasetLicenseError(
                f"Dataset {spec.dataset_id!r} ({lic}) prohibits commercial L0 firmware deployment."
            )
        if lic not in COMMERCIAL_SAFE_SPDX_LICENSES:
            raise DatasetLicenseError(
                f"Dataset {spec.dataset_id!r} license {lic!r} is not in allowlist "
                f"{sorted(COMMERCIAL_SAFE_SPDX_LICENSES)}."
            )


OPEN_EDGE_DATASETS: dict[str, OpenEdgeDataset] = {
    "mnist1d": OpenEdgeDataset(
        dataset_id="mnist1d",
        n_features=40,
        n_classes=10,
        has_temporal_drift_batches=False,
        url="https://raw.githubusercontent.com/greydanus/mnist1d/master/mnist1d_data.pkl",
        spdx_license="Apache-2.0",
        commercial_use_ok=True,
        title="MNIST-1D Compact 40-Point Signal Classification",
        modality="1d_signal",
        default_normalizer="zscore",
        landing_url="https://github.com/greydanus/mnist1d",
        archive_filename="mnist1d_data.pkl",
        citation="Greydanus & Kobak (ICML 2024), Scaling Down Deep Learning with MNIST-1D.",
    ),
    "uci_gas_sensor_drift": OpenEdgeDataset(
        dataset_id="uci_gas_sensor_drift",
        n_features=128,
        n_classes=6,
        has_temporal_drift_batches=True,
        url="https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="UCI Gas Sensor Array Drift Dataset (16 MOX Sensors, 36 Months, 10 Batches)",
        modality="chemical_gas_sensor_drift",
        default_normalizer="signed_log1p_zscore",
        landing_url="https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
        archive_filename="gas_sensor_array_drift_dataset.zip",
        citation="Vergara et al. (2012), Chemical gas sensor drift compensation using classifier ensembles, DOI:10.24432/C5RP6W.",
    ),
    "uci_har_inertial": OpenEdgeDataset(
        dataset_id="uci_har_inertial",
        n_features=561,
        n_classes=6,
        has_temporal_drift_batches=False,
        url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="UCI Inertial Accelerometer & Gyroscope 50 Hz Window Features (HAR)",
        modality="inertial_imu",
        default_normalizer="zscore",
        landing_url="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        archive_filename="human_activity_recognition_using_smartphones.zip",
        citation="Anguita et al. (ESANN 2013), Human Activity Recognition Using Smartphones, DOI:10.24432/C54S4K.",
    ),
    "cwru_bearing_vibration": OpenEdgeDataset(
        dataset_id="cwru_bearing_vibration",
        n_features=17,
        n_classes=4,
        has_temporal_drift_batches=True,
        url="https://huggingface.co/datasets/odysseywt/CWRU/resolve/main/CWRU.csv",
        spdx_license="MIT",
        commercial_use_ok=True,
        title="CWRU Rolling-Element Bearing Vibration Fault Features Across Spall Severities",
        modality="vibration_fault",
        default_normalizer="robust_iqr",
        landing_url="https://huggingface.co/datasets/odysseywt/CWRU",
        archive_filename="CWRU.csv",
        citation="Case Western Reserve University Bearing Data Center / odysseywt CWRU 17-feature benchmark.",
    ),
    "uci_hydraulic_vibration": OpenEdgeDataset(
        dataset_id="uci_hydraulic_vibration",
        n_features=60,
        n_classes=4,
        has_temporal_drift_batches=True,
        url="https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="UCI Hydraulic Test-Rig Mechanical Vibration (VS1) Valve Degradation & Thermal Drift",
        modality="vibration_fault",
        default_normalizer="zscore",
        landing_url="https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
        archive_filename="condition_monitoring_of_hydraulic_systems.zip",
        citation="Helwig, Pignanelli & Schutze (I2MTC 2015), Condition Monitoring of Hydraulic Systems, DOI:10.24432/C5CW21.",
    ),
    "sers_bacteria_raman_2026": OpenEdgeDataset(
        dataset_id="sers_bacteria_raman_2026",
        n_features=1215,
        n_classes=9,
        has_temporal_drift_batches=True,
        url="https://raw.githubusercontent.com/erkankaratas-eng/sers-gna-bacteria/main/data/sers_spectra_preprocessed.csv",
        spdx_license="MIT",
        commercial_use_ok=True,
        title="SERS Gold-Nanorod Raman Spectra of Nine Clinical Bacterial Strains (2026)",
        modality="raman_spectroscopy",
        default_normalizer="snv_zscore",
        landing_url="https://github.com/erkankaratas-eng/sers-gna-bacteria",
        archive_filename="sers_spectra_preprocessed.csv",
        citation="Gulmus et al. (2026), SERS spectra of nine bacterial strains on an anisotropic gold nanorod array.",
    ),
    "ethanol_nir_spectroscopy": OpenEdgeDataset(
        dataset_id="ethanol_nir_spectroscopy",
        n_features=1751,
        n_classes=4,
        has_temporal_drift_batches=True,
        url="https://www.timeseriesclassification.com/aeon-toolkit/EthanolLevel.zip",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="Non-Invasive Through-Glass NIR/Raman Ethanol Concentration Spectroscopy",
        modality="nir_spectroscopy",
        default_normalizer="snv_zscore",
        landing_url="https://zenodo.org/records/11190985",
        archive_filename="EthanolLevel.zip",
        citation="Lines, Taylor & Bagnall (IEEE ICDM 2016) / Scotch Whisky Research Institute EthanolLevel, Zenodo:11190985.",
    ),
}


def list_open_edge_datasets(
    *,
    commercial_only: bool = False,
    drift_capable_only: bool = False,
) -> list[OpenEdgeDataset]:
    """Return registered `OpenEdgeDataset` specifications in deterministic order."""
    out: list[OpenEdgeDataset] = []
    for spec in OPEN_EDGE_DATASETS.values():
        if commercial_only and not spec.commercial_use_ok:
            continue
        if drift_capable_only and not spec.has_temporal_drift_batches:
            continue
        out.append(spec)
    return out


def get_open_edge_dataset(dataset_id: str) -> OpenEdgeDataset:
    """Look up a registered `OpenEdgeDataset` by `dataset_id`."""
    if dataset_id not in OPEN_EDGE_DATASETS:
        known = ", ".join(sorted(OPEN_EDGE_DATASETS))
        raise KeyError(f"Unknown dataset_id {dataset_id!r}; registered IDs: {known}.")
    return OPEN_EDGE_DATASETS[dataset_id]


def _apply_rowwise_pretransform(X: np.ndarray, kind: NormalizerKind, eps: float) -> np.ndarray:
    """Apply stateless per-sample pre-transform (`signed_log1p` or `SNV`) in float64."""
    x64 = np.asarray(X, dtype=np.float64)
    if kind == "signed_log1p_zscore":
        return np.sign(x64) * np.log1p(np.abs(x64))
    if kind == "snv_zscore":
        row_mean = np.mean(x64, axis=1, keepdims=True)
        row_std = np.std(x64, axis=1, keepdims=True)
        safe_row_std = np.where(row_std > eps, row_std, 1.0)
        return (x64 - row_mean) / safe_row_std
    return x64


class DeterministicFeatureNormalizer:
    """Leakage-free feature normalizer fit strictly on training samples.

    Supports:
    - `zscore`: `(X - mean) / std` with constant-channel guard (`std <= eps -> scale = 1`).
    - `robust_iqr`: `(X - median) / IQR` (`q75 - q25`), resilient to heavy vibration spikes.
    - `minmax_symmetric`: maps training `[min, max]` into `[-1, +1]`.
    - `signed_log1p_zscore`: `sign(X) * log1p(|X|)` followed by training z-score (ideal for
      wide-dynamic-range MOX chemical resistance + EMA transient features).
    - `snv_zscore`: per-spectrum Standard Normal Variate (`SNV`) baseline/path-length removal
      followed by per-wavenumber training z-score (ideal for Raman & NIR spectra).
    """

    def __init__(self, kind: NormalizerKind = "zscore", *, eps: float = 1e-8) -> None:
        if kind not in VALID_NORMALIZERS:
            raise ValueError(f"Unsupported normalizer kind {kind!r}; expected one of {VALID_NORMALIZERS}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.kind: NormalizerKind = kind
        self.eps: float = float(eps)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "DeterministicFeatureNormalizer":
        x_arr = np.asarray(X_train, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[0] < 1 or x_arr.shape[1] < 1:
            raise ValueError(f"X_train must be a non-empty 2D array, got shape {x_arr.shape}.")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("X_train contains NaN or infinite values.")

        pre = _apply_rowwise_pretransform(x_arr, self.kind, self.eps)

        if self.kind in ("zscore", "signed_log1p_zscore", "snv_zscore"):
            center = np.mean(pre, axis=0)
            raw_scale = np.std(pre, axis=0)
            scale = np.where(raw_scale > self.eps, raw_scale, 1.0)
        elif self.kind == "robust_iqr":
            center = np.median(pre, axis=0)
            q25 = np.percentile(pre, 25.0, axis=0)
            q75 = np.percentile(pre, 75.0, axis=0)
            iqr = q75 - q25
            fallback = np.std(pre, axis=0)
            raw_scale = np.where(iqr > self.eps, iqr, fallback)
            scale = np.where(raw_scale > self.eps, raw_scale, 1.0)
        elif self.kind == "minmax_symmetric":
            vmin = np.min(pre, axis=0)
            vmax = np.max(pre, axis=0)
            center = 0.5 * (vmin + vmax)
            half_span = 0.5 * (vmax - vmin)
            scale = np.where(half_span > self.eps, half_span, 1.0)
        else:  # pragma: no cover
            raise ValueError(f"Unhandled normalizer kind {self.kind!r}.")

        self.center_ = np.ascontiguousarray(center, dtype=np.float64)
        self.scale_ = np.ascontiguousarray(scale, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("DeterministicFeatureNormalizer must be fit before calling transform().")
        x_arr = np.asarray(X, dtype=np.float64)
        squeeze = False
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
            squeeze = True
        if x_arr.ndim != 2 or x_arr.shape[1] != self.center_.size:
            raise ValueError(
                f"Expected 2D input with {self.center_.size} features, got shape {x_arr.shape}."
            )
        pre = _apply_rowwise_pretransform(x_arr, self.kind, self.eps)
        normed = ((pre - self.center_[None, :]) / self.scale_[None, :]).astype(np.float32)
        return normed[0] if squeeze else np.ascontiguousarray(normed, dtype=np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def state_sha256(self) -> str:
        """Compute canonical little-endian SHA-256 hex digest of fitted normalizer parameters."""
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("DeterministicFeatureNormalizer must be fit before state_sha256().")
        h = hashlib.sha256()
        h.update(b"LFNORM1:")
        h.update(self.kind.encode("ascii"))
        h.update(struct.pack("<I", int(self.center_.size)))
        h.update(np.ascontiguousarray(self.center_, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.scale_, dtype="<f8").tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class LoadedEdgeDataset:
    """In-memory edge dataset array bundle with optional drift batch IDs and source SHA-256."""

    spec: OpenEdgeDataset
    X: np.ndarray
    y: np.ndarray
    batch_ids: np.ndarray | None = None
    predefined_split: np.ndarray | None = None
    raw_sha256: str = ""

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.X.shape[1] != self.spec.n_features:
            raise ValueError(
                f"Loaded X shape {self.X.shape} does not match spec.n_features={self.spec.n_features}."
            )
        if self.y.ndim != 1 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"Loaded y shape {self.y.shape} does not align with X shape {self.X.shape}."
            )
        if self.batch_ids is not None and self.batch_ids.shape != (self.X.shape[0],):
            raise ValueError(
                f"batch_ids shape {self.batch_ids.shape} does not match sample count {self.X.shape[0]}."
            )
        if self.predefined_split is not None and self.predefined_split.shape != (self.X.shape[0],):
            raise ValueError(
                f"predefined_split shape {self.predefined_split.shape} does not match {self.X.shape[0]}."
            )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fetch_open_edge_dataset_archive(
    dataset_id: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    force: bool = False,
    require_commercial_use: bool = True,
    timeout_sec: float = 60.0,
) -> Path:
    """Download an `OpenEdgeDataset` archive on-demand into gitignored `data/<dataset_id>/`."""
    spec = get_open_edge_dataset(dataset_id)
    verify_dataset_license(spec, require_commercial_use=require_commercial_use)

    target_dir = Path(data_dir) / spec.dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / spec.resolved_archive_filename()
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest

    part_path = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "lab-feet-open-datasets/0.2.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp, part_path.open("wb") as out_fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out_fh.write(chunk)
    part_path.replace(dest)
    return dest


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Restricted unpickler permitting only NumPy ndarray reconstruction for `mnist1d_data.pkl`."""

    _SAFE_BUILTINS = {"slice": slice, "range": range, "complex": complex, "bytearray": bytearray}
    _SAFE_NUMPY = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in self._SAFE_BUILTINS:
            return self._SAFE_BUILTINS[name]
        if (module, name) in self._SAFE_NUMPY:
            mod = __import__(module, fromlist=[name])
            return getattr(mod, name)
        raise pickle.UnpicklingError(f"Disallowed pickle global {module}.{name} in MNIST-1D loader.")


def _parse_mnist1d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            x_tr = np.asarray(data["x"], dtype=np.float32)
            y_tr = np.asarray(data["y"], dtype=np.int64).reshape(-1)
            x_te = np.asarray(data["x_test"], dtype=np.float32)
            y_te = np.asarray(data["y_test"], dtype=np.int64).reshape(-1)
    else:
        with path.open("rb") as fh:
            payload = _RestrictedNumpyUnpickler(fh, encoding="latin1").load()
        x_tr = np.asarray(payload["x"], dtype=np.float32)
        y_tr = np.asarray(payload["y"], dtype=np.int64).reshape(-1)
        x_te = np.asarray(payload["x_test"], dtype=np.float32)
        y_te = np.asarray(payload["y_test"], dtype=np.int64).reshape(-1)

    X = np.vstack([x_tr, x_te]).astype(np.float32, copy=False)
    y = np.concatenate([y_tr, y_te]).astype(np.int64, copy=False)
    split = np.concatenate(
        [np.zeros(len(y_tr), dtype=np.int64), np.ones(len(y_te), dtype=np.int64)]
    )
    return X, y, None, split


def _parse_libsvm_gas_lines(lines: Sequence[str], n_features: int = 128) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        head = parts[0].split(";")[0]
        cls_1based = int(float(head))
        if not (1 <= cls_1based <= 6):
            raise ValueError(f"Unexpected UCI Gas Sensor class label {cls_1based} (expected 1..6).")
        vec = np.zeros(n_features, dtype=np.float32)
        for tok in parts[1:]:
            idx_s, val_s = tok.split(":", 1)
            f_idx = int(idx_s) - 1
            if 0 <= f_idx < n_features:
                vec[f_idx] = np.float32(float(val_s))
        xs.append(vec)
        ys.append(cls_1based - 1)
    if not xs:
        raise ValueError("No valid LIBSVM rows found in UCI Gas Sensor batch file.")
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)


def _extract_batch_number(member_name: str) -> int:
    stem = Path(member_name).stem.lower()
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def _parse_uci_gas_sensor_drift(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    batches_all: list[np.ndarray] = []

    with zipfile.ZipFile(path, "r") as zf:
        dat_members = [
            m
            for m in zf.namelist()
            if m.lower().endswith(".dat") and "batch" in Path(m).name.lower() and "__macosx" not in m.lower()
        ]
        if not dat_members:
            nested_zips = [m for m in zf.namelist() if m.lower().endswith(".zip")]
            if nested_zips:
                with zipfile.ZipFile(io.BytesIO(zf.read(nested_zips[0])), "r") as inner:
                    inner_members = sorted(
                        [
                            m
                            for m in inner.namelist()
                            if m.lower().endswith(".dat") and "batch" in Path(m).name.lower()
                        ],
                        key=_extract_batch_number,
                    )
                    for m in inner_members:
                        b_num = _extract_batch_number(m)
                        lines = inner.read(m).decode("utf-8", errors="replace").splitlines()
                        xb, yb = _parse_libsvm_gas_lines(lines, n_features=128)
                        xs_all.append(xb)
                        ys_all.append(yb)
                        batches_all.append(np.full(len(yb), b_num, dtype=np.int64))
        else:
            for m in sorted(dat_members, key=_extract_batch_number):
                b_num = _extract_batch_number(m)
                lines = zf.read(m).decode("utf-8", errors="replace").splitlines()
                xb, yb = _parse_libsvm_gas_lines(lines, n_features=128)
                xs_all.append(xb)
                ys_all.append(yb)
                batches_all.append(np.full(len(yb), b_num, dtype=np.int64))

    if not xs_all:
        raise ValueError(f"No batch*.dat files found inside {path}.")
    return (
        np.vstack(xs_all).astype(np.float32, copy=False),
        np.concatenate(ys_all).astype(np.int64, copy=False),
        np.concatenate(batches_all).astype(np.int64, copy=False),
        None,
    )


def _find_zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    suffix_low = suffix.lower()
    for name in zf.namelist():
        norm = name.replace("\\", "/")
        if "__macosx" in norm.lower() or Path(norm).name.startswith("._"):
            continue
        if norm.lower().endswith(suffix_low):
            return name
    raise KeyError(f"Could not locate archive member ending with {suffix!r}.")


def _parse_uci_har_inertial(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with zipfile.ZipFile(path, "r") as outer_zf:
        nested = [
            m
            for m in outer_zf.namelist()
            if m.lower().endswith(".zip") and "__macosx" not in m.lower()
        ]
        if nested:
            inner_bytes = outer_zf.read(nested[0])
            har_zf = zipfile.ZipFile(io.BytesIO(inner_bytes), "r")
            close_inner = True
        else:
            har_zf = outer_zf
            close_inner = False

        try:
            x_tr_txt = har_zf.read(_find_zip_member(har_zf, "train/X_train.txt")).decode("utf-8")
            y_tr_txt = har_zf.read(_find_zip_member(har_zf, "train/y_train.txt")).decode("utf-8")
            x_te_txt = har_zf.read(_find_zip_member(har_zf, "test/X_test.txt")).decode("utf-8")
            y_te_txt = har_zf.read(_find_zip_member(har_zf, "test/y_test.txt")).decode("utf-8")
        finally:
            if close_inner:
                har_zf.close()

    x_tr = np.loadtxt(io.StringIO(x_tr_txt), dtype=np.float32)
    y_tr = np.loadtxt(io.StringIO(y_tr_txt), dtype=np.int64).reshape(-1) - 1
    x_te = np.loadtxt(io.StringIO(x_te_txt), dtype=np.float32)
    y_te = np.loadtxt(io.StringIO(y_te_txt), dtype=np.int64).reshape(-1) - 1
    if x_tr.ndim == 1:
        x_tr = x_tr.reshape(1, -1)
    if x_te.ndim == 1:
        x_te = x_te.reshape(1, -1)

    X = np.vstack([x_tr, x_te]).astype(np.float32, copy=False)
    y = np.concatenate([y_tr, y_te]).astype(np.int64, copy=False)
    split = np.concatenate(
        [np.zeros(len(y_tr), dtype=np.int64), np.ones(len(y_te), dtype=np.int64)]
    )
    return X, y, None, split


def _parse_cwru_bearing_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    xs: list[list[float]] = []
    ys: list[int] = []
    fault_sizes: list[float] = []
    for row in reader:
        feats = json.loads(row["all_features"])
        xs.append([float(v) for v in feats])
        ys.append(int(row["label"]))
        fault_sizes.append(round(float(row["fault_size"]), 6))
    if not xs:
        raise ValueError(f"Empty CWRU bearing CSV at {path}.")
    unique_sizes = sorted(set(fault_sizes))
    size_to_batch = {sz: idx for idx, sz in enumerate(unique_sizes)}
    batches = np.asarray([size_to_batch[sz] for sz in fault_sizes], dtype=np.int64)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        batches,
        None,
    )


def _parse_uci_hydraulic_vibration(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    valve_map = {100: 0, 90: 1, 80: 2, 73: 3}
    cooler_batch_map = {100: 0, 20: 1, 3: 2}
    with zipfile.ZipFile(path, "r") as zf:
        vs1_txt = zf.read(_find_zip_member(zf, "VS1.txt")).decode("utf-8")
        prof_txt = zf.read(_find_zip_member(zf, "profile.txt")).decode("utf-8")

    X = np.loadtxt(io.StringIO(vs1_txt), dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    prof = np.loadtxt(io.StringIO(prof_txt), dtype=np.int64)
    if prof.ndim == 1:
        prof = prof.reshape(1, -1)

    cooler_raw = prof[:, 0]
    valve_raw = prof[:, 1]
    y = np.asarray([valve_map[int(v)] for v in valve_raw], dtype=np.int64)
    batches = np.asarray([cooler_batch_map[int(c)] for c in cooler_raw], dtype=np.int64)
    return X, y, batches, None


def _parse_sers_bacteria_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    spec_start = 6
    if len(header) <= spec_start:
        raise ValueError(f"Unexpected SERS bacteria CSV header with {len(header)} columns.")
    rows = [r for r in reader if r]
    if not rows:
        raise ValueError(f"Empty SERS bacteria CSV at {path}.")

    strain_keys = [f"{r[0].strip()}::{r[1].strip()}" for r in rows]
    unique_strains = sorted(set(strain_keys))
    strain_to_id = {s: idx for idx, s in enumerate(unique_strains)}

    acq_orders = np.asarray([int(float(r[4])) for r in rows], dtype=np.int64)
    # Partition each strain's acquisition order into 3 chronological measurement batches (0, 1, 2)
    batches = np.zeros(len(rows), dtype=np.int64)
    y = np.asarray([strain_to_id[k] for k in strain_keys], dtype=np.int64)
    for c_id in range(len(unique_strains)):
        mask = np.flatnonzero(y == c_id)
        if mask.size > 0:
            order_rank = np.argsort(np.argsort(acq_orders[mask], kind="mergesort"), kind="mergesort")
            batches[mask] = np.minimum(2, (order_rank * 3) // max(1, mask.size))

    X = np.asarray([[float(v) for v in r[spec_start:]] for r in rows], dtype=np.float32)
    return X, y, batches, None


def _parse_ethanol_nir_zip(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with zipfile.ZipFile(path, "r") as zf:
        tr_txt = zf.read(_find_zip_member(zf, "EthanolLevel_TRAIN.txt")).decode("utf-8")
        te_txt = zf.read(_find_zip_member(zf, "EthanolLevel_TEST.txt")).decode("utf-8")

    tr_mat = np.loadtxt(io.StringIO(tr_txt), dtype=np.float64)
    te_mat = np.loadtxt(io.StringIO(te_txt), dtype=np.float64)
    if tr_mat.ndim == 1:
        tr_mat = tr_mat.reshape(1, -1)
    if te_mat.ndim == 1:
        te_mat = te_mat.reshape(1, -1)

    y_tr = np.rint(tr_mat[:, 0]).astype(np.int64) - 1
    y_te = np.rint(te_mat[:, 0]).astype(np.int64) - 1
    x_tr = tr_mat[:, 1:].astype(np.float32)
    x_te = te_mat[:, 1:].astype(np.float32)

    X = np.vstack([x_tr, x_te]).astype(np.float32, copy=False)
    y = np.concatenate([y_tr, y_te]).astype(np.int64, copy=False)
    split = np.concatenate(
        [np.zeros(len(y_tr), dtype=np.int64), np.ones(len(y_te), dtype=np.int64)]
    )
    # Batch 0 = calibration bottle containers, Batch 1 = held-out unseen bottle container types
    batches = split.copy()
    return X, y, batches, split


_PARSERS = {
    "mnist1d": _parse_mnist1d,
    "uci_gas_sensor_drift": _parse_uci_gas_sensor_drift,
    "uci_har_inertial": _parse_uci_har_inertial,
    "cwru_bearing_vibration": _parse_cwru_bearing_csv,
    "uci_hydraulic_vibration": _parse_uci_hydraulic_vibration,
    "sers_bacteria_raman_2026": _parse_sers_bacteria_csv,
    "ethanol_nir_spectroscopy": _parse_ethanol_nir_zip,
}


def load_open_edge_dataset(
    dataset_id: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    archive_path: str | Path | None = None,
    auto_download: bool = False,
    require_commercial_use: bool = True,
) -> LoadedEdgeDataset:
    """Load and parse a registered `OpenEdgeDataset` from `archive_path` or `data/<dataset_id>/`."""
    spec = get_open_edge_dataset(dataset_id)
    verify_dataset_license(spec, require_commercial_use=require_commercial_use)

    if archive_path is not None:
        resolved = Path(archive_path)
    else:
        candidate = Path(data_dir) / spec.dataset_id / spec.resolved_archive_filename()
        if not candidate.is_file():
            if not auto_download:
                raise FileNotFoundError(
                    f"Archive for {dataset_id!r} not found at {candidate}. "
                    f"Run `python scripts/fetch_open_datasets.py --dataset {dataset_id}` "
                    f"or pass `auto_download=True`."
                )
            candidate = fetch_open_edge_dataset_archive(
                dataset_id,
                data_dir=data_dir,
                require_commercial_use=require_commercial_use,
            )
        resolved = candidate

    parser = _PARSERS[spec.dataset_id]
    X, y, batch_ids, predefined_split = parser(resolved)
    return LoadedEdgeDataset(
        spec=spec,
        X=np.ascontiguousarray(X, dtype=np.float32),
        y=np.ascontiguousarray(y, dtype=np.int64),
        batch_ids=batch_ids,
        predefined_split=predefined_split,
        raw_sha256=_sha256_file(resolved),
    )


def prepare_distill_quantize_splits(
    dataset: LoadedEdgeDataset,
    *,
    normalizer_kind: NormalizerKind | None = None,
    test_size: float = 0.25,
    val_fraction_of_dev: float = 0.20,
    seed: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    DeterministicFeatureNormalizer,
]:
    """Create leakage-free `(X_tr, y_tr, X_val, y_val, X_te, y_te, normalizer)` splits.

    Respects `dataset.predefined_split` (`0`=dev, `1`=test) when provided and fits
    `DeterministicFeatureNormalizer` strictly on `X_tr`.
    """
    eff_kind = dataset.spec.default_normalizer if normalizer_kind is None else normalizer_kind
    if dataset.predefined_split is not None and np.any(dataset.predefined_split == 0) and np.any(
        dataset.predefined_split == 1
    ):
        dev_mask = dataset.predefined_split == 0
        te_mask = dataset.predefined_split == 1
        X_dev, y_dev = dataset.X[dev_mask], dataset.y[dev_mask]
        X_te_raw, y_te = dataset.X[te_mask], dataset.y[te_mask]
    else:
        X_dev, X_te_raw, y_dev, y_te = train_test_split(
            dataset.X,
            dataset.y,
            test_size=test_size,
            random_state=seed,
            stratify=dataset.y,
        )

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=val_fraction_of_dev,
        random_state=seed,
        stratify=y_dev,
    )
    norm = DeterministicFeatureNormalizer(kind=eff_kind).fit(X_tr_raw)
    return (
        norm.transform(X_tr_raw),
        np.ascontiguousarray(y_tr, dtype=np.int64),
        norm.transform(X_val_raw),
        np.ascontiguousarray(y_val, dtype=np.int64),
        norm.transform(X_te_raw),
        np.ascontiguousarray(y_te, dtype=np.int64),
        norm,
    )


def run_open_dataset_distill_quantize(
    dataset: LoadedEdgeDataset,
    *,
    teacher_hidden: tuple[int, ...] = (128, 64),
    student_hidden: tuple[int, ...] = (32,),
    temperature: float = 4.0,
    alpha: float = 0.25,
    student_epochs: int = 180,
    teacher_max_iter: int = 260,
    normalizer_kind: NormalizerKind | None = None,
    require_commercial_use: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Run hybrid `KL(T^2)+CE` distillation + `int8` (`W8A8`) / `4-bit` (`LFT1`) quantization on `dataset`."""
    verify_dataset_license(dataset.spec, require_commercial_use=require_commercial_use)
    X_tr, y_tr, X_val, y_val, X_te, y_te, norm = prepare_distill_quantize_splits(
        dataset,
        normalizer_kind=normalizer_kind,
        seed=seed,
    )

    teacher = MLPClassifier(
        hidden_layer_sizes=teacher_hidden,
        max_iter=teacher_max_iter,
        random_state=seed,
    )
    teacher_diag = fit_sklearn_mlp(teacher, X_tr, y_tr)
    t_layers = get_layers(teacher)
    t_logits_tr = forward(t_layers, X_tr)

    kl_layers, kl_diag = distill_kl_student(
        X_tr,
        y_tr,
        t_logits_tr,
        X_val=X_val,
        y_val=y_val,
        hidden_sizes=student_hidden,
        temperature=temperature,
        alpha=alpha,
        epochs=student_epochs,
        seed=seed,
    )

    q_int8: QuantizedModel = quantize_model(
        kl_layers,
        8,
        per_channel=True,
        mode="symmetric",
        X_cal=X_val,
        calibrate_static_act_clips=True,
    )
    q_int4: QuantizedModel = quantize_model(
        kl_layers,
        4,
        per_channel=True,
        mode="symmetric",
        clip_search="activation_mse",
        X_cal=X_val,
        calibrate_static_act_clips=True,
    )

    lft1_blob = to_binary_buffer(q_int4)
    expected_lft1_bytes = compute_binary_header_bytes(len(q_int4.layers)) + q_int4.nbytes
    if len(lft1_blob) != expected_lft1_bytes:
        raise AssertionError(f"LFT1 size mismatch: {len(lft1_blob)} != {expected_lft1_bytes}.")
    reloaded_q4 = from_binary_buffer(lft1_blob)

    teacher_te_acc = accuracy(t_layers, X_te, y_te)
    kl_te_preds = forward(kl_layers, X_te).argmax(axis=1)
    kl_te_acc = float((kl_te_preds == y_te).mean())

    w8a8_te_preds = forward_int8(q_int8, X_te).argmax(axis=1)
    w8a8_te_acc = float((w8a8_te_preds == y_te).mean())

    int4_deq_preds = forward(reloaded_q4.dequantize(), X_te).argmax(axis=1)
    int4_w4a8_preds = forward_int8(reloaded_q4, X_te).argmax(axis=1)
    int4_te_acc = float((int4_deq_preds == y_te).mean())
    int4_w4a8_acc = float((int4_w4a8_preds == y_te).mean())

    return {
        "dataset_id": dataset.spec.dataset_id,
        "spdx_license": dataset.spec.spdx_license,
        "commercial_use_ok": dataset.spec.commercial_use_ok,
        "raw_sha256": dataset.raw_sha256,
        "normalizer": {
            "kind": norm.kind,
            "state_sha256": norm.state_sha256(),
        },
        "splits": {
            "train": int(len(X_tr)),
            "val": int(len(X_val)),
            "test": int(len(X_te)),
            "n_features": int(dataset.spec.n_features),
            "n_classes": int(dataset.spec.n_classes),
        },
        "teacher_fp32": {
            "test_acc": round(teacher_te_acc, 4),
            "params": n_params(t_layers),
            "weight_bytes": fp32_bytes(t_layers),
            "training": teacher_diag.to_dict(),
        },
        "student_kl_fp32": {
            "test_acc": round(kl_te_acc, 4),
            "params": n_params(kl_layers),
            "weight_bytes": fp32_bytes(kl_layers),
            "training": kl_diag.to_dict(),
        },
        "student_int8_w8a8": {
            "test_acc": round(w8a8_te_acc, 4),
            "weight_bytes": q_int8.nbytes,
            "agreement_with_kl_fp32": round(float((w8a8_te_preds == kl_te_preds).mean()), 4),
        },
        "student_int4_pc_mse": {
            "test_acc": round(int4_te_acc, 4),
            "w4a8_test_acc": round(int4_w4a8_acc, 4),
            "weight_bytes": reloaded_q4.nbytes,
            "lft1_total_bytes": len(lft1_blob),
            "agreement_with_kl_fp32": round(float((int4_deq_preds == kl_te_preds).mean()), 4),
        },
    }


def _load_drift_classifier_cls() -> Any:
    """Import `DriftAbstainingInt4Classifier` cleanly from `community_extensions`."""
    mod_name = "_lab_feet_datasets_drift_int4"
    existing = sys.modules.get(mod_name)
    if existing is not None:
        return existing.DriftAbstainingInt4Classifier
    mod_path = REPO_ROOT / "community_extensions" / "drift_abstaining_int4.py"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to locate {mod_path}.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.DriftAbstainingInt4Classifier


def evaluate_temporal_drift_abstaining_int4(
    dataset: LoadedEdgeDataset,
    *,
    source_batches: Sequence[int] | None = None,
    n_hidden: int = 32,
    max_monitor_features: int | None = None,
    normalizer_kind: NormalizerKind | None = None,
    require_commercial_use: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Train & calibrate `DriftAbstainingInt4Classifier` on early source batch(es) and evaluate multi-batch drift.

    Particularly tailored for chronological sensor-aging datasets such as `uci_gas_sensor_drift`
    (Batches `1..10` across 36 months) and progressive fault-severity datasets (`cwru_bearing_vibration`,
    `uci_hydraulic_vibration`). Fits the feature normalizer and optional top-variance sensor channel
    selector strictly on `source_batches` training data so zero future drift statistics leak into
    integer ADC scale or `L1`-Mahalanobis thresholds.
    """
    verify_dataset_license(dataset.spec, require_commercial_use=require_commercial_use)
    if not dataset.spec.has_temporal_drift_batches or dataset.batch_ids is None:
        raise ValueError(
            f"Dataset {dataset.spec.dataset_id!r} does not expose temporal/regime drift batches."
        )

    unique_batches = sorted(int(b) for b in np.unique(dataset.batch_ids))
    if len(unique_batches) < 2:
        raise ValueError(
            f"Need at least 2 distinct batches for temporal drift evaluation, got {unique_batches}."
        )

    src_set = {unique_batches[0]} if source_batches is None else {int(b) for b in source_batches}
    src_mask = np.isin(dataset.batch_ids, list(src_set))
    if not np.any(src_mask):
        raise ValueError(f"No samples matched source_batches={sorted(src_set)} in {unique_batches}.")

    X_src_all = dataset.X[src_mask]
    y_src_all = dataset.y[src_mask]

    # Split source batch(es) stratified into 60% train, 20% OOD gate calibration, 20% source ID test
    X_src_dev, X_src_te_raw, y_src_dev, y_src_te = train_test_split(
        X_src_all,
        y_src_all,
        test_size=0.20,
        random_state=seed,
        stratify=y_src_all,
    )
    X_tr_raw, X_cal_raw, y_tr, y_cal = train_test_split(
        X_src_dev,
        y_src_dev,
        test_size=0.25,
        random_state=seed,
        stratify=y_src_dev,
    )

    eff_kind = dataset.spec.default_normalizer if normalizer_kind is None else normalizer_kind
    norm = DeterministicFeatureNormalizer(kind=eff_kind).fit(X_tr_raw)
    X_tr_full = norm.transform(X_tr_raw)
    X_cal_full = norm.transform(X_cal_raw)
    X_src_te_full = norm.transform(X_src_te_raw)

    if max_monitor_features is not None and max_monitor_features < dataset.spec.n_features:
        if max_monitor_features < 1:
            raise ValueError(f"max_monitor_features must be >= 1, got {max_monitor_features}.")
        # Select highest-F-ratio discriminative channels strictly on source training split
        overall_mean = np.mean(X_tr_full, axis=0)
        between_var = np.zeros(dataset.spec.n_features, dtype=np.float64)
        for c in range(dataset.spec.n_classes):
            c_mask = y_tr == c
            if np.any(c_mask):
                diff = np.mean(X_tr_full[c_mask], axis=0) - overall_mean
                between_var += float(np.sum(c_mask)) * np.square(diff)
        feat_idx = np.sort(np.argsort(-between_var, kind="mergesort")[:max_monitor_features])
    else:
        feat_idx = np.arange(dataset.spec.n_features, dtype=np.int64)

    X_tr = X_tr_full[:, feat_idx]
    X_cal = X_cal_full[:, feat_idx]
    X_src_te = X_src_te_full[:, feat_idx]

    clf_cls = _load_drift_classifier_cls()
    clf = clf_cls(
        n_features=int(feat_idx.size),
        n_hidden=n_hidden,
        n_classes=dataset.spec.n_classes,
    )
    clf.fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, seed=seed)

    def _eval_split(X_eval: np.ndarray, y_eval: np.ndarray, tag: str, batch_num: int) -> dict[str, Any]:
        n = int(len(y_eval))
        naive_correct = 0
        guarded_correct = 0
        guarded_abstained = 0
        guarded_silent_err = 0
        l1_scores: list[int] = []
        int4_clips: list[int] = []
        in_clips: list[int] = []

        for idx in range(n):
            truth = int(y_eval[idx])
            p_naive = clf.predict_naive_int4_argmax(X_eval[idx])
            p_guard = clf.predict_drift_abstaining_int4(X_eval[idx])

            if p_naive.class_id == truth:
                naive_correct += 1
            if p_guard.abstained:
                guarded_abstained += 1
            elif p_guard.class_id == truth:
                guarded_correct += 1
            else:
                guarded_silent_err += 1

            l1_scores.append(p_guard.l1_mahalanobis_score)
            int4_clips.append(p_guard.int4_clip_count)
            in_clips.append(p_guard.int8_input_clip_count)

        accepted = n - guarded_abstained
        naive_acc = 100.0 * naive_correct / max(1, n)
        naive_silent_err = 100.0 - naive_acc
        guarded_sel_acc = (
            100.0 * guarded_correct / accepted if accepted > 0 else 100.0
        )
        return {
            "batch_tag": tag,
            "batch_id": int(batch_num),
            "n_samples": n,
            "naive_int4_accuracy_pct": round(naive_acc, 4),
            "naive_silent_error_rate_pct": round(naive_silent_err, 4),
            "guarded_accepted_accuracy_pct": round(guarded_sel_acc, 4),
            "guarded_safe_hold_rate_pct": round(100.0 * guarded_abstained / max(1, n), 4),
            "guarded_silent_error_rate_pct": round(100.0 * guarded_silent_err / max(1, n), 4),
            "mean_l1_mahalanobis_score": round(float(np.mean(l1_scores)), 2),
            "mean_int4_clips": round(float(np.mean(int4_clips)), 3),
            "mean_int8_input_clips": round(float(np.mean(in_clips)), 3),
        }

    source_holdout_metrics = _eval_split(X_src_te, y_src_te, "source_in_domain_holdout", min(src_set))

    drift_batches_metrics: list[dict[str, Any]] = []
    for b_id in unique_batches:
        if b_id in src_set:
            continue
        b_mask = dataset.batch_ids == b_id
        X_b = norm.transform(dataset.X[b_mask])[:, feat_idx]
        y_b = dataset.y[b_mask]
        drift_batches_metrics.append(_eval_split(X_b, y_b, f"batch_{b_id}", b_id))

    return {
        "dataset_id": dataset.spec.dataset_id,
        "spdx_license": dataset.spec.spdx_license,
        "source_batches": sorted(src_set),
        "evaluated_drift_batches": [m["batch_id"] for m in drift_batches_metrics],
        "n_selected_features": int(feat_idx.size),
        "packed_weight_bytes": clf.packed_weight_bytes,
        "extra_sram_bytes": clf.extra_sram_bytes,
        "normalizer": {
            "kind": norm.kind,
            "state_sha256": norm.state_sha256(),
        },
        "source_in_domain": source_holdout_metrics,
        "drift_batches": drift_batches_metrics,
    }


__all__ = [
    "COMMERCIAL_SAFE_SPDX_LICENSES",
    "DEFAULT_DATA_DIR",
    "OPEN_EDGE_DATASETS",
    "VALID_NORMALIZERS",
    "DatasetLicenseError",
    "DeterministicFeatureNormalizer",
    "LoadedEdgeDataset",
    "NormalizerKind",
    "OpenEdgeDataset",
    "evaluate_temporal_drift_abstaining_int4",
    "fetch_open_edge_dataset_archive",
    "get_open_edge_dataset",
    "list_open_edge_datasets",
    "load_open_edge_dataset",
    "prepare_distill_quantize_splits",
    "run_open_dataset_distill_quantize",
    "verify_dataset_license",
]
