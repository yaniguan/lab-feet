"""Frontier L0 quantization, 1D spectral Conv/ResNet backbone, and drift/OOD abstention methods.

Provides:
1. **Walsh-Hadamard Transform (`WHT` / SpinQuant-style) Outlier Rotation + GPTQ-lite Hessian Compensation**:
   - Exact orthogonal block-dyadic normalized Hadamard rotation `Q = Pi^T diag(s) H_blk` (`Q^T Q = I_d`)
     dispersing activation and weight channel spikes in `O(d log d)` butterfly operations without changing
     tensor shapes `(d_in, d_out)`.
   - Cholesky inverse-Hessian error compensation (`gptq_quantize_weights`, Frantar et al., ICLR 2023)
     paired with `W4A8` integer inference (`WHTQuantizedModel`, `quantize_model_wht_gptq`).
2. **1D Residual Spectral Conv/MLP Backbone (`SpectralResNet1D`, `15k-50k` parameters)**:
   - Multi-scale 1D convolutional stem + two residual spectral convolution blocks + multi-scale spectral
     pooling + bottleneck head (`~39k` parameters by default), trained with mini-batch Adam + cosine LR
     and hybrid `KL(T^2) + CE` distillation, and quantized end-to-end to `int8` (`W8A8`) and `4-bit`
     (`WHT + GPTQ W4A8`).
3. **Calibrated OOD Abstainers & Test-Time Drift Adaptation**:
   - Maximum Softmax Probability (`MSP`, Hendrycks & Gimpel, ICLR 2017)
   - Helmholtz Free Energy (`-logsumexp(logits)`, Liu et al., NeurIPS 2020)
   - `ReAct` penultimate activation rectification (Sun et al., NeurIPS 2021)
   - Ledoit-Wolf Shrinkage Mahalanobis OOD detector (`ShrinkageMahalanobisDetector`, Lee et al., NeurIPS 2018)
   - Area Under the Risk-Coverage Curve (`compute_aurc`, Geifman & El-Yaniv, NeurIPS 2017)
   - Bayesian-shrunk Test-Time Batch Normalization (`TTN`) + Conformal Risk Control (`CRC`,
     Angelopoulos et al., ICLR 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .distill import TrainingDiagnostics, _log_softmax, kl_divergence_soft_targets
from .quantize import (
    ClipSearchMode,
    QuantizedLayer,
    QuantizedModel,
    QuantMode,
    _quantize_activations_int8,
    _tiled_int32_matmul,
    max_safe_int32_fan_in,
    pack_4bit,
    quantize,
)
from .runtime import fp32_bytes, softmax


def _decompose_dyadic_blocks(d: int) -> tuple[int, ...]:
    """Decompose positive integer `d` into descending powers of two (`sum(blocks) == d`)."""
    if d < 1:
        raise ValueError(f"Dimension d must be >= 1, got {d}.")
    blocks: list[int] = []
    rem = int(d)
    bit = 1 << (rem.bit_length() - 1)
    while bit > 0:
        if rem & bit:
            blocks.append(bit)
            rem -= bit
        bit >>= 1
    return tuple(blocks)


def _fwht_pow2_normalized(x_block: np.ndarray) -> np.ndarray:
    """Normalized Fast Walsh-Hadamard Transform along the last axis for power-of-2 width `B`."""
    b = int(x_block.shape[-1])
    if b <= 1:
        return x_block.astype(np.float64, copy=True)
    if b & (b - 1) != 0:
        raise ValueError(f"_fwht_pow2_normalized requires power-of-2 width, got {b}.")

    out = x_block.astype(np.float64, copy=True)
    orig_shape = out.shape
    flat = out.reshape(-1, b)
    h = 1
    while h < b:
        reshaped = flat.reshape(-1, b // (2 * h), 2, h)
        u = reshaped[:, :, 0, :].copy()
        v = reshaped[:, :, 1, :].copy()
        reshaped[:, :, 0, :] = u + v
        reshaped[:, :, 1, :] = u - v
        h <<= 1
    flat *= 1.0 / np.sqrt(float(b))
    return flat.reshape(orig_shape)


@dataclass(frozen=True)
class WHTSpec:
    """Exact orthogonal block-dyadic Walsh-Hadamard rotation specification (`Q^T Q = I_d`)."""

    dim: int
    perm: np.ndarray
    inv_perm: np.ndarray
    signs: np.ndarray
    block_sizes: tuple[int, ...]


def build_wht_spec(
    dim: int,
    *,
    X_cal: np.ndarray | None = None,
    seed: int = 0,
) -> WHTSpec:
    """Build an exact `dim x dim` orthogonal Walsh-Hadamard rotation dispersing outlier channels."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}.")
    rng = np.random.default_rng(seed)
    blocks = _decompose_dyadic_blocks(dim)
    if X_cal is not None and np.asarray(X_cal).size > 0:
        x2d = np.asarray(X_cal, dtype=np.float64).reshape(-1, dim)
        ch_peak = np.max(np.abs(x2d), axis=0)
        # Route largest-peak outlier channels into the widest dyadic Hadamard block first
        perm = np.argsort(-ch_peak, kind="mergesort").astype(np.int64)
    else:
        perm = rng.permutation(dim).astype(np.int64)
    inv_perm = np.argsort(perm, kind="mergesort").astype(np.int64)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=dim)
    return WHTSpec(
        dim=int(dim),
        perm=perm,
        inv_perm=inv_perm,
        signs=signs,
        block_sizes=blocks,
    )


def walsh_hadamard_rotate(X: np.ndarray, spec: WHTSpec) -> np.ndarray:
    """Apply orthogonal rotation `X @ Q` in `O(d log d)` butterfly steps."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.shape[-1] != spec.dim:
        raise ValueError(f"Expected last dimension {spec.dim}, got shape {arr.shape}.")
    work = arr[..., spec.perm] * spec.signs
    out = np.empty_like(work)
    offset = 0
    for b in spec.block_sizes:
        out[..., offset : offset + b] = _fwht_pow2_normalized(work[..., offset : offset + b])
        offset += b
    return out.astype(np.float32)


def walsh_hadamard_unrotate(X_rot: np.ndarray, spec: WHTSpec) -> np.ndarray:
    """Apply exact inverse `X_rot @ Q^T` (`Q Q^T = I_d`)."""
    arr = np.asarray(X_rot, dtype=np.float64)
    if arr.shape[-1] != spec.dim:
        raise ValueError(f"Expected last dimension {spec.dim}, got shape {arr.shape}.")
    work = np.empty_like(arr)
    offset = 0
    for b in spec.block_sizes:
        work[..., offset : offset + b] = _fwht_pow2_normalized(arr[..., offset : offset + b])
        offset += b
    work *= spec.signs
    return work[..., spec.inv_perm].astype(np.float32)


def rotate_linear_weights_wht(W: np.ndarray, spec: WHTSpec) -> np.ndarray:
    """Transform weight matrix `W` `(d_in, d_out)` into `Q^T @ W` so `(X @ Q) @ (Q^T @ W) == X @ W`."""
    w_arr = np.asarray(W, dtype=np.float64)
    if w_arr.ndim != 2 or w_arr.shape[0] != spec.dim:
        raise ValueError(f"Expected 2D weight matrix ({spec.dim}, d_out), got {w_arr.shape}.")
    return walsh_hadamard_rotate(w_arr.T, spec).T


def gptq_quantize_weights(
    W: np.ndarray,
    bits: int = 4,
    *,
    X_cal: np.ndarray,
    per_channel: bool = True,
    mode: QuantMode = "symmetric_full",
    clip_search: ClipSearchMode = "activation_mse",
    damp_percent: float = 0.01,
    block_size: int = 32,
    actorder: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Quantize `W` `(d_in, d_out)` with Cholesky inverse-Hessian GPTQ error compensation."""
    W_f64 = np.asarray(W, dtype=np.float64).copy()
    X_f64 = np.asarray(X_cal, dtype=np.float64)
    if X_f64.ndim == 1:
        X_f64 = X_f64.reshape(1, -1)
    if W_f64.ndim != 2 or X_f64.shape[1] != W_f64.shape[0]:
        raise ValueError(f"Incompatible shapes for GPTQ: X_cal={X_f64.shape}, W={W_f64.shape}.")

    d_in, d_out = W_f64.shape
    if d_in == 0 or d_out == 0 or X_f64.shape[0] == 0:
        q_res = quantize(W_f64.astype(np.float32), bits, per_channel=per_channel, mode=mode, return_zero_point=(mode == "asymmetric"))
        if mode == "asymmetric":
            return q_res  # type: ignore[return-value]
        return q_res[0], q_res[1], None  # type: ignore[index]

    # Calibrate initial output-channel scales (and optional zero-points) via MSE grid search
    q_init_res = quantize(
        W_f64.astype(np.float32),
        bits,
        per_channel=per_channel,
        mode=mode,
        clip_search=clip_search,
        X_cal=X_f64.astype(np.float32),
        return_zero_point=(mode == "asymmetric"),
    )
    if mode == "asymmetric":
        _, scale_init, zp_init = q_init_res  # type: ignore[misc]
    else:
        _, scale_init = q_init_res  # type: ignore[misc]
        zp_init = None

    s_vec = (
        np.asarray(scale_init, dtype=np.float64).reshape(1, d_out)
        if per_channel
        else np.full((1, d_out), float(np.asarray(scale_init).reshape(-1)[0]), dtype=np.float64)
    )
    zp_vec = (
        np.asarray(zp_init, dtype=np.float64).reshape(1, d_out)
        if (zp_init is not None and per_channel)
        else (
            np.full((1, d_out), float(np.asarray(zp_init).reshape(-1)[0]), dtype=np.float64)
            if zp_init is not None
            else np.zeros((1, d_out), dtype=np.float64)
        )
    )
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax if mode == "symmetric" else -(1 << (bits - 1))

    # Empirical input Hessian H = (2 / N) X^T X
    n_cal = float(X_f64.shape[0])
    H = (2.0 / n_cal) * (X_f64.T @ X_f64)
    diag_h = np.diag(H).copy()
    dead = diag_h <= 1e-12
    if np.any(dead):
        W_f64[dead, :] = 0.0
        H[dead, :] = 0.0
        H[:, dead] = 0.0
        H[dead, dead] = 1.0
        diag_h[dead] = 1.0

    if actorder:
        perm = np.argsort(-diag_h, kind="mergesort")
        inv_perm = np.argsort(perm, kind="mergesort")
        W_work = W_f64[perm, :].copy()
        H_work = H[np.ix_(perm, perm)].copy()
    else:
        perm = np.arange(d_in, dtype=np.int64)
        inv_perm = perm
        W_work = W_f64.copy()
        H_work = H

    mean_diag = max(float(np.mean(np.diag(H_work))), 1e-8)
    damp = max(float(damp_percent), 1e-4) * mean_diag
    U: np.ndarray | None = None
    for mult in (1.0, 5.0, 25.0, 100.0):
        try:
            H_damped = H_work + (damp * mult) * np.eye(d_in, dtype=np.float64)
            L_h = np.linalg.cholesky(H_damped)
            L_inv = np.linalg.solve(L_h, np.eye(d_in, dtype=np.float64))
            H_inv = L_inv.T @ L_inv
            # Upper Cholesky U of H_inv such that U^T @ U == H_inv (exact Schur-complement OBQ update)
            U = np.linalg.cholesky(H_inv).T
            break
        except np.linalg.LinAlgError:
            continue

    if U is None:
        # Diagonal fallback if Hessian is numerically singular
        U = np.diag(1.0 / np.sqrt(np.maximum(np.diag(H_work), 1e-8)))

    Q_codes = np.zeros((d_in, d_out), dtype=np.int8)
    bsz = max(1, int(block_size))
    for i0 in range(0, d_in, bsz):
        i1 = min(i0 + bsz, d_in)
        W_blk = W_work[i0:i1, :].copy()
        Err_blk = np.zeros_like(W_blk)
        U_blk = U[i0:i1, i0:i1]

        for r in range(i1 - i0):
            w_row = W_blk[r : r + 1, :]
            q_row = np.clip(np.round(w_row / s_vec + zp_vec), qmin, qmax)
            w_hat = (q_row - zp_vec) * s_vec
            Q_codes[i0 + r, :] = q_row.astype(np.int8).reshape(-1)

            u_diag = max(float(U_blk[r, r]), 1e-12)
            err_row = (w_row - w_hat) / u_diag
            Err_blk[r : r + 1, :] = err_row
            if r + 1 < (i1 - i0):
                W_blk[r + 1 :, :] -= U_blk[r, r + 1 :][:, None] * err_row

        if i1 < d_in:
            W_work[i1:, :] -= U[i0:i1, i1:].T @ Err_blk

    Q_final = Q_codes[inv_perm, :]
    if dead.any():
        Q_final[dead, :] = np.clip(np.round(zp_vec), qmin, qmax).astype(np.int8)

    scale_out = (
        s_vec.astype(np.float32)
        if per_channel
        else np.float32(float(s_vec.reshape(-1)[0]))
    )
    zp_out = (
        zp_vec.astype(np.int8)
        if (zp_init is not None and per_channel)
        else (np.int8(int(zp_vec.reshape(-1)[0])) if zp_init is not None else None)
    )
    return Q_final, scale_out, zp_out


@dataclass
class WHTQuantizedModel:
    """Outlier-rotated (`WHT`) + `GPTQ`-compensated `QuantizedModel` executing `W4A8` / `W8A8`."""

    qmodel: QuantizedModel
    wht_specs: list[WHTSpec]

    @property
    def nbytes(self) -> int:
        # Each WHT layer stores deterministic seed/sign bitmask (`ceil(d/8)` B) + permutation indices (`2*d` B)
        rot_meta_bytes = sum(((s.dim + 7) // 8) + 2 * s.dim for s in self.wht_specs)
        return int(self.qmodel.nbytes + rot_meta_bytes)

    @property
    def packed_weight_bytes(self) -> int:
        return int(self.qmodel.nbytes)

    def forward_dequant(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        was_1d = x_arr.ndim == 1
        h = x_arr.reshape(1, -1) if was_1d else x_arr
        for idx, (layer, spec) in enumerate(zip(self.qmodel.layers, self.wht_specs)):
            h_rot = walsh_hadamard_rotate(h, spec)
            w_deq, b_deq = layer.dequantize()
            out = h_rot @ w_deq + b_deq
            if idx < len(self.qmodel.layers) - 1 and self.qmodel.activation == "relu":
                h = np.maximum(out, np.float32(0.0))
            else:
                h = out
        return h.reshape(-1) if was_1d else h

    def forward_int8(self, x: np.ndarray, *, act_bits: int = 8) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        was_1d = x_arr.ndim == 1
        h = x_arr.reshape(1, -1) if was_1d else x_arr
        clips = self.qmodel.calibrated_act_clip_max
        for idx, (qlayer, spec) in enumerate(zip(self.qmodel.layers, self.wht_specs)):
            h_rot = walsh_hadamard_rotate(h, spec)
            c_max = clips[idx] if (clips is not None and idx < len(clips)) else None
            h_q, s_h = _quantize_activations_int8(h_rot, act_bits=act_bits, act_clip_max=c_max)
            w_q = qlayer.unpack_codes()
            k_max = max_safe_int32_fan_in(
                act_bits=act_bits,
                weight_bits=qlayer.bits,
                asymmetric=(qlayer.zero_point is not None),
                symmetric_full=(qlayer.mode == "symmetric_full"),
            )
            acc = _tiled_int32_matmul(h_q, w_q, qlayer.zero_point, k_max=k_max)
            out = (
                acc.astype(np.float64) * (s_h.astype(np.float64) * qlayer.scale.astype(np.float64))
                + qlayer.bias.astype(np.float64)
            ).astype(np.float32)
            if idx < len(self.qmodel.layers) - 1 and self.qmodel.activation == "relu":
                h = np.maximum(out, np.float32(0.0))
            else:
                h = out
        return h.reshape(-1) if was_1d else h


def quantize_model_wht_gptq(
    layers: list[tuple[np.ndarray, np.ndarray]],
    bits: int = 4,
    *,
    X_cal: np.ndarray,
    activation: str = "relu",
    mode: QuantMode = "symmetric_full",
    damp_percent: float = 0.01,
    block_size: int = 32,
    calibrate_static_act_clips: bool = True,
    seed: int = 0,
) -> WHTQuantizedModel:
    """Quantize an MLP using per-layer Walsh-Hadamard outlier rotation + Cholesky GPTQ compensation."""
    h_cal = np.asarray(X_cal, dtype=np.float32)
    if h_cal.ndim == 1:
        h_cal = h_cal.reshape(1, -1)

    q_layers: list[QuantizedLayer] = []
    specs: list[WHTSpec] = []
    act_clips: list[float] = []

    for idx, (W, b) in enumerate(layers):
        W_f32 = np.asarray(W, dtype=np.float32)
        b_f32 = np.asarray(b, dtype=np.float32)
        spec = build_wht_spec(W_f32.shape[0], X_cal=h_cal, seed=seed + 97 * idx)
        h_rot = walsh_hadamard_rotate(h_cal, spec)
        W_rot = rotate_linear_weights_wht(W_f32, spec)

        if calibrate_static_act_clips and h_rot.size > 0:
            act_clips.append(float(np.percentile(np.abs(h_rot), 99.9) * 1.08))

        hq_cal, sh_cal = _quantize_activations_int8(
            h_rot,
            act_bits=8,
            act_clip_max=act_clips[-1] if act_clips else None,
        )
        x_cal_qdeq = (hq_cal.astype(np.float32) * sh_cal).astype(np.float32)

        q_codes, scale, zp = gptq_quantize_weights(
            W_rot,
            bits=bits,
            X_cal=x_cal_qdeq,
            per_channel=True,
            mode=mode,
            clip_search="activation_mse",
            damp_percent=damp_percent,
            block_size=block_size,
        )
        packed = pack_4bit(q_codes) if bits == 4 else np.ascontiguousarray(q_codes, dtype=np.int8)
        qlayer = QuantizedLayer(
            packed_weights=packed,  # type: ignore[arg-type]
            scale=np.asarray(scale, dtype=np.float32),
            bias=b_f32,
            shape=(W_f32.shape[0], W_f32.shape[1]),
            bits=bits,
            per_channel=True,
            mode=mode,
            zero_point=zp,
        )
        q_layers.append(qlayer)
        specs.append(spec)

        if idx < len(layers) - 1:
            w_deq, b_deq = qlayer.dequantize()
            out_cal = x_cal_qdeq @ w_deq + b_deq
            h_cal = np.maximum(out_cal, np.float32(0.0)) if activation == "relu" else out_cal

    qm = QuantizedModel(
        layers=q_layers,
        activation=activation,
        calibrated_act_clip_max=tuple(act_clips) if act_clips else None,
    )
    return WHTQuantizedModel(qmodel=qm, wht_specs=specs)


# ============================================================================
# 2. 1D Residual Spectral Conv/MLP Backbone (`15k-50k` parameters)
# ============================================================================


def _conv1d_im2col(x: np.ndarray, kernel_size: int, stride: int, pad: int) -> tuple[np.ndarray, int]:
    """Extract `(N, L_out, k * C_in)` patches from `(N, L, C_in)` using zero-copy strided views."""
    n, length, c_in = x.shape
    if pad > 0:
        x_pad = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="constant")
    else:
        x_pad = x
    # sliding_window_view over axis=1 gives shape (N, L_pad - k + 1, C_in, k)
    win = sliding_window_view(x_pad, window_shape=kernel_size, axis=1)[:, ::stride, :, :]
    l_out = int(win.shape[1])
    # Transpose (N, L_out, k, C_in) -> contiguous (N * L_out, k * C_in)
    cols = np.ascontiguousarray(win.transpose(0, 1, 3, 2)).reshape(n * l_out, kernel_size * c_in)
    return cols, l_out


def _conv1d_col2im(
    dcols: np.ndarray,
    x_shape: tuple[int, int, int],
    kernel_size: int,
    stride: int,
    pad: int,
    l_out: int,
) -> np.ndarray:
    """Scatter-accumulate `(N * L_out, k * C_in)` gradients back to `(N, L, C_in)`."""
    n, length, c_in = x_shape
    dx_pad = np.zeros((n, length + 2 * pad, c_in), dtype=np.float32)
    grad_4d = dcols.reshape(n, l_out, kernel_size, c_in)
    for k_idx in range(kernel_size):
        dx_pad[:, k_idx : k_idx + stride * l_out : stride, :] += grad_4d[:, :, k_idx, :]
    if pad > 0:
        return dx_pad[:, pad : pad + length, :]
    return dx_pad


def _adaptive_bins_forward(h: np.ndarray, n_bins: int = 4) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Pool `(N, L, C)` into `mean + max + n_bins` spatial-spectral segments `(N, C * (2 + n_bins))`."""
    n, length, c = h.shape
    mean_feat = np.mean(h, axis=1)
    max_feat = np.max(h, axis=1)
    bounds: list[tuple[int, int]] = []
    bin_feats: list[np.ndarray] = []
    for b_idx in range(n_bins):
        s0 = (b_idx * length) // n_bins
        s1 = max(s0 + 1, ((b_idx + 1) * length) // n_bins)
        s0 = min(s0, length - 1)
        s1 = min(max(s1, s0 + 1), length)
        bounds.append((s0, s1))
        bin_feats.append(np.mean(h[:, s0:s1, :], axis=1))
    return np.hstack([mean_feat, max_feat, *bin_feats]).astype(np.float32), bounds


def _adaptive_bins_backward(
    dpool: np.ndarray,
    h: np.ndarray,
    bounds: list[tuple[int, int]],
) -> np.ndarray:
    """Backpropagate gradient through `mean + max + n_bins` spectral pooling."""
    n, length, c = h.shape
    n_bins = len(bounds)
    dh = np.zeros_like(h, dtype=np.float32)

    d_mean = dpool[:, :c]
    dh += d_mean[:, None, :] / float(length)

    d_max = dpool[:, c : 2 * c]
    argmax_idx = np.argmax(h, axis=1)  # (N, C)
    n_grid = np.arange(n)[:, None]
    c_grid = np.arange(c)[None, :]
    dh[n_grid, argmax_idx, c_grid] += d_max

    for b_idx, (s0, s1) in enumerate(bounds):
        d_bin = dpool[:, (2 + b_idx) * c : (3 + b_idx) * c]
        span = float(max(1, s1 - s0))
        dh[:, s0:s1, :] += d_bin[:, None, :] / span
    return dh


@dataclass
class SpectralResNet1D:
    """1D Residual Spectral Conv/MLP classifier (`15k-50k` parameters) with W8A8 & WHT+GPTQ W4A8."""

    d_in: int
    n_classes: int
    stem_stride: int
    c1: int
    c2: int
    head_dim: int
    n_bins: int
    # Stored as 2D matrices `(k * C_in, C_out)` + 1D bias `(C_out,)` in exact topological order:
    # [0] stem_conv (5*1, c1)
    # [1] b1_conv1  (5*c1, c1)
    # [2] b1_conv2  (3*c1, c1)
    # [3] b2_conv1  (5*c1, c2) stride=2
    # [4] b2_conv2  (3*c2, c2)
    # [5] b2_skip   (1*c1, c2) stride=2
    # [6] fc1       (c2*(2+n_bins), head_dim)
    # [7] fc2       (head_dim, n_classes)
    layers: list[tuple[np.ndarray, np.ndarray]]

    @property
    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in self.layers))

    @property
    def fp32_bytes(self) -> int:
        return fp32_bytes(self.layers)

    def forward(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float32)
        was_1d = x_arr.ndim == 1
        x2d = x_arr.reshape(1, -1) if was_1d else x_arr
        n = x2d.shape[0]
        h0_in = x2d[:, :, None]

        # Stem
        col0, l0 = _conv1d_im2col(h0_in, 5, self.stem_stride, 2)
        h0 = np.maximum(col0 @ self.layers[0][0] + self.layers[0][1], np.float32(0.0)).reshape(n, l0, self.c1)

        # Residual Block 1
        col1a, l1 = _conv1d_im2col(h0, 5, 1, 2)
        h1a = np.maximum(col1a @ self.layers[1][0] + self.layers[1][1], np.float32(0.0)).reshape(n, l1, self.c1)
        col1b, _ = _conv1d_im2col(h1a, 3, 1, 1)
        z1b = (col1b @ self.layers[2][0] + self.layers[2][1]).reshape(n, l1, self.c1)
        h1 = np.maximum(h0 + z1b, np.float32(0.0))

        # Residual Block 2 (stride-2 downsampling)
        b2_stride = 2 if l1 >= 8 else 1
        col2a, l2 = _conv1d_im2col(h1, 5, b2_stride, 2)
        h2a = np.maximum(col2a @ self.layers[3][0] + self.layers[3][1], np.float32(0.0)).reshape(n, l2, self.c2)
        col2b, _ = _conv1d_im2col(h2a, 3, 1, 1)
        z2b = (col2b @ self.layers[4][0] + self.layers[4][1]).reshape(n, l2, self.c2)
        col2s, _ = _conv1d_im2col(h1, 1, b2_stride, 0)
        z2s = (col2s @ self.layers[5][0] + self.layers[5][1]).reshape(n, l2, self.c2)
        h2 = np.maximum(z2s + z2b, np.float32(0.0))

        # Multi-scale spectral pooling + MLP head
        pooled, _ = _adaptive_bins_forward(h2, self.n_bins)
        h_fc = np.maximum(pooled @ self.layers[6][0] + self.layers[6][1], np.float32(0.0))
        logits = h_fc @ self.layers[7][0] + self.layers[7][1]
        return logits.reshape(-1) if was_1d else logits

    def quantize_wht_gptq(
        self,
        X_cal: np.ndarray,
        *,
        bits: int = 4,
        use_wht: bool = True,
        seed: int = 0,
    ) -> "QuantizedSpectralResNet1D":
        """Quantize all 8 conv/linear operators with optional WHT outlier rotation + GPTQ compensation."""
        x_cal = np.asarray(X_cal, dtype=np.float32)
        if x_cal.ndim == 1:
            x_cal = x_cal.reshape(1, -1)
        n = x_cal.shape[0]

        def _quant_op(w: np.ndarray, b: np.ndarray, cols_cal: np.ndarray, op_idx: int) -> tuple[QuantizedLayer, WHTSpec | None]:
            if cols_cal.shape[0] > 2048:
                rng_sub = np.random.default_rng(seed + op_idx)
                sub = rng_sub.choice(cols_cal.shape[0], size=2048, replace=False)
                cal_sub = cols_cal[sub]
            else:
                cal_sub = cols_cal

            if use_wht:
                spec = build_wht_spec(w.shape[0], X_cal=cal_sub, seed=seed + 31 * op_idx)
                cal_in = walsh_hadamard_rotate(cal_sub, spec)
                w_in = rotate_linear_weights_wht(w, spec)
            else:
                spec = None
                cal_in = cal_sub
                w_in = w

            q_codes, scale, zp = gptq_quantize_weights(
                w_in,
                bits=bits,
                X_cal=cal_in,
                per_channel=True,
                mode="symmetric_full" if bits == 4 else "symmetric",
                clip_search="activation_mse",
            )
            packed = pack_4bit(q_codes) if bits == 4 else np.ascontiguousarray(q_codes, dtype=np.int8)
            ql = QuantizedLayer(
                packed_weights=packed,  # type: ignore[arg-type]
                scale=np.asarray(scale, dtype=np.float32),
                bias=b.astype(np.float32),
                shape=(w.shape[0], w.shape[1]),
                bits=bits,
                per_channel=True,
                mode="symmetric_full" if bits == 4 else "symmetric",
                zero_point=zp,
            )
            return ql, spec

        qlayers: list[QuantizedLayer] = []
        specs: list[WHTSpec | None] = []

        # Collect calibration patch matrices in exact forward order
        col0, l0 = _conv1d_im2col(x_cal[:, :, None], 5, self.stem_stride, 2)
        ql0, sp0 = _quant_op(*self.layers[0], col0, 0)
        qlayers.append(ql0)
        specs.append(sp0)
        h0 = np.maximum(_run_qop_int8(col0, ql0, sp0), np.float32(0.0)).reshape(n, l0, self.c1)

        col1a, l1 = _conv1d_im2col(h0, 5, 1, 2)
        ql1, sp1 = _quant_op(*self.layers[1], col1a, 1)
        qlayers.append(ql1)
        specs.append(sp1)
        h1a = np.maximum(_run_qop_int8(col1a, ql1, sp1), np.float32(0.0)).reshape(n, l1, self.c1)

        col1b, _ = _conv1d_im2col(h1a, 3, 1, 1)
        ql2, sp2 = _quant_op(*self.layers[2], col1b, 2)
        qlayers.append(ql2)
        specs.append(sp2)
        h1 = np.maximum(h0 + _run_qop_int8(col1b, ql2, sp2).reshape(n, l1, self.c1), np.float32(0.0))

        b2_stride = 2 if l1 >= 8 else 1
        col2a, l2 = _conv1d_im2col(h1, 5, b2_stride, 2)
        ql3, sp3 = _quant_op(*self.layers[3], col2a, 3)
        qlayers.append(ql3)
        specs.append(sp3)
        h2a = np.maximum(_run_qop_int8(col2a, ql3, sp3), np.float32(0.0)).reshape(n, l2, self.c2)

        col2b, _ = _conv1d_im2col(h2a, 3, 1, 1)
        ql4, sp4 = _quant_op(*self.layers[4], col2b, 4)
        qlayers.append(ql4)
        specs.append(sp4)

        col2s, _ = _conv1d_im2col(h1, 1, b2_stride, 0)
        ql5, sp5 = _quant_op(*self.layers[5], col2s, 5)
        qlayers.append(ql5)
        specs.append(sp5)

        h2 = np.maximum(
            _run_qop_int8(col2s, ql5, sp5).reshape(n, l2, self.c2)
            + _run_qop_int8(col2b, ql4, sp4).reshape(n, l2, self.c2),
            np.float32(0.0),
        )

        pooled, _ = _adaptive_bins_forward(h2, self.n_bins)
        ql6, sp6 = _quant_op(*self.layers[6], pooled, 6)
        qlayers.append(ql6)
        specs.append(sp6)
        h_fc = np.maximum(_run_qop_int8(pooled, ql6, sp6), np.float32(0.0))

        ql7, sp7 = _quant_op(*self.layers[7], h_fc, 7)
        qlayers.append(ql7)
        specs.append(sp7)

        return QuantizedSpectralResNet1D(
            d_in=self.d_in,
            n_classes=self.n_classes,
            stem_stride=self.stem_stride,
            c1=self.c1,
            c2=self.c2,
            head_dim=self.head_dim,
            n_bins=self.n_bins,
            qlayers=qlayers,
            wht_specs=specs,
        )


def _run_qop_int8(cols: np.ndarray, qlayer: QuantizedLayer, spec: WHTSpec | None, *, act_bits: int = 8) -> np.ndarray:
    in_mat = walsh_hadamard_rotate(cols, spec) if spec is not None else cols
    h_q, s_h = _quantize_activations_int8(in_mat, act_bits=act_bits)
    w_q = qlayer.unpack_codes()
    k_max = max_safe_int32_fan_in(
        act_bits=act_bits,
        weight_bits=qlayer.bits,
        asymmetric=(qlayer.zero_point is not None),
        symmetric_full=(qlayer.mode == "symmetric_full"),
    )
    acc = _tiled_int32_matmul(h_q, w_q, qlayer.zero_point, k_max=k_max)
    return (
        acc.astype(np.float64) * (s_h.astype(np.float64) * qlayer.scale.astype(np.float64))
        + qlayer.bias.astype(np.float64)
    ).astype(np.float32)


@dataclass
class QuantizedSpectralResNet1D:
    """Quantized `int8` / `4-bit` 1D Residual Spectral Conv/MLP model executing integer MACs."""

    d_in: int
    n_classes: int
    stem_stride: int
    c1: int
    c2: int
    head_dim: int
    n_bins: int
    qlayers: list[QuantizedLayer]
    wht_specs: list[WHTSpec | None]

    @property
    def packed_weight_bytes(self) -> int:
        return int(sum(ql.nbytes for ql in self.qlayers))

    @property
    def n_params(self) -> int:
        return int(sum(ql.n_params for ql in self.qlayers))

    def to_quantized_model(self) -> QuantizedModel:
        """Expose underlying 8 packed `QuantizedLayer` tensors for `LFT1` binary serialization."""
        return QuantizedModel(layers=list(self.qlayers), activation="relu")

    def forward_int8(self, X: np.ndarray, *, act_bits: int = 8) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float32)
        was_1d = x_arr.ndim == 1
        x2d = x_arr.reshape(1, -1) if was_1d else x_arr
        n = x2d.shape[0]

        col0, l0 = _conv1d_im2col(x2d[:, :, None], 5, self.stem_stride, 2)
        h0 = np.maximum(_run_qop_int8(col0, self.qlayers[0], self.wht_specs[0], act_bits=act_bits), np.float32(0.0)).reshape(n, l0, self.c1)

        col1a, l1 = _conv1d_im2col(h0, 5, 1, 2)
        h1a = np.maximum(_run_qop_int8(col1a, self.qlayers[1], self.wht_specs[1], act_bits=act_bits), np.float32(0.0)).reshape(n, l1, self.c1)
        col1b, _ = _conv1d_im2col(h1a, 3, 1, 1)
        z1b = _run_qop_int8(col1b, self.qlayers[2], self.wht_specs[2], act_bits=act_bits).reshape(n, l1, self.c1)
        h1 = np.maximum(h0 + z1b, np.float32(0.0))

        b2_stride = 2 if l1 >= 8 else 1
        col2a, l2 = _conv1d_im2col(h1, 5, b2_stride, 2)
        h2a = np.maximum(_run_qop_int8(col2a, self.qlayers[3], self.wht_specs[3], act_bits=act_bits), np.float32(0.0)).reshape(n, l2, self.c2)
        col2b, _ = _conv1d_im2col(h2a, 3, 1, 1)
        z2b = _run_qop_int8(col2b, self.qlayers[4], self.wht_specs[4], act_bits=act_bits).reshape(n, l2, self.c2)
        col2s, _ = _conv1d_im2col(h1, 1, b2_stride, 0)
        z2s = _run_qop_int8(col2s, self.qlayers[5], self.wht_specs[5], act_bits=act_bits).reshape(n, l2, self.c2)
        h2 = np.maximum(z2s + z2b, np.float32(0.0))

        pooled, _ = _adaptive_bins_forward(h2, self.n_bins)
        h_fc = np.maximum(_run_qop_int8(pooled, self.qlayers[6], self.wht_specs[6], act_bits=act_bits), np.float32(0.0))
        logits = _run_qop_int8(h_fc, self.qlayers[7], self.wht_specs[7], act_bits=act_bits)
        return logits.reshape(-1) if was_1d else logits


def train_spectral_resnet1d(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    teacher_logits_tr: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    c1: int = 32,
    c2: int = 48,
    head_dim: int = 48,
    n_bins: int = 4,
    temperature: float = 4.0,
    alpha: float = 0.25,
    lr: float = 2.5e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 90,
    seed: int = 0,
) -> tuple[SpectralResNet1D, TrainingDiagnostics]:
    """Train a `15k-50k` param 1D Residual Spectral Conv/MLP backbone via Adam + optional KL(T^2)+CE."""
    rng = np.random.default_rng(seed)
    X_f = np.asarray(X_tr, dtype=np.float32)
    y_i = np.asarray(y_tr, dtype=np.int64).reshape(-1)
    n_samples, d_in = X_f.shape
    n_classes = int(teacher_logits_tr.shape[1]) if teacher_logits_tr is not None else int(np.max(y_i) + 1)

    use_kl = teacher_logits_tr is not None and alpha < 1.0
    p_teacher = (
        softmax(np.asarray(teacher_logits_tr, dtype=np.float64) / temperature).astype(np.float32)
        if use_kl and teacher_logits_tr is not None
        else None
    )

    stem_stride = max(1, (d_in + 95) // 96)
    pooled_dim = c2 * (2 + n_bins)
    shapes = [
        (5 * 1, c1),
        (5 * c1, c1),
        (3 * c1, c1),
        (5 * c1, c2),
        (3 * c2, c2),
        (1 * c1, c2),
        (pooled_dim, head_dim),
        (head_dim, n_classes),
    ]
    Ws = [
        rng.normal(0.0, np.sqrt(2.0 / s_in), size=(s_in, s_out)).astype(np.float32)
        for s_in, s_out in shapes
    ]
    # Zero-init residual branch second convs slightly scaled down for stable residual learning
    Ws[2] *= np.float32(0.5)
    Ws[4] *= np.float32(0.5)
    bs = [np.zeros(s_out, dtype=np.float32) for _, s_out in shapes]

    mW = [np.zeros_like(w) for w in Ws]
    vW = [np.zeros_like(w) for w in Ws]
    mb = [np.zeros_like(b) for b in bs]
    vb = [np.zeros_like(b) for b in bs]
    beta1, beta2, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)

    step = 0
    best_val_score = -float("inf")
    best_val_acc = -float("inf")
    best_loss = float("inf")
    last_improve_epoch = 0
    best_layers = [(w.copy(), b.copy()) for w, b in zip(Ws, bs)]
    final_loss = float("nan")

    eff_bsz = max(int(batch_size), 192) if n_samples >= 1024 else int(batch_size)
    eff_lr_base = float(lr) * (1.35 if eff_bsz > batch_size else 1.0)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_samples)
        epoch_loss_sum = 0.0
        cos_decay = np.float32(0.5 * (1.0 + np.cos(np.pi * (epoch - 1) / max(1, epochs))))
        eff_lr = np.float32(eff_lr_base) * (np.float32(0.15) + np.float32(0.85) * cos_decay)

        for start in range(0, n_samples, eff_bsz):
            idx_b = perm[start : start + eff_bsz]
            xb = X_f[idx_b]
            yb = y_i[idx_b]
            nb = int(len(idx_b))
            bsz_f = np.float32(nb)

            # Forward pass with saved intermediates
            x3d = xb[:, :, None]
            col0, l0 = _conv1d_im2col(x3d, 5, stem_stride, 2)
            z0 = (col0 @ Ws[0] + bs[0]).reshape(nb, l0, c1)
            h0 = np.maximum(z0, np.float32(0.0))

            col1a, l1 = _conv1d_im2col(h0, 5, 1, 2)
            z1a = (col1a @ Ws[1] + bs[1]).reshape(nb, l1, c1)
            h1a = np.maximum(z1a, np.float32(0.0))
            col1b, _ = _conv1d_im2col(h1a, 3, 1, 1)
            z1b = (col1b @ Ws[2] + bs[2]).reshape(nb, l1, c1)
            pre_h1 = h0 + z1b
            h1 = np.maximum(pre_h1, np.float32(0.0))

            b2_stride = 2 if l1 >= 8 else 1
            col2a, l2 = _conv1d_im2col(h1, 5, b2_stride, 2)
            z2a = (col2a @ Ws[3] + bs[3]).reshape(nb, l2, c2)
            h2a = np.maximum(z2a, np.float32(0.0))
            col2b, _ = _conv1d_im2col(h2a, 3, 1, 1)
            z2b = (col2b @ Ws[4] + bs[4]).reshape(nb, l2, c2)
            col2s, _ = _conv1d_im2col(h1, 1, b2_stride, 0)
            z2s = (col2s @ Ws[5] + bs[5]).reshape(nb, l2, c2)
            pre_h2 = z2s + z2b
            h2 = np.maximum(pre_h2, np.float32(0.0))

            pooled, bounds = _adaptive_bins_forward(h2, n_bins)
            z_fc = pooled @ Ws[6] + bs[6]
            h_fc = np.maximum(z_fc, np.float32(0.0))
            z_out = h_fc @ Ws[7] + bs[7]

            q_hard = softmax(z_out).astype(np.float32)
            grad_ce = q_hard.copy()
            grad_ce[np.arange(nb), yb] -= np.float32(1.0)
            ce_val = float(-np.mean(_log_softmax(z_out)[np.arange(nb), yb]))

            if use_kl and p_teacher is not None and teacher_logits_tr is not None:
                pt_b = p_teacher[idx_b]
                q_soft = softmax(z_out / np.float32(temperature)).astype(np.float32)
                grad_kl = np.float32(temperature) * (q_soft - pt_b)
                dz_out = ((np.float32(1.0 - alpha) * grad_kl) + (np.float32(alpha) * grad_ce)) / bsz_f
                kl_val = kl_divergence_soft_targets(teacher_logits_tr[idx_b], z_out, temperature)
                epoch_loss_sum += ((1.0 - alpha) * kl_val + alpha * ce_val) * float(nb)
            else:
                dz_out = grad_ce / bsz_f
                epoch_loss_sum += ce_val * float(nb)

            # Backward pass across MLP head, adaptive spectral pooling, and residual 1D conv blocks
            gWs: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * 8
            gbs: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * 8

            gWs[7] = h_fc.T @ dz_out + np.float32(weight_decay) * Ws[7]
            gbs[7] = np.sum(dz_out, axis=0)
            dh_fc = dz_out @ Ws[7].T
            dz_fc = dh_fc * (z_fc > 0.0).astype(np.float32)

            gWs[6] = pooled.T @ dz_fc + np.float32(weight_decay) * Ws[6]
            gbs[6] = np.sum(dz_fc, axis=0)
            dpooled = dz_fc @ Ws[6].T

            dh2 = _adaptive_bins_backward(dpooled, h2, bounds)
            dpre_h2 = dh2 * (pre_h2 > 0.0).astype(np.float32)
            dpre_h2_2d = dpre_h2.reshape(nb * l2, c2)

            # Skip branch of Block 2
            gWs[5] = col2s.T @ dpre_h2_2d + np.float32(weight_decay) * Ws[5]
            gbs[5] = np.sum(dpre_h2_2d, axis=0)
            dcol2s = dpre_h2_2d @ Ws[5].T
            dh1_from_skip = _conv1d_col2im(dcol2s, h1.shape, 1, b2_stride, 0, l2)

            # Main branch of Block 2
            gWs[4] = col2b.T @ dpre_h2_2d + np.float32(weight_decay) * Ws[4]
            gbs[4] = np.sum(dpre_h2_2d, axis=0)
            dcol2b = dpre_h2_2d @ Ws[4].T
            dh2a = _conv1d_col2im(dcol2b, h2a.shape, 3, 1, 1, l2)
            dz2a_2d = (dh2a * (z2a > 0.0).astype(np.float32)).reshape(nb * l2, c2)

            gWs[3] = col2a.T @ dz2a_2d + np.float32(weight_decay) * Ws[3]
            gbs[3] = np.sum(dz2a_2d, axis=0)
            dcol2a = dz2a_2d @ Ws[3].T
            dh1 = dh1_from_skip + _conv1d_col2im(dcol2a, h1.shape, 5, b2_stride, 2, l2)

            # Block 1
            dpre_h1 = dh1 * (pre_h1 > 0.0).astype(np.float32)
            dpre_h1_2d = dpre_h1.reshape(nb * l1, c1)
            gWs[2] = col1b.T @ dpre_h1_2d + np.float32(weight_decay) * Ws[2]
            gbs[2] = np.sum(dpre_h1_2d, axis=0)
            dcol1b = dpre_h1_2d @ Ws[2].T
            dh1a = _conv1d_col2im(dcol1b, h1a.shape, 3, 1, 1, l1)
            dz1a_2d = (dh1a * (z1a > 0.0).astype(np.float32)).reshape(nb * l1, c1)

            gWs[1] = col1a.T @ dz1a_2d + np.float32(weight_decay) * Ws[1]
            gbs[1] = np.sum(dz1a_2d, axis=0)
            dcol1a = dz1a_2d @ Ws[1].T
            dh0 = dpre_h1 + _conv1d_col2im(dcol1a, h0.shape, 5, 1, 2, l1)

            # Stem
            dz0_2d = (dh0 * (z0 > 0.0).astype(np.float32)).reshape(nb * l0, c1)
            gWs[0] = col0.T @ dz0_2d + np.float32(weight_decay) * Ws[0]
            gbs[0] = np.sum(dz0_2d, axis=0)

            step += 1
            bc1 = np.float32(1.0 - float(beta1) ** step)
            bc2 = np.float32(1.0 - float(beta2) ** step)
            lr_t = eff_lr * np.sqrt(bc2) / bc1
            for l_i in range(8):
                mW[l_i] = beta1 * mW[l_i] + (np.float32(1.0) - beta1) * gWs[l_i]
                vW[l_i] = beta2 * vW[l_i] + (np.float32(1.0) - beta2) * (gWs[l_i] * gWs[l_i])
                Ws[l_i] -= lr_t * mW[l_i] / (np.sqrt(vW[l_i]) + eps)

                mb[l_i] = beta1 * mb[l_i] + (np.float32(1.0) - beta1) * gbs[l_i]
                vb[l_i] = beta2 * vb[l_i] + (np.float32(1.0) - beta2) * (gbs[l_i] * gbs[l_i])
                bs[l_i] -= lr_t * mb[l_i] / (np.sqrt(vb[l_i]) + eps)

        final_loss = epoch_loss_sum / float(n_samples)
        cur_model = SpectralResNet1D(
            d_in=d_in,
            n_classes=n_classes,
            stem_stride=stem_stride,
            c1=c1,
            c2=c2,
            head_dim=head_dim,
            n_bins=n_bins,
            layers=[(w.copy(), b.copy()) for w, b in zip(Ws, bs)],
        )
        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_logits = cur_model.forward(X_val)
            val_acc = float((val_logits.argmax(axis=1) == y_val).mean())
            val_ce = float(-np.mean(_log_softmax(val_logits)[np.arange(len(y_val)), y_val]))
            score = val_acc - 1e-4 * val_ce
            if val_acc > best_val_acc + 1e-6:
                best_val_acc = val_acc
                last_improve_epoch = epoch
            if score >= best_val_score:
                best_val_score = score
                best_layers = cur_model.layers
        else:
            if final_loss < best_loss - 1e-3:
                best_loss = final_loss
                last_improve_epoch = epoch
            best_layers = cur_model.layers

    stagnation_k = max(3, min(15, epochs // 8))
    converged = bool(epochs > stagnation_k and (epochs - last_improve_epoch) >= stagnation_k)
    trained = SpectralResNet1D(
        d_in=d_in,
        n_classes=n_classes,
        stem_stride=stem_stride,
        c1=c1,
        c2=c2,
        head_dim=head_dim,
        n_bins=n_bins,
        layers=best_layers,
    )
    return trained, TrainingDiagnostics(
        n_iter=epochs,
        final_loss=float(final_loss),
        converged=converged,
    )


# ============================================================================
# 3. OOD / Drift Abstainers, AURC, Shrinkage Mahalanobis, TTN & Conformal Risk
# ============================================================================


def msp_uncertainty_score(logits: np.ndarray) -> np.ndarray:
    """1 - Max Softmax Probability (`1 - max_c softmax(z)_c`, Hendrycks & Gimpel, ICLR 2017).

    Higher values denote greater predictive uncertainty (abstain when `score > threshold`).
    """
    probs = softmax(np.asarray(logits, dtype=np.float64))
    max_prob = np.max(probs, axis=-1)
    return (1.0 - max_prob).astype(np.float64)


def energy_ood_score(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Helmholtz free energy score `-T * logsumexp(logits / T)` (Liu et al., NeurIPS 2020).

    Higher (less negative) energy indicates lower log-partition density / higher OOD risk.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z_max = np.max(z, axis=-1, keepdims=True)
    lse = np.squeeze(z_max, axis=-1) + np.log(np.sum(np.exp(z - z_max), axis=-1))
    return (-float(temperature) * lse).astype(np.float64)


def compute_aurc(uncertainty_scores: np.ndarray, correct_mask: np.ndarray) -> float:
    """Compute exact Area Under the Risk-Coverage Curve (`AURC` in `[0, 1]`, lower is better)."""
    u = np.asarray(uncertainty_scores, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct_mask, dtype=bool).reshape(-1)
    if u.size == 0:
        return 0.0
    if u.size != corr.size:
        raise ValueError(f"Length mismatch: scores={u.size}, correct_mask={corr.size}.")
    # Sort from lowest uncertainty (most confident) to highest uncertainty; stable tie-break
    order = np.argsort(u, kind="mergesort")
    errors_sorted = (~corr[order]).astype(np.float64)
    cum_errors = np.cumsum(errors_sorted)
    coverages_k = np.arange(1, u.size + 1, dtype=np.float64)
    selective_risks = cum_errors / coverages_k
    return float(np.mean(selective_risks))


def calibrate_matched_abstain_threshold(
    cal_uncertainty: np.ndarray,
    target_abstain_rate: float,
) -> float:
    """Choose threshold `tau` so `mean(cal_uncertainty > tau)` matches `target_abstain_rate`."""
    u = np.asarray(cal_uncertainty, dtype=np.float64).reshape(-1)
    if u.size == 0:
        return float("inf")
    r = float(np.clip(target_abstain_rate, 0.0, 1.0))
    if r <= 0.0:
        return float(np.max(u) + 1e-9)
    if r >= 1.0:
        return float(np.min(u) - 1e-9)
    n = u.size
    k_abstain = int(round(r * n))
    if k_abstain <= 0:
        return float(np.max(u) + 1e-9)
    if k_abstain >= n:
        return float(np.min(u) - 1e-9)
    sorted_u = np.sort(u)
    idx_keep_last = n - k_abstain - 1
    # Place threshold midpoint between last accepted and first abstained calibration score
    return float(0.5 * (sorted_u[idx_keep_last] + sorted_u[idx_keep_last + 1]))


def calibrate_conformal_quantile(
    cal_uncertainty: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Finite-sample Split Conformal / CRC threshold at level `1 - alpha` (Angelopoulos et al., 2024)."""
    u = np.asarray(cal_uncertainty, dtype=np.float64).reshape(-1)
    if u.size == 0:
        return float("inf")
    a = float(np.clip(alpha, 0.0, 1.0))
    n = u.size
    rank = int(np.ceil((n + 1) * (1.0 - a))) - 1
    if rank >= n:
        return float(np.max(u) + 1e-9)
    if rank < 0:
        return float(np.min(u) - 1e-9)
    return float(np.sort(u)[rank])


@dataclass
class ShrinkageMahalanobisDetector:
    """Ledoit-Wolf / ridge-shrunk class-conditional Gaussian Mahalanobis OOD detector."""

    class_means: np.ndarray
    precision: np.ndarray
    shrinkage: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, n_classes: int) -> "ShrinkageMahalanobisDetector":
        x64 = np.asarray(X, dtype=np.float64)
        y_i = np.asarray(y, dtype=np.int64).reshape(-1)
        n, d = x64.shape
        means = np.zeros((n_classes, d), dtype=np.float64)
        global_mean = np.mean(x64, axis=0)
        centered = np.empty_like(x64)

        for c in range(n_classes):
            mask = y_i == c
            if np.any(mask):
                means[c] = np.mean(x64[mask], axis=0)
                centered[mask] = x64[mask] - means[c]
            else:
                means[c] = global_mean

        # Analytical Ledoit-Wolf shrinkage intensity toward scaled identity (tr(S)/d) * I
        emp_cov = (centered.T @ centered) / max(1.0, float(n))
        mu_trace = float(np.trace(emp_cov)) / max(1.0, float(d))
        target = mu_trace * np.eye(d, dtype=np.float64)
        delta_sq = float(np.sum(np.square(emp_cov - target)))
        if delta_sq <= 1e-15 or n <= 1:
            lam = 1.0
        else:
            x_sq = np.square(centered)
            sum_row_gram_sq = float(np.sum(np.square(x_sq.T @ x_sq))) / (float(n) ** 2)
            beta_hat = max(0.0, (sum_row_gram_sq - float(np.sum(np.square(emp_cov))) / float(n)))
            lam = float(np.clip(beta_hat / delta_sq, 0.05, 1.0))

        shrunk_cov = (1.0 - lam) * emp_cov + lam * target + 1e-6 * np.eye(d, dtype=np.float64)
        prec = np.linalg.pinv(shrunk_cov, hermitian=True)
        return cls(class_means=means, precision=prec, shrinkage=lam)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute minimum squared Mahalanobis distance `min_c (x - mu_c)^T Sigma^-1 (x - mu_c)`."""
        x64 = np.asarray(X, dtype=np.float64)
        squeeze = x64.ndim == 1
        if squeeze:
            x64 = x64.reshape(1, -1)
        # diff: (N, C, d)
        diff = x64[:, None, :] - self.class_means[None, :, :]
        proj = np.einsum("ncd,de->nce", diff, self.precision, optimize=True)
        d2 = np.maximum(0.0, np.sum(proj * diff, axis=-1))
        min_d2 = np.min(d2, axis=1)
        return min_d2[0] if squeeze else min_d2


def test_time_batch_normalize(
    X_batch: np.ndarray,
    *,
    source_mean: np.ndarray | None = None,
    source_std: np.ndarray | None = None,
    prior_strength: float = 16.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """Unlabeled Test-Time Normalization (`TTN` / AdaBN) with empirical-Bayes small-batch shrinkage.

    Re-aligns drifted sensor batch statistics toward the source calibration distribution `(source_mean,
    source_std)` using ONLY unlabeled features `X_batch` (`w = n / (n + prior_strength)`).
    """
    x64 = np.asarray(X_batch, dtype=np.float64)
    squeeze = x64.ndim == 1
    if squeeze:
        x64 = x64.reshape(1, -1)
    n, d = x64.shape
    src_mu = np.zeros(d, dtype=np.float64) if source_mean is None else np.asarray(source_mean, dtype=np.float64).reshape(d)
    src_sd = np.ones(d, dtype=np.float64) if source_std is None else np.maximum(np.asarray(source_std, dtype=np.float64).reshape(d), eps)

    if n < 2:
        return x64[0].astype(np.float32) if squeeze else x64.astype(np.float32)

    w = float(n) / (float(n) + max(0.0, float(prior_strength)))
    b_mu = np.mean(x64, axis=0)
    b_sd = np.std(x64, axis=0)
    b_sd = np.where(b_sd > eps, b_sd, src_sd)

    eff_mu = w * b_mu + (1.0 - w) * src_mu
    eff_sd = np.maximum(w * b_sd + (1.0 - w) * src_sd, eps)
    aligned = ((x64 - eff_mu[None, :]) / eff_sd[None, :]) * src_sd[None, :] + src_mu[None, :]
    out = aligned.astype(np.float32)
    return out[0] if squeeze else out


__all__ = [
    "QuantizedSpectralResNet1D",
    "ShrinkageMahalanobisDetector",
    "SpectralResNet1D",
    "WHTQuantizedModel",
    "WHTSpec",
    "build_wht_spec",
    "calibrate_conformal_quantile",
    "calibrate_matched_abstain_threshold",
    "compute_aurc",
    "energy_ood_score",
    "gptq_quantize_weights",
    "msp_uncertainty_score",
    "quantize_model_wht_gptq",
    "rotate_linear_weights_wht",
    "test_time_batch_normalize",
    "train_spectral_resnet1d",
    "walsh_hadamard_rotate",
    "walsh_hadamard_unrotate",
]
