"""SO(3)-equivariant scalar irrep weight quantization for L3 YLM molecular force heads."""

from __future__ import annotations

import numpy as np

from .quantize import QuantizedLayer, pack_4bit, quantize


def guard_equivariant_spatial_quantization(
    W: np.ndarray,
    *,
    irrep_l: int,
    expected_channel_shape: tuple[int, int] | None = None,
) -> None:
    """Reject naive unstructured elementwise quantization across spherical-harmonic (2*l + 1) axes.

    For l >= 1 irreps (e.g., l=1 Cartesian force/dipole vectors in R^3 or l=2 quadrupoles in R^5),
    applying independent elementwise rounding or per-Cartesian-column scaling to a flattened
    `(C_in * (2l+1), C_out * (2l+1))` matrix breaks exact SO(3) rotation equivariance `f(x R^T) = f(x) R^T`.
    Only invariant scalar channel/radial path weights `W_invariant` of shape `(C_in, C_out)` commute
    with Wigner-D matrices `D^(l)(R)`.
    """
    if irrep_l < 0:
        raise ValueError(f"Spherical harmonic degree irrep_l must be >= 0, got {irrep_l}.")
    W_arr = np.asarray(W)
    if W_arr.ndim != 2:
        raise ValueError(
            f"Equivariant channel weight matrix must be 2D `(C_in, C_out)`, got shape {W_arr.shape} "
            "(do not pass 3D/4D spatial basis tensors directly to elementwise rounding)."
        )
    if expected_channel_shape is not None and W_arr.shape != tuple(expected_channel_shape):
        spatial_dim = 2 * irrep_l + 1
        c_in, c_out = expected_channel_shape
        if W_arr.shape == (c_in * spatial_dim, c_out * spatial_dim) and irrep_l >= 1:
            raise ValueError(
                f"Refusing to quantize flattened spatial matrix of shape {W_arr.shape} for l={irrep_l} "
                f"(2l+1={spatial_dim}): elementwise rounding across spatial components breaks SO(3) "
                f"rotation equivariance. Pass invariant channel weights of shape {(c_in, c_out)} instead."
            )
        raise ValueError(
            f"Weight shape {W_arr.shape} does not match expected_channel_shape={expected_channel_shape}."
        )


def quantize_equivariant_irrep_weights(
    W_invariant: np.ndarray,
    bits: int = 8,
    *,
    irrep_l: int = 1,
    per_channel: bool = True,
    expected_channel_shape: tuple[int, int] | None = None,
) -> QuantizedLayer:
    """Quantize invariant scalar channel-mixing weights `W_invariant` `(C_in, C_out)` for irrep `l`.

    Because biases on `l >= 1` non-scalar irreps would break rotational equivariance (`0 * R != b`),
    `QuantizedLayer.bias` is enforced to exact zero (`np.zeros(C_out, dtype=np.float32)`).
    """
    guard_equivariant_spatial_quantization(
        W_invariant,
        irrep_l=irrep_l,
        expected_channel_shape=expected_channel_shape,
    )
    W_f32 = np.asarray(W_invariant, dtype=np.float32)
    q_codes, scale = quantize(W_f32, bits=bits, per_channel=per_channel, mode="symmetric")  # type: ignore[misc]
    packed = pack_4bit(q_codes) if bits == 4 else np.ascontiguousarray(q_codes, dtype=np.int8)
    zero_bias = np.zeros(W_f32.shape[1], dtype=np.float32)
    return QuantizedLayer(
        packed_weights=packed,
        scale=np.asarray(scale, dtype=np.float32),
        bias=zero_bias,
        shape=(W_f32.shape[0], W_f32.shape[1]),
        bits=bits,
        per_channel=per_channel,
        mode="symmetric",
        zero_point=None,
    )


def equivariant_irrep_linear(X_irrep: np.ndarray, qlayer_or_W: QuantizedLayer | np.ndarray) -> np.ndarray:
    """Apply equivariant channel mixing `Y_{n,o,m} = sum_c X_{n,c,m} * W_{c,o}` across `(N, C_in, 2l+1)`."""
    X_f32 = np.asarray(X_irrep, dtype=np.float32)
    if X_f32.ndim != 3:
        raise ValueError(f"Expected X_irrep of shape (N, C_in, 2l+1), got {X_f32.shape}.")
    if isinstance(qlayer_or_W, QuantizedLayer):
        W_deq, _ = qlayer_or_W.dequantize()
    else:
        W_deq = np.asarray(qlayer_or_W, dtype=np.float32)
    return np.einsum("ncm,co->nom", X_f32, W_deq, optimize=True).astype(np.float32)


def random_so3_rotation(seed_or_rng: int | np.random.Generator = 0) -> np.ndarray:
    """Sample a proper 3D rotation matrix `R in SO(3)` (`R^T R = I`, `det(R) = +1`)."""
    rng = (
        seed_or_rng
        if isinstance(seed_or_rng, np.random.Generator)
        else np.random.default_rng(seed_or_rng)
    )
    A = rng.normal(size=(3, 3))
    Q, R_tri = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R_tri)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.astype(np.float32)


def max_so3_equivariance_error(
    X_vec: np.ndarray,
    qlayer_or_W: QuantizedLayer | np.ndarray,
    R: np.ndarray,
) -> float:
    """Measure `max |f(X @ R^T) - f(X) @ R^T|` on l=1 3D vector features `(N, C_in, 3)`."""
    R_f32 = np.asarray(R, dtype=np.float32)
    lhs = equivariant_irrep_linear(X_vec @ R_f32.T, qlayer_or_W)
    rhs = equivariant_irrep_linear(X_vec, qlayer_or_W) @ R_f32.T
    return float(np.max(np.abs(lhs - rhs)))
