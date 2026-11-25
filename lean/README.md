# `LabFeetLean` -- Formal L0 Edge Quantization, Nibble Round-Trip, Saturating Requantization & 5-Gate Monitor Proofs

`LabFeetLean` (`leanprover/lean4:v4.36.0`, zero external dependencies, `0 sorry`) machine-checks the quantization scale, two's-complement 4-bit nibble packing, widening saturating requantization, `int32` MAC accumulator bounds, edge SRAM budget, and 5-gate low-bit drift abstention invariants of `lab-feet`:

- **Two's-Complement `4-Bit` Nibble Packing (`packInt4Pair`, `unpackInt4Lo`, `unpackInt4Hi`, `packInt4Single`)**: Proves exact bitwise round-trip over `[-8, 7]` (`unpackInt4_packInt4_id`), canonical zero high-nibble padding on odd-length arrays (`packInt4Single_zero_high_nibble`), and `[-8, 7]` output bounds (`unpackInt4_in_range`).
- **Widening Fixed-Point Saturating Requantization (`requantSaturate`, `requantSaturateTiesToEven`)**: Proves output saturation bounds `[qmin, qmax]` (`requantSaturate_in_bounds`), rounding-mode parity `|roundHalfUp - roundTiesToEven| <= 1` (`requantSaturate_rounding_diff_le_one`), and `int32` MAC non-overflow (`int32_mac_no_overflow`).
- **Core Edge Student Gate (`checkLabFeet`, `checkLabFeet_safe_for_edge`)**: Verifies bit-width `[2, 8]` (`checkLabFeet_valid_bitwidth`), strictly positive quantization scale `0 < scaleMicro`, SRAM footprint `packedWeightBytes <= maxEdgeSramBytes`, and teacher-student agreement floor.
- **Signed 5-Gate Quantized OOD & Clipping Monitor (`checkDriftAbstainingEdge`, `decideDriftAbstainingInt4`)**: Proves `< 512 B` extra SRAM overhead (`driftMonitor_extra_sram_under_512_bytes`), mandatory `safeHold` across all 5 runtime triggers (`L1` Mahalanobis OOD, `int4` hidden saturation, `int8` input ADC rail clips, low energy margin, and signed `peakLogit < minPeakLogit`), and constructive soundness (`decideDriftAbstainingInt4_sound`).

## Build

```bash
cd lean
lake build
```
