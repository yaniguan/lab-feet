# `LabFeetLean` -- Formal L0 Edge Quantization & OOD/Saturation Abstention Proofs

`LabFeetLean` (`leanprover/lean4:v4.36.0`, zero external dependencies) machine-checks the quantization scale, edge SRAM budget, teacher-student fidelity, and low-bit drift abstention invariants of `lab-feet`:

- **Core Edge Student Gate (`checkLabFeet`, `checkLabFeet_safe_for_edge`)**: Verifies bit-width `[2, 8]` (`checkLabFeet_valid_bitwidth`), strictly positive quantization scale `0 < scaleMicro`, SRAM footprint `packedWeightBytes <= maxEdgeSramBytes`, and teacher-student agreement floor.
- **Quantized Activation-Energy OOD & Clipping Monitor (`checkDriftAbstainingEdge`)**: Proves `< 512 B` extra SRAM overhead (`driftMonitor_extra_sram_under_512_bytes`), mandatory `safeHold` abstention under `L1` feature-Mahalanobis drift (`driftMonitor_safe_hold_on_l1_ood`) or `int4` saturation clipping (`driftMonitor_safe_hold_on_int4_saturation`), and constructive soundness of `decideDriftAbstainingInt4` (`decideDriftAbstainingInt4_sound`).

## Build

```bash
cd lean
lake build
```
