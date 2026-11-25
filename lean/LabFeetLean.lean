/-!
# LabFeetLean: L0 Edge Quantization, Two's-Complement Nibble Packing, Saturating Requantization & 5-Gate OOD Proofs

Formalizes the `lab-feet` (Layer L0) edge distillation and low-bit quantization guardrails:
1. **Bit-Width, Scale Positivity & `int32` MAC Bounds**: `bitWidth in [2, 8]`, `0 < scaleMicro`,
   and `fanIn * (actQmax * weightQmax) <= 2147483647` preventing 32-bit signed MAC wraparound.
2. **Two's-Complement `4-Bit` Nibble Packing (`[-8, 7]`)**: Bit-exact `UInt8` packing/unpacking
   (`packInt4Pair`, `unpackInt4Lo`, `unpackInt4Hi`, `packInt4Single`) with zero-`sorry` round-trip
   and odd-length high-nibble zero-padding theorems.
3. **Widening Fixed-Point Saturating Requantization**: Formalizes `roundHalfUpShift` (`+ 2^(shift-1)`)
   and `roundTiesToEvenShift`, proving `qmin <= requantSaturate <= qmax` and bounded rounding-mode
   discrepancy `|roundHalfUp - roundTiesToEven| <= 1`.
4. **Signed 5-Gate Quantized OOD & Clipping Monitor (Idea #20)**: Verifies `< 512 B` extra SRAM and
   proves mandatory `safeHold` abstention across all 5 runtime triggers (`L1` Mahalanobis OOD,
   `int4` hidden saturation, `int8` ADC input rail clips, low energy margin, and signed `peakLogit < minPeakLogit`).
-/
namespace LabFeetLean

/-- Structured boundary violation reporting repository, rule identifier, and detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

/-- Return the first violation in a list of `(holds, violation)` checks. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, _⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## 1. Two's-Complement 4-Bit Nibble Packing & Odd-Length Zero Padding -/

/-- Encode signed 4-bit integer `q ∈ [-8, 7]` as its 4-bit two's-complement nibble in `0..15`. -/
def encodeInt4Nibble (q : Int) : UInt8 :=
  UInt8.ofNat (q % 16).toNat

/-- Decode low 4 bits of `u` from two's-complement nibble `0..15` back into signed `[-8, 7]`. -/
def decodeInt4Nibble (u : UInt8) : Int :=
  let raw : Int := Int.ofNat (u &&& (0x0F : UInt8)).toNat
  if raw ≥ 8 then raw - 16 else raw

/-- Pack two signed 4-bit codes `q0, q1 ∈ [-8, 7]` into one `UInt8` (`q0` low nibble, `q1` high nibble). -/
def packInt4Pair (q0 q1 : Int) : UInt8 :=
  (encodeInt4Nibble q0 &&& (0x0F : UInt8)) ||| ((encodeInt4Nibble q1 &&& (0x0F : UInt8)) <<< (4 : UInt8))

/-- Pack a single trailing odd-element 4-bit code `q0` with canonical `0` high-nibble padding. -/
def packInt4Single (q0 : Int) : UInt8 :=
  packInt4Pair q0 0

/-- Unpack low signed 4-bit code (`bits 0..3`) from a packed `UInt8` byte. -/
def unpackInt4Lo (b : UInt8) : Int :=
  decodeInt4Nibble (b &&& (0x0F : UInt8))

/-- Unpack high signed 4-bit code (`bits 4..7`) from a packed `UInt8` byte. -/
def unpackInt4Hi (b : UInt8) : Int :=
  decodeInt4Nibble ((b >>> (4 : UInt8)) &&& (0x0F : UInt8))

private theorem packInt4_fin16_roundtrip :
    ∀ i : Fin 16, ∀ j : Fin 16,
      let q0 : Int := (i.val : Int) - 8
      let q1 : Int := (j.val : Int) - 8
      unpackInt4Lo (packInt4Pair q0 q1) = q0 ∧
      unpackInt4Hi (packInt4Pair q0 q1) = q1 := by
  decide

/-- Exact Nibble Round-Trip Theorem: Any signed 4-bit pair `q0, q1 ∈ [-8, 7]` packs into one `UInt8`
    and unpacks without loss via `unpackInt4Lo` and `unpackInt4Hi`. -/
theorem unpackInt4_packInt4_id (q0 q1 : Int)
    (h0 : -8 ≤ q0 ∧ q0 ≤ 7) (h1 : -8 ≤ q1 ∧ q1 ≤ 7) :
    unpackInt4Lo (packInt4Pair q0 q1) = q0 ∧
    unpackInt4Hi (packInt4Pair q0 q1) = q1 := by
  let i : Fin 16 := ⟨(q0 + 8).toNat, by omega⟩
  let j : Fin 16 := ⟨(q1 + 8).toNat, by omega⟩
  have hi : (i.val : Int) - 8 = q0 := by dsimp [i]; omega
  have hj : (j.val : Int) - 8 = q1 := by dsimp [j]; omega
  have hfin := packInt4_fin16_roundtrip i j
  dsimp at hfin
  rw [hi, hj] at hfin
  exact hfin

private theorem packInt4Single_fin16 :
    ∀ i : Fin 16,
      let q0 : Int := (i.val : Int) - 8
      ((packInt4Single q0 >>> (4 : UInt8)) &&& (0x0F : UInt8)) = (0 : UInt8) ∧
      unpackInt4Lo (packInt4Single q0) = q0 := by
  decide

/-- Odd-Element Canonical Zero-Padding Theorem: Packing a trailing odd element `q0 ∈ [-8, 7]`
    guarantees a zero high nibble `((b >>> 4) &&& 0x0F) = 0` and exact low-nibble recovery. -/
theorem packInt4Single_zero_high_nibble (q0 : Int)
    (h0 : -8 ≤ q0 ∧ q0 ≤ 7) :
    ((packInt4Single q0 >>> (4 : UInt8)) &&& (0x0F : UInt8)) = (0 : UInt8) ∧
    unpackInt4Lo (packInt4Single q0) = q0 := by
  let i : Fin 16 := ⟨(q0 + 8).toNat, by omega⟩
  have hi : (i.val : Int) - 8 = q0 := by dsimp [i]; omega
  have hfin := packInt4Single_fin16 i
  dsimp at hfin
  rw [hi] at hfin
  exact hfin

private theorem decodeInt4Nibble_bounds (u : UInt8) :
    -8 ≤ decodeInt4Nibble u ∧ decodeInt4Nibble u ≤ 7 := by
  unfold decodeInt4Nibble
  have hlt : (u &&& (0x0F : UInt8)).toNat < 16 := by
    change u.toNat &&& 15 < 2 ^ 4
    exact Nat.and_lt_two_pow u.toNat (by decide)
  dsimp
  omega

/-- Unpacked Range Theorem: Unpacking any `UInt8` byte always yields signed `[-8, 7]` codes. -/
theorem unpackInt4_in_range (b : UInt8) :
    (-8 ≤ unpackInt4Lo b ∧ unpackInt4Lo b ≤ 7) ∧
    (-8 ≤ unpackInt4Hi b ∧ unpackInt4Hi b ≤ 7) :=
  ⟨decodeInt4Nibble_bounds (b &&& (0x0F : UInt8)),
   decodeInt4Nibble_bounds ((b >>> (4 : UInt8)) &&& (0x0F : UInt8))⟩

/-! ## 2. Widening Fixed-Point Saturating Requantization & Rounding-Mode Parity -/

/-- Clamp signed integer `x` into closed interval `[qmin, qmax]`. -/
def clampInt (x qmin qmax : Int) : Int :=
  max qmin (min qmax x)

theorem clampInt_bounds (x qmin qmax : Int) (h : qmin ≤ qmax) :
    qmin ≤ clampInt x qmin qmax ∧ clampInt x qmin qmax ≤ qmax := by
  unfold clampInt
  omega

/-- Nearest-integer right-shift using round-half-up (`+ 2^(shift - 1)` before Euclidean/floor shift). -/
def roundHalfUpShift (prod : Int) (shift : Nat) : Int :=
  match shift with
  | 0 => prod
  | k + 1 => (prod + (2 ^ k : Nat)) / (2 ^ (k + 1) : Nat)

/-- Nearest-integer right-shift using IEEE-754 round-ties-to-even. -/
def roundTiesToEvenShift (prod : Int) (shift : Nat) : Int :=
  match shift with
  | 0 => prod
  | k + 1 =>
    let d : Int := (2 ^ (k + 1) : Nat)
    let half : Int := (2 ^ k : Nat)
    let qFloor : Int := prod / d
    let rem : Int := prod % d
    if rem < half then
      qFloor
    else if half < rem then
      qFloor + 1
    else if qFloor % 2 == 0 then
      qFloor
    else
      qFloor + 1

/-- Widening fixed-point multiply + round-half-up shift + `[qmin, qmax]` saturation. -/
def requantSaturate (acc : Int) (mult shift : Nat) (qmin qmax : Int) : Int :=
  clampInt (roundHalfUpShift (acc * (mult : Int)) shift) qmin qmax

/-- Widening fixed-point multiply + round-ties-to-even shift + `[qmin, qmax]` saturation. -/
def requantSaturateTiesToEven (acc : Int) (mult shift : Nat) (qmin qmax : Int) : Int :=
  clampInt (roundTiesToEvenShift (acc * (mult : Int)) shift) qmin qmax

/-- Saturation Range Theorem: `requantSaturate` always lies within `[qmin, qmax]`. -/
theorem requantSaturate_in_bounds (acc : Int) (mult shift : Nat) (qmin qmax : Int)
    (h : qmin ≤ qmax) :
    qmin ≤ requantSaturate acc mult shift qmin qmax ∧
    requantSaturate acc mult shift qmin qmax ≤ qmax :=
  clampInt_bounds _ qmin qmax h

/-- Saturation Range Theorem (Ties-to-Even): `requantSaturateTiesToEven` lies within `[qmin, qmax]`. -/
theorem requantSaturateTiesToEven_in_bounds (acc : Int) (mult shift : Nat) (qmin qmax : Int)
    (h : qmin ≤ qmax) :
    qmin ≤ requantSaturateTiesToEven acc mult shift qmin qmax ∧
    requantSaturateTiesToEven acc mult shift qmin qmax ≤ qmax :=
  clampInt_bounds _ qmin qmax h

private theorem two_pow_pos_int (k : Nat) : (0 : Int) < (2 ^ k : Nat) := by
  have hp : 0 < 2 ^ k := Nat.two_pow_pos k
  omega

private theorem two_pow_succ_int (k : Nat) :
    ((2 ^ (k + 1) : Nat) : Int) = 2 * ((2 ^ k : Nat) : Int) := by
  have h : 2 ^ (k + 1) = 2 * 2 ^ k := by
    rw [Nat.pow_succ]
    omega
  omega

/-- Exact Unclamped Rounding-Mode Difference Theorem: `roundHalfUpShift` and `roundTiesToEvenShift`
    agree identically off halfway ties and differ by at most `1` LSB everywhere. -/
theorem roundHalfUp_sub_roundTiesToEven_bounds (prod : Int) (shift : Nat) :
    0 ≤ roundHalfUpShift prod shift - roundTiesToEvenShift prod shift ∧
    roundHalfUpShift prod shift - roundTiesToEvenShift prod shift ≤ 1 := by
  cases shift with
  | zero =>
    simp [roundHalfUpShift, roundTiesToEvenShift]
  | succ k =>
    simp only [roundHalfUpShift, roundTiesToEvenShift]
    let half : Int := ((2 ^ k : Nat) : Int)
    let d : Int := ((2 ^ (k + 1) : Nat) : Int)
    change 0 ≤ (prod + half) / d -
        (if prod % d < half then prod / d
         else if half < prod % d then prod / d + 1
         else if (prod / d % 2 == 0) = true then prod / d else prod / d + 1) ∧
      (prod + half) / d -
        (if prod % d < half then prod / d
         else if half < prod % d then prod / d + 1
         else if (prod / d % 2 == 0) = true then prod / d else prod / d + 1) ≤ 1
    have hhalf_pos : 0 < half := two_pow_pos_int k
    have hd_eq : d = 2 * half := two_pow_succ_int k
    have hd_pos : 0 < d := by omega
    have hd_ne : d ≠ 0 := by omega
    have hdivmod : d * (prod / d) + prod % d = prod := Int.mul_ediv_add_emod prod d
    have hrem_nonneg : 0 ≤ prod % d := Int.emod_nonneg prod hd_ne
    have hrem_lt : prod % d < d := Int.emod_lt_of_pos prod hd_pos
    by_cases hlt : prod % d < half
    · have hsum_eq : prod + half = (prod % d + half) + d * (prod / d) := by omega
      have hsmall : (prod % d + half) / d = 0 :=
        Int.ediv_eq_zero_of_lt (by omega) (by omega)
      have hhu : (prod + half) / d = prod / d := by
        rw [hsum_eq, Int.add_mul_ediv_left _ _ hd_ne, hsmall]
        omega
      simp [hlt, hhu]
    · have hsum_eq : prod + half = (prod % d + half - d) + d * (prod / d + 1) := by
        have h1 : d * (prod / d + 1) = d * (prod / d) + d := by
          rw [Int.mul_add, Int.mul_one]
        omega
      have hsmall : (prod % d + half - d) / d = 0 :=
        Int.ediv_eq_zero_of_lt (by omega) (by omega)
      have hhu : (prod + half) / d = prod / d + 1 := by
        rw [hsum_eq, Int.add_mul_ediv_left _ _ hd_ne, hsmall]
        omega
      by_cases hgt : half < prod % d
      · simp [hlt, hgt, hhu]
      · by_cases heven : (prod / d % 2 == 0) = true
        · simp [hlt, hgt, heven, hhu]; omega
        · simp [hlt, hgt, heven, hhu]

/-- Rounding-Mode Parity Theorem: Clamped `requantSaturate` (`roundHalfUp`) and
    `requantSaturateTiesToEven` (`np.rint`) differ by at most `1` quantized code unit. -/
theorem requantSaturate_rounding_diff_le_one
    (acc : Int) (mult shift : Nat) (qmin qmax : Int) :
    (requantSaturate acc mult shift qmin qmax -
      requantSaturateTiesToEven acc mult shift qmin qmax).natAbs ≤ 1 := by
  have h := roundHalfUp_sub_roundTiesToEven_bounds (acc * (mult : Int)) shift
  unfold requantSaturate requantSaturateTiesToEven clampInt
  omega

/-- Signed `int32` MAC Worst-Case Bound Theorem: When `fanIn * (actQmax * weightQmax) ≤ 2147483647`
    and an accumulator magnitude is bounded by `fanIn * (actQmax * weightQmax)`, the accumulator
    strictly fits inside signed 32-bit `[-2147483648, 2147483647]` without wraparound. -/
theorem int32_mac_no_overflow
    (acc : Int) (fanIn actQmax weightQmax : Nat)
    (hbound : acc.natAbs ≤ fanIn * (actQmax * weightQmax))
    (hsafe : fanIn * (actQmax * weightQmax) ≤ 2147483647) :
    -2147483648 ≤ acc ∧ acc ≤ 2147483647 := by
  omega

/-! ## 3. Core L0 Edge Student Gate -/

/-- Quantized L0 edge student verification record. -/
structure EdgeStudentCheck where
  bitWidth          : Nat
  scaleMicro        : Nat     -- quantization scale in 10^-6 units (must be > 0)
  packedWeightBytes : Nat
  maxEdgeSramBytes  : Nat
  agreementPermille : Nat     -- agreement with FP32 teacher/student (e.g., >= 980)
  minAgreePermille  : Nat
  deriving DecidableEq, Repr

/-- Core L0 `lab-feet` edge student quantization checker. -/
def checkLabFeet (e : EdgeStudentCheck) : Option RegimeViolation :=
  firstViolation
    [ (decide (2 ≤ e.bitWidth ∧ e.bitWidth ≤ 8),
        { repo := "lab-feet", rule := "supported_quant_bitwidth",
          detail := "quantization bit-width must lie in [2, 8]" }),
      (decide (0 < e.scaleMicro),
        { repo := "lab-feet", rule := "nonzero_quantization_scale",
          detail := "zero quantization scale causes division-by-zero collapse on constant tensors" }),
      (decide (e.packedWeightBytes ≤ e.maxEdgeSramBytes),
        { repo := "lab-feet", rule := "edge_memory_budget",
          detail := "packed student footprint exceeds target edge board SRAM budget" }),
      (decide (e.minAgreePermille ≤ e.agreementPermille),
        { repo := "lab-feet", rule := "teacher_student_fidelity",
          detail := "quantized edge student agreement fell below certified floor" }) ]

/-- Edge Safety Theorem: Accepted `lab-feet` quantized students have strictly positive
    quantization scale, fit inside target SRAM, and meet teacher-student agreement floor. -/
theorem checkLabFeet_safe_for_edge (e : EdgeStudentCheck)
    (hok : checkLabFeet e = none) :
    0 < e.scaleMicro ∧ e.packedWeightBytes ≤ e.maxEdgeSramBytes ∧
    e.minAgreePermille ≤ e.agreementPermille := by
  unfold checkLabFeet at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
  rcases hok with ⟨_, h2, h3, h4⟩
  exact ⟨h2, h3, h4⟩

/-- Bit-Width Range Theorem: Accepted students use supported integer bit-width `[2, 8]`. -/
theorem checkLabFeet_valid_bitwidth (e : EdgeStudentCheck)
    (hok : checkLabFeet e = none) :
    2 ≤ e.bitWidth ∧ e.bitWidth ≤ 8 := by
  unfold checkLabFeet at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
  exact hok.1

/-! ## 4. Signed 5-Gate Drift-Abstaining Int4 Monitor (Idea #20) -/

/-- Edge inference output decision: actionable class index or deterministic `safeHold`. -/
inductive EdgeInferenceAction where
  | emitClass (classIdx : Nat)
  | safeHold
  deriving DecidableEq, Repr

/-- Complete 5-gate runtime OOD + saturation + signed-peak monitor check matching `drift_abstaining_int4.py`. -/
structure DriftAbstainMonitorCheck where
  student             : EdgeStudentCheck
  extraSramBytes      : Nat          -- integer monitor table + scratch SRAM (< 512 B)
  l1MahalanobisScore  : Nat          -- Gate 1: integer L1 feature-Mahalanobis distance
  l1Threshold         : Nat
  int4SaturationClips : Nat          -- Gate 2: count of hidden units hitting int4 +7 rail
  maxClipCount        : Nat
  int8InputClips      : Nat := 0     -- Gate 3: count of input ADC channels hitting +/-127 rails
  maxInputClips       : Nat := 2
  energyMarginScore   : Nat          -- Gate 4: quantized top-1 vs runner-up logit margin
  minEnergyMargin     : Nat
  peakLogit           : Int := 20    -- Gate 5: signed peak quantized logit (can be < 0 on dead cable)
  minPeakLogit        : Int := 12
  action              : EdgeInferenceAction
  deriving DecidableEq, Repr

/-- Deterministic 5-gate edge inference policy routing drifted, rail-clipped, or flatlined inputs to `safeHold`. -/
def decideDriftAbstainingInt4
    (argmaxClass : Nat)
    (l1Score l1Thresh clipCount maxClips inClips maxInClips energyMargin minEnergy : Nat)
    (peakLogit minPeakLogit : Int) :
    EdgeInferenceAction :=
  if decide (l1Thresh < l1Score ∨
             maxClips < clipCount ∨
             maxInClips < inClips ∨
             energyMargin < minEnergy ∨
             peakLogit < minPeakLogit) then
    .safeHold
  else
    .emitClass argmaxClass

/-- Combined L0 student + `< 512 B` 5-gate OOD/saturation abstention verifier. -/
def checkDriftAbstainingEdge (m : DriftAbstainMonitorCheck) : Option RegimeViolation :=
  match checkLabFeet m.student with
  | some v => some v
  | none =>
    let drifted : Bool :=
      decide (m.l1Threshold < m.l1MahalanobisScore ∨
              m.maxClipCount < m.int4SaturationClips ∨
              m.maxInputClips < m.int8InputClips ∨
              m.energyMarginScore < m.minEnergyMargin ∨
              m.peakLogit < m.minPeakLogit)
    firstViolation
      [ (decide (m.extraSramBytes < 512),
          { repo := "lab-feet", rule := "monitor_sram_under_512_bytes",
            detail := "OOD and saturation monitor exceeds 512-byte SRAM overhead cap" }),
        (decide (m.student.packedWeightBytes + m.extraSramBytes ≤ m.student.maxEdgeSramBytes),
          { repo := "lab-feet", rule := "combined_student_monitor_sram_budget",
            detail := "packed student weights plus OOD monitor exceed total edge SRAM budget" }),
        (decide (drifted = true → m.action = .safeHold),
          { repo := "lab-feet", rule := "mandatory_safe_hold_on_drift_or_clip",
            detail := "drifted, rail-saturated, or low-peak int4 input emitted unhalted class prediction" }) ]

/-- Monitor SRAM Budget Theorem: Certified edge monitors strictly use `< 512 B` extra SRAM
    and preserve the combined board SRAM ceiling. -/
theorem driftMonitor_extra_sram_under_512_bytes (m : DriftAbstainMonitorCheck)
    (hok : checkDriftAbstainingEdge m = none) :
    m.extraSramBytes < 512 ∧
    m.student.packedWeightBytes + m.extraSramBytes ≤ m.student.maxEdgeSramBytes := by
  unfold checkDriftAbstainingEdge at hok
  cases hs : checkLabFeet m.student with
  | some v => simp [hs] at hok
  | none =>
    simp only [hs, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    exact ⟨hok.1, hok.2.1⟩

/-- Gate 1 Theorem (`L1` Mahalanobis): Exceeding `l1Threshold` forces `safeHold`. -/
theorem driftMonitor_safe_hold_on_l1_ood (m : DriftAbstainMonitorCheck)
    (hok : checkDriftAbstainingEdge m = none)
    (hood : m.l1Threshold < m.l1MahalanobisScore) :
    m.action = .safeHold := by
  unfold checkDriftAbstainingEdge at hok
  cases hs : checkLabFeet m.student with
  | some v => simp [hs] at hok
  | none =>
    simp only [hs, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    apply hok.2.2
    exact Or.inl hood

/-- Gate 2 Theorem (`int4` Hidden Saturation): Exceeding `maxClipCount` forces `safeHold`. -/
theorem driftMonitor_safe_hold_on_int4_saturation (m : DriftAbstainMonitorCheck)
    (hok : checkDriftAbstainingEdge m = none)
    (hclip : m.maxClipCount < m.int4SaturationClips) :
    m.action = .safeHold := by
  unfold checkDriftAbstainingEdge at hok
  cases hs : checkLabFeet m.student with
  | some v => simp [hs] at hok
  | none =>
    simp only [hs, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    apply hok.2.2
    exact Or.inr (Or.inl hclip)

/-- Gate 3 Theorem (`int8` Input ADC Rail Clips): Exceeding `maxInputClips` forces `safeHold`. -/
theorem driftMonitor_safe_hold_on_int8_input_clips (m : DriftAbstainMonitorCheck)
    (hok : checkDriftAbstainingEdge m = none)
    (hin : m.maxInputClips < m.int8InputClips) :
    m.action = .safeHold := by
  unfold checkDriftAbstainingEdge at hok
  cases hs : checkLabFeet m.student with
  | some v => simp [hs] at hok
  | none =>
    simp only [hs, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    apply hok.2.2
    exact Or.inr (Or.inr (Or.inl hin))

/-- Gate 4 Theorem (Energy Margin): Falling below `minEnergyMargin` forces `safeHold`. -/
theorem driftMonitor_safe_hold_on_low_energy_margin (m : DriftAbstainMonitorCheck)
    (hok : checkDriftAbstainingEdge m = none)
    (hmargin : m.energyMarginScore < m.minEnergyMargin) :
    m.action = .safeHold := by
  unfold checkDriftAbstainingEdge at hok
  cases hs : checkLabFeet m.student with
  | some v => simp [hs] at hok
  | none =>
    simp only [hs, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    apply hok.2.2
    exact Or.inr (Or.inr (Or.inr (Or.inl hmargin)))

/-- Gate 5 Theorem (Signed Peak Logit): Signed `peakLogit < minPeakLogit` (including negative
    flatline logits) deterministically forces `safeHold`. -/
theorem driftMonitor_safe_hold_on_low_peak_logit (m : DriftAbstainMonitorCheck)
    (hok : checkDriftAbstainingEdge m = none)
    (hpeak : m.peakLogit < m.minPeakLogit) :
    m.action = .safeHold := by
  unfold checkDriftAbstainingEdge at hok
  cases hs : checkLabFeet m.student with
  | some v => simp [hs] at hok
  | none =>
    simp only [hs, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    apply hok.2.2
    exact Or.inr (Or.inr (Or.inr (Or.inr hpeak)))

/-- Policy Soundness Theorem: `decideDriftAbstainingInt4` satisfies `checkDriftAbstainingEdge`
    across all 5 gates whenever `checkLabFeet` and the `< 512 B` SRAM bounds hold. -/
theorem decideDriftAbstainingInt4_sound
    (student : EdgeStudentCheck) (extraSram argmaxClass : Nat)
    (l1Score l1Thresh clipCount maxClips inClips maxInClips energyMargin minEnergy : Nat)
    (peakLogit minPeakLogit : Int)
    (hfeet : checkLabFeet student = none)
    (hsram512 : extraSram < 512)
    (htotal : student.packedWeightBytes + extraSram ≤ student.maxEdgeSramBytes) :
    checkDriftAbstainingEdge
      { student             := student
        extraSramBytes      := extraSram
        l1MahalanobisScore  := l1Score
        l1Threshold         := l1Thresh
        int4SaturationClips := clipCount
        maxClipCount        := maxClips
        int8InputClips      := inClips
        maxInputClips       := maxInClips
        energyMarginScore   := energyMargin
        minEnergyMargin     := minEnergy
        peakLogit           := peakLogit
        minPeakLogit        := minPeakLogit
        action              := decideDriftAbstainingInt4 argmaxClass
                                 l1Score l1Thresh clipCount maxClips
                                 inClips maxInClips energyMargin minEnergy
                                 peakLogit minPeakLogit } = none := by
  unfold checkDriftAbstainingEdge decideDriftAbstainingInt4
  simp only [hfeet, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq]
  refine ⟨hsram512, htotal, ?_⟩
  intro hdrift
  simp [hdrift]

/-! ## 5. Compile-Time `#guard` Witnesses -/

def validEdgeStudent : EdgeStudentCheck :=
  { bitWidth          := 4
    scaleMicro        := 12500
    packedWeightBytes := 1520
    maxEdgeSramBytes  := 4096
    agreementPermille := 996
    minAgreePermille  := 980 }

def validInDomainMonitor : DriftAbstainMonitorCheck :=
  { student             := validEdgeStudent
    extraSramBytes      := 220
    l1MahalanobisScore  := 410
    l1Threshold         := 780
    int4SaturationClips := 2
    maxClipCount        := 8
    int8InputClips      := 0
    maxInputClips       := 2
    energyMarginScore   := 45
    minEnergyMargin     := 12
    peakLogit           := 38
    minPeakLogit        := 12
    action              := .emitClass 3 }

def validDriftedAbstain : DriftAbstainMonitorCheck :=
  { validInDomainMonitor with
    l1MahalanobisScore  := 1450
    int4SaturationClips := 19
    action              := .safeHold }

-- Nibble round-trip and odd-length canonical zero-padding witnesses
#guard packInt4Pair (-8) 7 == 0x78
#guard unpackInt4Lo (packInt4Pair (-8) 7) == -8
#guard unpackInt4Hi (packInt4Pair (-8) 7) == 7
#guard ((packInt4Single (-3) >>> (4 : UInt8)) &&& (0x0F : UInt8)) == (0 : UInt8)
#guard unpackInt4Lo (packInt4Single (-3)) == -3

-- Nearest-integer requantization witnesses (1023 / 1024 -> 1 instead of floor-truncated 0)
#guard roundHalfUpShift (1 * 1023) 10 == 1
#guard requantSaturate 113792 32767 10 0 7 == 7
#guard (requantSaturate 1 512 10 (-8) 7 - requantSaturateTiesToEven 1 512 10 (-8) 7).natAbs ≤ 1

-- Student & 5-gate monitor checks
#guard checkLabFeet validEdgeStudent == none
#guard checkDriftAbstainingEdge validInDomainMonitor == none
#guard checkDriftAbstainingEdge validDriftedAbstain == none

-- Zero quantization scale (`scaleMicro = 0`) and unsupported bit-width are rejected
#guard (checkLabFeet { validEdgeStudent with scaleMicro := 0 }).isSome == true
#guard (checkLabFeet { validEdgeStudent with bitWidth := 1 }).isSome == true

-- Monitor SRAM >= 512 B is rejected
#guard (checkDriftAbstainingEdge { validInDomainMonitor with extraSramBytes := 512 }).isSome == true

-- All 5 individual drift/saturation/flatline gates reject unhalted class emission
#guard (checkDriftAbstainingEdge { validInDomainMonitor with
                                   l1MahalanobisScore := 1450,
                                   action := .emitClass 3 }).isSome == true
#guard (checkDriftAbstainingEdge { validInDomainMonitor with
                                   int4SaturationClips := 19,
                                   action := .emitClass 1 }).isSome == true
#guard (checkDriftAbstainingEdge { validInDomainMonitor with
                                   int8InputClips := 24,
                                   action := .emitClass 2 }).isSome == true
#guard (checkDriftAbstainingEdge { validInDomainMonitor with
                                   energyMarginScore := 4,
                                   action := .emitClass 0 }).isSome == true
#guard (checkDriftAbstainingEdge { validInDomainMonitor with
                                   peakLogit := -18,
                                   action := .emitClass 4 }).isSome == true

-- Negative peakLogit (-18 < 12) routes deterministically to safeHold
#guard decideDriftAbstainingInt4 4 200 780 0 8 0 2 20 12 (-18) 12 == .safeHold
#guard decideDriftAbstainingInt4 4 200 780 0 8 24 2 20 12 30 12 == .safeHold
#guard decideDriftAbstainingInt4 4 200 780 0 8 0 2 20 12 30 12 == .emitClass 4

end LabFeetLean
