/-!
# LabFeetLean: L0 Edge Quantization Scale, SRAM Footprint & OOD/Clipping Abstention Proofs

Formalizes the `lab-feet` (Layer L0) edge distillation and low-bit quantization guardrails:
1. **Bit-Width & Scale Positivity**: `bitWidth in [2, 8]` and `0 < scaleMicro` (preventing
   zero-scale collapse on constant tensors).
2. **Edge SRAM Budget & Fidelity**: `packedWeightBytes <= maxEdgeSramBytes` and
   `minAgreePermille <= agreementPermille`.
3. **Quantized Activation-Energy OOD & Clipping Monitor (Idea #20)**: Pairs the `int4`/`int8`
   student with an `< 512 B` integer `L1` feature-Mahalanobis + `int4` rail-saturation
   counter that deterministically triggers `safeHold` abstention instead of emitting
   overconfident clipped class predictions under benchtop sensor drift.
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

/-- Edge inference output decision: actionable class index or deterministic `safeHold`. -/
inductive EdgeInferenceAction where
  | emitClass (classIdx : Nat)
  | safeHold
  deriving DecidableEq, Repr

/-- Runtime OOD + int4 saturation clipping monitor check (Idea #20). -/
structure DriftAbstainMonitorCheck where
  student             : EdgeStudentCheck
  extraSramBytes      : Nat     -- integer monitor table + scratch SRAM (< 512 B)
  l1MahalanobisScore  : Nat     -- integer L1 feature-Mahalanobis distance
  l1Threshold         : Nat
  int4SaturationClips : Nat     -- count of hidden/input rails hitting int4/int8 clip limits
  maxClipCount        : Nat
  energyMarginScore   : Nat     -- quantized activation/logit energy margin
  minEnergyMargin     : Nat
  action              : EdgeInferenceAction
  deriving DecidableEq, Repr

/-- Deterministic edge inference policy routing drifted or saturated inputs to `safeHold`. -/
def decideDriftAbstainingInt4
    (argmaxClass : Nat)
    (l1Score l1Thresh clipCount maxClips energyMargin minEnergy : Nat) :
    EdgeInferenceAction :=
  if decide (l1Thresh < l1Score ∨ maxClips < clipCount ∨ energyMargin < minEnergy) then
    .safeHold
  else
    .emitClass argmaxClass

/-- Combined L0 student + `< 512 B` OOD/saturation abstention gate. -/
def checkDriftAbstainingEdge (m : DriftAbstainMonitorCheck) : Option RegimeViolation :=
  match checkLabFeet m.student with
  | some v => some v
  | none =>
    let drifted : Bool :=
      decide (m.l1Threshold < m.l1MahalanobisScore ∨
              m.maxClipCount < m.int4SaturationClips ∨
              m.energyMarginScore < m.minEnergyMargin)
    firstViolation
      [ (decide (m.extraSramBytes < 512),
          { repo := "lab-feet", rule := "monitor_sram_under_512_bytes",
            detail := "OOD and saturation monitor exceeds 512-byte SRAM overhead cap" }),
        (decide (m.student.packedWeightBytes + m.extraSramBytes ≤ m.student.maxEdgeSramBytes),
          { repo := "lab-feet", rule := "combined_student_monitor_sram_budget",
            detail := "packed student weights plus OOD monitor exceed total edge SRAM budget" }),
        (decide (drifted = true → m.action = .safeHold),
          { repo := "lab-feet", rule := "mandatory_safe_hold_on_drift_or_clip",
            detail := "drifted or saturated int4 input emitted silent overconfident class prediction" }) ]

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

/-- L1-Mahalanobis OOD Abstention Theorem: Exceeding `l1Threshold` deterministically forces `safeHold`. -/
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

/-- Int4 Saturation Abstention Theorem: Exceeding `maxClipCount` deterministically forces `safeHold`. -/
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

/-- Policy Soundness Theorem: `decideDriftAbstainingInt4` always satisfies `checkDriftAbstainingEdge`
    whenever `checkLabFeet` and the `< 512 B` SRAM bounds hold. -/
theorem decideDriftAbstainingInt4_sound
    (student : EdgeStudentCheck) (extraSram argmaxClass : Nat)
    (l1Score l1Thresh clipCount maxClips energyMargin minEnergy : Nat)
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
        energyMarginScore   := energyMargin
        minEnergyMargin     := minEnergy
        action              := decideDriftAbstainingInt4 argmaxClass
                                 l1Score l1Thresh clipCount maxClips energyMargin minEnergy } = none := by
  unfold checkDriftAbstainingEdge decideDriftAbstainingInt4
  simp only [hfeet, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq]
  refine ⟨hsram512, htotal, ?_⟩
  intro hdrift
  simp [hdrift]

/-! ## Compile-Time `#guard` Witnesses -/

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
    energyMarginScore   := 45
    minEnergyMargin     := 12
    action              := .emitClass 3 }

def validDriftedAbstain : DriftAbstainMonitorCheck :=
  { validInDomainMonitor with
    l1MahalanobisScore  := 1450
    int4SaturationClips := 19
    action              := .safeHold }

#guard checkLabFeet validEdgeStudent == none
#guard checkDriftAbstainingEdge validInDomainMonitor == none
#guard checkDriftAbstainingEdge validDriftedAbstain == none

-- Zero quantization scale (`scaleMicro = 0`) is rejected
#guard (checkLabFeet { validEdgeStudent with scaleMicro := 0 }).isSome == true

-- Unsupported 1-bit or 16-bit width outside `[2, 8]` is rejected
#guard (checkLabFeet { validEdgeStudent with bitWidth := 1 }).isSome == true

-- Monitor SRAM >= 512 B is rejected
#guard (checkDriftAbstainingEdge { validInDomainMonitor with extraSramBytes := 512 }).isSome == true

-- Emitting a class on OOD L1 distance (`1450 > 780`) or int4 saturation (`19 > 8`) is rejected
#guard (checkDriftAbstainingEdge { validInDomainMonitor with
                                   l1MahalanobisScore := 1450,
                                   action := .emitClass 3 }).isSome == true
#guard (checkDriftAbstainingEdge { validInDomainMonitor with
                                   int4SaturationClips := 19,
                                   action := .emitClass 1 }).isSome == true

end LabFeetLean
