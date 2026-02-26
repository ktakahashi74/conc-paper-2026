# Pitch Dynamics Issue: Inseparable Consonance Seeking and Density Repulsion

## Summary

The conchordal Population API's pitch scoring function conflates two conceptually independent forces—consonance-based landscape seeking and density-based perceptual repulsion—into a single scalar, making it impossible to enable one without the other. This prevents clean experimental ablation and contradicts the paper's conceptual model where these forces are orthogonal.

## Evidence from paper experiments

### E2 (Self-Organized Polyphony) — custom bin-level simulation

E2 uses a **custom simulation** in `experiments/paper_plots.rs` (not the Population API) that implements pitch dynamics at the bin level. It cleanly separates three conditions:

| Condition | Hill-climbing (consonance) | Repulsion (density) |
|---|---|---|
| Baseline | ON | ON |
| No hill-climb | OFF | ON |
| No repulsion | ON | OFF |

**Key finding**: hill-climbing and repulsion are independent, additive forces. Removing either one produces qualitatively different outcomes (entropy shift, pitch collapse). This is the paper's central ablation result for self-organization.

### E6 (Hereditary Adaptation) — uses Population API

E6 needed "repulsion ON, hill-climbing OFF" to isolate hereditary selection as the sole source of consonance improvement. This proved impossible with the Population API:

- **PitchMode::Lock**: No pitch movement at all → no repulsion possible → all converging seeds collapse to octave (±12 st), the single strongest attractor
- **PitchMode::Free + PeakSampler**: Agents move, but `adjusted_pitch_score` always includes landscape consonance → agents actively seek consonant peaks → hill-climbing is inseparable from repulsion → confounds hereditary effect with individual-level consonance seeking

## Root cause

### File: `src/life/pitch_core.rs`, line 563–589

```rust
fn adjusted_pitch_score(
    pitch_log2: f32,
    current_pitch_log2: f32,
    integration_window: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    move_cost_coeff: f32,
    move_cost_exp: u8,
    landscape: &Landscape,
    perceptual: &PerceptualContext,
) -> f32 {
    let score = landscape.evaluate_pitch_score_log2(clamped);  // ← consonance
    let penalty = dist_cost * integration_window * move_cost_coeff.max(0.0);
    let gravity_penalty = dist * dist * tessitura_gravity;
    let base = score - penalty - gravity_penalty;
    base + perceptual.score_adjustment(idx)  // ← repulsion
}
```

The landscape consonance score and perceptual repulsion adjustment are summed into a single scalar. Both `PitchHillClimbCore` (line 176) and `PitchPeakSamplerCore` (line 356) call this function for all candidate scoring. There is no code path that uses perceptual repulsion without landscape consonance.

### Used by both pitch core implementations

- **HillClimb** (`src/life/pitch_core.rs:129–221`): Deterministic acceptance based on `adjusted_pitch_score` improvement
- **PeakSampler** (`src/life/pitch_core.rs:310–412`): Softmax sampling over `adjusted_pitch_score` values

Both always include landscape consonance in the score. Even with `exploration=0.95` and high effective temperature, the landscape score biases softmax probabilities toward consonant peaks.

## Conceptual misalignment with DCC

The paper's conceptual model (derived from DCC) treats the pitch landscape as having two distinct perceptual axes:

1. **Consonance field** — "where is it good to be?" (landscape attractors at integer ratios)
2. **Density/novelty** — "where is it crowded?" (perceptual boredom/familiarity)

These correspond to different levels of auditory processing: consonance arises from cochlear/periodicity mechanisms, while density-based avoidance reflects higher-order habituation. Collapsing them into one score prevents the system from expressing behaviors where agents avoid crowding without seeking consonance (or vice versa).

E2's ablation results empirically confirm that these are independent forces with distinct, measurable effects. The Population API should reflect this independence.

## Proposed solution direction

### Option A: Separate scoring components

Refactor `adjusted_pitch_score` to return a struct with independent components:

```rust
struct PitchScore {
    landscape: f32,      // consonance field evaluation
    perceptual: f32,     // density/novelty adjustment
    move_cost: f32,      // distance penalty
    gravity: f32,        // tessitura pull
}
```

Then let PitchCoreKind (or a new configuration field) control which components are combined for candidate evaluation. For example:

```rust
pub struct PitchControl {
    // ... existing fields ...
    pub use_landscape_score: bool,  // enable/disable consonance seeking
    // perceptual repulsion is controlled via PerceptualControl.enabled
}
```

### Option B: New PitchCoreKind variant

Add `PitchCoreKind::RepulsionOnly` that uses only `perceptual.score_adjustment(idx)` (plus move cost and gravity) without `landscape.evaluate_pitch_score_log2()`.

### Option C: Weight-based blending

Add a `landscape_weight: f32` parameter (default 1.0) to PitchControl:

```rust
let base = landscape_weight * score - penalty - gravity_penalty;
base + perceptual.score_adjustment(idx)
```

Setting `landscape_weight = 0.0` gives pure repulsion dynamics. This is the least invasive change.

## Relevant files

| File | Role |
|---|---|
| `src/life/pitch_core.rs` | PitchHillClimbCore, PitchPeakSamplerCore, adjusted_pitch_score |
| `src/life/perceptual.rs` | PerceptualContext, score_adjustment (density/novelty) |
| `src/life/control.rs` | PitchMode, PitchCoreKind, PitchControl, PerceptualControl |
| `src/life/control_adapters.rs` | Parameter conversion from control to config |
| `src/life/pitch_controller.rs` | Pitch proposal and perceptual update gating |

## Impact on paper

Until this is resolved, E6 (Hereditary Adaptation) uses `PitchMode::Lock` as a workaround. This isolates hereditary selection cleanly but prevents repulsion, resulting in convergence only to the single strongest attractor (octave). The experiment demonstrates heredity works but cannot show polyphonic hereditary convergence to diverse consonant ratios.
