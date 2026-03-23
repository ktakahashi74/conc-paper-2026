# Audio Supplement — Conchordal

Qualitative audio demonstrations accompanying the paper
*"Conchordal: A Bio-Acoustic Artificial Life Instrument"*.

**[Listen in browser](https://ktakahashi74.github.io/conc-paper/supplementary_audio/)**

The paper-facing audio tracks correspond to **Experiment 1**, **Experiment 4**,
and **Experiment 3**. There is **no standalone scene replay for Experiment 2**,
because that assay is primarily about selection pressure and lifetime statistics
rather than a stable emergent scene.

## Track listing

| File | Role | Duration | Description |
|------|------|----------|-------------|
| `00_quicklisten_showcase.wav` | Orientation | ~21 s | Two-segment showcase: strongest current examples from Exp. 1 and Exp. 4 |
| `01_quicklisten_controls.wav` | Orientation | ~21 s | Two-segment control montage: matched ablations for Exp. 1 and Exp. 4 |
| `10_exp1_polyphony.wav` | Main-paper evidence (Exp. 1; internal `E2`) | ~35 s | Self-organized polyphony replay: local-search vs. no-hill across seed 0 and seed 10 |
| `40_e6b_polyphony.wav` | Main-paper evidence (Exp. 4; internal `E6b`) | ~107 s | Hereditary adaptation replay: heredity vs. matched-random, selection on/off, seeds 0 and 10 |
| `30_e3_shared.wav`, `30_e3_scrambled.wav`, `30_e3_off.wav` | Main-paper evidence (Exp. 3; historical `e3` filename prefix) | ~8 s each | Temporal scaffold render: shared vs. scrambled vs. off |

### 00_quicklisten_showcase — Orientation montage (showcase)

For quick orientation. Two short segments present the strongest current
examples:

1. **Exp. 1 local-search** — score-based local search + repulsion → consonant polyphony
2. **Exp. 4 H+S** — family-biased respawn + local azimuth search → hereditary polyphony

The Exp. 1 excerpt is cropped to **2 s before** and **4 s after** the phase switch.
Each segment uses a 1 s fade-in and fade-out, with 2 s silence between segments.

### 01_quicklisten_controls — Orientation montage (controls)

Matched control versions of the same two assays:

1. **Exp. 1 no-hill** — crowding only, no hill-climb
2. **Exp. 4 neither** — matched random respawn, no selection

The Exp. 1 control excerpt is cropped to the same **2 s pre-switch / 4 s post-switch**
window for direct comparison.

### 10_exp1_polyphony — Experiment 1 (internal ID: E2)

Two conditions × two seeds (index 0, 10). Each segment ~8 s:

| Condition | landscape_weight | repulsion |
|-----------|-----------------|-----------|
| local-search | 1.0 | 0.15 @ 72 cents |
| no-hill | 0.0 | 0.15 @ 72 cents |

**Expected:** baseline preserves the current hill-climb + crowding regime;
no-hill removes local search while keeping the same crowding field.

### 40_e6b_polyphony — Experiment 4 (internal ID: E6b)

Four conditions × two seeds (index 0, 10). Each segment ~12 s. Within each seed,
segments are ordered from strongest hereditary fusion to null control:

| Condition | respawn | selection |
|-----------|---------|-----------|
| H+S | family-biased respawn + local azimuth search | on |
| H-only | family-biased respawn + local azimuth search | off |
| S-only | matched log-random candidates | on |
| neither | matched log-random candidates | off |

This replay uses the current paper-facing hereditary regime: 16-agent population,
adult pitch locked, a brief 2-tick slot-local juvenile settling phase, and
family-biased respawn whose final azimuth is chosen by local search rather than
inherited directly from the parent. Sonification again uses slowed per-life
harmonic voices.

**Expected:** H+S should be the clearest multi-band polyphony, H-only should retain
some family structure without the same metabolic sharpening, and the matched
random controls should remain broader and more diffuse.

### 30_exp3_temporal_scaffold — Experiment 3 (historical `e3` filename prefix)

Three qualitative renders contrast the temporal scaffold conditions used
in the main paper's Experiment 3:

| Condition | Drive | Coupling |
|-----------|-------|----------|
| `shared` | single continuous 2 Hz phase reference | fixed |
| `scrambled` | 2 Hz cycles preserved, phase reset each cycle | fixed |
| `off` | no shared phase drive | 0 |

All three tracks use the same pitch inventory and intrinsic-frequency
jitter; only the temporal scaffold differs.
No external drone or metronome is mixed into track 30.

**Expected:** `shared` produces the clearest beat-aligned pulse,
`scrambled` is less stable, and `off` remains diffuse.

## Methodology

### Seed selection

Seeds are taken by ordinal index from the experiment-specific seed lists in
`experiments/paper_plots.rs`, with no cherry-picking. The main-paper replay
tracks use the current `E2` and `E6b` seed schedules; track `30` is a fixed
illustrative temporal render.

### Rendering chain

All tracks are rendered with `conchordal-render` using the same pipeline:

- Primary voices: `harmonic` (neutral harmonic series)
- Fixed reference drones, where present: `sine`
- Output: mono WAV, 48 kHz
- No reverb, no mastering, no per-file loudness optimisation
- Identical rendering chain across all tracks and segments

### Reproduction

```bash
cd supplementary_audio
bash render.sh
```

Requires the `conchordal` crate at `../../conchordal/` with
`conchordal-render` binary.

## Caveats

These audio demonstrations are **qualitative support only**. Quantitative
claims rest on the paper's simulation statistics (Section 5).

| Aspect | Paper simulation (statistics) | Audio supplement (this) |
|--------|-------------------------------|------------------------|
| Landscape | Computed from known frequencies | Computed from audio signal (NSGT) |
| Update timing | Discrete steps | Real-time hop rate |
| Exp. 1 local-search | Discrete grid + LOO | Continuous frequency + native pitch_core |
| Exp. 4 heredity | Family-biased respawn + local azimuth search | RuntimeGroup-based |
| Qualitative behaviour | Same attractors | Same attractors |
| Quantitative match | — | Not guaranteed |

The audio is generated by the instrument's native Rhai scenario system,
not by replaying simulation event logs. The underlying mechanisms are
the same, but timing and update granularity differ.

## Manifest

See `manifest.csv` for per-segment metadata (file, segment index, label,
internal experiment ID, paper experiment label, condition, seed, start/end
times, and whether the track is an orientation clip or main-paper evidence).
