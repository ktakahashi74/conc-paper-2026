# Audio Supplement — Conchordal

Qualitative audio demonstrations accompanying the paper
*"Conchordal: A Bio-Acoustic Artificial Life Instrument"*.

**[Listen in browser](https://ktakahashi74.github.io/conc-paper/supplementary_audio/)**

## Track listing

| File | Role | Duration | Description |
|------|------|----------|-------------|
| `00_quicklisten.wav` | Orientation | ~77 s | Curated montage: best-case emergence vs. ablation controls |
| `10_exp1_polyphony.wav` | Evidence (E2) | ~35 s | Current E2 replay montage: baseline vs. no-hill across seed 0 and seed 10 |
| `20_integration.wav` | Evidence (E6) | ~107 s | Current E6 integration montage: full 32-agent paper regime, slowed for listening |
| `40_e6b_polyphony.wav` | Evidence (E6b) | ~107 s | E6b hereditary polyphony replay: heredity/random × selection on/off across seed 0 and seed 10 |
| `30_e3_shared.wav`, `30_e3_scrambled.wav`, `30_e3_off.wav` | Evidence (main E3) | ~8 s each | Agent-only render of shared temporal scaffold vs. scrambled/off controls |

### 00_quicklisten — Orientation montage

For quick orientation. Four short segments contrast the strongest
emergent behaviour with ablation controls:

1. **E2 local-search** — score-based local search + repulsion → consonant polyphony
2. **E6 both** — hereditary respawn + hill-climbing → integration
3. **E2 random-walk** — repulsion only, matched random local walk
4. **E6 neither** — random respawn, no landscape → random control

Each segment uses a 1 s fade-in and fade-out, with 2 s silence between
segments.

### 10_exp1_polyphony — Experiment 2 (E2)

Two conditions × two seeds (index 0, 10). Each segment ~8 s:

| Condition | landscape_weight | repulsion |
|-----------|-----------------|-----------|
| local-search | 1.0 | 0.15 @ 72 cents |
| no-hill | 0.0 | 0.15 @ 72 cents |

**Expected:** baseline preserves the current hill-climb + crowding regime;
no-hill removes local search while keeping the same crowding field.

### 20_integration — Experiment 6 (E6)

Four conditions × two seeds (index 0, 10). Each segment ~12 s. Within each seed,
segments are ordered from most integrated to random. The replay uses the full
32-agent paper regime, but the sonification is slowed and spectrally thinned so
the pitch structure is still audible:

| Condition | respawn | landscape_weight |
|-----------|---------|-----------------|
| both | parent-only H heredity | 1.0 |
| H-only | parent-only H heredity | 0.0 |
| hill-only | random | 1.0 |
| neither | random | 0.0 |

The E6 supplementary render uses parent-only harmonicity for hereditary respawn
and the same shared hill-climb profile, replayed as a 32-agent per-life montage
with slower updates and near-sine voices for audibility.

**Expected:** "both" achieves the highest consonance score, with H-only above
the random respawn controls.

### 40_e6b_polyphony — Experiment 6b (E6b)

Four conditions × two seeds (index 0, 10). Each segment ~12 s. Within each seed,
segments are ordered from strongest hereditary fusion to null control:

| Condition | respawn | selection |
|-----------|---------|-----------|
| H+S | hereditary family replay | on |
| H-only | hereditary family replay | off |
| S-only | random | on |
| neither | random | off |

The E6b replay uses the current hereditary-polyphony regime: 16-agent population,
adult pitch locked, juvenile settlement enabled, and hereditary respawn centered
on family/azimuth replay. Sonification again uses slowed per-life harmonic voices.

**Expected:** H+S and H-only should concentrate into a few stable pitch bands,
while the random controls remain broad and diffuse. H+S is the intended
selection-assisted version of the hereditary fuse pattern.

### 30_exp3_temporal_scaffold — Experiment 3 (main paper)

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

Seeds are taken by ordinal index (0 and 10) from the paper's shared seed
lists (`E2_SEEDS`, `E6_SEEDS`, `E6B_SEEDS` in `experiments/paper_plots.rs`). No
cherry-picking.

### Rendering chain

All tracks are rendered with `conchordal-render` using the same pipeline:

- Primary voices: `harmonic` (neutral harmonic series; E6 uses a 1-partial voice for clarity)
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
| E2 local-search | Discrete grid + LOO | Continuous frequency + native pitch_core |
| E6 heredity | Custom parent selection | RuntimeGroup-based |
| Qualitative behaviour | Same attractors | Same attractors |
| Quantitative match | — | Not guaranteed |

The audio is generated by the instrument's native Rhai scenario system,
not by replaying simulation event logs. The underlying mechanisms are
the same, but timing and update granularity differ.

## Manifest

See `manifest.csv` for per-segment metadata (file, segment index, label,
experiment, condition, seed, start/end times).
