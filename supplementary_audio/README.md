# Audio Supplement — Conchordal

Qualitative audio demonstrations accompanying the paper
*"Conchordal: A Bio-Acoustic Artificial Life Instrument"*.

**[Listen in browser](https://ktakahashi74.github.io/conc-paper/supplementary_audio/)**

Quick links:

- [Audio supplement page](https://ktakahashi74.github.io/conc-paper/supplementary_audio/)
- [Paper PDF](../main.pdf)
- [Supplementary PDF](../supplementary.pdf)
- [Generated Data README](../experiments/plots/README.md)
- [Manifest](manifest.csv)
- [Repository README](../README.md)

The paper-facing audio tracks correspond to three assay families:

- **Self-Organized Polyphony**
- **Hereditary Adaptation**
- **Temporal Scaffold**

There is no standalone audio track for the **selection-pressure assay**,
because that result is primarily about lifetime statistics rather than a stable
replayable scene.

## Track listing

| File | Role | Duration | Description |
|------|------|----------|-------------|
| `showcase.wav` | Quick listen | ~22 s | Strongest current hereditary and local-search examples |
| `controls.wav` | Quick listen | ~22 s | Matched control excerpts: neither and no-hill |
| `self_organized_polyphony.wav` | Main-paper evidence | ~16 s | Local-search across seed 0 and seed 10 |
| `self_organized_polyphony_no_hill.wav` | Main-paper evidence | ~16 s | No-hill control across seed 0 and seed 10 |
| `hereditary_adaptation.wav` | Main-paper evidence | ~52 s | Heredity + selection and heredity only across seed 0 and seed 10 |
| `hereditary_adaptation_controls.wav` | Main-paper evidence | ~52 s | Selection only and neither across seed 0 and seed 10 |
| `temporal_scaffold_shared.wav`, `temporal_scaffold_scrambled.wav`, `temporal_scaffold_off.wav` | Main-paper evidence | ~8 s each | Shared vs. scrambled vs. off |

### Quick Listen

For quick orientation, use the paired `showcase.wav` and `controls.wav` tracks.

`showcase.wav` contains:
1. **Heredity + selection** — family-biased respawn plus local azimuth search yields hereditary polyphony
2. **Local search** — score-based local search plus repulsion yields consonant polyphony

The self-organized polyphony excerpt is cropped to **2 s before** and **6 s after**
the phase switch. Each segment uses a 1 s fade-in and fade-out, with 2 s silence
between segments.

`controls.wav` contains the matched control versions of the same two assays:

1. **Neither** — matched random respawn, no selection
2. **No hill-climb** — crowding only, no local hill-climb

The self-organized polyphony control excerpt uses the same **2 s pre-switch / 6 s post-switch**
window for direct comparison.

### Self-Organized Polyphony

Two companion tracks cover the same two seeds (index 0 and 10). Each segment is about 8 s:

| Condition | landscape_weight | repulsion |
|-----------|------------------|-----------|
| local-search | 1.0 | 0.15 @ 72 cents |
| no-hill | 0.0 | 0.15 @ 72 cents |

`self_organized_polyphony.wav` contains only the adaptive **local-search** condition.
`self_organized_polyphony_no_hill.wav` contains only the **no-hill** control.

**Expected:** local search preserves the hill-climb plus crowding regime; no-hill removes
local search while keeping the same crowding field.

### Hereditary Adaptation

Two companion tracks cover the same two seeds (index 0 and 10). Each segment is about 12 s.

| Condition | respawn | selection |
|-----------|---------|-----------|
| heredity + selection | family-biased respawn + local azimuth search | on |
| heredity only | family-biased respawn + local azimuth search | off |
| selection only | matched log-random candidates | on |
| neither | matched log-random candidates | off |

`hereditary_adaptation.wav` contains the two **hereditary** conditions:
heredity + selection and heredity only, for seed 0 and seed 10.

`hereditary_adaptation_controls.wav` contains the matched **control** conditions:
selection only and neither, for seed 0 and seed 10.

This replay uses the current paper-facing hereditary regime: a 16-agent population,
adult pitch locked, a brief slot-local juvenile settling phase, and family-biased
respawn whose final azimuth is chosen by local search rather than inherited directly
from the parent.

**Expected:** heredity + selection should be the clearest multi-band polyphony;
heredity only retains some family structure without the same metabolic sharpening;
the matched-random controls remain broader and less musically focused.

### Temporal Scaffold

Three qualitative renders contrast the temporal scaffold conditions used in the paper:

| Condition | Drive | Coupling |
|-----------|-------|----------|
| `shared` | single continuous 2 Hz phase reference | fixed |
| `scrambled` | 2 Hz cycles preserved, phase reset each cycle | fixed |
| `off` | no shared phase drive | 0 |

All three tracks use the same pitch inventory and intrinsic-frequency jitter; only
the temporal scaffold differs. No external drone or metronome is mixed into these
examples.

**Expected:** `shared` produces the clearest beat-aligned pulse, `scrambled` is less
stable, and `off` remains diffuse.

## Methodology

### Seed selection

Seeds are taken by ordinal index from the paper's shared seed lists, with no
cherry-picking. The self-organized polyphony and hereditary adaptation replays
use the current paper-facing seed schedules; the temporal scaffold tracks are a
fixed illustrative render.

### Rendering chain

The orientation clips and the main replays for self-organized polyphony and
hereditary adaptation are rendered from generated scenario scripts with
`conchordal-render`. The temporal scaffold tracks are rendered directly by the
paper tool. Across all tracks:

- Primary voices: harmonic or sine, depending on the assay
- Fixed reference drones, where present: sine
- Output: mono WAV, 48 kHz
- No reverb, no mastering, no per-file loudness optimisation
- Assay-specific synthesis, but no added production effects

### Reproduction

```bash
cd supplementary_audio
bash render.sh
```

This requires the `conchordal` crate at `../../conchordal/` with the
`conchordal-render` binary for the scenario-based tracks.

## Caveats

These audio demonstrations are **qualitative support only**. Quantitative
claims rest on the paper's simulation statistics.

| Aspect | Paper simulation (statistics) | Audio supplement |
|--------|-------------------------------|------------------|
| Landscape | Computed from known frequencies | Computed from audio signal or rendered event stream |
| Update timing | Discrete steps | Real-time hop rate or rendered note stream |
| Self-organized polyphony | Discrete grid + LOO | Continuous replay with native pitch tracking |
| Hereditary adaptation | Family-biased respawn + local azimuth search | Scenario-based replay with slowed per-life voices |
| Temporal scaffold | Timing statistics over event schedules | Direct monitor render of the same schedule |
| Qualitative behaviour | Same attractors | Same attractors |
| Quantitative match | — | Not guaranteed |

These tracks are listening aids derived from the same experimental settings,
not the source of the paper's quantitative analyses.

## Manifest

See `manifest.csv` for per-segment metadata: file, segment index, label,
condition, seed, start/end times, assay name, and whether the track belongs to
quick listening or main-paper evidence.
