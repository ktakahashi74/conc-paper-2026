# Audio Supplement Reproduction Guide

This file is for **reproducing / re-rendering the audio files used in the
supplement**. It is not the listening page itself.

If you want to listen first, start from the public-facing page:

- [Audio Supplement Page](https://ktakahashi74.github.io/conc-paper-2026/supplementary_audio/)
- [Supplementary PDF](../supplementary.pdf)
- [Experiments README](../experiments/README.md)
- [Top-level README](../README.md)

## What lives here

- `index.html`: reviewer-facing listening page
- `manifest.csv`: per-segment metadata for regeneration and verification
- `audio/`: rendered WAV files served by the listening page
- `scenarios/`: generated Rhai scenarios for the scenario-based tracks
- `render.sh`: convenience wrapper for the scenario-based renders

## Interpretation boundary

Audio supplement provides qualitative corroboration of the reported quantitative
results; it is not a behavioural listening study.

## Public track set

The listening page serves these files:

- `showcase.wav`
- `controls.wav`
- `self_organized_polyphony.wav`
- `self_organized_polyphony_no_hill.wav`
- `hereditary_adaptation.wav`
- `hereditary_adaptation_controls.wav`
- `temporal_scaffold_shared.wav`
- `temporal_scaffold_scrambled.wav`
- `temporal_scaffold_off.wav`

Reviewer-facing interpretation lives on `index.html`, where the public track set
is organized as:

- orientation
- Consonance Search evidence
- Hereditary Adaptation evidence
- Temporal Scaffold evidence

## Rendering paths

There are two rendering paths.

There are two rendering paths for the public track set:

| Track(s) | Render method | How to reproduce |
| --- | --- | --- |
| `showcase.wav`<br/>`controls.wav`<br/>`self_organized_polyphony*.wav`<br/>`hereditary_adaptation*.wav` | `conchordal-render` (Conchordal 0.3.0) | `target/release/paper --audio-rhai` then `bash supplementary_audio/render.sh` |
| `temporal_scaffold_shared.wav`<br/>`temporal_scaffold_scrambled.wav`<br/>`temporal_scaffold_off.wav` | Paper internal direct synthesis (`render_e3_audio`) | `target/release/paper --e3-audio` |

The three temporal-scaffold tracks are generated from Rhai scene descriptions for consistency, but rendered by the paper binary directly rather than by `conchordal-render`.

## Prerequisites

- Rust toolchain
- sibling checkout of `../conchordal`
- `conchordal-render` available via the sibling checkout

Build the paper binary first:

```bash
cargo build --release --bin paper
```

## Regenerate everything

From the repository root:

```bash
target/release/paper --audio-rhai
bash supplementary_audio/render.sh
target/release/paper --e3-audio
target/release/paper --postprocess-quicklisten
target/release/paper --postprocess-e6b
```

This sequence:

- regenerates the Rhai scenarios
- renders the scenario-based WAV files
- regenerates the temporal scaffold WAV files
- applies the post-processing used by the public tracks

## Regenerate a subset

Scenario files:

```bash
target/release/paper --audio-rhai
```

Scenario-based WAV files:

```bash
bash supplementary_audio/render.sh
```

Temporal scaffold only:

```bash
target/release/paper --e3-audio
```

## Notes

- The public-facing descriptions live on [index.html](index.html), not here.
- `manifest.csv` remains available as auxiliary metadata for segment boundaries and labels.
- The audio supplement is qualitative support; the quantitative results live in
  the experiment outputs under [../experiments/plots/README.md](../experiments/plots/README.md).
- There is no separate public track for the supplementary spectral-to-temporal
  bridge check.
