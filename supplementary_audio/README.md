# Audio Supplement Reproduction Guide

This file is for **reproducing / re-rendering the audio files used in the
supplement**. It is not the listening page itself.

If you want to listen first, start from the public-facing page:

- [Audio Supplement Page](https://ktakahashi74.github.io/conc-paper-2026/supplementary_audio/)
- [Supplementary PDF](../supplementary.pdf)
- [Experiments README](../experiments/README.md)
- [Generated Data Guide](../experiments/plots/README.md)
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

All listening tracks are rendered with Conchordal version 0.3.0.
Use `bash supplementary_audio/render.sh` from the repository root to regenerate
the full listening page payload.

## Prerequisites

- Rust toolchain
- sibling checkout of `../conchordal`
- `conchordal-render` available via the sibling checkout

Build the paper binary first:

```bash
cargo build --release --bin paper
```

If you also need the PDFs from a clean checkout, use the top-level recipes:

```bash
just latex-main
just supplementary
```

## Regenerate the public track set

From the repository root:

```bash
bash supplementary_audio/render.sh
```

This sequence:

- regenerates the Rhai scenarios
- renders all public WAV files with `conchordal-render`
- applies the post-processing used by the orientation and hereditary comparison tracks

## Regenerate a subset

Scenario files only:

```bash
target/release/paper --audio-rhai
```

Full public track set:

```bash
bash supplementary_audio/render.sh
```

Notes:

- `render.sh` is the canonical reproduction entrypoint for the public listening page.
- It regenerates the Rhai inputs before rendering, so a separate `--audio-rhai` step is only needed when you want to inspect or diff the generated scenarios without rebuilding the WAVs.
- Temporal Scaffold is rendered through the same renderer pipeline as the other public tracks.

## Notes

- The public-facing descriptions live on [index.html](index.html), not here.
- `manifest.csv` remains available as auxiliary metadata for segment boundaries and labels.
- The audio supplement is qualitative support; the quantitative results live in
  the experiment outputs under [../experiments/plots/README.md](../experiments/plots/README.md).
- There is no separate public track for the supplementary spectral-to-temporal
  bridge check.
