# Experiments and Plot Generation

This directory contains the experiment runner and plot generator used to produce
the paper's figures, summaries, and exported data tables.

Quick links:

- [Top-level README](../README.md)
- [Generated Data README](plots/README.md)
- [Paper PDF](../main.pdf)
- [Supplementary PDF](../supplementary.pdf)
- [Audio Supplement Page](../supplementary_audio/index.html)

## Paper-facing assay names and internal IDs

The manuscript uses paper-facing assay names. The code keeps historical internal
IDs (`E1`...`E7`) for continuity.

| Paper-facing assay | Main figure | Internal ID(s) | Notes |
|--------------------|-------------|----------------|-------|
| Self-Organized Polyphony | Fig. 2 | `E1`, `E2` | landscape attractors and local-search polyphony |
| Consonance as Selection Pressure | Fig. 3 | `E3` | metabolic selection assay |
| Hereditary Adaptation | Fig. 4 | `E6b` | current paper-facing hereditary assay |
| Temporal Scaffold | Fig. 5 | `E7` | timing scaffold assay |
| Legacy hereditary assay | — | `E6` | previous hereditary implementation, kept for reference |
| Supplementary mechanism check | — | `E5` | spectral–temporal bridge check |
| Skipped internal slot | — | `E4` | not used in the current paper |

## Quick start

```bash
# Run all paper-facing experiments, convert SVG→PDF, and build the paper
just all

# Run experiments only (release build)
just paper

# Run a subset
just paper --exp e2
just paper --exp e6b,e7

# Clean generated outputs first
just paper --clean --exp e2
```

When `--exp` is omitted, the default set `e1,e2,e3,e6b,e7` is run. This covers
the current paper-facing assay families and leaves the legacy hereditary assay
(`e6`) out of the default pipeline.

## Running directly

```bash
cargo run --release --bin paper -- --exp e2
cargo run --release --bin paper -- --exp e6b,e7
cargo run --release --bin paper -- --clean --exp e2
```

## Generated outputs

All generated artifacts are written under `experiments/plots/<id>/`.

For the output layout, file types, and the key paper-facing figure/table files,
see [plots/README.md](plots/README.md).

In brief:

- `.svg` — source vector plots
- `.pdf` — converted figure PDFs
- `.csv` — exported numeric tables
- `.txt` — summary reports and comparison notes

## Build and test

```bash
cargo check --bin paper
cargo test --bin paper
```

## Concurrency lock

`paper` rejects concurrent runs via a lock at `experiments/.paper_plots.lock`.
If a previous run crashed, remove that directory and retry.

## Headless demo scenarios

Lightweight Rhai scripts for quick behavioural checks live in
`experiments/scenarios/`. They are not the figure-generation pipeline.

Example:

```bash
cargo run -- --nogui experiments/scenarios/e1_landscape_scan_demo.rhai
```
