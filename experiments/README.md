# Experiments, Code, and Data

This directory contains the experiment runner and plot generator used to produce
the paper's figures, summaries, and exported data tables.

Quick links:

- [Top-level README](../README.md)
- [Generated Data README](plots/README.md)
- [Supplementary PDF](../supplementary.pdf)
- [Audio Supplement Page](https://ktakahashi74.github.io/conc-paper-2026/supplementary_audio/)
- [Audio Rendering README](../supplementary_audio/README.md)

## What this directory contains

The main paper pipeline lives here.

- `main.rs`: CLI entry point
- `paper_plots.rs`: experiment orchestration, figure generation, CSV export, and audio-scenario generation
- `sim.rs`: shared simulation harness for the paper assays
- `scenarios/`: lightweight Rhai demos and checks
- `plots/`: generated figures, CSVs, summaries, and tables

If you want to understand how the code maps to the paper, this is the main
entry point. If you want only the rendered outputs, go straight to
[plots/README.md](plots/README.md).

## Paper-facing assay names and internal IDs

The manuscript uses paper-facing assay names. The code keeps historical internal
IDs (`E1`...`E7`) for continuity.

| Paper-facing assay | Main figure | Internal ID(s) | Notes |
|--------------------|-------------|----------------|-------|
| Self-Organized Polyphony | Fig. 2 | `E1`, `E2` | landscape attractors and local-search polyphony |
| Consonance as Selection Pressure | Fig. 3 | `E3` | metabolic selection assay |
| Hereditary Adaptation | Fig. 4 | `E6b` | current paper-facing hereditary assay |
| Temporal Scaffold | Fig. 5 | `E7` | timing scaffold assay |
| Supplementary mechanism check | — | `E5` | spectral–temporal bridge check |

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
the current paper-facing assay families. Legacy internal assays removed from the
current paper pipeline are intentionally not runnable through `--exp`.

## Running directly

```bash
cargo run --release --bin paper -- --exp e2
cargo run --release --bin paper -- --exp e6b,e7
cargo run --release --bin paper -- --clean --exp e2
```

## Generated data overview

All generated artifacts are written under `experiments/plots/<id>/`.

The current paper-facing figure mapping is:

| Figure | Assay name | Output directory | Files to inspect first |
|--------|------------|------------------|------------------------|
| Fig. 2 | Self-Organized Polyphony | `plots/e2/` | `paper_e2_figure_e2_1.pdf`, `paper_e2_figure_e2_2.pdf`, `paper_e2_summary.csv` |
| Fig. 3 | Consonance as Selection Pressure | `plots/e3/` | `paper_e3_figure4.pdf`, `paper_e3_lifetimes.csv` |
| Fig. 4 | Hereditary Adaptation | `plots/e6b/` | `paper_e6b_figure.pdf`, `paper_e6b_summary.txt`, `paper_e6b_endpoint_metrics.csv` |
| Fig. 5 | Temporal Scaffold | `plots/e7/` | `paper_e7_figure.pdf`, `paper_e7_summary.csv` |

Common file types:

- `.svg`: source vector plots written by the Rust pipeline
- `.pdf`: paper-facing figure PDFs
- `.csv`: numeric tables and time series
- `.txt`: compact summaries and comparison notes

For a directory-by-directory index, including legacy outputs, see
[plots/README.md](plots/README.md).

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
