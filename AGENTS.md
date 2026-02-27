# AGENTS.md — conc-paper

## Project Overview

ALIFE conference paper for **Conchordal**: an artificial life system where sonic agents inhabit a psychoacoustic fitness landscape.

## Reference Materials

Conchordal source repository is located at `../conchordal/` (relative to this repo root).

### Concept & Design Documents

| Document | Path | Contents |
|---|---|---|
| Manifesto | `../conchordal/web/content/manifesto.md` | Philosophical background, Direct Cognitive Coupling principle, historical context from Pythagoras to spectralism |
| Technical Note | `../conchordal/web/content/technote.md` | Full technical details: Log2Space, ERB, NSGT, Roughness (Plomp-Levelt), Harmonicity (Sibling Projection), ALife engine, Kuramoto entrainment, Rust architecture |

Japanese translations are also available at the same location (`manifesto.ja.md`, `technote.ja.md`).

### Source Code

Conchordal is a Rust project (`../conchordal/Cargo.toml`). Main source is under `../conchordal/src/`.

| Area | Path | Description |
|---|---|---|
| Core DSP | `../conchordal/src/core/` | Landscape, NSGT, roughness/harmonicity kernels, ERB |
| ALife Engine | `../conchordal/src/life/` | Individual, PitchCore, ArticulationCore, metabolism |
| Scenarios | `../conchordal/examples/` | General Rhai scripts used by the system |

### Paper Experiment Code

Experiment runner lives in this repository at `experiments/`.

| File | Role |
|---|---|
| `experiments/paper_plots.rs` | Plot generation logic for all experiments |
| `experiments/sim.rs` | Simulation harness shared across experiments |
| `experiments/main.rs` | CLI entry point (`cargo run --bin paper -- --exp <name>`) |
| `experiments/scenarios/*.rhai` | Rhai scenario scripts used by each experiment |

See `experiments/README.md` for run instructions and available options.

### Experiment ID Mapping

The paper uses sequential numbering (Experiment 1–5). The internal experiment IDs in the codebase differ. E4 is excluded from the paper.

| Paper | Internal ID | Topic |
|---|---|---|
| Experiment 1 | E1 | Landscape Attractors |
| Experiment 2 | E2 | Self-Organized Polyphony |
| Experiment 3 | E3 | Consonance as Selection Pressure |
| Experiment 4 | E5 | Rhythmic Entrainment |
| Experiment 5 | E6 | Hereditary Adaptation |

## Workflow

All build tasks are defined in the top-level `justfile`. Always use `--release` for experiments (debug builds are prohibitively slow); the justfile handles this automatically.

| Command | What it does |
|---|---|
| `just all` | Full pipeline: run experiments → SVG→PDF → pdflatex |
| `just paper-pdf` | Run experiments + SVG→PDF conversion |
| `just paper` | Run experiments only (release build) |
| `just svg2pdf` | Convert SVG plots to PDF (rsvg-convert or inkscape) |
| `just latex` | Build `main.pdf` with pdflatex |
| `just paper-check` | `cargo check` + `cargo test` |
| `just clean` | Remove `experiments/plots` and stale lock |

Pass extra args to experiment recipes, e.g.:

```bash
just all --clean --exp e2
```

**Important**: After regenerating any figure (SVG→PDF), always run `just latex` to rebuild the paper PDF. Never leave figures updated without rebuilding latex.
