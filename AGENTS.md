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
| Scenarios | `../conchordal/examples/` | Rhai scripts for experiments |

### Paper Experiment Code

Experiment runner lives at `../conchordal/examples/paper/`. Key files:

| File | Role |
|---|---|
| `paper_plots.rs` | Plot generation logic for all experiments |
| `sim.rs` | Simulation harness shared across experiments |
| `main.rs` | CLI entry point (`cargo run --example paper -- --exp <name>`) |
| `scenarios/*.rhai` | Rhai scenario scripts used by each experiment |

See `../conchordal/examples/paper/README.md` for run instructions and available options.

### Plots Symlink

`plots/` in this repo is a local symlink to `../conchordal/examples/paper/plots/`. It is listed in `.gitignore` and not tracked. Each collaborator should create the symlink locally or copy figures as needed.
