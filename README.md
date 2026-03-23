# Conchordal: Emergent Harmony via Direct Cognitive Coupling in a Psychoacoustic Landscape

Paper, generated data, audio supplement, and experiment code.

## Quick Start

- **[Supplementary PDF](supplementary.pdf)** — extended methods, ablations, and tables
- **[Audio Supplement](supplementary_audio/index.html)** — browser-based listening examples with track notes and links back to the paper and data
- **[Experiments](experiments/README.md)** — experiment-code overview, data layout, and reproduction commands

## Abstract

Conchordal is a bio-acoustic instrument for generative composition whose sonic agents are governed by artificial life dynamics within a psychoacoustic fitness landscape defined in log-frequency space. Four experiments demonstrate self-organised polyphony, consonance-driven selection, consonance-gated entrainment, and hereditary pitch adaptation.

## Repository structure

```
main.tex                  Main paper
supplementary.tex         Supplementary material
conc.bib                  Bibliography
alifeconf.sty             Conference style
experiments/
  README.md               Experiment runner guide
  main.rs                 CLI entry point
  paper_plots.rs          All experiment and plot logic
  sim.rs                  Simulation harness
  scenarios/              Rhai demo scripts for headless checks
  plots/                  Generated figures, tables, and summaries
supplementary_audio/
  README.md               Audio rendering and reproduction guide
  index.html              Browser-facing audio page
  audio/                  Rendered WAV files
justfile                  Build recipes
```

## Navigation

- For paper-facing methods and ablations, start from [supplementary.pdf](supplementary.pdf).
- For listening, start from [supplementary_audio/index.html](supplementary_audio/index.html).
- For code, generated data, and exact run commands, start from [experiments/README.md](experiments/README.md).

## Dependencies

- **Rust** (edition 2024)
- **Conchordal** v0.3.0: <https://github.com/ktakahashi74/conchordal>
  Clone to a sibling directory and check out the `v0.3.0` tag:
  ```bash
  git clone https://github.com/ktakahashi74/conchordal.git ../conchordal
  cd ../conchordal && git checkout v0.3.0
  ```
- **pdflatex** with standard packages (natbib, hyperref, booktabs, amsmath, cleveref)
- **rsvg-convert** or **inkscape** for SVG-to-PDF conversion
- [**just**](https://github.com/casey/just) task runner (optional but recommended)

## Reproducing the paper

### Full pipeline

```bash
just all
```

This runs all experiments (release build), converts SVG plots to PDF, and builds `main.pdf`.
Generated figures, CSVs, and summary tables are written under `experiments/plots/`.
Audio tracks are written under `supplementary_audio/audio/`.

### Step by step

```bash
# Run experiments (outputs to experiments/plots/)
just paper

# Convert SVG to PDF
just svg2pdf

# Build paper PDF
just latex
```

### Individual experiments

```bash
just paper --exp e2          # Self-Organised Polyphony only
just paper --exp e1,e3       # Multiple experiments
just paper --clean --exp e2  # Clean plots directory first
```

Historical internal experiment IDs (`E1`...`E7`) are still used in the code and
output directories. The paper-facing assay names and their internal mappings are
documented in [experiments/README.md](experiments/README.md).

### Verify

```bash
cargo check --bin paper
cargo test --bin paper
```

## License

See the Conchordal repository for license details.
