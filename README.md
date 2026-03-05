# Conchordal: Emergent Harmony via Direct Cognitive Coupling in a Psychoacoustic Landscape

Paper and experiment code.

**[Supplementary Material (PDF)](supplementary.pdf)** — terrain controls, ablation details, and additional tables (S1–S7).

## Abstract

Conchordal is a bio-acoustic instrument for generative composition whose sonic agents are governed by artificial life dynamics within a psychoacoustic fitness landscape defined in log-frequency space. Four experiments demonstrate self-organised polyphony, consonance-driven selection, consonance-gated entrainment, and hereditary pitch adaptation.

## Repository structure

```
main.tex                  Main paper
supplementary.tex         Supplementary material
conc.bib                  Bibliography
alifeconf.sty             Conference style
experiments/
  main.rs                 CLI entry point
  paper_plots.rs          All experiment and plot logic
  sim.rs                  Simulation harness
  scenarios/              Rhai demo scripts for headless checks
justfile                  Build recipes
```

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

### Experiment mapping

| Paper        | Internal ID | Topic                        |
|--------------|-------------|------------------------------|
| Experiment 1 | E1, E2      | Landscape Attractors & Self-Organised Polyphony |
| Experiment 2 | E3          | Consonance as Selection      |
| Experiment 3 | E5          | Rhythmic Entrainment         |
| *(E4 skipped)* | E4        | *(excluded from paper)*      |
| Experiment 4 | E6          | Hereditary Adaptation        |

### Verify

```bash
cargo check --bin paper
cargo test --bin paper
```

## License

See the Conchordal repository for license details.
