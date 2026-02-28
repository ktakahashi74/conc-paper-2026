# Paper Experiments

Experiment runner and plot generator for all figures and statistical analyses in the paper.

## Experiment mapping

| Paper        | Internal ID | Topic                                          |
|--------------|-------------|------------------------------------------------|
| Experiment 1 | E1, E2      | Landscape Attractors & Self-Organised Polyphony |
| Experiment 2 | E3          | Consonance as Selection Pressure               |
| Experiment 3 | E5          | Rhythmic Entrainment                           |
| *(skipped)*  | E4          | *(excluded from paper)*                        |
| Experiment 4 | E6          | Hereditary Adaptation                          |

## Quick start

```bash
# Run all experiments (release build) + convert SVG→PDF + build paper
just all

# Run a single experiment
just paper --exp e2

# Run experiments + SVG→PDF conversion (no LaTeX)
just paper-pdf --exp e2
```

When `--exp` is omitted, the default set `e1,e2,e3,e5,e6` is run.

## Build and test

```bash
cargo check --bin paper
cargo test --bin paper
```

## Options

```bash
cargo run --release --bin paper -- --exp e3        # Single experiment
cargo run --release --bin paper -- --exp e1,e3     # Multiple experiments
cargo run --release --bin paper -- --clean --exp e2 # Clean output first
```

## Output

All outputs are written to `experiments/plots/<exp>/` (e.g., `experiments/plots/e2/`).

- `.svg` — vector plots
- `.pdf` — converted from SVG via `rsvg-convert` or `inkscape`
- `.csv` — raw data tables
- `.txt` — statistical summaries and comparison reports

Key output files per experiment:

| Experiment | Main figure | Key data files |
|------------|-------------|----------------|
| E1 | `paper_e1_landscape_scan_anchor220.svg` | `paper_e1_anchor_robustness.txt` |
| E2 | `paper_e2_figure_e2_1.svg`, `paper_e2_figure_e2_2.svg` | `paper_e2_shuffled_comparison.txt`, `paper_e2_terrain_controls.txt`, `paper_e2_coefficient_sweep.txt` |
| E3 | `paper_e3_figure4.svg` | `paper_e3_lifetimes.csv` |
| E5 | `paper_e5_figure.svg` | `paper_e5_summary.csv` |
| E6 | `paper_e6_figure.svg`, `paper_e6_integration_figure.svg` | `paper_e6_summary.csv` |

Use `--clean` when you need strict reproducibility from a fully fresh output tree.

## Concurrency lock

`paper` rejects concurrent runs via a lock at `experiments/.paper_plots.lock`.
If a previous run crashed, remove that lock directory and retry.

## Headless demo scenarios

Lightweight Rhai scripts for quick behavioural checks (not for plot generation):

```bash
cargo run -- --nogui experiments/scenarios/e1_landscape_scan_demo.rhai
```

Available scenarios:
- `e1_landscape_scan_demo.rhai`
- `e2_emergent_harmony_demo.rhai`
- `e3_metabolic_selection_demo.rhai`
- `e5_rhythmic_entrainment_demo.rhai`
