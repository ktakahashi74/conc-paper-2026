# Paper experiments

Example runner for generating plots used in the paper.

Paper scenarios used for headless behavioral checks are under `experiments/scenarios/`.

## Run

```bash
cargo run --bin paper
```

`--exp` を省略した場合は論文向けの既定セット `e1,e2,e3,e5` のみ実行します。
`e4` は `--exp e4` を明示した場合にのみ実行されます。

## Just recipes

```bash
just --justfile experiments/justfile paper --exp e2
```

```bash
just --justfile experiments/justfile paper-pdf --exp e2
```

`paper-pdf` runs the plot generator and converts all emitted SVG files in
`experiments/plots/` to PDF. Conversion uses `rsvg-convert` or `inkscape`.
`paper` rejects concurrent runs with a lock at `experiments/.paper_plots.lock`.
If a previous run crashed, remove that lock directory and retry.

## Build check

```bash
cargo check --bin paper
```

```bash
cargo check --bins
```

```bash
cargo check --all-targets
```

```bash
cargo test --bins
```

## Options

```bash
cargo run --bin paper -- --exp e3
```

```bash
cargo run --bin paper -- --exp e4 --e4-hist on
```

```bash
cargo run --bin paper -- --exp e4 --e4-debug-fit-metrics on
```

```bash
cargo run --bin paper -- --exp e4 --e4-env-partials 9 --e4-env-decay 0.8
```

```bash
cargo run --bin paper -- --exp e4 --e4-dyn-exploration 0.9 --e4-dyn-persistence 0.1 --e4-dyn-step-cents 75
```

E2 uses the `dissonance_then_consonance` phase schedule.
E4 fit debug CSV outputs are disabled by default and emitted only with `--e4-debug-fit-metrics on`.

Outputs are written to `experiments/plots/<exp>/` (for example, `experiments/plots/e2/`).
Plot images are emitted as `.svg` files (vector output).
Default behavior: only selected experiment directories are cleared and regenerated.
Use `--clean` to clear `experiments/plots` entirely before generation.

## Manual verification

```bash
cargo check --bin paper
cargo check --bins
cargo check --all-targets
cargo test --bins
cargo run --bin paper -- --clean --exp e4
find experiments/plots/e4 -maxdepth 1 -type f | rg 'binding_metrics|binding_summary|harmonic_tilt|binding_phase_diagram' || true
```

Primary E4 outputs to verify:
- `paper_e4_binding_metrics_raw.csv`
- `paper_e4_binding_metrics_summary.csv`
- `paper_e4_harmonic_tilt.png`
- `paper_e4_binding_phase_diagram.png`

Use `--clean` when you need strict reproducibility from a fully fresh `plots` tree.

## Paper scenarios (headless demos)

```bash
cargo run -- --nogui experiments/scenarios/e1_landscape_scan_demo.rhai
```

Files:
- `e1_landscape_scan_demo.rhai`
- `e2_emergent_harmony_demo.rhai`
- `e3_metabolic_selection_demo.rhai`
- `e4_mirror_sweep_demo.rhai`
- `e4_mirror_sweep_between_runs.rhai`
- `e5_rhythmic_entrainment_demo.rhai`
