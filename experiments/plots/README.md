# Generated Data and Figures

This directory holds generated outputs from the paper experiments. These files
are derived artifacts: they can be regenerated from the code in the repository
root and the commands documented in [`../README.md`](../README.md).

Quick links:

- [Top-level README](../../README.md)
- [Experiments README](../README.md)
- [Paper PDF](../../main.pdf)
- [Supplementary PDF](../../supplementary.pdf)
- [Audio Supplement Page](../../supplementary_audio/index.html)

## Directory layout

Each subdirectory corresponds to an internal experiment ID:

| Directory | Paper-facing role | Typical outputs to inspect first |
|-----------|-------------------|----------------------------------|
| `e1/` | Landscape attractors supplementing Self-Organized Polyphony | `paper_e1_landscape_scan_anchor220.pdf`, `paper_e1_anchor_robustness.txt` |
| `e2/` | Self-Organized Polyphony (Fig. 2) | `paper_e2_figure_e2_1.pdf`, `paper_e2_figure_e2_2.pdf`, `paper_e2_summary.csv` |
| `e3/` | Consonance as Selection Pressure (Fig. 3) | `paper_e3_figure4.pdf`, `paper_e3_lifetimes.csv` |
| `e5/` | Supplementary mechanism check | `paper_e5_figure.pdf`, `paper_e5_summary.csv` |
| `e6/` | Legacy hereditary assay | `paper_e6_figure.pdf`, `paper_e6_summary.csv` |
| `e6b/` | Hereditary Adaptation (Fig. 4) | `paper_e6b_figure.pdf`, `paper_e6b_summary.txt`, `paper_e6b_endpoint_metrics.csv` |
| `e7/` | Temporal Scaffold (Fig. 5) | `paper_e7_figure.pdf`, `paper_e7_summary.csv` |

## File types

- `.svg` — source vector plots produced by the Rust pipeline
- `.pdf` — paper-ready figure PDFs converted from SVG
- `.csv` — exported numeric tables and time series
- `.txt` — compact human-readable summaries

The paper itself embeds figure PDFs after SVG→PDF conversion. If you want the
same files the LaTeX build uses, inspect the `.pdf` files first.

## Paper-facing path through the data

If you are reviewing the paper and want the shortest route from manuscript to
data:

1. Read the figure caption in [main.pdf](../../main.pdf).
2. Open the corresponding `paper_*.pdf` in the matching subdirectory here.
3. Open the accompanying `.csv` or `.txt` summary in the same directory.

The current main-paper mappings are:

- Fig. 2 → `e2/`
- Fig. 3 → `e3/`
- Fig. 4 → `e6b/`
- Fig. 5 → `e7/`

## Regeneration

From the repository root:

```bash
# Regenerate paper-facing experiment outputs
just paper

# Convert SVG plots to PDF
just svg2pdf
```

For exact run flags and internal-ID mappings, see
[`../README.md`](../README.md).

## Notes

- Some directories contain exploratory or supplementary outputs alongside the
  main figure files.
- Audio files are **not** stored here; they live under
  [`../../supplementary_audio/`](../../supplementary_audio/README.md).
- The internal directory names are historical; the paper-facing assay names are
  documented in [`../README.md`](../README.md).
