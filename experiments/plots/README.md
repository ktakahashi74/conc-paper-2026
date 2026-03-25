# Generated Data Guide

This directory holds the generated figures, CSV tables, and text summaries
produced by the paper pipeline.

Quick links:

- [Top-level README](../../README.md)
- [Experiments README](../README.md)
- [Supplementary PDF](../../supplementary.pdf)
- [Audio Supplement Page](https://ktakahashi74.github.io/conc-paper-2026/supplementary_audio/)

## Directory map

| Directory | Paper-facing assay | Main output(s) |
|-----------|--------------------|----------------|
| `e1/` | landscape scan / attractor context | `paper_e1_landscape_scan_anchor220.{svg,pdf}` |
| `e2/` | Consonance Search | `paper_e2_figure_e2_1.{svg,pdf}`, `paper_e2_figure_e2_2.{svg,pdf}`, `paper_e2_summary.csv` |
| `e3/` | Consonance as Selection Pressure | `paper_e3_figure4.{svg,pdf}`, `paper_e3_summary_by_seed.csv`, `paper_e3_seed_level_stats.txt` |
| `e5/` | supplementary bridge check | `paper_e5_figure.{svg,pdf}`, `paper_e5_summary.csv` |
| `e6b/` | Hereditary Adaptation | `paper_e6b_figure.{svg,pdf}`, `paper_e6b_endpoint_metrics.csv`, `paper_e6b_timeseries.csv` |
| `e7/` | Temporal Scaffold | `paper_e7_figure.{svg,pdf}`, `paper_e7_summary.csv` |

## Canonical regeneration

From the repository root:

```bash
# Main paper-facing experiment set
just paper

# Fig. 2 supplementary controls and terrain diagnostics
just paper --exp e2 --e2-diagnostics

# Convert SVG outputs to PDF
just svg2pdf
```

## Notes

- `e2/` contains both the main Fig. 2 outputs and the supplementary control /
  terrain dumps used in the supplement.
- Many directories also contain helper SVG files and CSV tables that are useful
  for verification but are not directly included in the manuscript.
- The repository tracks only the paper-facing figure PDFs. Bulk PDFs produced by
  `just svg2pdf` for auxiliary SVGs are intentionally ignored in Git.
