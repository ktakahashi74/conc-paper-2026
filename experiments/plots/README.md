# Generated Data

This directory holds the generated figures, CSV tables, and text summaries
written by the paper experiment pipeline.

Quick links:

- [Top-level README](../../README.md)
- [Experiments, Code, and Data](../README.md)
- [Supplementary PDF](../../supplementary.pdf)
- [Audio Supplement Page](../../supplementary_audio/)
- [Supplementary Audio README](../../supplementary_audio/README.md)

## Current paper-facing directories

The current manuscript uses the following generated output directories:

| Figure | Assay name | Directory | First files to inspect |
|--------|------------|-----------|------------------------|
| Fig. 2 | Consonance Search | `e2/` | `paper_e2_figure_e2_1.pdf`, `paper_e2_figure_e2_2.pdf`, `paper_e2_summary.csv` |
| Fig. 3 | Consonance as Selection Pressure | `e3/` | `paper_e3_figure4.pdf`, `paper_e3_summary_by_seed.csv`, `paper_e3_seed_level_stats.txt` |
| Fig. 4 | Hereditary Adaptation | `e6b/` | `paper_e6b_figure.pdf`, `paper_e6b_endpoint_metrics.csv`, `paper_e6b_timeseries.csv` |
| Fig. 5 | Temporal Scaffold | `e7/` | `paper_e7_figure.pdf`, `paper_e7_summary.csv` |
| Supplementary mechanism check | `e5/` | `paper_e5_figure.pdf`, `paper_e5_summary.csv` |
| Supplementary landscape analysis | `e1/` | `paper_e1_landscape_scan_anchor220.pdf`, `paper_e1_anchor_robustness.txt` |

## File types

- `.svg`: vector plots written directly by the Rust pipeline
- `.pdf`: paper-facing figure exports
- `.csv`: numeric tables and seed-level detail
- `.txt`: compact summaries, representative seeds, and notes

## Legacy directories

Some old branches or historical working trees may also contain directories such
as `e4/` or `e6/`. These are legacy internal assays and are not part of the
current paper-facing pipeline.
