# Architecture Cleanup Report

This file records the non-destructive cleanup performed while introducing the
`magnetic_cilium/` architecture facade.

## Result directories moved

Runtime result directories were moved out of the source package and into the
repository-level `results/` directory. Existing source configs and scripts now
refer to those paths through `../results/...` when launched from
`magnetic_cilium_pipeline/`.

Duplicate legacy result trees named `magnetic_cilium_3d_results_final` were
kept under `results/archives/` instead of being overwritten.

## Files not required for numerical calculations

The following file groups are not part of mechanics, magnetostatics, sweeps, or
postprocessing runs:

- Python bytecode caches: `__pycache__/`, `*.pyc`.
- LaTeX build byproducts: `*.aux`, `*.log`, `*.fls`, `*.fdb_latexmk`,
  `*.synctex.gz`, `*.toc`, `*.out`.
- Built report artifacts such as generated `*.pdf` and `*.docx` files.
- Historical prototype material under `legecy/`.
- Literature/reference PDFs under `source/`.

They were identified, but not deleted, to keep this migration reversible.
