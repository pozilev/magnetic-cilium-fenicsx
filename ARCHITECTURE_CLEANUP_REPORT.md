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

## Report materials moved

LaTeX report materials were moved out of the source package:

- Old location: `magnetic_cilium_pipeline/magnetic_cilium_tex/`.
- New location: `../magnetic_cilium/magnetic_cilium_tex/`.

This keeps `magnetic_cilium_pipeline/` focused on executable source code,
configuration, tests, and launch scripts. The report sources and generated
documents remain available at the project root.

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

## Architecture layer status

The current source package now includes the following architecture-level APIs:

- `magnetic_cilium.config.adapters`: validated config to legacy solver params.
- `magnetic_cilium.mechanics.state`: typed `MechanicsState` and `MechanicsResult`.
- `magnetic_cilium.magnetics.state`: typed magnetic response objects.
- `magnetic_cilium.pipeline.full`: new orchestration layer with dry-run planning.
- `magnetic_cilium.io.restart`: explicit restart directory contract and manifest.
- `magnetic_cilium.postprocess.quality`: common quality report for mechanics and magnetic outputs.
- `magnetic_cilium.io.master_table`: optional `master.parquet` synchronization.
- `magnetic_cilium.visualization`: plot specifications for report figures.

The numerical FEM formulas are still delegated to the validated legacy backend
until each solver block is migrated and regression-tested independently.
