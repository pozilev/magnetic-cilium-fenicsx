# Architecture Cleanup Report

This file records the non-destructive cleanup performed while introducing the
`magnetic_cilium/` architecture facade.

## Result directories moved

Runtime result directories were moved out of the source package and into the
repository-level `results/` directory. Existing source configs and scripts now
refer to those paths through `../results/...` when launched from
`magnetic_cilium_pipeline/`.

Duplicate historical result trees named `magnetic_cilium_3d_results_final` were
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
- Historical prototype material formerly stored under `legecy/` was removed
  during the cleanup.
- Literature/reference PDFs under `source/`.

Reproducible caches/build byproducts and historical prototype code were
removed. Report deliverables and literature PDFs were kept because they are
useful diploma materials, not runtime source code.

## Removed obsolete compatibility files

After `magnetic_cilium/` became self-contained, the old compatibility modules
were removed from the source package root:

- `cli/`
- `config/`
- `io/`
- `tests/`
- root `__init__.py`

The root archive `magnetic_cilium_pipeline.zip` and the historical prototype
directory `legecy/` were also removed.

## Backend code moved into package

The old top-level runtime/backend modules were moved into architecture modules:

- `params.py` -> `magnetic_cilium/config/params.py`
- `logging_utils.py` -> `magnetic_cilium/io/logging_utils.py`
- `mechanics_model.py` -> `magnetic_cilium/mechanics/backend.py`
- `magnetics_dipoles.py` -> `magnetic_cilium/magnetics/dipole.py`
- `magnetics_fem.py` -> `magnetic_cilium/magnetics/fem_scalar_potential.py`
- `sensor_sampling.py` -> `magnetic_cilium/magnetics/sensor.py`
- `magnetic_results.py` -> `magnetic_cilium/io/master_table.py`
- `magnetic_interpolation.py` -> `magnetic_cilium/postprocess/interpolation.py`
- `pipeline.py` -> `magnetic_cilium/pipeline/execution.py`
- `main.py` -> `magnetic_cilium/cli/runtime.py`

The compatibility shim `magnetic_cilium/_compat.py` was removed after all
imports were rewired to package-local modules.

## Architecture layer status

The current source package now includes the following architecture-level APIs:

- `magnetic_cilium.config.adapters`: validated config to runtime solver params.
- `magnetic_cilium.mechanics.state`: typed `MechanicsState` and `MechanicsResult`.
- `magnetic_cilium.magnetics.state`: typed magnetic response objects.
- `magnetic_cilium.pipeline.full`: new orchestration layer with dry-run planning.
- `magnetic_cilium.io.restart`: explicit restart directory contract and manifest.
- `magnetic_cilium.postprocess.quality`: common quality report for mechanics and magnetic outputs.
- `magnetic_cilium.io.master_table`: optional `master.parquet` synchronization.
- `magnetic_cilium.visualization`: plot specifications for report figures.

The numerical FEM formulas now live inside the `magnetic_cilium/` package.
