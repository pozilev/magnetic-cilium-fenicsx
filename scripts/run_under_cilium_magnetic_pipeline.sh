#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SENSOR_AREA_CONFIG="${SENSOR_AREA_CONFIG:-configs/magnetic_under_cilium_sensor_area_sweep.yaml}"
BR_AREA_CONFIG="${BR_AREA_CONFIG:-configs/magnetic_under_cilium_br_z_sweep.yaml}"
FEM_COMPARISON_CONFIG="${FEM_COMPARISON_CONFIG:-configs/magnetic_under_cilium_fem_dipole_comparison.yaml}"
DIPOLE_COMPARISON_CONFIG="${DIPOLE_COMPARISON_CONFIG:-configs/magnetic_under_cilium_dipole_comparison.yaml}"
MASTER_CSV="${MASTER_CSV:-results/magnetic_results_master.csv}"

# Set RUN_SENSOR_AREA=0, RUN_BR_AREA=0, RUN_COMPARISON=0 to skip stages.
RUN_SENSOR_AREA="${RUN_SENSOR_AREA:-1}"
RUN_BR_AREA="${RUN_BR_AREA:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$MASTER_CSV")"

echo "Under-cilium magnetic experiment pipeline"
echo "project_dir = $PROJECT_DIR"
echo "master_csv  = $MASTER_CSV"
echo

echo "Approximate runtime on Ryzen 5 3500U / 8 GB RAM:"
echo "  sensor-area FEM sweep: 5 cases  -> about 10-20 min"
echo "  Br x area FEM sweep:   25 cases -> about 50-100 min"
echo "  FEM + dipole compare:  1 FEM + 1 dipole -> about 2-5 min"
echo "  total if all enabled: about 1-2 hours, depending on mesh size and swapping"
echo

if [[ "$RUN_SENSOR_AREA" == "1" ]]; then
  echo "Running sensor-area FEM sweep..."
  python main.py --config "$SENSOR_AREA_CONFIG" --master-csv-path "$MASTER_CSV"
else
  echo "Skipping sensor-area FEM sweep."
fi

if [[ "$RUN_BR_AREA" == "1" ]]; then
  echo "Running Br x sensor-area FEM sweep..."
  python main.py --config "$BR_AREA_CONFIG" --master-csv-path "$MASTER_CSV"
else
  echo "Skipping Br x sensor-area FEM sweep."
fi

if [[ "$RUN_COMPARISON" == "1" ]]; then
  echo "Running under-cilium FEM comparison case..."
  python main.py --config "$FEM_COMPARISON_CONFIG" --master-csv-path "$MASTER_CSV"

  echo "Running under-cilium dipole comparison case..."
  python main.py --config "$DIPOLE_COMPARISON_CONFIG" --master-csv-path "$MASTER_CSV"
else
  echo "Skipping FEM/dipole comparison."
fi

echo
echo "Done."
echo "master_csv = $MASTER_CSV"
echo "sensor_area_summary = ../magnetic_fem_under_cilium_sensor_area_results/magnetic_under_cilium_sensor_area_summary.csv"
echo "br_area_summary = ../magnetic_fem_under_cilium_br_sensor_area_results/magnetic_under_cilium_br_sensor_area_summary.csv"
echo "fem_comparison_summary = ../magnetic_fem_under_cilium_comparison_results/magnetic_under_cilium_fem_comparison_summary.csv"
echo "dipole_summary = ../magnetic_cilium_3d_results_final/magnetic_summary.csv"
