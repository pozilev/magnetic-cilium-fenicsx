#!/usr/bin/env bash
set -euo pipefail

MAG_SWEEP_CONFIG="${MAG_SWEEP_CONFIG:-configs/magnetic_sensor_position_sweep.yaml}"
INTERP_CONFIG="${INTERP_CONFIG:-configs/interpolate_sensor_position.yaml}"
MASTER_CSV="${MASTER_CSV:-../results/magnetic_results_master.csv}"
INTERP_OUT="${INTERP_OUT:-../results/interpolation/sensor_position}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$MASTER_CSV")" "$INTERP_OUT"

echo "Running magnetic sweep..."
python -m magnetic_cilium.cli.main sweep "$MAG_SWEEP_CONFIG" -- --master-csv-path "$MASTER_CSV"

echo "Running interpolation..."
python -m magnetic_cilium.cli.main interpolate "$INTERP_CONFIG" -- --input-csv "$MASTER_CSV" --output-dir "$INTERP_OUT"

echo "Done."
echo "master_csv = $MASTER_CSV"
echo "interpolation_report = $INTERP_OUT/interpolation_report.json"
echo "next_points = $INTERP_OUT/next_points.csv"
