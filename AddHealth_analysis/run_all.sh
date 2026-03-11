#!/usr/bin/env bash
# run_all.sh
# Run the MCMC sampler on every .gml file in dominance_networks/ and write
# samples to samples/.
#
# Usage (from AddHealth_analysis/):
#   ./run_all.sh [--num-samples N] [--sweeps-per-sample N] [--seed N]
# ./run_all.sh --num-samples 1000 --sweeps-per-sample 100
# The binary must already be compiled; see README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/run_sampler_to_csv"
INPUT_DIR="$SCRIPT_DIR/dominance_networks"
OUTPUT_DIR="$SCRIPT_DIR/samples"

# Forward any extra arguments (--num-samples, --sweeps-per-sample, --seed)
# directly to the binary.
EXTRA_ARGS=("$@")

# Check binary exists
if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found at $BINARY"
    echo "Compile first – see README.md."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Collect .gml files
shopt -s nullglob
gml_files=("$INPUT_DIR"/*.gml)

if [[ ${#gml_files[@]} -eq 0 ]]; then
    echo "No .gml files found in $INPUT_DIR"
    exit 0
fi

echo "Found ${#gml_files[@]} network(s). Writing samples to $OUTPUT_DIR/"
echo ""

for gml in "${gml_files[@]}"; do
    name="$(basename "$gml" .gml)"
    out="$OUTPUT_DIR/${name}_samples.csv"
    echo "==> $name"
    "$BINARY" --input-gml "$gml" --output-csv "$out" "${EXTRA_ARGS[@]}"
    echo ""
done

echo "Done. All samples written to $OUTPUT_DIR/"
