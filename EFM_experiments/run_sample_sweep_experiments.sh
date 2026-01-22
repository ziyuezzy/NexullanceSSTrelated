#!/bin/bash
# Run EFM sample sweep experiments
# Reproduces archive/EFM_experiments/RRG_4CPE with the new framework
#
# Usage:
#   ./run_sample_sweep_experiments.sh                    # Use defaults (RRG, V=36, D=5)
#   ./run_sample_sweep_experiments.sh --topo DDF --V 40  # Custom topology
#   ./run_sample_sweep_experiments.sh --help             # Show all options

cd "$(dirname "$0")"
python3.12 run_sample_sweep_experiments.py "$@"
