#!/bin/bash
# Run all EFM benchmark experiments
# Reproduces experiments from archive/EFM_experiments/RRG_4CPE/run_all.sh

echo "================================================================================"
echo "EFM BENCHMARK EXPERIMENTS - ROUTING COMPARISON"
echo "================================================================================"
echo "Testing three routing methods across multiple MPI benchmarks:"
echo "  - shortest_path (baseline)"
echo "  - ugal (adaptive routing)"
echo "  - nexullance (optimized routing)"
echo ""
echo "Topology: RRG (V=36, D=5) with 4 cores per endpoint"
echo "================================================================================"
echo ""

# Run the main experiment script
python3.12 run_experiments.py

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================================"
