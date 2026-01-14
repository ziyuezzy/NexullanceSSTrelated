#!/usr/bin/env python3
"""
Test script for SST-EFM (Ember+Firefly+Merlin) simulations with Nexullance optimization.

This is the entry point for running the complete 5-step workflow:
  1. Run baseline EFM simulation to collect traffic demand matrix from MPI benchmark
  2. Collect baseline throughput statistics
  3. Run Nexullance optimization using collected demand
  4. Run optimized simulation with Nexullance routing
  5. Calculate and compare throughput improvements
"""

import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import run_ember_simulation, run_ember_experiment_with_nexullance


def main():
    """
    Main entry point for the complete Nexullance optimization workflow with EFM.
    
    Uncomment the example you want to run:
    - Example 1: Simple traffic demand collection only
    - Example 2: Complete 5-step workflow with optimization and comparison
    """
    
    # # Example 1: Simple traffic demand collection
    # print("\n" + "="*80)
    # print("Example 1: Collecting traffic demand matrix only")
    # print("="*80)
    # run_ember_simulation(
    #     topo_name="RRG",
    #     V=16,
    #     D=5,
    #     benchmark="Allreduce",
    #     bench_args=" iterations=50 count=512",
    #     cores_per_ep=1,
    #     link_bw=16,
    #     enable_traffic_trace=True
    # )
    
    # Example 2: Complete 5-step experiment with Nexullance optimization
    print("\n" + "="*80)
    print("COMPLETE EFM NEXULLANCE OPTIMIZATION WORKFLOW (5 STEPS)")
    print("="*80)
    print("This workflow will:")
    print("  Step 1: Run baseline EFM simulation and collect traffic demand")
    print("  Step 2: Run Nexullance optimization with collected demand")
    print("  Step 3: Calculate and compare throughput improvements")
    print("="*80 + "\n")
    
    results = run_ember_experiment_with_nexullance(
        topo_name="RRG",
        V=16,
        D=5,
        benchmark="Allreduce",
        bench_args=" iterations=50 count=512",
        cores_per_ep=1,
        link_bw=16,
        num_threads=8,
        traffic_collection_rate="10us",
        # Cap_core and Cap_access default to link_bw if not specified
        demand_scaling_factor=1.0
    )
    
    if results:
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"Demand matrix:           {results['demand_file']}")
        if results.get('baseline_sim_time_ms') is not None:
            print(f"Baseline sim time:       {results['baseline_sim_time_ms']:.4f} ms")
        else:
            print(f"Baseline sim time:       Not available")
        if results.get('optimized_sim_time_ms') is not None:
            print(f"Optimized sim time:      {results['optimized_sim_time_ms']:.4f} ms")
        else:
            print(f"Optimized sim time:      Not available")
        if results.get('speedup'):
            print(f"Speedup:                 {results['speedup']:.4f}x")
            print(f"Improvement:             {results['improvement_percent']:+.2f}%")
        else:
            print(f"Speedup:                 Could not calculate (missing baseline or optimized time)")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("EXPERIMENT FAILED!")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
 
