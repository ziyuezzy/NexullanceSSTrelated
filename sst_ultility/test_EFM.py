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
    Main entry point for testing EFM simulations with all routing methods.
    
    Tests all three routing methods:
    - Example 1: Shortest-path routing (baseline)
    - Example 2: UGAL adaptive routing
    - Example 3: Complete Nexullance optimization workflow
    """
    
    # Common configuration
    topo_name = "RRG"
    V = 16
    D = 5
    benchmark = "Allreduce"
    bench_args = " iterations=50 count=512"
    cores_per_ep = 1
    link_bw = 16
    num_threads = 8
    
    # Example 1: Test shortest-path routing (baseline)
    print("\n" + "="*80)
    print("Example 1: Testing SHORTEST-PATH routing (baseline)")
    print("="*80)
    success_sp = run_ember_simulation(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark=benchmark,
        bench_args=bench_args,
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        enable_traffic_trace=False,
        routing_method="shortest_path"
    )
    print(f"\n{'✓' if success_sp else '✗'} Shortest-path routing test {'PASSED' if success_sp else 'FAILED'}")
    
    # Example 2: Test UGAL adaptive routing
    print("\n" + "="*80)
    print("Example 2: Testing UGAL adaptive routing")
    print("="*80)
    success_ugal = run_ember_simulation(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark=benchmark,
        bench_args=bench_args,
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        enable_traffic_trace=False,
        routing_method="ugal"
    )
    print(f"\n{'✓' if success_ugal else '✗'} UGAL routing test {'PASSED' if success_ugal else 'FAILED'}")
    
    # Example 3: Complete Nexullance optimization workflow
    print("\n" + "="*80)
    print("Example 3: Testing NEXULLANCE optimization workflow")
    print("="*80)
    print("This workflow will:")
    print("  Step 1: Run shortest-path baseline and collect traffic demand")
    print("  Step 2: Run Nexullance-optimized simulation")
    print("  Step 3: Calculate and compare simulation times")
    print("="*80 + "\n")
    
    results = run_ember_experiment_with_nexullance(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark=benchmark,
        bench_args=bench_args,
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        traffic_collection_rate="10us",
        # Cap_core and Cap_access default to link_bw if not specified
        demand_scaling_factor=10.0
    )
    
    success_nexullance = results is not None
    
    if results:
        print("\n" + "="*80)
        print("NEXULLANCE WORKFLOW SUMMARY")
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
    else:
        print("\n" + "="*80)
        print("NEXULLANCE WORKFLOW FAILED!")
        print("="*80)
    
    print(f"\n{'✓' if success_nexullance else '✗'} Nexullance workflow test {'PASSED' if success_nexullance else 'FAILED'}")
    
    # Overall summary
    print("\n" + "="*80)
    print("ALL TESTS SUMMARY")
    print("="*80)
    print(f"Shortest-path routing:   {'✓ PASSED' if success_sp else '✗ FAILED'}")
    print(f"UGAL routing:            {'✓ PASSED' if success_ugal else '✗ FAILED'}")
    print(f"Nexullance workflow:     {'✓ PASSED' if success_nexullance else '✗ FAILED'}")
    print("="*80)
    
    all_passed = success_sp and success_ugal and success_nexullance
    if all_passed:
        print("\n✓ ALL TESTS PASSED!\n")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
 
