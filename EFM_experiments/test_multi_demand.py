#!/usr/bin/env python3
"""
Test multi-demand Nexullance for EFM experiments.
Tests both SD (single-demand) and MD (multi-demand) methods.
"""

import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import run_ember_experiment_with_nexullance


def main():
    """Test both single-demand and multi-demand Nexullance."""
    print("\n" + "="*80)
    print("MULTI-DEMAND NEXULLANCE TEST FOR EFM")
    print("="*80)
    print("This will test both Nexullance optimization methods:")
    print("  1. SD (Single-Demand): Optimizes for accumulated demand")
    print("  2. MD (Multi-Demand): Optimizes for multiple sampled demands")
    print("="*80)
    
    # Common configuration
    topo_name = "RRG"
    V = 36
    D = 5
    benchmark = "Allreduce"
    bench_args = " iterations=10 count=256"
    cores_per_ep = 4
    link_bw = 16
    num_threads = 8
    
    # Test 1: Single-Demand Nexullance SD (default)
    print("\n" + "="*80)
    print("Test 1: Single-Demand Nexullance SD")
    print("="*80)
    results_sd = run_ember_experiment_with_nexullance(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark=benchmark,
        bench_args=bench_args,
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        traffic_collection_rate="10us",
        demand_scaling_factor=10.0,
        nexullance_method="SD"  # Single-demand
    )
    
    success_sd = results_sd is not None
    if success_sd:
        print(f"\n✓ SD Method - Speedup: {results_sd.get('speedup', 'N/A'):.4f}x")
    else:
        print("\n✗ SD Method FAILED")
    
    # Test 2: Multi-Demand Nexullance MD
    print("\n" + "="*80)
    print("Test 2: Multi-Demand Nexullance MD")
    print("="*80)
    results_md = run_ember_experiment_with_nexullance(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark=benchmark,
        bench_args=bench_args,
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        traffic_collection_rate="10us",
        demand_scaling_factor=10.0,
        nexullance_method="MD",  # Multi-demand
        num_demand_samples=4,     # Number of demand samples
        max_path_length=4         # Maximum path length for MD
    )
    
    success_md = results_md is not None
    if success_md:
        print(f"\n✓ MD Method - Speedup: {results_md.get('speedup', 'N/A'):.4f}x")
    else:
        print("\n✗ MD Method FAILED")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"SD (Single-Demand):  {'✓ PASSED' if success_sd else '✗ FAILED'}")
    if success_sd:
        print(f"  Speedup: {results_sd.get('speedup', 'N/A'):.4f}x")
    
    print(f"MD (Multi-Demand):   {'✓ PASSED' if success_md else '✗ FAILED'}")
    if success_md:
        print(f"  Speedup: {results_md.get('speedup', 'N/A'):.4f}x")
    
    print("="*80)
    
    if success_sd and success_md:
        print("\n✓ ALL TESTS PASSED!\n")
        
        # Compare methods
        if results_sd and results_md:
            sd_speedup = results_sd.get('speedup', 0)
            md_speedup = results_md.get('speedup', 0)
            
            print("Method Comparison:")
            print(f"  SD Speedup: {sd_speedup:.4f}x")
            print(f"  MD Speedup: {md_speedup:.4f}x")
            
            if md_speedup > sd_speedup:
                improvement = ((md_speedup - sd_speedup) / sd_speedup) * 100
                print(f"  MD is {improvement:.2f}% better than SD")
            elif sd_speedup > md_speedup:
                improvement = ((sd_speedup - md_speedup) / md_speedup) * 100
                print(f"  SD is {improvement:.2f}% better than MD")
            else:
                print("  Both methods achieve similar performance")
        
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
