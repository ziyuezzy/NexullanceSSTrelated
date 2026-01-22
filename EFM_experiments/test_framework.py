#!/usr/bin/env python3
"""
Quick test of EFM experiments framework.
Runs a single small test case for each benchmark.
"""

import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import run_ember_simulation, run_ember_experiment_with_nexullance


def main():
    """Run quick test of EFM framework."""
    print("\n" + "="*80)
    print("EFM EXPERIMENTS FRAMEWORK TEST")
    print("="*80)
    print("Running small test cases to verify framework functionality")
    print("="*80)
    
    # Common configuration
    topo_name = "RRG"
    V = 36
    D = 5
    cores_per_ep = 4
    link_bw = 16
    num_threads = 8
    
    # Test 1: Allreduce with shortest-path
    print("\n" + "="*80)
    print("Test 1: Allreduce with shortest-path routing")
    print("="*80)
    success1 = run_ember_simulation(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark="Allreduce",
        bench_args=" iterations=10 count=256",
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        enable_traffic_trace=False,
        routing_method="shortest_path"
    )
    print(f"{'✓ PASSED' if success1 else '✗ FAILED'}")
    
    # Test 2: Allreduce with UGAL
    print("\n" + "="*80)
    print("Test 2: Allreduce with UGAL routing")
    print("="*80)
    success2 = run_ember_simulation(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark="Allreduce",
        bench_args=" iterations=10 count=256",
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        enable_traffic_trace=False,
        routing_method="ugal"
    )
    print(f"{'✓ PASSED' if success2 else '✗ FAILED'}")
    
    # Test 3: Allreduce with Nexullance (full workflow)
    print("\n" + "="*80)
    print("Test 3: Allreduce with Nexullance optimization")
    print("="*80)
    results = run_ember_experiment_with_nexullance(
        topo_name=topo_name,
        V=V,
        D=D,
        benchmark="Allreduce",
        bench_args=" iterations=10 count=256",
        cores_per_ep=cores_per_ep,
        link_bw=link_bw,
        num_threads=num_threads,
        traffic_collection_rate="10us",
        demand_scaling_factor=10.0
    )
    success3 = results is not None
    if success3:
        print(f"Speedup: {results.get('speedup', 'N/A'):.4f}x")
    print(f"{'✓ PASSED' if success3 else '✗ FAILED'}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Shortest-path test:  {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"UGAL test:           {'✓ PASSED' if success2 else '✗ FAILED'}")
    print(f"Nexullance test:     {'✓ PASSED' if success3 else '✗ FAILED'}")
    print("="*80)
    
    all_passed = success1 and success2 and success3
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Framework is ready!\n")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Check errors above\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
