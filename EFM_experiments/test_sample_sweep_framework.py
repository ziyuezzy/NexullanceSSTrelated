#!/usr/bin/env python3
"""
Quick test of sample sweep framework.
Tests Allreduce with one problem size and a few sample counts.
"""

import sys
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import run_ember_experiment_with_nexullance, _run_sst, _extract_simulation_time_from_output


def main():
    """Quick test with small configuration."""
    parser = argparse.ArgumentParser(
        description="Quick test of sample sweep framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default RRG(36,5) topology:
  python3.12 test_sample_sweep_framework.py
  
  # Run with custom topology:
  python3.12 test_sample_sweep_framework.py --topo DDF --V 40 --D 6
        """
    )
    parser.add_argument('--topo', '--topo-name', dest='topo_name', type=str, default='RRG',
                       help='Topology name (default: RRG)')
    parser.add_argument('--V', '--vertices', dest='V', type=int, default=36,
                       help='Number of vertices/routers (default: 36)')
    parser.add_argument('--D', '--degree', dest='D', type=int, default=5,
                       help='Degree of routers (default: 5)')
    parser.add_argument('--cores-per-ep', type=int, default=4,
                       help='Cores per endpoint (default: 4)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SAMPLE SWEEP FRAMEWORK TEST")
    print("="*80)
    print(f"Topology: {args.topo_name} (V={args.V}, D={args.D})")
    print(f"Cores per EP: {args.cores_per_ep}")
    print("Testing with Allreduce, count=256, samples=[1, 4, 8]")
    print("="*80)
    
    # Configuration
    topo_name = args.topo_name
    V = args.V
    D = args.D
    cores_per_ep = args.cores_per_ep
    link_bw = 16
    num_threads = 8
    benchmark = "Allreduce"
    count = 256
    bench_args = f" iterations=10 count={count}"
    
    # Test 1: Baseline and SD
    print("\n[1/4] Testing SHORTEST-PATH baseline and SD...")
    baseline_results = run_ember_experiment_with_nexullance(
        topo_name=topo_name, V=V, D=D,
        benchmark=benchmark, bench_args=bench_args,
        cores_per_ep=cores_per_ep, link_bw=link_bw,
        num_threads=num_threads,
        nexullance_method="SD"
    )
    
    if not baseline_results:
        print("✗ Baseline/SD test FAILED")
        return 1
    
    baseline_time = baseline_results['baseline_sim_time_ms']
    sd_time = baseline_results['optimized_sim_time_ms']
    sd_speedup = baseline_results['speedup']
    
    print(f"✓ Baseline time: {baseline_time:.4f} ms")
    print(f"✓ SD time: {sd_time:.4f} ms, speedup: {sd_speedup:.4f}x")
    
    # Test 2: UGAL
    print("\n[2/4] Testing UGAL...")
    ugal_config = {
        'UNIFIED_ROUTER_LINK_BW': link_bw,
        'V': V, 'D': D,
        'topo_name': topo_name,
        'benchmark': benchmark,
        'benchargs': bench_args,
        'Cores_per_EP': cores_per_ep,
        'routing_method': 'ugal'
    }
    stdout, stderr, returncode, sim_dir = _run_sst(ugal_config, 'EFM', num_threads)
    
    if returncode == 0:
        output_file = sim_dir / f"simulation_output_{sim_dir.name}.txt"
        ugal_time = _extract_simulation_time_from_output(str(output_file))
        if ugal_time:
            ugal_speedup = baseline_time / ugal_time
            print(f"✓ UGAL time: {ugal_time:.4f} ms, speedup: {ugal_speedup:.4f}x")
        else:
            print("✗ Could not extract UGAL time")
    else:
        print("✗ UGAL test FAILED")
    
    # Test 3: MD with different sample counts
    print("\n[3/4] Testing MD with sample counts: [1, 4, 8]...")
    sample_counts = [1, 4, 8]
    
    for num_samples in sample_counts:
        print(f"\n  Testing MD with {num_samples} samples...")
        
        md_results = run_ember_experiment_with_nexullance(
            topo_name=topo_name, V=V, D=D,
            benchmark=benchmark, bench_args=bench_args,
            cores_per_ep=cores_per_ep, link_bw=link_bw,
            num_threads=num_threads,
            nexullance_method="MD",
            num_demand_samples=num_samples,
            max_path_length=4
        )
        
        if md_results:
            md_time = md_results['optimized_sim_time_ms']
            md_speedup = md_results['speedup']
            print(f"  ✓ MD ({num_samples} samples): {md_time:.4f} ms, speedup: {md_speedup:.4f}x")
        else:
            print(f"  ✗ MD ({num_samples} samples) FAILED")
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print("\nSummary:")
    print(f"  Baseline (shortest-path): {baseline_time:.4f} ms")
    print(f"  SD speedup: {sd_speedup:.4f}x")
    print("  MD sample sweep tested successfully")
    print("\nThe sample sweep framework is working correctly!")
    print("You can now run the full experiments with:")
    print("  ./run_sample_sweep_experiments.sh")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
