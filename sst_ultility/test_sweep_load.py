#!/usr/bin/env python3
"""
Load sweep script for SST-Merlin simulations with Nexullance optimization.

Sweeps offered load from 0.1 to 1.0 with step 0.1 and compares baseline vs optimized performance.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import run_merlin_experiment_with_nexullance


def main():
    """
    Sweep load from 0.1 to 1.0 and collect performance metrics.
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Sweep offered load and compare baseline vs Nexullance-optimized routing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--traffic-pattern', '-p', type=str, default='uniform',
                        help='Traffic pattern (uniform, shift_X, tornado, etc.)')
    parser.add_argument('--topo-name', '-t', type=str, default='RRG',
                        help='Topology name (RRG, Slimfly, DDF, etc.)')
    parser.add_argument('--V', type=int, default=16,
                        help='Number of vertices/routers')
    parser.add_argument('--D', type=int, default=5,
                        help='Degree of routers')
    parser.add_argument('--link-bw', type=int, default=16,
                        help='Link bandwidth in Gbps')
    parser.add_argument('--load-start', type=float, default=0.1,
                        help='Starting load')
    parser.add_argument('--load-end', type=float, default=1.0,
                        help='Ending load')
    parser.add_argument('--load-step', type=float, default=0.1,
                        help='Load increment step')
    parser.add_argument('--num-threads', type=int, default=8,
                        help='Number of SST threads')
    
    args = parser.parse_args()
    
    # Configuration from command-line arguments
    topo_name = args.topo_name
    V = args.V
    D = args.D
    traffic_pattern = args.traffic_pattern
    link_bw = args.link_bw
    num_threads = args.num_threads
    traffic_collection_rate = "10us"
    demand_scaling_factor = 1.0
    
    # Load sweep parameters
    load_start = args.load_start
    load_end = args.load_end
    load_step = args.load_step
    
    print("\n" + "="*80)
    print("LOAD SWEEP EXPERIMENT")
    print("="*80)
    print(f"Topology:        {topo_name} (V={V}, D={D})")
    print(f"Traffic Pattern: {traffic_pattern}")
    print(f"Load Range:      {load_start} to {load_end} (step={load_step})")
    print(f"Link Bandwidth:  {link_bw} Gbps")
    print("="*80 + "\n")
    
    # Store results
    results_list = []
    
    # Sweep loads
    load = load_start
    while load <= load_end + 0.001:  # Small epsilon for floating point comparison
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT: Load = {load:.1f}")
        print("="*80)
        
        result = run_merlin_experiment_with_nexullance(
            topo_name=topo_name,
            V=V,
            D=D,
            load=load,
            traffic_pattern=traffic_pattern,
            link_bw=link_bw,
            num_threads=num_threads,
            traffic_collection_rate=traffic_collection_rate,
            demand_scaling_factor=demand_scaling_factor
        )
        
        if result:
            results_list.append({
                'load': load,
                'baseline_throughput_gbps': result['baseline_throughput_gbps'],
                'optimized_throughput_gbps': result['optimized_throughput_gbps'],
                'speedup': result.get('speedup', None),
                'improvement_percent': result.get('improvement_percent', None),
                'demand_file': result['demand_file'],
                'baseline_throughput_file': result['baseline_throughput_file'],
                'throughput_file': result['throughput_file']
            })
            print(f"\n✓ Load {load:.1f} completed successfully")
        else:
            print(f"\n✗ Load {load:.1f} FAILED!")
            results_list.append({
                'load': load,
                'baseline_throughput_gbps': None,
                'optimized_throughput_gbps': None,
                'speedup': None,
                'improvement_percent': None,
                'demand_file': None,
                'baseline_throughput_file': None,
                'throughput_file': None
            })
        
        load += load_step
    
    # Create results summary
    print("\n" + "="*80)
    print("LOAD SWEEP COMPLETE - SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results_list)
    
    # Display summary table
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Load':<8} {'Baseline (Gbps)':<18} {'Optimized (Gbps)':<18} {'Speedup':<10} {'Improvement':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        if row['baseline_throughput_gbps'] is not None:
            print(f"{row['load']:<8.1f} {row['baseline_throughput_gbps']:<18.4f} "
                  f"{row['optimized_throughput_gbps']:<18.4f} {row['speedup']:<10.4f} "
                  f"{row['improvement_percent']:>+11.2f}%")
        else:
            print(f"{row['load']:<8.1f} {'FAILED':<18} {'FAILED':<18} {'N/A':<10} {'N/A':<12}")
    
    print("-" * 80)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = SCRIPT_DIR.parent / "simulation_results" / f"load_sweep_{topo_name}_V{V}_D{D}_{traffic_pattern}_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    
    print(f"\nResults saved to: {results_csv}")
    
    # Print statistics
    successful_runs = df[df['baseline_throughput_gbps'].notna()]
    if len(successful_runs) > 0:
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Successful runs:     {len(successful_runs)} / {len(df)}")
        print(f"Average speedup:     {successful_runs['speedup'].mean():.4f}x")
        print(f"Best speedup:        {successful_runs['speedup'].max():.4f}x (at load {successful_runs.loc[successful_runs['speedup'].idxmax(), 'load']:.1f})")
        print(f"Worst speedup:       {successful_runs['speedup'].min():.4f}x (at load {successful_runs.loc[successful_runs['speedup'].idxmin(), 'load']:.1f})")
        print(f"Avg improvement:     {successful_runs['improvement_percent'].mean():+.2f}%")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
