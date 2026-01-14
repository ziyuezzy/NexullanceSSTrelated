#!/usr/bin/env python3
"""
Benchmark sweep script for SST-EFM simulations with Nexullance optimization.

Sweeps through different MPI benchmarks or benchmark configurations and compares 
baseline vs optimized performance.
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

from sst_ultility.ultility import run_ember_experiment_with_nexullance


def main():
    """
    Sweep through MPI benchmarks and collect performance metrics.
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Sweep MPI benchmarks and compare baseline vs Nexullance-optimized routing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--benchmark', '-b', type=str, default='Allreduce',
                        help='Ember benchmark name (Allreduce, Alltoall, FFT3D, etc.)')
    parser.add_argument('--bench-args', '-a', type=str, default=' iterations=50 count=512',
                        help='Benchmark arguments')
    parser.add_argument('--topo-name', '-t', type=str, default='RRG',
                        help='Topology name (RRG, Slimfly, DDF, etc.)')
    parser.add_argument('--V', type=int, default=16,
                        help='Number of vertices/routers')
    parser.add_argument('--D', type=int, default=5,
                        help='Degree of routers')
    parser.add_argument('--cores-per-ep', type=int, default=1,
                        help='Number of cores per endpoint')
    parser.add_argument('--link-bw', type=int, default=16,
                        help='Link bandwidth in Gbps')
    parser.add_argument('--num-threads', type=int, default=8,
                        help='Number of SST threads')
    parser.add_argument('--sweep-mode', choices=['count', 'iterations', 'custom'], default='count',
                        help='Parameter to sweep: count (message size), iterations, or custom list')
    parser.add_argument('--sweep-values', type=str, default='128,256,512,1024,2048',
                        help='Comma-separated values to sweep (e.g., "128,256,512,1024")')
    
    args = parser.parse_args()
    
    # Configuration from command-line arguments
    topo_name = args.topo_name
    V = args.V
    D = args.D
    benchmark = args.benchmark
    base_bench_args = args.bench_args
    cores_per_ep = args.cores_per_ep
    link_bw = args.link_bw
    num_threads = args.num_threads
    traffic_collection_rate = "10us"
    demand_scaling_factor = 1.0
    
    # Parse sweep values
    sweep_values = [int(v.strip()) if v.strip().isdigit() else v.strip() 
                    for v in args.sweep_values.split(',')]
    
    print("\n" + "="*80)
    print("EFM BENCHMARK SWEEP EXPERIMENT")
    print("="*80)
    print(f"Topology:        {topo_name} (V={V}, D={D})")
    print(f"Benchmark:       {benchmark}")
    print(f"Base Args:       {base_bench_args}")
    print(f"Sweep Mode:      {args.sweep_mode}")
    print(f"Sweep Values:    {sweep_values}")
    print(f"Cores per EP:    {cores_per_ep}")
    print(f"Link Bandwidth:  {link_bw} Gbps")
    print("="*80 + "\n")
    
    # Store results
    results_list = []
    
    # Sweep through values
    for value in sweep_values:
        # Generate benchmark arguments based on sweep mode
        if args.sweep_mode == 'count':
            bench_args = base_bench_args.replace('count=512', f'count={value}')
            if 'count=' not in base_bench_args:
                bench_args = base_bench_args + f' count={value}'
            param_label = f'count={value}'
        elif args.sweep_mode == 'iterations':
            bench_args = base_bench_args.replace('iterations=50', f'iterations={value}')
            if 'iterations=' not in base_bench_args:
                bench_args = base_bench_args + f' iterations={value}'
            param_label = f'iterations={value}'
        else:
            bench_args = str(value)
            param_label = str(value)
        
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT: {benchmark} {param_label}")
        print("="*80)
        
        result = run_ember_experiment_with_nexullance(
            topo_name=topo_name,
            V=V,
            D=D,
            benchmark=benchmark,
            bench_args=bench_args,
            cores_per_ep=cores_per_ep,
            link_bw=link_bw,
            num_threads=num_threads,
            traffic_collection_rate=traffic_collection_rate,
            demand_scaling_factor=demand_scaling_factor
        )
        
        if result:
            results_list.append({
                'benchmark': benchmark,
                'bench_args': bench_args,
                'param_label': param_label,
                'baseline_sim_time_ms': result.get('baseline_sim_time_ms'),
                'optimized_sim_time_ms': result.get('optimized_sim_time_ms'),
                'speedup': result.get('speedup', None),
                'improvement_percent': result.get('improvement_percent', None),
                'demand_file': result['demand_file'],
                'baseline_output_file': result.get('baseline_output_file'),
                'optimized_output_file': result.get('optimized_output_file')
            })
            print(f"\n✓ {param_label} completed successfully")
        else:
            print(f"\n✗ {param_label} FAILED!")
            results_list.append({
                'benchmark': benchmark,
                'bench_args': bench_args,
                'param_label': param_label,
                'baseline_sim_time_ms': None,
                'optimized_sim_time_ms': None,
                'speedup': None,
                'improvement_percent': None,
                'demand_file': None,
                'baseline_output_file': None,
                'optimized_output_file': None
            })
    
    # Create results summary
    print("\n" + "="*80)
    print("BENCHMARK SWEEP COMPLETE - SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results_list)
    
    # Display summary table
    print("\nPerformance Summary:")
    print("-" * 100)
    print(f"{'Parameter':<20} {'Baseline (ms)':<18} {'Optimized (ms)':<18} {'Speedup':<10} {'Improvement':<12}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        if row['baseline_sim_time_ms'] is not None:
            print(f"{row['param_label']:<20} {row['baseline_sim_time_ms']:<18.4f} "
                  f"{row['optimized_sim_time_ms']:<18.4f} {row['speedup']:<10.4f} "
                  f"{row['improvement_percent']:>+11.2f}%")
        else:
            print(f"{row['param_label']:<20} {'FAILED':<18} {'FAILED':<18} {'N/A':<10} {'N/A':<12}")
    
    print("-" * 100)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_clean = benchmark.replace(' ', '_')
    results_csv = SCRIPT_DIR.parent / "simulation_results" / f"efm_sweep_{topo_name}_V{V}_D{D}_{bench_clean}_{args.sweep_mode}_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    
    print(f"\nResults saved to: {results_csv}")
    
    # Print statistics
    successful_runs = df[df['baseline_sim_time_ms'].notna()]
    if len(successful_runs) > 0:
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Successful runs:     {len(successful_runs)} / {len(df)}")
        print(f"Average speedup:     {successful_runs['speedup'].mean():.4f}x")
        print(f"Best speedup:        {successful_runs['speedup'].max():.4f}x ({successful_runs.loc[successful_runs['speedup'].idxmax(), 'param_label']})")
        print(f"Worst speedup:       {successful_runs['speedup'].min():.4f}x ({successful_runs.loc[successful_runs['speedup'].idxmin(), 'param_label']})")
        print(f"Avg improvement:     {successful_runs['improvement_percent'].mean():+.2f}%")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
