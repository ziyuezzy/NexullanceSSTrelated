#!/usr/bin/env python3
"""
EFM Benchmark Experiments - Comparing Routing Methods
Reproduces experiments from archive/EFM_experiments with the new framework.

Compares three routing methods across multiple MPI benchmarks:
- shortest_path: Standard shortest path routing (baseline, formerly ECMP_ASP)
- ugal: Universal Globally-Adaptive Load-balancing routing  
- nexullance: Nexullance-optimized routing (formerly MD_IT/MD_MP)
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import csv

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import run_ember_simulation, run_ember_experiment_with_nexullance


def write_csv_row(filename: str, content_row: list):
    """Append a row to a CSV file."""
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(content_row)
        csv_file.flush()


def run_benchmark_comparison(benchmark: str, bench_args_list: list, 
                             param_name: str, param_label: str,
                             topo_name: str = "RRG", V: int = 36, D: int = 5,
                             cores_per_ep: int = 4, link_bw: int = 16,
                             num_threads: int = 8,
                             nexullance_method: str = "SD",
                             num_demand_samples: int = 1,
                             max_path_length: int = 4):
    """
    Run a complete benchmark comparison across all routing methods.
    
    Args:
        benchmark: Ember benchmark name (e.g., "Allreduce", "Alltoall", "FFT3D")
        bench_args_list: List of (param_value, bench_args) tuples
        param_name: Parameter being varied (for CSV header)
        param_label: Parameter label for filenames
        topo_name: Topology name (default: "RRG")
        V: Number of vertices (default: 36)
        D: Router degree (default: 5)
        cores_per_ep: Cores per endpoint (default: 4)
        link_bw: Link bandwidth in Gbps (default: 16)
        num_threads: SST threads (default: 8)
        nexullance_method: Nexullance method ('SD' or 'MD', default: 'SD')
        num_demand_samples: Number of demand samples for MD (default: 1)
        max_path_length: Max path length for MD (default: 4)
        
    Returns:
        Dictionary of results CSV filenames
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = SCRIPT_DIR / f"{topo_name}_V{V}_D{D}_CPE{cores_per_ep}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output CSV files
    csv_files = {
        "shortest_path": output_dir / f"{benchmark}_{param_label}_shortest_path_result_{timestamp}.csv",
        "ugal": output_dir / f"{benchmark}_{param_label}_ugal_result_{timestamp}.csv",
        "nexullance": output_dir / f"{benchmark}_{param_label}_nexullance_result_{timestamp}.csv"
    }
    
    # Write CSV headers
    write_csv_row(str(csv_files["shortest_path"]), [param_name, "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["ugal"]), [param_name, "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["nexullance"]), 
                  [param_name, "baseline_sim_time_ms", "optimized_sim_time_ms", "speedup", "improvement_percent"])
    
    print("\n" + "="*80)
    print(f"BENCHMARK COMPARISON EXPERIMENT: {benchmark}")
    print("="*80)
    print(f"Topology:        {topo_name} (V={V}, D={D})")
    print(f"Cores per EP:    {cores_per_ep}")
    print(f"Link Bandwidth:  {link_bw} Gbps")
    print(f"Parameter:       {param_name}")
    print(f"Test Cases:      {len(bench_args_list)}")
    print("="*80 + "\n")
    
    # Run experiments for each parameter value
    for param_value, bench_args in bench_args_list:
        print("\n" + "="*80)
        print(f"Testing: {benchmark} with {param_name}={param_value}")
        print("="*80)
        
        # Step 1: Run shortest-path baseline
        print(f"\n[1/3] Running SHORTEST-PATH baseline...")
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
        
        if success_sp:
            # Extract simulation time from output
            from sst_ultility.ultility import _extract_simulation_time_from_output
            from paths import SIMULATION_RESULTS_DIR
            
            # Find the most recent simulation directory
            sim_dirs = sorted(SIMULATION_RESULTS_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if sim_dirs:
                output_file = sim_dirs[0] / f"simulation_output_{sim_dirs[0].name}.txt"
                baseline_time = _extract_simulation_time_from_output(str(output_file))
                if baseline_time:
                    print(f"✓ Shortest-path sim time: {baseline_time:.4f} ms")
                    write_csv_row(str(csv_files["shortest_path"]), [param_value, baseline_time, 1.0])
                else:
                    print("✗ Could not extract simulation time")
                    baseline_time = None
            else:
                baseline_time = None
        else:
            print("✗ Shortest-path simulation FAILED")
            baseline_time = None
        
        # Step 2: Run UGAL adaptive routing
        print(f"\n[2/3] Running UGAL adaptive routing...")
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
        
        if success_ugal:
            # Extract simulation time
            sim_dirs = sorted(SIMULATION_RESULTS_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if sim_dirs:
                output_file = sim_dirs[0] / f"simulation_output_{sim_dirs[0].name}.txt"
                ugal_time = _extract_simulation_time_from_output(str(output_file))
                if ugal_time and baseline_time:
                    speedup = baseline_time / ugal_time
                    print(f"✓ UGAL sim time: {ugal_time:.4f} ms (speedup: {speedup:.4f}x)")
                    write_csv_row(str(csv_files["ugal"]), [param_value, ugal_time, speedup])
                elif ugal_time:
                    print(f"✓ UGAL sim time: {ugal_time:.4f} ms (baseline unavailable)")
                    write_csv_row(str(csv_files["ugal"]), [param_value, ugal_time, None])
                else:
                    print("✗ Could not extract simulation time")
        else:
            print("✗ UGAL simulation FAILED")
        
        # Step 3: Run Nexullance optimization workflow
        print(f"\n[3/3] Running NEXULLANCE optimization workflow ({nexullance_method})...")
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
            demand_scaling_factor=10.0,
            nexullance_method=nexullance_method,
            num_demand_samples=num_demand_samples,
            max_path_length=max_path_length
        )
        
        if results:
            baseline_sim_time = results.get('baseline_sim_time_ms')
            optimized_sim_time = results.get('optimized_sim_time_ms')
            speedup = results.get('speedup')
            improvement = results.get('improvement_percent')
            
            print(f"✓ Nexullance baseline: {baseline_sim_time:.4f} ms, "
                  f"optimized: {optimized_sim_time:.4f} ms (speedup: {speedup:.4f}x)")
            
            write_csv_row(str(csv_files["nexullance"]), 
                         [param_value, baseline_sim_time, optimized_sim_time, speedup, improvement])
        else:
            print("✗ Nexullance workflow FAILED")
            write_csv_row(str(csv_files["nexullance"]), 
                         [param_value, None, None, None, None])
        
        print(f"\nCompleted {param_name}={param_value}")
    
    print("\n" + "="*80)
    print(f"BENCHMARK COMPARISON COMPLETE: {benchmark}")
    print("="*80)
    print("Results saved to:")
    for method, filename in csv_files.items():
        print(f"  {method:<15} {filename}")
    print("="*80)
    
    return csv_files


def run_allreduce_experiments():
    """Run Allreduce benchmark experiments with varying message counts."""
    print("\n" + "="*80)
    print("ALLREDUCE EXPERIMENTS")
    print("="*80)
    
    # Test with different message counts
    problem_sizes = [256, 512, 1024, 2048]
    bench_args_list = [(size, f" iterations=10 count={size}") for size in problem_sizes]
    
    return run_benchmark_comparison(
        benchmark="Allreduce",
        bench_args_list=bench_args_list,
        param_name="count",
        param_label="count_x"
    )


def run_alltoall_experiments():
    """Run Alltoall benchmark experiments with varying message sizes."""
    print("\n" + "="*80)
    print("ALLTOALL EXPERIMENTS")
    print("="*80)
    
    # Test with different byte counts
    problem_sizes = [1, 8, 64]
    bench_args_list = [(size, f" bytes={size}") for size in problem_sizes]
    
    return run_benchmark_comparison(
        benchmark="Alltoall",
        bench_args_list=bench_args_list,
        param_name="bytes",
        param_label="bytes_x"
    )


def run_fft3d_experiments():
    """Run FFT3D benchmark experiments with varying problem sizes."""
    print("\n" + "="*80)
    print("FFT3D EXPERIMENTS")
    print("="*80)
    
    # Test with different grid sizes
    problem_sizes = [256, 512, 1024, 2048]
    bench_args_list = [(size, f" nx={size} ny={size} nz={size} npRow=12") 
                       for size in problem_sizes]
    
    return run_benchmark_comparison(
        benchmark="FFT3D",
        bench_args_list=bench_args_list,
        param_name="nx",
        param_label="nx_ny_nz_npRow_12"
    )


def main():
    """Run all EFM benchmark experiments."""
    print("\n" + "="*80)
    print("EFM BENCHMARK EXPERIMENTS - ROUTING COMPARISON")
    print("="*80)
    print("This script reproduces the experiments from archive/EFM_experiments")
    print("using the new unified framework.")
    print()
    print("Routing methods tested:")
    print("  - shortest_path: Standard shortest path routing (baseline)")
    print("  - ugal: Universal Globally-Adaptive Load-balancing")
    print("  - nexullance: Nexullance-optimized weighted routing")
    print()
    print("Benchmarks:")
    print("  - Allreduce: Collective communication with varying message counts")
    print("  - Alltoall: All-to-all communication with varying message sizes")
    print("  - FFT3D: 3D FFT with varying problem sizes")
    print("="*80)
    
    # Run all experiments
    allreduce_results = run_allreduce_experiments()
    alltoall_results = run_alltoall_experiments()
    fft3d_results = run_fft3d_experiments()
    
    # Final summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nResults Summary:")
    print("\nAllreduce:")
    for method, filename in allreduce_results.items():
        print(f"  {method:<15} {filename.name}")
    print("\nAlltoall:")
    for method, filename in alltoall_results.items():
        print(f"  {method:<15} {filename.name}")
    print("\nFFT3D:")
    for method, filename in fft3d_results.items():
        print(f"  {method:<15} {filename.name}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
