#!/usr/bin/env python3
"""
EFM Sample Sweep Experiments - Reproducing archive/EFM_experiments/RRG_4CPE
Compares four routing methods across multiple MPI benchmarks with MD sample sweep:
- shortest_path: Standard shortest path routing (baseline)
- ugal: Universal Globally-Adaptive Load-balancing routing
- nexullance_SD: Single-demand Nexullance optimization
- nexullance_MD: Multi-demand Nexullance with varying number of samples
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import csv
import argparse

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import (
    run_ember_simulation, 
    run_ember_experiment_with_nexullance,
    _extract_simulation_time_from_output
)


def write_csv_row(filename: str, content_row: list):
    """Append a row to a CSV file."""
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(content_row)
        csv_file.flush()


def run_allreduce_sample_sweep(topo_name="RRG", V=36, D=5, cores_per_ep=4):
    """
    Run Allreduce experiments with sample sweep for MD method.
    
    Args:
        topo_name: Topology name (default: "RRG")
        V: Number of vertices/routers (default: 36)
        D: Degree of routers (default: 5)
        cores_per_ep: Cores per endpoint (default: 4)
    
    Problem sizes: [256, 512, 1024, 2048]
    Sample counts: [1, 2, 4, 8, 16, 32, 64, 128]
    """
    print("\n" + "="*80)
    print("ALLREDUCE SAMPLE SWEEP EXPERIMENTS")
    print("="*80)
    
    # Configuration
    link_bw = 16
    num_threads = 8
    benchmark = "Allreduce"
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / f"{topo_name}_V{V}_D{D}_CPE{cores_per_ep}_sample_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV files
    csv_files = {
        "shortest_path": output_dir / f"Allreduce_count_x_shortest_path_result_{timestamp}.csv",
        "ugal": output_dir / f"Allreduce_count_x_ugal_result_{timestamp}.csv",
        "nexullance_SD": output_dir / f"Allreduce_count_x_nexullance_SD_result_{timestamp}.csv",
        "nexullance_MD": output_dir / f"Allreduce_count_x_nexullance_MD_sample_sweep_{timestamp}.csv"
    }
    
    # Write headers
    write_csv_row(str(csv_files["shortest_path"]), ["count", "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["ugal"]), ["count", "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["nexullance_SD"]), ["count", "baseline_sim_time_ms", "optimized_sim_time_ms", "speedup", "improvement_percent"])
    write_csv_row(str(csv_files["nexullance_MD"]), ["count", "num_samples", "baseline_sim_time_ms", "optimized_sim_time_ms", "speedup", "improvement_percent"])
    
    # Problem sizes and sample counts
    problem_sizes = [256, 512, 1024, 2048]
    sample_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for count in problem_sizes:
        print(f"\n{'='*80}")
        print(f"Problem Size: count={count}")
        print(f"{'='*80}")
        
        bench_args = f" iterations=10 count={count}"
        
        # 1. Shortest-path baseline
        print(f"\n[1/4] Running SHORTEST-PATH baseline...")
        baseline_results = run_ember_experiment_with_nexullance(
            topo_name=topo_name, V=V, D=D,
            benchmark=benchmark, bench_args=bench_args,
            cores_per_ep=cores_per_ep, link_bw=link_bw,
            num_threads=num_threads,
            nexullance_method="SD"  # We'll use the demand from this
        )
        
        if not baseline_results:
            print(f"ERROR: Baseline failed for count={count}")
            continue
        
        baseline_time = baseline_results['baseline_sim_time_ms']
        write_csv_row(str(csv_files["shortest_path"]), [count, baseline_time, 1.0])
        print(f"✓ Baseline time: {baseline_time:.4f} ms")
        
        # 2. UGAL routing
        print(f"\n[2/4] Running UGAL...")
        # Run UGAL simulation directly
        from sst_ultility.ultility import _run_sst
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
                write_csv_row(str(csv_files["ugal"]), [count, ugal_time, ugal_speedup])
                print(f"✓ UGAL time: {ugal_time:.4f} ms, speedup: {ugal_speedup:.4f}x")
            else:
                print(f"ERROR: Could not extract UGAL time for count={count}")
        else:
            print(f"ERROR: UGAL simulation failed for count={count}")
        
        # 3. Single-demand Nexullance (SD)
        print(f"\n[3/4] Running SINGLE-DEMAND Nexullance (SD)...")
        # We already have this from baseline_results
        sd_time = baseline_results['optimized_sim_time_ms']
        sd_speedup = baseline_results['speedup']
        sd_improvement = baseline_results['improvement_percent']
        write_csv_row(str(csv_files["nexullance_SD"]), 
                     [count, baseline_time, sd_time, sd_speedup, sd_improvement])
        print(f"✓ SD time: {sd_time:.4f} ms, speedup: {sd_speedup:.4f}x")
        
        # 4. Multi-demand Nexullance (MD) with sample sweep
        print(f"\n[4/4] Running MULTI-DEMAND Nexullance (MD) - SAMPLE SWEEP...")
        for num_samples in sample_counts:
            print(f"  Testing with {num_samples} samples...")
            
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
                md_improvement = md_results['improvement_percent']
                write_csv_row(str(csv_files["nexullance_MD"]),
                             [count, num_samples, baseline_time, md_time, md_speedup, md_improvement])
                print(f"    ✓ MD ({num_samples} samples): {md_time:.4f} ms, speedup: {md_speedup:.4f}x")
            else:
                print(f"    ✗ MD failed with {num_samples} samples")
    
    print(f"\n{'='*80}")
    print("ALLREDUCE EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    return csv_files


def run_alltoall_sample_sweep(topo_name="RRG", V=36, D=5, cores_per_ep=4):
    """
    Run Alltoall experiments with sample sweep for MD method.
    
    Args:
        topo_name: Topology name (default: "RRG")
        V: Number of vertices/routers (default: 36)
        D: Degree of routers (default: 5)
        cores_per_ep: Cores per endpoint (default: 4)
    
    Message sizes: [1, 8, 64] bytes
    Sample counts: [1, 2, 4, 8, 16, 32, 64, 128]
    """
    print("\n" + "="*80)
    print("ALLTOALL SAMPLE SWEEP EXPERIMENTS")
    print("="*80)
    
    # Configuration
    link_bw = 16
    num_threads = 8
    benchmark = "Alltoall"
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / f"{topo_name}_V{V}_D{D}_CPE{cores_per_ep}_sample_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV files
    csv_files = {
        "shortest_path": output_dir / f"Alltoall_bytes_x_shortest_path_result_{timestamp}.csv",
        "ugal": output_dir / f"Alltoall_bytes_x_ugal_result_{timestamp}.csv",
        "nexullance_SD": output_dir / f"Alltoall_bytes_x_nexullance_SD_result_{timestamp}.csv",
        "nexullance_MD": output_dir / f"Alltoall_bytes_x_nexullance_MD_sample_sweep_{timestamp}.csv"
    }
    
    # Write headers
    write_csv_row(str(csv_files["shortest_path"]), ["bytes", "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["ugal"]), ["bytes", "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["nexullance_SD"]), ["bytes", "baseline_sim_time_ms", "optimized_sim_time_ms", "speedup", "improvement_percent"])
    write_csv_row(str(csv_files["nexullance_MD"]), ["bytes", "num_samples", "baseline_sim_time_ms", "optimized_sim_time_ms", "speedup", "improvement_percent"])
    
    # Message sizes and sample counts
    message_sizes = [1, 8, 64]
    sample_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for bytes_size in message_sizes:
        print(f"\n{'='*80}")
        print(f"Message Size: bytes={bytes_size}")
        print(f"{'='*80}")
        
        bench_args = f" bytes={bytes_size}"
        
        # 1. Shortest-path baseline
        print(f"\n[1/4] Running SHORTEST-PATH baseline...")
        baseline_results = run_ember_experiment_with_nexullance(
            topo_name=topo_name, V=V, D=D,
            benchmark=benchmark, bench_args=bench_args,
            cores_per_ep=cores_per_ep, link_bw=link_bw,
            num_threads=num_threads,
            nexullance_method="SD"
        )
        
        if not baseline_results:
            print(f"ERROR: Baseline failed for bytes={bytes_size}")
            continue
        
        baseline_time = baseline_results['baseline_sim_time_ms']
        write_csv_row(str(csv_files["shortest_path"]), [bytes_size, baseline_time, 1.0])
        print(f"✓ Baseline time: {baseline_time:.4f} ms")
        
        # 2. UGAL routing
        print(f"\n[2/4] Running UGAL...")
        from sst_ultility.ultility import _run_sst
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
                write_csv_row(str(csv_files["ugal"]), [bytes_size, ugal_time, ugal_speedup])
                print(f"✓ UGAL time: {ugal_time:.4f} ms, speedup: {ugal_speedup:.4f}x")
        
        # 3. Single-demand Nexullance (SD)
        print(f"\n[3/4] Running SINGLE-DEMAND Nexullance (SD)...")
        sd_time = baseline_results['optimized_sim_time_ms']
        sd_speedup = baseline_results['speedup']
        sd_improvement = baseline_results['improvement_percent']
        write_csv_row(str(csv_files["nexullance_SD"]), 
                     [bytes_size, baseline_time, sd_time, sd_speedup, sd_improvement])
        print(f"✓ SD time: {sd_time:.4f} ms, speedup: {sd_speedup:.4f}x")
        
        # 4. Multi-demand Nexullance (MD) with sample sweep
        print(f"\n[4/4] Running MULTI-DEMAND Nexullance (MD) - SAMPLE SWEEP...")
        for num_samples in sample_counts:
            print(f"  Testing with {num_samples} samples...")
            
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
                md_improvement = md_results['improvement_percent']
                write_csv_row(str(csv_files["nexullance_MD"]),
                             [bytes_size, num_samples, baseline_time, md_time, md_speedup, md_improvement])
                print(f"    ✓ MD ({num_samples} samples): {md_time:.4f} ms, speedup: {md_speedup:.4f}x")
            else:
                print(f"    ✗ MD failed with {num_samples} samples")
    
    print(f"\n{'='*80}")
    print("ALLTOALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    return csv_files


def run_fft3d_sample_sweep(topo_name="RRG", V=36, D=5, cores_per_ep=4):
    """
    Run FFT3D experiments with sample sweep for MD method.
    
    Args:
        topo_name: Topology name (default: "RRG")
        V: Number of vertices/routers (default: 36)
        D: Degree of routers (default: 5)
        cores_per_ep: Cores per endpoint (default: 4)
    
    Problem sizes: [256, 512, 1024, 2048]
    Sample counts: [1, 2, 4, 8, 16, 32, 64, 128]
    """
    print("\n" + "="*80)
    print("FFT3D SAMPLE SWEEP EXPERIMENTS")
    print("="*80)
    
    # Configuration
    link_bw = 16
    num_threads = 8
    benchmark = "FFT3D"
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / f"{topo_name}_V{V}_D{D}_CPE{cores_per_ep}_sample_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV files
    csv_files = {
        "shortest_path": output_dir / f"FFT3D_nx_ny_nz_npRow_12_shortest_path_result_{timestamp}.csv",
        "ugal": output_dir / f"FFT3D_nx_ny_nz_npRow_12_ugal_result_{timestamp}.csv",
        "nexullance_SD": output_dir / f"FFT3D_nx_ny_nz_npRow_12_nexullance_SD_result_{timestamp}.csv",
        "nexullance_MD": output_dir / f"FFT3D_nx_ny_nz_npRow_12_nexullance_MD_sample_sweep_{timestamp}.csv"
    }
    
    # Write headers
    write_csv_row(str(csv_files["shortest_path"]), ["nx", "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["ugal"]), ["nx", "sim_time_ms", "speedup"])
    write_csv_row(str(csv_files["nexullance_SD"]), ["nx", "baseline_sim_time_ms", "optimized_sim_time_ms", "speedup", "improvement_percent"])
    write_csv_row(str(csv_files["nexullance_MD"]), ["nx", "num_samples", "baseline_sim_time_ms", "optimized_sim_time_ms", "speedup", "improvement_percent"])
    
    # Problem sizes and sample counts
    problem_sizes = [256, 512, 1024, 2048]
    sample_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for size in problem_sizes:
        print(f"\n{'='*80}")
        print(f"Problem Size: nx=ny=nz={size}, npRow=12")
        print(f"{'='*80}")
        
        bench_args = f" nx={size} ny={size} nz={size} npRow=12"
        
        # 1. Shortest-path baseline
        print(f"\n[1/4] Running SHORTEST-PATH baseline...")
        baseline_results = run_ember_experiment_with_nexullance(
            topo_name=topo_name, V=V, D=D,
            benchmark=benchmark, bench_args=bench_args,
            cores_per_ep=cores_per_ep, link_bw=link_bw,
            num_threads=num_threads,
            nexullance_method="SD"
        )
        
        if not baseline_results:
            print(f"ERROR: Baseline failed for nx={size}")
            continue
        
        baseline_time = baseline_results['baseline_sim_time_ms']
        write_csv_row(str(csv_files["shortest_path"]), [size, baseline_time, 1.0])
        print(f"✓ Baseline time: {baseline_time:.4f} ms")
        
        # 2. UGAL routing
        print(f"\n[2/4] Running UGAL...")
        from sst_ultility.ultility import _run_sst
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
                write_csv_row(str(csv_files["ugal"]), [size, ugal_time, ugal_speedup])
                print(f"✓ UGAL time: {ugal_time:.4f} ms, speedup: {ugal_speedup:.4f}x")
        
        # 3. Single-demand Nexullance (SD)
        print(f"\n[3/4] Running SINGLE-DEMAND Nexullance (SD)...")
        sd_time = baseline_results['optimized_sim_time_ms']
        sd_speedup = baseline_results['speedup']
        sd_improvement = baseline_results['improvement_percent']
        write_csv_row(str(csv_files["nexullance_SD"]), 
                     [size, baseline_time, sd_time, sd_speedup, sd_improvement])
        print(f"✓ SD time: {sd_time:.4f} ms, speedup: {sd_speedup:.4f}x")
        
        # 4. Multi-demand Nexullance (MD) with sample sweep
        print(f"\n[4/4] Running MULTI-DEMAND Nexullance (MD) - SAMPLE SWEEP...")
        for num_samples in sample_counts:
            print(f"  Testing with {num_samples} samples...")
            
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
                md_improvement = md_results['improvement_percent']
                write_csv_row(str(csv_files["nexullance_MD"]),
                             [size, num_samples, baseline_time, md_time, md_speedup, md_improvement])
                print(f"    ✓ MD ({num_samples} samples): {md_time:.4f} ms, speedup: {md_speedup:.4f}x")
            else:
                print(f"    ✗ MD failed with {num_samples} samples")
    
    print(f"\n{'='*80}")
    print("FFT3D EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    return csv_files


def main():
    """Run all sample sweep experiments."""
    parser = argparse.ArgumentParser(
        description="EFM Sample Sweep Experiments - Compare routing methods with varying MD sample counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default RRG(36,5) topology:
  python3.12 run_sample_sweep_experiments.py
  
  # Run with custom topology:
  python3.12 run_sample_sweep_experiments.py --topo DDF --V 40 --D 6
  
  # Run with different cores per endpoint:
  python3.12 run_sample_sweep_experiments.py --cores-per-ep 8
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
    parser.add_argument('--no-prompt', action='store_true',
                       help='Skip confirmation prompt and start immediately')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("EFM SAMPLE SWEEP EXPERIMENTS")
    print("Reproducing archive/EFM_experiments/RRG_4CPE with new framework")
    print("="*80)
    print(f"\nTopology Configuration:")
    print(f"  Topology: {args.topo_name}")
    print(f"  Vertices (V): {args.V}")
    print(f"  Degree (D): {args.D}")
    print(f"  Cores per EP: {args.cores_per_ep}")
    print(f"  Endpoints: {args.V * ((args.D + 1) // 2)}")
    print(f"  Total Cores: {args.V * ((args.D + 1) // 2) * args.cores_per_ep}")
    print("\nBenchmarks to run:")
    print("  - Allreduce (count: 256, 512, 1024, 2048)")
    print("  - Alltoall (bytes: 1, 8, 64)")
    print("  - FFT3D (nx=ny=nz: 256, 512, 1024, 2048, npRow=12)")
    print("\nFor each benchmark:")
    print("  1. shortest_path (baseline)")
    print("  2. ugal")
    print("  3. nexullance_SD (single-demand)")
    print("  4. nexullance_MD (multi-demand, sweep samples: 1,2,4,8,16,32,64,128)")
    print("="*80)
    
    if not args.no_prompt:
        input("\nPress Enter to start experiments (this will take several hours)...")
    
    # Run all experiments with specified topology
    run_allreduce_sample_sweep(args.topo_name, args.V, args.D, args.cores_per_ep)
    run_alltoall_sample_sweep(args.topo_name, args.V, args.D, args.cores_per_ep)
    run_fft3d_sample_sweep(args.topo_name, args.V, args.D, args.cores_per_ep)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    sys.exit(main())
