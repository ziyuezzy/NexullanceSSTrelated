#!/usr/bin/env python3
"""
Analyze and visualize results from EFM benchmark experiments.
Creates comparison plots and summary statistics.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_results(output_dir: Path, benchmark: str, param_label: str):
    """
    Load experiment results from CSV files.
    
    Args:
        output_dir: Directory containing result CSVs
        benchmark: Benchmark name
        param_label: Parameter label in filenames
        
    Returns:
        Dictionary of DataFrames for each routing method
    """
    results = {}
    
    # Find the most recent CSV files for each routing method
    for method in ['shortest_path', 'ugal', 'nexullance']:
        pattern = f"{benchmark}_{param_label}_{method}_result_*.csv"
        csv_files = sorted(output_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            results[method] = df
            print(f"Loaded {method}: {csv_files[0].name}")
        else:
            print(f"Warning: No results found for {method}")
            results[method] = None
    
    return results


def plot_simulation_time_comparison(results: dict, benchmark: str, param_name: str, 
                                    output_file: Path):
    """Plot simulation time comparison across routing methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot shortest-path baseline
    if results['shortest_path'] is not None:
        df = results['shortest_path']
        ax.plot(df[param_name], df['sim_time_ms'], 'o-', label='Shortest-path (baseline)', 
                linewidth=2, markersize=8)
    
    # Plot UGAL
    if results['ugal'] is not None:
        df = results['ugal']
        ax.plot(df[param_name], df['sim_time_ms'], 's-', label='UGAL', 
                linewidth=2, markersize=8)
    
    # Plot Nexullance (use optimized time)
    if results['nexullance'] is not None:
        df = results['nexullance']
        ax.plot(df[param_name], df['optimized_sim_time_ms'], '^-', label='Nexullance', 
                linewidth=2, markersize=8)
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Simulation Time (ms)', fontsize=12)
    ax.set_title(f'{benchmark} Simulation Time Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_speedup_comparison(results: dict, benchmark: str, param_name: str, 
                           output_file: Path):
    """Plot speedup comparison (relative to shortest-path baseline)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Baseline is always 1.0
    if results['shortest_path'] is not None:
        df = results['shortest_path']
        ax.plot(df[param_name], [1.0] * len(df), 'o-', label='Shortest-path (baseline)', 
                linewidth=2, markersize=8)
    
    # Plot UGAL speedup
    if results['ugal'] is not None:
        df = results['ugal']
        if 'speedup' in df.columns:
            ax.plot(df[param_name], df['speedup'], 's-', label='UGAL', 
                    linewidth=2, markersize=8)
    
    # Plot Nexullance speedup
    if results['nexullance'] is not None:
        df = results['nexullance']
        if 'speedup' in df.columns:
            ax.plot(df[param_name], df['speedup'], '^-', label='Nexullance', 
                    linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Speedup vs Shortest-path', fontsize=12)
    ax.set_title(f'{benchmark} Speedup Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def generate_summary_table(results: dict, benchmark: str, param_name: str):
    """Generate summary statistics table."""
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS: {benchmark}")
    print("="*80)
    
    for method, df in results.items():
        if df is not None and 'speedup' in df.columns:
            speedups = df['speedup'].dropna()
            if len(speedups) > 0:
                print(f"\n{method.upper()}:")
                print(f"  Average speedup:    {speedups.mean():.4f}x")
                print(f"  Max speedup:        {speedups.max():.4f}x")
                print(f"  Min speedup:        {speedups.min():.4f}x")
                print(f"  Std deviation:      {speedups.std():.4f}")
    
    print("="*80)


def analyze_benchmark(output_dir: Path, benchmark: str, param_label: str, param_name: str):
    """Analyze results for a single benchmark."""
    print("\n" + "="*80)
    print(f"ANALYZING: {benchmark}")
    print("="*80)
    
    # Load results
    results = load_results(output_dir, benchmark, param_label)
    
    # Create plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    plot_simulation_time_comparison(
        results, benchmark, param_name,
        plot_dir / f"{benchmark}_simulation_time.png"
    )
    
    plot_speedup_comparison(
        results, benchmark, param_name,
        plot_dir / f"{benchmark}_speedup.png"
    )
    
    # Generate summary statistics
    generate_summary_table(results, benchmark, param_name)


def main():
    """Analyze all EFM benchmark results."""
    print("\n" + "="*80)
    print("EFM BENCHMARK RESULTS ANALYSIS")
    print("="*80)
    
    # Find the output directory (most recent)
    output_dirs = sorted(SCRIPT_DIR.glob("RRG_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not output_dirs:
        print("Error: No results directory found!")
        print("Please run experiments first: ./run_experiments.sh")
        return 1
    
    output_dir = output_dirs[0]
    print(f"Analyzing results from: {output_dir}")
    
    # Analyze each benchmark
    analyze_benchmark(output_dir, "Allreduce", "count_x", "count")
    analyze_benchmark(output_dir, "Alltoall", "bytes_x", "bytes")
    analyze_benchmark(output_dir, "FFT3D", "nx_ny_nz_npRow_12", "nx")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults and plots saved to: {output_dir}")
    print(f"Plots directory: {output_dir / 'plots'}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
