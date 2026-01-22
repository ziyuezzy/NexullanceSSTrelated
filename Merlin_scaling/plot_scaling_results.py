#!/usr/bin/env python3
"""
Plot scaling experiment results: performance speedup and runtime scaling
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent


def load_scaling_results(results_file: str):
    """Load scaling experiment results from CSV."""
    df = pd.read_csv(results_file)
    print(f"\nLoaded {len(df)} configurations from {results_file}")
    
    # Identify routing methods from column names
    routing_methods = []
    for col in df.columns:
        if col.endswith('_throughput'):
            method = col.replace('_throughput', '')
            routing_methods.append(method)
    
    print(f"Routing methods: {', '.join(routing_methods)}")
    
    return df, routing_methods


def plot_performance_scaling(df: pd.DataFrame, routing_methods: list, topo_name: str, output_dir: Path):
    """Plot throughput vs network size for different routing methods."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = {'shortest_path': 'o', 'ugal': 's', 'nexullance': '^'}
    colors = {'shortest_path': 'blue', 'ugal': 'orange', 'nexullance': 'green'}
    labels = {'shortest_path': 'Shortest Path', 'ugal': 'UGAL', 'nexullance': 'Nexullance'}
    
    for method in routing_methods:
        throughput_col = f'{method}_throughput'
        success_col = f'{method}_success'
        
        if throughput_col in df.columns:
            # Filter successful experiments
            method_df = df[df[success_col] == True].copy()
            
            if len(method_df) > 0:
                ax.plot(method_df['V'], method_df[throughput_col],
                       marker=markers.get(method, 'o'),
                       color=colors.get(method, 'gray'),
                       label=labels.get(method, method),
                       linewidth=2, markersize=8, alpha=0.7)
    
    ax.set_xlabel('Number of Routers (V)', fontsize=12)
    ax.set_ylabel('Network Throughput (Gbps)', fontsize=12)
    ax.set_title(f'{topo_name} Network: Throughput vs Network Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f'scaling_throughput_{topo_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved throughput plot: {output_file}")
    plt.close()


def plot_speedup_scaling(df: pd.DataFrame, routing_methods: list, topo_name: str, output_dir: Path):
    """Plot speedup vs network size for advanced routing methods."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = {'ugal': 's', 'nexullance': '^'}
    colors = {'ugal': 'orange', 'nexullance': 'green'}
    labels = {'ugal': 'UGAL', 'nexullance': 'Nexullance'}
    
    has_data = False
    
    for method in routing_methods:
        if method == 'shortest_path':
            continue
            
        speedup_col = f'{method}_speedup'
        
        if speedup_col in df.columns:
            # Filter valid speedup data
            method_df = df[df[speedup_col].notna()].copy()
            
            if len(method_df) > 0:
                has_data = True
                ax.plot(method_df['V'], method_df[speedup_col],
                       marker=markers.get(method, 'o'),
                       color=colors.get(method, 'gray'),
                       label=labels.get(method, method),
                       linewidth=2, markersize=8, alpha=0.7)
    
    if has_data:
        # Add baseline line at speedup=1.0
        ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=1.5, 
                   label='Shortest Path (baseline)', alpha=0.7)
        
        ax.set_xlabel('Number of Routers (V)', fontsize=12)
        ax.set_ylabel('Speedup vs Shortest Path', fontsize=12)
        ax.set_title(f'{topo_name} Network: Performance Speedup vs Network Size', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'scaling_speedup_{topo_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved speedup plot: {output_file}")
    else:
        print("⚠ No speedup data available to plot")
    
    plt.close()


def plot_runtime_scaling(df: pd.DataFrame, routing_methods: list, topo_name: str, output_dir: Path):
    """Plot simulation runtime vs network size."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = {'shortest_path': 'o', 'ugal': 's', 'nexullance': '^'}
    colors = {'shortest_path': 'blue', 'ugal': 'orange', 'nexullance': 'green'}
    labels = {'shortest_path': 'Shortest Path', 'ugal': 'UGAL', 'nexullance': 'Nexullance'}
    
    for method in routing_methods:
        runtime_col = f'{method}_runtime'
        
        if runtime_col in df.columns:
            # Filter valid runtime data
            method_df = df[df[runtime_col].notna()].copy()
            
            if len(method_df) > 0:
                ax.plot(method_df['V'], method_df[runtime_col],
                       marker=markers.get(method, 'o'),
                       color=colors.get(method, 'gray'),
                       label=labels.get(method, method),
                       linewidth=2, markersize=8, alpha=0.7)
    
    ax.set_xlabel('Number of Routers (V)', fontsize=12)
    ax.set_ylabel('Simulation Runtime (seconds)', fontsize=12)
    ax.set_title(f'{topo_name} Network: Runtime Scaling with Network Size', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for runtime
    
    plt.tight_layout()
    output_file = output_dir / f'scaling_runtime_{topo_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved runtime plot: {output_file}")
    plt.close()


def plot_combined_comparison(df: pd.DataFrame, routing_methods: list, topo_name: str, output_dir: Path):
    """Create a combined figure with multiple subplots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    markers = {'shortest_path': 'o', 'ugal': 's', 'nexullance': '^'}
    colors = {'shortest_path': 'blue', 'ugal': 'orange', 'nexullance': 'green'}
    labels = {'shortest_path': 'Shortest Path', 'ugal': 'UGAL', 'nexullance': 'Nexullance'}
    
    # 1. Throughput vs Network Size
    ax1 = axes[0, 0]
    for method in routing_methods:
        throughput_col = f'{method}_throughput'
        success_col = f'{method}_success'
        
        if throughput_col in df.columns:
            method_df = df[df[success_col] == True].copy()
            if len(method_df) > 0:
                ax1.plot(method_df['V'], method_df[throughput_col],
                        marker=markers.get(method, 'o'),
                        color=colors.get(method, 'gray'),
                        label=labels.get(method, method),
                        linewidth=2, markersize=6, alpha=0.7)
    
    ax1.set_xlabel('Number of Routers (V)', fontsize=11)
    ax1.set_ylabel('Throughput (Gbps)', fontsize=11)
    ax1.set_title('(a) Network Throughput', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup vs Network Size
    ax2 = axes[0, 1]
    for method in routing_methods:
        if method == 'shortest_path':
            continue
        speedup_col = f'{method}_speedup'
        if speedup_col in df.columns:
            method_df = df[df[speedup_col].notna()].copy()
            if len(method_df) > 0:
                ax2.plot(method_df['V'], method_df[speedup_col],
                        marker=markers.get(method, 'o'),
                        color=colors.get(method, 'gray'),
                        label=labels.get(method, method),
                        linewidth=2, markersize=6, alpha=0.7)
    
    ax2.axhline(y=1.0, color='blue', linestyle='--', linewidth=1.5, 
               label='Baseline', alpha=0.7)
    ax2.set_xlabel('Number of Routers (V)', fontsize=11)
    ax2.set_ylabel('Speedup vs Shortest Path', fontsize=11)
    ax2.set_title('(b) Performance Speedup', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Runtime vs Network Size
    ax3 = axes[1, 0]
    for method in routing_methods:
        runtime_col = f'{method}_runtime'
        if runtime_col in df.columns:
            method_df = df[df[runtime_col].notna()].copy()
            if len(method_df) > 0:
                ax3.plot(method_df['V'], method_df[runtime_col],
                        marker=markers.get(method, 'o'),
                        color=colors.get(method, 'gray'),
                        label=labels.get(method, method),
                        linewidth=2, markersize=6, alpha=0.7)
    
    ax3.set_xlabel('Number of Routers (V)', fontsize=11)
    ax3.set_ylabel('Runtime (seconds)', fontsize=11)
    ax3.set_title('(c) Simulation Runtime', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Runtime per endpoint
    ax4 = axes[1, 1]
    for method in routing_methods:
        runtime_col = f'{method}_runtime'
        if runtime_col in df.columns:
            method_df = df[df[runtime_col].notna()].copy()
            if len(method_df) > 0:
                runtime_per_ep = method_df[runtime_col] / method_df['num_endpoints']
                ax4.plot(method_df['V'], runtime_per_ep,
                        marker=markers.get(method, 'o'),
                        color=colors.get(method, 'gray'),
                        label=labels.get(method, method),
                        linewidth=2, markersize=6, alpha=0.7)
    
    ax4.set_xlabel('Number of Routers (V)', fontsize=11)
    ax4.set_ylabel('Runtime per Endpoint (s)', fontsize=11)
    ax4.set_title('(d) Normalized Runtime', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{topo_name} Network Scaling Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'scaling_combined_{topo_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot network scaling experiment results'
    )
    
    parser.add_argument('results_file', type=str,
                        help='Path to scaling results CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as results file)')
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return 1
    
    df, routing_methods = load_scaling_results(results_path)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract topology name from filename
    topo_name = "Unknown"
    for topo in ['Slimfly', 'DDF', 'Polarfly']:
        if topo.lower() in results_path.stem.lower():
            topo_name = topo
            break
    
    print(f"\nGenerating plots for {topo_name}...")
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    plot_performance_scaling(df, routing_methods, topo_name, output_dir)
    plot_speedup_scaling(df, routing_methods, topo_name, output_dir)
    plot_runtime_scaling(df, routing_methods, topo_name, output_dir)
    plot_combined_comparison(df, routing_methods, topo_name, output_dir)
    
    print("\n✓ All plots generated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
