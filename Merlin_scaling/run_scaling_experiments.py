#!/usr/bin/env python3
"""
Network Scaling Experiments: Performance and Runtime Analysis

For a given topology type (Slimfly, DDF, Polarfly), sweep through different network sizes
and measure:
1. Performance speedup of Nexullance and UGAL vs shortest path
2. Simulation runtime scaling with network size

Uses topology configurations from global_helpers.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import time
import json

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import (
    run_merlin_experiment_with_nexullance,
    run_merlin_simulation
)
from topoResearch.global_helpers import sf_configs, ddf_configs, pf_configs


def get_topology_configs(topo_name: str, max_routers: int = 1000):
    """
    Get topology configurations for the specified topology type.
    
    Args:
        topo_name: Topology name (Slimfly, DDF, Polarfly)
        max_routers: Maximum number of routers to include
        
    Returns:
        List of (V, D) tuples
    """
    topo_name_lower = topo_name.lower()
    
    if 'slimfly' in topo_name_lower or topo_name_lower == 'sf':
        configs = sf_configs
    elif 'ddf' in topo_name_lower:
        configs = ddf_configs
    elif 'polarfly' in topo_name_lower or topo_name_lower == 'pf':
        configs = pf_configs
    else:
        raise ValueError(f"Unknown topology: {topo_name}. Use 'Slimfly', 'DDF', or 'Polarfly'")
    
    # Filter by max_routers
    filtered_configs = [(v, d) for v, d in configs if v <= max_routers]
    
    print(f"\n{topo_name} configurations (V ≤ {max_routers}):")
    print(f"  Number of configs: {len(filtered_configs)}")
    print(f"  Router range: {filtered_configs[0][0]} to {filtered_configs[-1][0]}")
    
    return filtered_configs


def run_single_experiment(topo_name: str, V: int, D: int, routing_method: str,
                         traffic_pattern: str, load: float, link_bw: int, 
                         num_threads: int):
    """
    Run a single experiment and measure both performance and runtime.
    
    Returns:
        dict with throughput_gbps, runtime_seconds, and other metrics
    """
    start_time = time.time()
    
    try:
        if routing_method == 'nexullance':
            result = run_merlin_experiment_with_nexullance(
                topo_name=topo_name,
                V=V,
                D=D,
                load=load,
                traffic_pattern=traffic_pattern,
                link_bw=link_bw,
                num_threads=num_threads,
                traffic_collection_rate="200us",
                demand_scaling_factor=10.0
            )
            
            if result:
                throughput = result['optimized_throughput_gbps']
            else:
                throughput = None
                
        else:
            result = run_merlin_simulation(
                topo_name=topo_name,
                V=V,
                D=D,
                load=load,
                traffic_pattern=traffic_pattern,
                routing_method=routing_method,
                link_bw=link_bw,
                num_threads=num_threads,
                calculate_throughput=True
            )
            
            if result:
                throughput = result['throughput_gbps']
            else:
                throughput = None
        
        runtime_seconds = time.time() - start_time
        
        return {
            'success': throughput is not None,
            'throughput_gbps': throughput,
            'runtime_seconds': runtime_seconds
        }
        
    except Exception as e:
        runtime_seconds = time.time() - start_time
        print(f"ERROR in {routing_method}: {str(e)}")
        return {
            'success': False,
            'throughput_gbps': None,
            'runtime_seconds': runtime_seconds,
            'error': str(e)
        }


def run_scaling_experiment(topo_name: str,
                          traffic_pattern: str = "uniform",
                          load: float = 0.5,
                          link_bw: int = 16,
                          num_threads: int = 8,
                          max_routers: int = 1000,
                          routing_methods: list = None):
    """
    Run scaling experiments across different network sizes.
    
    Args:
        topo_name: Topology name (Slimfly, DDF, Polarfly)
        traffic_pattern: Traffic pattern (uniform, shift_1, shift_half)
        load: Offered load (0.0-1.0)
        link_bw: Link bandwidth in Gbps
        num_threads: Number of SST threads
        max_routers: Maximum number of routers
        routing_methods: List of routing methods ['shortest_path', 'ugal', 'nexullance']
        
    Returns:
        DataFrame with scaling results
    """
    if routing_methods is None:
        routing_methods = ['shortest_path', 'ugal', 'nexullance']
    
    # Get topology configurations
    configs = get_topology_configs(topo_name, max_routers)
    
    print("\n" + "="*80)
    print("NETWORK SCALING EXPERIMENT")
    print("="*80)
    print(f"Topology:         {topo_name}")
    print(f"Configurations:   {len(configs)} network sizes")
    print(f"Traffic Pattern:  {traffic_pattern}")
    print(f"Load:             {load}")
    print(f"Link Bandwidth:   {link_bw} Gbps")
    print(f"Routing Methods:  {', '.join(routing_methods)}")
    print("="*80 + "\n")
    
    all_results = []
    
    for config_idx, (V, D) in enumerate(configs):
        EPR = (D + 1) // 2
        num_endpoints = V * EPR
        num_cores = num_endpoints  # Assuming 1 core per endpoint for scaling
        
        print("\n" + "="*80)
        print(f"Configuration {config_idx+1}/{len(configs)}: V={V}, D={D}, EPR={EPR}")
        print(f"  Endpoints: {num_endpoints}, Total cores: {num_cores}")
        print("="*80)
        
        config_results = {
            'V': V,
            'D': D,
            'EPR': EPR,
            'num_endpoints': num_endpoints,
            'num_cores': num_cores
        }
        
        # Run each routing method
        for routing_method in routing_methods:
            print(f"\n  Testing {routing_method}...", end=" ", flush=True)
            
            result = run_single_experiment(
                topo_name=topo_name,
                V=V,
                D=D,
                routing_method=routing_method,
                traffic_pattern=traffic_pattern,
                load=load,
                link_bw=link_bw,
                num_threads=num_threads
            )
            
            config_results[f'{routing_method}_throughput'] = result['throughput_gbps']
            config_results[f'{routing_method}_runtime'] = result['runtime_seconds']
            config_results[f'{routing_method}_success'] = result['success']
            
            if result['success']:
                print(f"✓ {result['throughput_gbps']:.4f} Gbps ({result['runtime_seconds']:.1f}s)")
            else:
                print(f"✗ FAILED ({result['runtime_seconds']:.1f}s)")
        
        # Calculate speedups (if shortest_path succeeded)
        if config_results.get('shortest_path_success', False):
            baseline_throughput = config_results['shortest_path_throughput']
            
            for method in routing_methods:
                if method != 'shortest_path' and config_results.get(f'{method}_success', False):
                    speedup = config_results[f'{method}_throughput'] / baseline_throughput
                    config_results[f'{method}_speedup'] = speedup
                else:
                    config_results[f'{method}_speedup'] = None
        
        all_results.append(config_results)
        
        # Save intermediate results
        df_intermediate = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_csv = SCRIPT_DIR / f"scaling_{topo_name}_intermediate_{timestamp}.csv"
        df_intermediate.to_csv(intermediate_csv, index=False)
    
    # Create final results DataFrame
    df = pd.DataFrame(all_results)
    
    # Print summary
    print_scaling_summary(df, topo_name, routing_methods)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = SCRIPT_DIR / f"scaling_{topo_name}_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    print(f"\n✓ Final results saved to: {results_csv}")
    
    # Save metadata
    metadata = {
        'topo_name': topo_name,
        'traffic_pattern': traffic_pattern,
        'load': load,
        'link_bw': link_bw,
        'num_threads': num_threads,
        'max_routers': max_routers,
        'routing_methods': routing_methods,
        'num_configs': len(configs),
        'timestamp': timestamp
    }
    metadata_json = SCRIPT_DIR / f"scaling_{topo_name}_{timestamp}_metadata.json"
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return df


def print_scaling_summary(df: pd.DataFrame, topo_name: str, routing_methods: list):
    """Print a summary of scaling experiment results."""
    
    print("\n" + "="*80)
    print("SCALING EXPERIMENT SUMMARY")
    print("="*80)
    
    # Performance summary
    print(f"\nPerformance Summary for {topo_name}:")
    print("-" * 80)
    
    successful_df = df[df['shortest_path_success'] == True].copy()
    
    if len(successful_df) > 0:
        print(f"\nSuccessful experiments: {len(successful_df)} / {len(df)}")
        print(f"Network size range: V={successful_df['V'].min()} to {successful_df['V'].max()}")
        
        for method in routing_methods:
            throughput_col = f'{method}_throughput'
            if throughput_col in successful_df.columns:
                method_success = successful_df[successful_df[f'{method}_success'] == True]
                if len(method_success) > 0:
                    print(f"\n{method.upper()}:")
                    print(f"  Success rate:       {len(method_success)} / {len(successful_df)}")
                    print(f"  Avg throughput:     {method_success[throughput_col].mean():.4f} Gbps")
                    print(f"  Throughput range:   {method_success[throughput_col].min():.4f} - {method_success[throughput_col].max():.4f} Gbps")
                    
                    if method != 'shortest_path':
                        speedup_col = f'{method}_speedup'
                        if speedup_col in method_success.columns:
                            valid_speedups = method_success[method_success[speedup_col].notna()]
                            if len(valid_speedups) > 0:
                                print(f"  Avg speedup:        {valid_speedups[speedup_col].mean():.4f}x")
                                print(f"  Speedup range:      {valid_speedups[speedup_col].min():.4f}x - {valid_speedups[speedup_col].max():.4f}x")
    
    # Runtime summary
    print("\n" + "-" * 80)
    print("Runtime Scaling Summary:")
    print("-" * 80)
    
    for method in routing_methods:
        runtime_col = f'{method}_runtime'
        if runtime_col in df.columns:
            valid_runtimes = df[df[runtime_col].notna()]
            if len(valid_runtimes) > 0:
                print(f"\n{method.upper()}:")
                print(f"  Total runtime:      {valid_runtimes[runtime_col].sum():.1f} seconds ({valid_runtimes[runtime_col].sum()/60:.1f} minutes)")
                print(f"  Avg runtime:        {valid_runtimes[runtime_col].mean():.1f} seconds")
                print(f"  Runtime range:      {valid_runtimes[runtime_col].min():.1f}s - {valid_runtimes[runtime_col].max():.1f}s")
    
    print("\n" + "="*80)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Network scaling experiments: measure performance and runtime scaling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Topology parameters
    parser.add_argument('--topo-name', '-t', type=str, required=True,
                        help='Topology name (Slimfly/SF, DDF, Polarfly/PF)')
    parser.add_argument('--max-routers', type=int, default=1000,
                        help='Maximum number of routers to test')
    
    # Traffic parameters
    parser.add_argument('--traffic-pattern', '-p', type=str, default='uniform',
                        help='Traffic pattern (uniform, shift_1, shift_half)')
    parser.add_argument('--load', type=float, default=0.5,
                        help='Offered load (0.0-1.0)')
    parser.add_argument('--link-bw', type=int, default=16,
                        help='Link bandwidth in Gbps')
    
    # Routing methods
    parser.add_argument('--routing-methods', nargs='+', 
                        default=['shortest_path', 'ugal', 'nexullance'],
                        choices=['shortest_path', 'ugal', 'nexullance'],
                        help='Routing methods to compare')
    
    # System parameters
    parser.add_argument('--num-threads', type=int, default=8,
                        help='Number of SST threads')
    
    args = parser.parse_args()
    
    # Run scaling experiment
    df = run_scaling_experiment(
        topo_name=args.topo_name,
        traffic_pattern=args.traffic_pattern,
        load=args.load,
        link_bw=args.link_bw,
        num_threads=args.num_threads,
        max_routers=args.max_routers,
        routing_methods=args.routing_methods
    )
    
    print("\n✓ Scaling experiments complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
