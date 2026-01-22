#!/usr/bin/env python3
"""
Quick test script to verify the routing comparison framework is working.

Runs a small test with RRG topology (V=16, D=5) at a single load point.
"""

import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Merlin_experiments.run_experiments import run_comparison_experiment


def main():
    """Run a quick test experiment."""
    print("\n" + "="*80)
    print("QUICK TEST: Routing Comparison Framework")
    print("="*80)
    print("This will test all three routing methods on a small topology:")
    print("  - Topology: RRG (V=16, D=5)")
    print("  - Traffic: uniform")
    print("  - Load: 0.5 (single test point)")
    print("="*80 + "\n")
    
    # Run test experiment with small configuration
    df = run_comparison_experiment(
        topo_name="RRG",
        V=16,
        D=5,
        traffic_pattern="uniform",
        link_bw=16,
        num_threads=8,
        load_start=0.5,
        load_end=0.5,
        load_step=0.1,
        routing_methods=['shortest_path', 'nexullance', 'ugal']
    )
    
    if df is not None and len(df) > 0:
        print("\n" + "="*80)
        print("TEST SUCCESSFUL!")
        print("="*80)
        print("\nResults:")
        print(df.to_string(index=False))
        print("\n" + "="*80)
        print("Framework is ready to use!")
        print("Run './run_experiments.sh' to execute full experiments.")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("TEST FAILED!")
        print("="*80)
        print("Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
