#!/usr/bin/env python3
"""
Test script for the new demand_matrix_analyser class.
Demonstrates how to use the aggregated demand matrix format.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from traffic_analyser import demand_matrix_analyser

def test_basic_usage():
    """Test basic functionality of the demand matrix analyzer."""
    
    # Path to example CSV
    csv_path = os.path.join(
        os.path.dirname(__file__), 
        '../traffic_traces/demand_matrix_csv_example.csv'
    )
    
    if not os.path.exists(csv_path):
        print(f"Error: Example CSV not found at {csv_path}")
        return
    
    print("=" * 60)
    print("Testing demand_matrix_analyser")
    print("=" * 60)
    
    # Initialize the analyzer
    # Note: These parameters should match your topology
    analyzer = demand_matrix_analyser(
        input_csv_path=csv_path,
        V=36,  # Number of vertices (example)
        D=5,   # Degree/diameter parameter (example)
        topo_name="RRG",  # Topology name (example)
        EPR=1,  # Endpoints per router
        Cap_core=16.0,  # Core link capacity in Gbps
        Cap_access=16.0,  # Access link capacity in Gbps
        suffix="test"
    )
    
    # Print statistics
    analyzer.print_statistics()
    
    # Test sampling with different numbers of samples
    print("\nTesting sampling functionality:")
    stats = analyzer.get_statistics()
    num_matrices = stats['num_matrices']
    print(f"Total matrices available: {num_matrices}")
    
    # Find valid sample counts (divisors of total matrices)
    valid_samples = [i for i in [1, 2, 5, 10, 20, 50, 100] if num_matrices % i == 0]
    print(f"Valid sample counts: {valid_samples}")
    
    if valid_samples:
        num_samples = valid_samples[0]  # Use the first valid count
        print(f"\nSampling with {num_samples} samples:")
        
        matrices, weights, interval_us = analyzer.sample_traffic(
            num_samples=num_samples,
            filtering_threshold=0.0,  # No filtering
            auto_scale=False
        )
        
        print(f"  Sampled {len(matrices)} matrices")
        print(f"  Weights: {weights}")
        print(f"  Sampling interval: {interval_us:.2f} us")
        
        # Test with auto-scaling
        print(f"\nSampling with auto-scaling:")
        try:
            matrices_scaled, weights_scaled, interval_us, scale_factor = analyzer.sample_traffic(
                num_samples=num_samples,
                filtering_threshold=0.0,
                auto_scale=True
            )
            print(f"  Scaling factor: {scale_factor:.4f}")
        except Exception as e:
            print(f"  Warning: Auto-scaling failed (topology not available): {e}")
            print(f"  This is expected if topology files are not accessible.")
    
    # Get accumulated demand matrix
    print("\nGetting accumulated demand matrix:")
    acc_matrix = analyzer.get_accumulated_demand_matrix(plot=False)
    print(f"  Shape: {acc_matrix.shape}")
    print(f"  Total traffic: {acc_matrix.sum():.2f} Gbps")
    print(f"  Max flow: {acc_matrix.max():.4f} Gbps")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

def test_error_handling():
    """Test error handling for invalid sample counts."""
    
    csv_path = os.path.join(
        os.path.dirname(__file__), 
        '../traffic_traces/demand_matrix_csv_example.csv'
    )
    
    if not os.path.exists(csv_path):
        return
    
    print("\n" + "=" * 60)
    print("Testing error handling")
    print("=" * 60)
    
    analyzer = demand_matrix_analyser(
        input_csv_path=csv_path,
        V=36, D=5, topo_name="RRG", EPR=1,
        Cap_core=16.0, Cap_access=16.0, suffix="test"
    )
    
    # Try invalid sample count
    try:
        print("\nAttempting to sample with invalid count (should raise error):")
        matrices, weights, interval = analyzer.sample_traffic(num_samples=7)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_basic_usage()
    test_error_handling()
