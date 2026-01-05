
import os
import subprocess
import sys
from pathlib import Path

# Import the simplified anytopo utilities
try:
    from .anytopo_utils import create_base_config, run_merlin_experiment, run_efm_experiment
except ImportError:
    from anytopo_utils import create_base_config, run_merlin_experiment, run_efm_experiment

# Convenience functions for common use cases
def run_ember_simulation(topo_name: str, V: int, D: int, benchmark: str,
                        bench_args: str = "", traffic_trace_file: str = "",
                        cores_per_ep: int = 1, link_bw: int = 16, num_threads: int = 8):
    """
    Convenient function to run Ember MPI benchmarks with anytopo.
    """
    base_config = create_base_config(topo_name, V, D, link_bw)
    
    # Add traffic trace file if specified
    additional_config = {}
    if traffic_trace_file:
        additional_config['traffic_trace_file'] = traffic_trace_file
    
    stdout, stderr, returncode = run_efm_experiment(
        base_config=base_config,
        benchmark=benchmark,
        bench_args=bench_args,
        cores_per_ep=cores_per_ep,
        num_threads=num_threads,
        **additional_config
    )
    return returncode == 0

def run_synthetic_simulation(topo_name: str, V: int, D: int, load: float,
                           traffic_pattern: str = "uniform", link_bw: int = 16,
                           num_threads: int = 8):
    """
    Convenient function to run synthetic traffic simulations with anytopo.
    """
    base_config = create_base_config(topo_name, V, D, link_bw)
    
    stdout, stderr, returncode = run_merlin_experiment(
        base_config=base_config,
        load=load,
        traffic_pattern=traffic_pattern,
        num_threads=num_threads
    )
    return returncode == 0


