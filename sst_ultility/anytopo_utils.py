#!/usr/bin/env python3
"""
SST AnyTopo Utilities - Simplified interface for SST simulations.
Provides two main experiment types: Merlin (synthetic traffic) and EFM (Ember+Firefly+Merlin).
"""

import os
import subprocess
import sys
from pathlib import Path

def create_base_config(topo_name: str, V: int, D: int, link_bw: int = 16):
    """
    Create base anytopo configuration that can be extended for specific experiments.
    
    Args:
        topo_name: Name of topology (e.g., "RRG", "Slimfly", "DDF")
        V: Number of vertices/routers
        D: Degree of routers
        link_bw: Link bandwidth in Gbps
        
    Returns:
        Base configuration dictionary
    """
    return {
        'UNIFIED_ROUTER_LINK_BW': link_bw,
        'V': V,
        'D': D,
        'topo_name': topo_name,
        'identifier': 'anytopo'
    }

def run_merlin_experiment(base_config: dict, load: float, traffic_pattern: str = "uniform", 
                         num_threads: int = 8, **additional_config):
    """
    Run Merlin synthetic traffic experiment.
    
    Args:
        base_config: Base configuration from create_base_config()
        load: Offered load (0.0 to 1.0)
        traffic_pattern: Traffic pattern ("uniform", "shift_X", etc.)
        num_threads: SST threads
        **additional_config: Additional configuration parameters
        
    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    # Create full config for Merlin experiment
    config = base_config.copy()
    config.update({
        'LOAD': load,
        'paths': 'source_routing',
        'routing_algo': 'nonadaptive',
        'traffic_pattern': traffic_pattern
    })
    config.update(additional_config)
    
    return _run_sst_simulation(config, 'merlin', num_threads)

def run_efm_experiment(base_config: dict, benchmark: str, bench_args: str = "",
                      cores_per_ep: int = 1, num_threads: int = 8, **additional_config):
    """
    Run EFM (Ember+Firefly+Merlin) experiment.
    
    Args:
        base_config: Base configuration from create_base_config()
        benchmark: Ember benchmark name (e.g., "FFT3D", "Allreduce")
        bench_args: Benchmark arguments (e.g., " nx=100 ny=100 nz=100 npRow=12")
        cores_per_ep: Number of cores per endpoint
        num_threads: SST threads
        **additional_config: Additional configuration parameters (e.g., traffic_trace_file)
        
    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    # Create full config for EFM experiment
    config = base_config.copy()
    config.update({
        'routing_algo': 'source_routing',
        'benchmark': benchmark,
        'benchargs': bench_args,
        'Cores_per_EP': cores_per_ep
    })
    config.update(additional_config)
    
    return _run_sst_simulation(config, 'EFM', num_threads)

def _run_sst_simulation(config_dict: dict, template_name: str, num_threads: int = 8):
    """
    Internal function to run SST simulation with given config and template.
    
    Args:
        config_dict: Complete configuration dictionary
        template_name: Template type ('EFM' or 'merlin')
        num_threads: Number of SST threads
        
    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    # Determine template file path
    template_map = {
        'EFM': 'sst_EFM_config_template.py',
        'merlin': 'sst_merlin_config_template.py'
    }
    
    if template_name not in template_map:
        raise ValueError(f"Unknown template: {template_name}. Use 'EFM' or 'merlin'")
    
    script_dir = Path(__file__).parent
    template_file = script_dir / template_map[template_name]
    
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    # Create temporary config file
    config_filename = f"sst_config_{template_name}_{os.getpid()}.py"
    config_file_path = script_dir / config_filename
    
    try:
        # Write configuration file
        with open(config_file_path, 'w') as file:
            file.write("import numpy as np\n")
            file.write("config_dict = ")
            file.write(repr(config_dict))
            file.write("\n\n")
            
            # Append template content
            with open(template_file, 'r') as template_file_handle:
                file.write(template_file_handle.read())
        
        # Run SST simulation
        cmd = ["sst", "-n", f"{num_threads}", str(config_file_path)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        print("SST Simulation Output:")
        print(stdout)
        if stderr:
            print("SST Simulation Errors:")
            print(stderr)
            
        return stdout, stderr, process.returncode
        
    finally:
        # Clean up temporary config file
        if config_file_path.exists():
            config_file_path.unlink()