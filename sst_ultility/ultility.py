
#!/usr/bin/env python3
"""
SST Simulation Utilities
Provides functions for Merlin (synthetic traffic) and EFM (Ember+Firefly+Merlin) experiments.
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import SIMULATION_RESULTS_DIR, TOPOLOGIES_DIR, TRAFFIC_TRACES_DIR


def _generate_traffic_trace_filename(config_dict: dict, template_type: str) -> str:
    """
    Generate a unique, descriptive filename for traffic trace based on simulation parameters.
    
    Args:
        config_dict: Configuration dictionary
        template_type: 'merlin' or 'EFM'
        
    Returns:
        Absolute path to traffic trace file
    """
    V = config_dict['V']
    D = config_dict['D']
    EPR = (D + 1) // 2
    topo_name = config_dict['topo_name']
    
    if template_type == 'merlin':
        load = config_dict['LOAD']
        pattern = config_dict['traffic_pattern']
        # Format load to avoid decimal points in filename
        load_str = f"{load:.2f}".replace('.', 'p')
        filename = f"traffic_merlin_{topo_name}_V{V}_D{D}_EPR{EPR}_load{load_str}_{pattern}.csv"
    else:  # EFM
        benchmark = config_dict['benchmark']
        bench_args = config_dict.get('benchargs', '')
        # Clean bench args for filename (remove spaces, special chars)
        bench_args_clean = bench_args.strip().replace(' ', '_').replace('=', '')
        cores_per_ep = config_dict.get('Cores_per_EP', 1)
        filename = f"traffic_efm_{topo_name}_V{V}_D{D}_EPR{EPR}_CPE{cores_per_ep}_{benchmark}{bench_args_clean}.csv"
    
    return str(TRAFFIC_TRACES_DIR / filename)


def run_merlin_simulation(topo_name: str, V: int, D: int, load: float,
                          traffic_pattern: str = "uniform", link_bw: int = 16,
                          num_threads: int = 8, traffic_trace_file: str = "",
                          enable_traffic_trace: bool = True, **additional_config):
    """
    Run Merlin synthetic traffic simulation.
    
    Args:
        topo_name: Topology name (e.g., "RRG", "Slimfly", "DDF")
        V: Number of vertices/routers
        D: Degree of routers
        load: Offered load (0.0 to 1.0)
        traffic_pattern: Traffic pattern ("uniform", "shift_X", etc.)
        link_bw: Link bandwidth in Gbps (default: 16)
        num_threads: SST threads (default: 8)
        traffic_trace_file: Custom path for traffic trace (optional, auto-generated if empty)
        enable_traffic_trace: Whether to generate traffic trace (default: True)
        **additional_config: Additional configuration parameters
        
    Returns:
        True if successful, False otherwise
    """
    config = {
        'UNIFIED_ROUTER_LINK_BW': link_bw,
        'V': V,
        'D': D,
        'topo_name': topo_name,
        'LOAD': load,
        'traffic_pattern': traffic_pattern
    }
    
    # Generate traffic trace filename if tracing is enabled
    if enable_traffic_trace:
        if not traffic_trace_file:
            traffic_trace_file = _generate_traffic_trace_filename(config, 'merlin')
        config['traffic_trace_file'] = traffic_trace_file
        print(f"Traffic trace will be saved to: {traffic_trace_file}")
    
    config.update(additional_config)
    
    stdout, stderr, returncode = _run_sst(config, 'merlin', num_threads)
    return returncode == 0


def run_ember_simulation(topo_name: str, V: int, D: int, benchmark: str,
                        bench_args: str = "", traffic_trace_file: str = "",
                        cores_per_ep: int = 1, link_bw: int = 16, num_threads: int = 8,
                        enable_traffic_trace: bool = True, **additional_config):
    """
    Run EFM (Ember+Firefly+Merlin) MPI benchmark simulation.
    
    Args:
        topo_name: Topology name (e.g., "RRG", "Slimfly", "DDF")
        V: Number of vertices/routers
        D: Degree of routers
        benchmark: Ember benchmark name (e.g., "FFT3D", "Allreduce")
        bench_args: Benchmark arguments (e.g., " nx=100 ny=100 nz=100 npRow=12")
        traffic_trace_file: Custom path for traffic trace (optional, auto-generated if empty)
        cores_per_ep: Number of cores per endpoint (default: 1)
        link_bw: Link bandwidth in Gbps (default: 16)
        num_threads: SST threads (default: 8)
        enable_traffic_trace: Whether to generate traffic trace (default: True)
        **additional_config: Additional configuration parameters
        
    Returns:
        True if successful, False otherwise
    """
    config = {
        'UNIFIED_ROUTER_LINK_BW': link_bw,
        'V': V,
        'D': D,
        'topo_name': topo_name,
        'benchmark': benchmark,
        'benchargs': bench_args,
        'Cores_per_EP': cores_per_ep
    }
    
    # Generate traffic trace filename if tracing is enabled
    if enable_traffic_trace:
        if not traffic_trace_file:
            traffic_trace_file = _generate_traffic_trace_filename(config, 'EFM')
        config['traffic_trace_file'] = traffic_trace_file
        print(f"Traffic trace will be saved to: {traffic_trace_file}")
    
    config.update(additional_config)
    
    stdout, stderr, returncode = _run_sst(config, 'EFM', num_threads)
    return returncode == 0


def _run_sst(config_dict: dict, template_type: str, num_threads: int = 8):
    """
    Internal function to run SST simulation with given config and template.
    
    Args:
        config_dict: Complete configuration dictionary
        template_type: Template type ('EFM' or 'merlin')
        num_threads: Number of SST threads
        
    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    template_map = {
        'EFM': 'sst_EFM_config_template.py',
        'merlin': 'sst_merlin_config_template.py'
    }
    
    if template_type not in template_map:
        raise ValueError(f"Unknown template: {template_type}. Use 'EFM' or 'merlin'")
    
    script_dir = Path(__file__).parent
    template_file = script_dir / template_map[template_type]
    
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    # Create unique simulation directory using timestamp and PID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_id = f"{timestamp}_{os.getpid()}"
    sim_dir = SIMULATION_RESULTS_DIR / sim_id
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config file in simulation directory
    config_filename = f"sst_config_{template_type}_{sim_id}.py"
    config_file_path = sim_dir / config_filename
    
    # Write configuration file
    with open(config_file_path, 'w') as file:
        file.write("import numpy as np\n")
        file.write("import sys\n")
        file.write(f"sys.path.insert(0, r'{TOPOLOGIES_DIR}')\n\n")
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
    
    # Save simulation output to files
    output_file = sim_dir / f"simulation_output_{sim_id}.txt"
    with open(output_file, 'w') as f:
        f.write(stdout)
    
    if stderr:
        error_file = sim_dir / f"simulation_errors_{sim_id}.txt"
        with open(error_file, 'w') as f:
            f.write(stderr)
    
    # Print summary
    print(f"SST Simulation Complete:")
    print(f"  - Results saved to: {sim_dir}")
    print(f"  - Config file: {config_filename}")
    print(f"  - Output file: {output_file.name}")
    if stderr:
        print(f"  - Error file: {error_file.name}")
    print(f"\nSST Output:\n{stdout}")
    if stderr:
        print(f"\nSST Errors:\n{stderr}")
        
    return stdout, stderr, process.returncode


