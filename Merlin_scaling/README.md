# Network Scaling Experiments

This directory contains scripts for running network scaling experiments that measure:
1. **Performance speedup**: How Nexullance and UGAL compare to shortest path routing as network size increases
2. **Runtime scaling**: How simulation runtime scales with network size

## Files

- `run_scaling_experiments.py` - Main script to run scaling experiments
- `plot_scaling_results.py` - Script to visualize results
- `README.md` - This file

## Usage

### Running Scaling Experiments

```bash
# Run scaling experiments for Slimfly topology
python3 run_scaling_experiments.py --topo-name Slimfly

# Run for DDF with custom parameters
python3 run_scaling_experiments.py --topo-name DDF --max-routers 500 --load 0.7

# Run with specific routing methods
python3 run_scaling_experiments.py --topo-name Polarfly --routing-methods shortest_path nexullance

# Show all options
python3 run_scaling_experiments.py --help
```

### Command-line Arguments

**Topology Parameters:**
- `--topo-name`, `-t` (required): Topology name (Slimfly/SF, DDF, Polarfly/PF)
- `--max-routers`: Maximum number of routers to test (default: 1000)

**Traffic Parameters:**
- `--traffic-pattern`, `-p`: Traffic pattern (default: uniform)
- `--load`: Offered load, 0.0-1.0 (default: 0.5)
- `--link-bw`: Link bandwidth in Gbps (default: 16)

**Routing Methods:**
- `--routing-methods`: List of methods to compare (default: shortest_path ugal nexullance)

**System Parameters:**
- `--num-threads`: Number of SST threads (default: 8)

### Topology Configurations

The script uses topology configurations from `topoResearch/global_helpers.py`:

**Slimfly (sf_configs):** 27 configurations, V from 18 to 8978
**DDF (ddf_configs):** 12 configurations, V from 36 to 8814
**Polarfly (pf_configs):** 18 configurations, V from 7 to 1057

### Plotting Results

```bash
# Plot results from a scaling experiment
python3 plot_scaling_results.py scaling_Slimfly_20260122_120000.csv

# Specify custom output directory
python3 plot_scaling_results.py scaling_DDF_20260122_120000.csv --output-dir ./plots/
```

## Output Files

### CSV Results
- `scaling_{topo}_{timestamp}.csv` - Complete experimental results
- `scaling_{topo}_{timestamp}_metadata.json` - Experiment configuration metadata
- `scaling_{topo}_intermediate_{timestamp}.csv` - Intermediate results (saved during experiment)

### Plots
- `scaling_throughput_{topo}.png` - Network throughput vs network size
- `scaling_speedup_{topo}.png` - Speedup vs network size
- `scaling_runtime_{topo}.png` - Simulation runtime vs network size
- `scaling_combined_{topo}.png` - Combined 4-panel comparison

## Results Columns

The CSV output contains:
- `V`, `D`, `EPR` - Topology parameters
- `num_endpoints`, `num_cores` - Derived network size metrics
- `{method}_throughput` - Throughput in Gbps for each routing method
- `{method}_runtime` - Simulation runtime in seconds
- `{method}_success` - Whether the experiment succeeded
- `{method}_speedup` - Speedup relative to shortest path (for advanced methods)

## Example Workflows

### Full scaling study for all topologies
```bash
# Slimfly
python3 run_scaling_experiments.py --topo-name Slimfly --max-routers 1000
python3 plot_scaling_results.py scaling_Slimfly_*.csv

# DDF
python3 run_scaling_experiments.py --topo-name DDF --max-routers 1000
python3 plot_scaling_results.py scaling_DDF_*.csv

# Polarfly
python3 run_scaling_experiments.py --topo-name Polarfly --max-routers 1000
python3 plot_scaling_results.py scaling_Polarfly_*.csv
```

### Quick test with smaller networks
```bash
python3 run_scaling_experiments.py --topo-name Slimfly --max-routers 200 --load 0.5
```

### Compare only specific routing methods
```bash
# Only shortest path vs Nexullance
python3 run_scaling_experiments.py --topo-name DDF --routing-methods shortest_path nexullance
```

## Notes

- Experiments run sequentially through all network sizes
- Intermediate results are saved periodically to prevent data loss
- Runtime can be significant for large networks with Nexullance optimization
- Failed experiments are recorded but don't stop the overall sweep
- All timing includes both optimization and simulation time

## Interpretation

**Performance Speedup:**
- Values > 1.0 indicate improvement over shortest path
- Nexullance typically shows higher speedup on congested networks
- UGAL provides adaptive routing without offline optimization

**Runtime Scaling:**
- Should grow approximately linearly with network size
- Nexullance has higher overhead due to optimization
- Log-scale plots help visualize scaling trends
