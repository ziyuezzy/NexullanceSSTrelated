# Merlin Experiments: Routing Comparison Framework

This directory contains the new experimental framework for comparing network routing methods in SST-Merlin simulations.

## Overview

This framework compares three routing methods:
1. **Shortest-path routing** - Traditional shortest path routing (ASP)
2. **Nexullance IT optimization** - Traffic-aware optimized routing using Nexullance
3. **UGAL adaptive routing** - Universal Globally-Adaptive Load-balancing

## Quick Start

### Run All Experiments

Execute all experiments for the three topologies (RRG, DDF, Slimfly) with multiple traffic patterns:

```bash
cd Merlin_experiments
./run_experiments.sh
```

This will run experiments for:
- **Topologies**: RRG (36,5), DDF (36,5), Slimfly (32,6)
- **Traffic patterns**: uniform, shift_1, shift_half
- **Load range**: 0.1 to 1.0 (step 0.1)

### Run Individual Experiment

Run a specific experiment configuration:

```bash
python3.12 run_experiments.py \
    --topo-name RRG \
    --V 16 \
    --D 5 \
    --traffic-pattern uniform \
    --routing-methods shortest_path nexullance ugal \
    --load-start 0.1 \
    --load-end 1.0 \
    --load-step 0.1
```

### Analyze Results

Generate comparison plots and statistics:

```bash
python3.12 analyze_results.py
```

This will:
- Load all result CSV files from `simulation_results/`
- Generate throughput comparison plots
- Calculate speedup relative to shortest-path routing
- Create summary statistics tables
- Save plots to `Merlin_experiments/plots/`

## File Structure

```
Merlin_experiments/
├── run_experiments.py      # Main experiment script (Python)
├── run_experiments.sh      # Batch experiment launcher (Bash)
├── analyze_results.py      # Results analysis and plotting
├── README.md              # This file
└── plots/                 # Generated plots (created automatically)
```

## Experiment Workflow

### 1. Shortest-Path Routing
- Uses standard shortest path routing
- Serves as baseline for comparison
- No optimization required

### 2. Nexullance IT Optimization
- Step 1: Run baseline simulation to collect traffic demand matrix
- Step 2: Run Nexullance IT optimizer with collected demand
- Step 3: Run optimized simulation with Nexullance routing table
- Step 4: Calculate and compare throughput improvements

### 3. UGAL Adaptive Routing
- Uses adaptive routing with UGAL algorithm
- No pre-computed routing table required
- Dynamically balances load across available paths

## Command-Line Options

### run_experiments.py

```
--topo-name, -t        Topology name (RRG, Slimfly, DDF)
--V                    Number of vertices/routers
--D                    Degree of routers
--traffic-pattern, -p  Traffic pattern (uniform, shift_1, shift_half)
--link-bw              Link bandwidth in Gbps (default: 16)
--load-start           Starting load (default: 0.1)
--load-end             Ending load (default: 1.0)
--load-step            Load increment step (default: 0.1)
--routing-methods      Routing methods to compare (default: all three)
--num-threads          Number of SST threads (default: 8)
```

### analyze_results.py

```
--input-dir            Directory containing result CSV files
--output-dir           Directory to save plots and analysis
--result-files         Specific result CSV files to analyze
```

## Results

Results are saved to `../simulation_results/`:
- `routing_comparison_{topo}_{traffic}_{timestamp}.csv` - Raw experiment data
- Each row contains: load, routing_method, throughput_gbps, result_file

Plots are saved to `plots/`:
- `comparison_{topo}_{traffic}.png` - Throughput and speedup comparison
- `summary_statistics.csv` - Overall statistics

## Comparison with Archive

This new framework replaces the old archive experiments with:
- **Unified workflow**: Single script handles all three routing methods
- **Better organization**: Results clearly labeled and organized
- **Automated analysis**: Built-in plotting and comparison
- **Flexible configuration**: Easy to add new topologies or traffic patterns
- **UGAL support**: Added support for adaptive routing

## Example Output

```
================================================================================
ROUTING COMPARISON EXPERIMENT
================================================================================
Topology:         RRG (V=36, D=5)
Traffic Pattern:  uniform
Load Range:       0.1 to 1.0 (step=0.1)
Link Bandwidth:   16 Gbps
Routing Methods:  shortest_path, nexullance, ugal
================================================================================

Load = 0.1
✓ shortest_path: 12.3456 Gbps
✓ nexullance: 13.2345 Gbps
✓ ugal: 12.8765 Gbps

...

================================================================================
COMPARISON SUMMARY
================================================================================
Throughput (Gbps) by Load and Routing Method:
--------------------------------------------------------------------------------
routing_method  shortest_path  nexullance     ugal
load                                              
0.1                   12.3456     13.2345  12.8765
0.2                   23.4567     25.1234  24.3210
...
================================================================================
```

## Notes

- Results may vary based on system performance and SST configuration
- Large experiments (high V, D values) may take significant time
- Ensure sufficient disk space for simulation results
- Check `simulation_results/` for detailed SST output logs

## Troubleshooting

If experiments fail:
1. Check that SST is properly installed and in PATH
2. Verify Python dependencies are installed
3. Check disk space in `simulation_results/`
4. Review SST output logs in simulation directories
5. Ensure topology files are available in `topologies/`

For questions or issues, refer to the main project README.
