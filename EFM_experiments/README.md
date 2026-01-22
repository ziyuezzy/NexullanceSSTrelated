# EFM Experiments

Reproduction of experiments from `archive/EFM_experiments` using the new unified routing framework.

## Overview

This directory contains experiments comparing three routing methods across multiple MPI benchmarks:

- **shortest_path**: Standard shortest path routing (baseline, formerly ECMP_ASP)
- **ugal**: Universal Globally-Adaptive Load-balancing routing
- **nexullance**: Nexullance-optimized weighted routing (formerly MD_IT/MD_MP)
  - **SD (Single-Demand)**: Optimizes for accumulated traffic demand
  - **MD (Multi-Demand)**: Optimizes for multiple sampled traffic demands

## Topology Configuration

- **Topology**: RRG (Random Regular Graph)
- **Vertices (V)**: 36
- **Degree (D)**: 5
- **Endpoints per Router (EPR)**: 3
- **Cores per Endpoint**: 4
- **Link Bandwidth**: 16 Gbps

## Benchmarks

### 1. Allreduce
Tests collective reduction operations with varying message counts.

**Parameters**:
- iterations: 10
- count: [256, 512, 1024, 2048]

### 2. Alltoall
Tests all-to-all communication patterns with varying message sizes.

**Parameters**:
- bytes: [1, 8, 64]

### 3. FFT3D
Tests 3D Fast Fourier Transform with varying problem sizes.

**Parameters**:
- nx, ny, nz: [256, 512, 1024, 2048]
- npRow: 12

## Running Experiments

### Run All Experiments
```bash
./run_experiments.sh
```

Or run the Python script directly:
```bash
python3.12 run_experiments.py
```

### Analyze Results
After experiments complete, analyze and visualize results:
```bash
python3.12 analyze_results.py
```

This will generate:
- Simulation time comparison plots
- Speedup comparison plots
- Summary statistics tables

### Test Multi-Demand Nexullance
Test both SD (single-demand) and MD (multi-demand) Nexullance methods:
```bash
python3.12 test_multi_demand.py
```

This will compare:
- SD method: Single accumulated demand matrix
- MD method: Multiple sampled demand matrices

### Run Sample Sweep Experiments

Reproduce experiments from `archive/EFM_experiments/RRG_4CPE` with sample sweep for MD method:

**Default configuration (RRG, V=36, D=5):**
```bash
./run_sample_sweep_experiments.sh
```

**Custom topology:**
```bash
./run_sample_sweep_experiments.sh --topo DDF --V 40 --D 6
```

**All options:**
```bash
python3.12 run_sample_sweep_experiments.py --help
```

Available options:
- `--topo NAME`: Topology name (default: RRG)
- `--V NUM`: Number of vertices/routers (default: 36)
- `--D NUM`: Degree of routers (default: 5)
- `--cores-per-ep NUM`: Cores per endpoint (default: 4)
- `--no-prompt`: Skip confirmation and start immediately

This will run all three benchmarks (Allreduce, Alltoall, FFT3D) with:
- **Shortest-path**: Baseline routing
- **UGAL**: Adaptive load-balancing
- **Nexullance SD**: Single-demand optimization
- **Nexullance MD**: Multi-demand with sample sweep (1, 2, 4, 8, 16, 32, 64, 128 samples)

#### Analyze Sample Sweep Results

After sample sweep experiments complete:

```bash
python3.12 analyze_sample_sweep_results.py
```

This will generate:
- Sample sweep speedup plots for each benchmark
- Optimal sample count analysis
- Comprehensive summary comparison table

## Output Structure

```
EFM_experiments/
├── run_experiments.py          # Main experiment script
├── run_experiments.sh          # Bash launcher
├── run_sample_sweep_experiments.py  # Sample sweep experiments (archive/RRG_4CPE)
├── run_sample_sweep_experiments.sh  # Bash launcher for sample sweep
├── analyze_results.py          # Results analysis
├── analyze_sample_sweep_results.py  # Sample sweep analysis
├── test_multi_demand.py        # Test SD vs MD methods
├── README.md                   # This file
├── RRG_V36_D5_CPE4/           # Basic experiment results
│   ├── Allreduce_count_x_shortest_path_result_<timestamp>.csv
│   ├── Allreduce_count_x_ugal_result_<timestamp>.csv
│   ├── Allreduce_count_x_nexullance_result_<timestamp>.csv
│   └── plots/
│       ├── Allreduce_simulation_time.png
│       └── Allreduce_speedup.png
└── RRG_V36_D5_CPE4_sample_sweep/  # Sample sweep results
    ├── Allreduce_count_x_shortest_path_result_<timestamp>.csv
    ├── Allreduce_count_x_ugal_result_<timestamp>.csv
    ├── Allreduce_count_x_nexullance_SD_result_<timestamp>.csv
    ├── Allreduce_count_x_nexullance_MD_sample_sweep_<timestamp>.csv
    ├── summary_comparison.csv
    └── plots/
        ├── Allreduce_sample_sweep_speedup.png
        ├── Alltoall_sample_sweep_speedup.png
        ├── FFT3D_sample_sweep_speedup.png
        └── optimal_samples_comparison.png
```

## CSV File Formats

### Shortest-path and UGAL Results
```
param_name,sim_time_ms,speedup
256,5.0899,1.0
512,7.2341,1.0
...
```

### Nexullance Results
```

### Nexullance MD Sample Sweep Results
```
param_name,num_samples,baseline_sim_time_ms,optimized_sim_time_ms,speedup,improvement_percent
256,1,5.0899,4.2100,1.2088,20.88
256,2,5.0899,4.1800,1.2176,21.76
256,4,5.0899,4.1234,1.2345,23.45
256,8,5.0899,4.1500,1.2264,22.64
...
```
param_name,baseline_sim_time_ms,optimized_sim_time_ms,speedup,improvement_percent
256,5.0899,4.1234,1.2345,23.45
512,7.2341,5.8976,1.2267,22.67
...
```

## Key Differences from Archive

1. **Unified Framework**: Uses new `run_ember_simulation()` and `run_ember_experiment_with_nexullance()` functions
2. **Simplified Routing**: Three clear routing methods instead of multiple MD variants
3. **Cleaner Code**: No complex wrapper classes, direct function calls
4. **Better Tracking**: All simulations tracked in `simulation_results/` directory
5. **Consistent Naming**: `shortest_path` (baseline) instead of `ECMP_ASP`

## Performance Metrics

- **Simulation Time**: Time to complete MPI benchmark (in milliseconds)
- **Speedup**: Ratio of baseline time to optimized time (speedup > 1.0 means improvement)
- **Improvement**: Percentage improvement over baseline

## Notes

- The Nexullance workflow includes traffic demand collection, optimization, and execution
- UGAL provides adaptive load-balancing without requiring traffic demand collection
- Shortest-path serves as the baseline for all comparisons
- Results may vary depending on traffic patterns and network conditions
