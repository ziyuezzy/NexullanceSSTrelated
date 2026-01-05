# Nexullance SST Related Utilities

This repository contains utilities and experiments for the SST (Structural Simulation Toolkit) network simulator, specifically designed to work with our enhanced SST fork featuring the new `anytopo` topology and `endpointNIC` system.

## New Features (2026 Update)

This utility repository has been **completely refactored** to work with the latest SST fork that includes:

- **AnyTopo System**: Arbitrary topology support with NetworkX integration
- **EndpointNIC**: Pluggable network interface architecture  
- **Source Routing Plugin**: Dynamic routing table calculation
- **Traffic Tracing Plugin**: Standardized packet-level tracing
- **Simplified Configuration**: No more pickle files or manual path dictionaries

For migrating from the old system, see [MIGRATION.md](MIGRATION.md).

## Prerequisites

1. **SST Installation**: Install our enhanced sst-core and sst-elements fork
2. **Dependencies**: NetworkX, NumPy, Pandas
3. **Environment**: Add SST bin path to $PATH environment variable

```bash
# Install dependencies
pip install networkx numpy pandas matplotlib

# Set environment (add to ~/.bashrc)
export PATH="/path/to/sst/bin:$PATH"
```

## Quick Start

The utility provides two main experiment types with a clean, simple API:

### 1. EFM Experiments (Ember + Firefly + Merlin)
```python
from sst_ultility.anytopo_utils import create_base_config, run_efm_experiment

# Create base topology configuration
base_config = create_base_config(topo_name="RRG", V=36, D=5, link_bw=16)

# Run FFT3D benchmark with traffic tracing
stdout, stderr, returncode = run_efm_experiment(
    base_config=base_config,
    benchmark="FFT3D", 
    bench_args=" nx=100 ny=100 nz=100 npRow=12",
    cores_per_ep=1,
    traffic_trace_file="fft3d_trace.csv"
)
```

### 2. Merlin Experiments (Synthetic Traffic)
```python
from sst_ultility.anytopo_utils import create_base_config, run_merlin_experiment

# Create base topology configuration  
base_config = create_base_config(topo_name="Slimfly", V=18, D=5)

# Run uniform traffic pattern at 50% load
stdout, stderr, returncode = run_merlin_experiment(
    base_config=base_config,
    load=0.5,
    traffic_pattern="uniform"
)
```

### 3. Analyze Traffic Traces
```python
from traffic_analyser.traffic_analyser import traffic_analyser

# Analyze packet-level traces from new TrafficTracingPlugin
analyzer = traffic_analyser(
    input_csv_path="fft3d_trace.csv",
    V=36, D=5,
    topo_name="RRG", 
    EPR=3,  # endpoints per router
    Cap_core=10, Cap_access=10,
    suffix="test"
)

# Get processed packet data
pkt_data = analyzer.pkt_data  
print(f"Analyzed {len(pkt_data)} packets")
```

## Directory Structure

```
NexullanceSSTrelated/
├── sst_ultility/              # SST simulation utilities
│   ├── anytopo_utils.py       # High-level anytopo functions
│   ├── ultility.py           # Convenience wrapper functions
│   ├── sst_EFM_config_template.py    # Ember config template
│   └── sst_merlin_config_template.py # Merlin config template
├── traffic_analyser/          # Traffic analysis tools
│   ├── traffic_analyser.py   # Enhanced analyzer (supports new CSV format)
│   └── helpers.py            # Filename parsing utilities
├── EFM_experiments/          # Ember MPI benchmark experiments
├── Merlin_experiments/       # Merlin synthetic traffic experiments  
├── topoResearch/            # Topology research and optimization
└── MIGRATION.md             # Migration guide from old system
```

## Supported Topologies

The new anytopo system supports any topology that can be represented as a NetworkX graph:

- **Slim Fly**: High-performance, low-diameter topology
- **Dragonfly**: Hierarchical topology for large-scale systems
- **Jellyfish**: Random regular graph topology
- **Polar Fly**: Cost-effective topology based on projective geometry
- **Custom**: Any topology defined as NetworkX graph

## Key Improvements

### 1. **Simplified Configuration**
- No more pickle files for edge lists and routing paths
- Direct NetworkX graph import
- Automatic routing table calculation

### 2. **Pluggable Architecture**  
- Modular plugin system for NIC functionality
- Easy to extend with new features
- Clean separation of concerns

### 3. **Enhanced Traffic Tracing**
- Standardized CSV output format
- Thread-safe operation
- Automatic packet ID assignment

### 4. **Better Performance**
- Optimized routing algorithms
- Reduced memory usage
- Faster simulation startup

## Example Usage Scenarios

### Research Experiments
```python
# Compare routing algorithms on different topologies
for topo in ["Slimfly", "Dragonfly", "FatTree"]:
    for routing in ["source_routing", "dest_tag_routing"]:
        run_ember_experiment(
            topo_name=topo, V=36, D=5,
            benchmark="Allreduce",
            traffic_trace_file=f"{topo}_{routing}_trace.csv"
        )
```

### Performance Analysis
```python
# Sweep load levels
loads = [0.1, 0.3, 0.5, 0.7, 0.9]
for load in loads:
    run_synthetic_experiment(
        topo_name="RRG", V=36, D=5,
        load=load, traffic_pattern="uniform"
    )
```

## Getting Started

1. **Install Dependencies**: `pip install networkx numpy pandas matplotlib`
2. **Set Environment**: Add SST bin path to `$PATH`
3. **Test Installation**: Run `python test_migration.py`
4. **Run Experiments**: Use the example code above

## Support

See [MIGRATION.md](MIGRATION.md) for detailed information about transitioning from older SST systems.