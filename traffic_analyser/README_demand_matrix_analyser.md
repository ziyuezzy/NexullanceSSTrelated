# Demand Matrix Analyser - Usage Guide

## Overview

The `demand_matrix_analyser` class is designed to handle aggregated traffic demand matrix data from SST simulations, which is more scalable than the packet-level event format used by the original `traffic_analyser` class.

## Key Differences from `traffic_analyser`

### Data Format

**Old format (`traffic_analyser`):**
- Individual packet events (in/out of network)
- Columns: pkt_id, type, time_ns, srcNIC, destNIC, Size_Bytes
- Very detailed but large file size

**New format (`demand_matrix_analyser`):**
- Aggregated statistics per source-destination pair
- Columns: ComponentName, StatisticName, StatisticSubId, SimTime, Sum.u64, etc.
- Compact and scalable

### Sampling Behavior

**Old format:**
- Can sample at any arbitrary interval
- Sampling interval is flexible

**New format:**
- Data is pre-aggregated at a fixed interval (e.g., 10ms)
- Can only resample at integer multiples of the original interval
- If original interval is 10ms, you can sample at 10ms, 20ms, 30ms, etc.
- The class validates that `num_samples` is a divisor of the total number of matrices

## API Compatibility

The new class maintains the same key APIs as the old one:

### Common Methods

```python
# Initialize
analyzer = demand_matrix_analyser(
    input_csv_path="path/to/demand_matrix.csv",
    V=36,                # Number of vertices
    D=5,                 # Degree parameter
    topo_name="RRG",     # Topology name
    EPR=1,               # Endpoints per router
    Cap_core=16.0,       # Core link capacity (Gbps)
    Cap_access=16.0,     # Access link capacity (Gbps)
    suffix="experiment1"
)

# Sample traffic (main API)
matrices, weights, interval_us = analyzer.sample_traffic(
    num_samples=10,              # Must be a divisor of total matrices
    filtering_threshold=0.0,     # Min max link load threshold
    auto_scale=False             # Auto-scale to target load
)

# With auto-scaling
matrices, weights, interval_us, scale_factor = analyzer.sample_traffic(
    num_samples=10,
    filtering_threshold=1.0,
    auto_scale=True
)

# Get accumulated demand matrix
acc_matrix = analyzer.get_accumulated_demand_matrix(plot=False)

# Filter matrices
filtered_matrices, filtered_weights = analyzer.filter_traffic_demand_matrices(
    matrices, weights, max_link_load_threshold=1.0
)

# Auto-scale matrices
scaled_matrices, scale_factor = analyzer.auto_scale(matrices)
```

### Differences

1. **`get_ave_message_lat_us()`**: 
   - Old: Returns average packet latency
   - New: Returns `None` with a warning (latency not available from aggregated data)

2. **Sampling constraints**:
   - Old: Any number of samples
   - New: `num_samples` must be a divisor of total available matrices

3. **Visualization methods**:
   - Old: Multiple methods (visualize_enroute_data, visualize_ranged_sent_data, etc.)
   - New: `visualize_demand_evolution()` for time-series animation

## New Features

### Statistics

```python
# Get detailed statistics
stats = analyzer.get_statistics()
# Returns dict with:
# - num_matrices
# - sampling_interval_ns/us
# - total_traffic_gbps (mean, std, min, max)
# - max_flow_gbps (mean, std, min, max)
# - sparsity (mean, std, min, max)

# Print formatted statistics
analyzer.print_statistics()
```

### Demand Evolution Visualization

```python
# Create animation of demand matrix over time
ani = analyzer.visualize_demand_evolution(vmax=1.0)
# Save animation
ani.save('demand_evolution.mp4', writer='ffmpeg', fps=5)
```

## Example Usage

### Basic Example

```python
from traffic_analyser import demand_matrix_analyser

# Initialize
analyzer = demand_matrix_analyser(
    input_csv_path="simulation_results/demand_matrix.csv",
    V=36, D=5, topo_name="RRG", EPR=1,
    Cap_core=16.0, Cap_access=16.0,
    suffix="FFT3D_test"
)

# Print statistics
analyzer.print_statistics()

# Sample traffic
matrices, weights, interval_us = analyzer.sample_traffic(
    num_samples=10,
    filtering_threshold=0.0
)

print(f"Sampled {len(matrices)} matrices")
print(f"Sampling interval: {interval_us} us")
```

### Determine Valid Sample Counts

```python
# Get statistics to find total matrices
stats = analyzer.get_statistics()
num_matrices = stats['num_matrices']

# Find valid divisors
valid_samples = [i for i in range(1, num_matrices + 1) 
                 if num_matrices % i == 0]
print(f"Valid sample counts: {valid_samples}")
```

### Error Handling

```python
try:
    matrices, weights, interval = analyzer.sample_traffic(num_samples=7)
except ValueError as e:
    print(f"Invalid sample count: {e}")
    # Error message will suggest valid values
```

## Migration Guide

If you have code using the old `traffic_analyser`:

1. **Check your CSV format**: If it has packet events, use `traffic_analyser`. If it has aggregated statistics, use `demand_matrix_analyser`.

2. **Update sampling calls**: Ensure `num_samples` is a valid divisor:
   ```python
   # Old: any number works
   matrices, weights, interval = old_analyzer.sample_traffic(num_samples=17)
   
   # New: must be divisor
   stats = new_analyzer.get_statistics()
   num_matrices = stats['num_matrices']
   valid_count = num_matrices // 2  # or any divisor
   matrices, weights, interval = new_analyzer.sample_traffic(num_samples=valid_count)
   ```

3. **Handle missing latency**: If you use `get_ave_message_lat_us()`, it will return `None` with the new class.

4. **Update visualization**: Use `visualize_demand_evolution()` instead of the old visualization methods.

## Performance Comparison

**Old format advantages:**
- Fine-grained timing information
- Can compute packet-level latency
- Flexible sampling intervals

**New format advantages:**
- Much smaller file size (10-100x reduction)
- Faster to load and process
- Scalable to large simulations
- Pre-aggregated statistics available

**Recommendation:** Use the new format for production simulations and large-scale experiments. Use the old format only when packet-level detail is specifically needed.
