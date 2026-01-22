# Nexullance Module Configuration

This project is designed to work with different implementations of the Nexullance routing optimization algorithm. The Nexullance module path is configurable to allow easy switching between implementations.

## Configuration

The Nexullance module is configured in `paths.py`:

```python
# Current default configuration
NEXULLANCE_MODULE_PATH = "topoResearch.nexullance.ultility"
NEXULLANCE_CONTAINER_CLASS = "nexullance_exp_container"
```

To use an alternative Nexullance implementation, simply change these values in `paths.py`.

## Required API

Any alternative Nexullance implementation must provide a container class with the following API:

### Class Constructor

```python
class nexullance_exp_container:
    def __init__(self, topo_name: str, V: int, D: int, EPR: int,
                 Cap_core: float, Cap_access: float,
                 Demand_scaling_factor: float):
        """
        Initialize the Nexullance container.
        
        Args:
            topo_name: Topology name (e.g., "RRG", "Slimfly", "DDF")
            V: Number of vertices/routers
            D: Degree of routers
            EPR: Endpoints per router
            Cap_core: Core link capacity in Gbps
            Cap_access: Access link capacity in Gbps
            Demand_scaling_factor: Scaling factor for demand matrices
        """
        pass
```

### Single-Demand (SD) Optimization

```python
def run_nexullance_IT_return_RT(self, M_EPs: np.ndarray, 
                                traffic_name: str, 
                                _debug: bool = False):
    """
    Run single-demand Nexullance optimization.
    
    Args:
        M_EPs: Demand matrix (num_endpoints × num_endpoints) in Gbps
        traffic_name: Name/identifier for this traffic pattern
        _debug: Enable debug output
        
    Returns:
        dict: Routing table in format {(src, dst): [(path, weight), ...]}
              where path is a list of vertices including source
    """
    pass
```

### Multi-Demand (MD) Optimization

```python
def run_MD_nexullance_IT_return_RT(self, M_EPs_s: list[np.ndarray], 
                                   M_EPs_weights: list[float],
                                   _debug: bool = False):
    """
    Run multi-demand Nexullance optimization.
    
    Args:
        M_EPs_s: List of demand matrices (each num_endpoints × num_endpoints) in Gbps
        M_EPs_weights: List of weights for each demand matrix (must sum to 1.0)
        _debug: Enable debug output
        
    Returns:
        tuple: (objective_value, routing_table)
            - objective_value: float, optimization objective value
            - routing_table: dict, format {(src, dst): [(path, weight), ...]}
    """
    pass
```

## Routing Table Format

The routing table returned by Nexullance must follow this format:

```python
{
    (src_ep, dst_ep): [
        (path, weight),
        (path, weight),
        ...
    ],
    ...
}
```

Where:
- `src_ep`, `dst_ep`: Endpoint IDs (integers)
- `path`: List of vertex IDs including source (e.g., `[0, 2, 5, 7]`)
- `weight`: Float in range [0, 1], weights for each (src, dst) pair should sum to 1.0

**Example:**
```python
{
    (0, 3): [
        ([0, 1, 3], 0.6),
        ([0, 2, 3], 0.4)
    ],
    (0, 5): [
        ([0, 1, 4, 5], 1.0)
    ]
}
```

The system will automatically convert this to SST's required format.

## Usage Examples

### Using Default Nexullance Implementation

No configuration changes needed. The system uses `topoResearch.nexullance.ultility` by default.

### Using an Alternative Implementation

1. Create your alternative implementation with the required API:

```python
# my_nexullance/improved_optimizer.py

class ImprovedNexullanceContainer:
    def __init__(self, topo_name, V, D, EPR, Cap_core, Cap_access, Demand_scaling_factor):
        # Your initialization code
        pass
    
    def run_nexullance_IT_return_RT(self, M_EPs, traffic_name, _debug=False):
        # Your SD optimization code
        return routing_table
    
    def run_MD_nexullance_IT_return_RT(self, M_EPs_s, M_EPs_weights, _debug=False):
        # Your MD optimization code
        return obj_value, routing_table
```

2. Update `paths.py`:

```python
NEXULLANCE_MODULE_PATH = "my_nexullance.improved_optimizer"
NEXULLANCE_CONTAINER_CLASS = "ImprovedNexullanceContainer"
```

3. Run experiments as usual - the system will automatically use your implementation.

## Testing Alternative Implementations

After configuring a new Nexullance module:

```bash
# Test with EFM experiments
cd EFM_experiments
python3 test_sample_sweep_framework.py

# Test with Merlin experiments  
cd ../Merlin_experiments
python3 run_experiments.py --topo-name RRG --V 16 --D 5

# Test scaling experiments
cd ../Merlin_scaling
python3 run_scaling_experiments.py --topo-name Slimfly --max-routers 100
```

## Compatibility Notes

- The alternative implementation must be importable from the project root
- All methods must accept the parameters shown in the API specification
- Return values must match the specified formats
- The routing table format is critical for correct SST simulation

## Troubleshooting

**Import Error:**
- Ensure your module is in the Python path
- Check that the module and class names in `paths.py` are correct
- Verify the module structure matches the import path

**API Mismatch:**
- Review the required API signatures above
- Check that parameter names and types match
- Ensure return value formats are correct

**Routing Table Issues:**
- Verify paths include the source vertex
- Check that weights sum to 1.0 for each (src, dst) pair
- Ensure vertex IDs are valid for the topology
