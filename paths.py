from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]

SST_EFM_TEMP_PATH = REPO_ROOT / "sst_ultility" / "sst_EFM_config_template.py"
SST_MERLIN_TEMP_PATH = REPO_ROOT / "sst_ultility" / "sst_merlin_config_template.py"
TRAFFIC_TRACES_DIR = REPO_ROOT / "traffic_traces"
SIMULATION_RESULTS_DIR = REPO_ROOT / "simulation_results"
TOPOLOGIES_DIR = REPO_ROOT / "topoResearch" / "topologies"

# Nexullance module configuration
# Change this to use alternative Nexullance implementations with compatible APIs
# Expected API: nexullance_exp_container class with methods:
#   - run_nexullance_IT_return_RT(M_EPs, traffic_name, _debug)
#   - run_MD_nexullance_IT_return_RT(M_EPs_s, M_EPs_weights, _debug)
NEXULLANCE_MODULE_PATH = "topoResearch.nexullance.ultility"
NEXULLANCE_CONTAINER_CLASS = "nexullance_exp_container"

# Ensure directories exist
TRAFFIC_TRACES_DIR.mkdir(parents=True, exist_ok=True)
SIMULATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)