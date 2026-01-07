import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sst_ultility.ultility import run_merlin_simulation
from paths import SST_MERLIN_TEMP_PATH

config_dict = {
    'LOAD':1.0,
    'UNIFIED_ROUTER_LINK_BW':16,  #Gbps
    'V':16,
    'D':5,
    'topo_name':"RRG",
    'paths':"ASP",
    'routing_algo': "nonadaptive", # same for two subnets
    'traffic_pattern': "uniform"
}

# Fixed: use run_merlin_simulation instead of run_sst
run_merlin_simulation(
    topo_name=config_dict['topo_name'],
    V=config_dict['V'],
    D=config_dict['D'],
    load=config_dict['LOAD'],
    traffic_pattern=config_dict['traffic_pattern'],
    link_bw=config_dict['UNIFIED_ROUTER_LINK_BW']
) 

# config_dict = {
#     'LOAD':0.1,
#     'UNIFIED_ROUTER_LINK_BW':16,  #Gbps
#     'V':36,
#     'D':5,
#     'topo_name':"RRG",
#     'paths':"ASP",
#     'routing_algo': "nonadaptive", # same for two subnets
#     'traffic_pattern': "uniform"
# }

# # Fixed: use run_merlin_simulation instead of run_sst
# run_merlin_simulation(
#     topo_name=config_dict['topo_name'],
#     V=config_dict['V'],
#     D=config_dict['D'],
#     load=config_dict['LOAD'],
#     traffic_pattern=config_dict['traffic_pattern'],
#     link_bw=config_dict['UNIFIED_ROUTER_LINK_BW']
# ) 

