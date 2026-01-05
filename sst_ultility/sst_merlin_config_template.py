
import os
import sys

# Add topologies directory to path for HPC topology classes
topologies_path = os.path.join(os.path.dirname(__file__), '..', 'topoResearch', 'topologies')
sys.path.insert(0, topologies_path)

# Copyright 2009-2024 NTESS. Under the terms
# of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Copyright (c) 2009-2024, NTESS
# All rights reserved.
#
# This file is part of the SST software package. For license
# information, see the LICENSE file in the top level directory of the
# distribution.

if __name__ == "__main__":
    import sst
    from sst.merlin.base import *
    from sst.merlin.endpoint import *
    from sst.merlin.interface import *
    from sst.merlin.topology import *
    from sst.merlin.targetgen import *
    
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required. Please install NetworkX and try again.")
        sys.exit(1)
    
    # Import HPC topology utilities
    from HPC_topo import *

    LOAD=config_dict['LOAD']
    UNIFIED_ROUTER_LINK_BW=config_dict['UNIFIED_ROUTER_LINK_BW']
    V=config_dict['V']
    D=config_dict['D']
    EPR=(D+1)//2
    topo_name=config_dict['topo_name']
    Paths=config_dict['paths']
    routing_algo=config_dict['routing_algo']

    topo_full_name=f"({V},{D}){topo_name}"

    ### Setup the topology using new anytopo system
    # Create the HPC topology instance and get NetworkX graph
    hpc_topo = HPC_topo.initialize_child_instance(topo_name + "topo", V, D)
    hpc_topo.set_endpoints_per_router(EPR)
    G = hpc_topo.get_nx_graph()
    
    # Setup anytopo
    topo = topoAny()
    topo.routing_mode = "source_routing"  # Use source routing mode
    topo.topo_name = topo_full_name
    topo.import_graph(G)
    
    # Calculate routing table
    routing_table = topo.calculate_routing_table()
    
    # Set up the routers
    router = hr_router()
    router.link_bw = f"{UNIFIED_ROUTER_LINK_BW}Gb/s"
    router.flit_size = "32B"
    router.xbar_bw = f"{UNIFIED_ROUTER_LINK_BW*2}Gb/s" # 2x crossbar speedup
    router.input_latency = "20ns"
    router.output_latency = "20ns"
    router.input_buf_size = "32kB"
    router.output_buf_size = "32kB"   
    router.num_vns = 2
    router.xbar_arb = "merlin.xbar_arb_rr"

    topo.router = router
    topo.link_latency = "20ns"
    topo.host_link_latency = "10ns"
    
    ### set up the endpoint with endpointNIC and plugins
    endpointNIC = EndpointNIC(use_reorderLinkControl=False, topo=topo)
    
    # Add source routing plugin
    endpointNIC.addPlugin("sourceRoutingPlugin", routing_table=routing_table)
    
    # Configure network interface parameters
    endpointNIC.link_bw = f"{UNIFIED_ROUTER_LINK_BW}Gb/s"
    endpointNIC.input_buf_size = "32kB"
    endpointNIC.output_buf_size = "32kB"
    endpointNIC.vn_remap = [0]
    
    targetgen = 0
    traffic_pattern = config_dict['traffic_pattern']
    if traffic_pattern == 'uniform':
        targetgen=UniformTarget()
    elif traffic_pattern.startswith("shift_"):
        shift_bits = traffic_pattern[6:]
        if shift_bits == "half":
            shift_bits = (V*EPR)//2
        else:
            try:
                shift_bits = int(shift_bits)
            except ValueError:
                print(f"Error: traffic pattern: ", traffic_pattern)
                sys.exit(1)
        targetgen=ShiftTarget()
        targetgen.shift=shift_bits
    else:
        print(f"Error: invalid traffic pattern: {traffic_pattern}")
        sys.exit(1)

    ep = OfferedLoadJob(0,topo.getNumNodes())
    ep.setEndpointNIC(endpointNIC)
    ep.pattern=targetgen
    ep.offered_load = LOAD
    ep.link_bw = f"{UNIFIED_ROUTER_LINK_BW}Gb/s"
    ep.message_size = "32B"
    ep.collect_time = "200us"
    ep.warmup_time = "200us"
    ep.drain_time = "1000us" 

    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep,"linear")

    system.build()

    # sst.setStatisticLoadLevel(10)
    # sst.setStatisticOutput("sst.statOutputCSV");
    # sst.setStatisticOutputOptions({
    #     "filepath" : f"load_{LOAD}.csv",
    #     "separator" : ", "
    # })
    # # sst.enableAllStatisticsForComponentType("merlin.linkcontrol", {"type":"sst.AccumulatorStatistic","rate":"0ns"})
    # sst.enableAllStatisticsForAllComponents({"type":"sst.AccumulatorStatistic","rate":"0us"})

    # delete python objects:
    del topo
    del router
    del networkif
    del targetgen
    del ep
    del system