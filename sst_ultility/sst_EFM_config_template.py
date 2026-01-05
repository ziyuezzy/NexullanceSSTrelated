
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
    from sst.ember import *
    
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required. Please install NetworkX and try again.")
        sys.exit(1)
    
    # Import HPC topology utilities
    from HPC_topo import *

    UNIFIED_ROUTER_LINK_BW=config_dict['UNIFIED_ROUTER_LINK_BW']
    V=config_dict['V']
    D=config_dict['D']
    EPR=(D+1)//2
    topo_name=config_dict['topo_name']
    identifier=config_dict['identifier']
    routing_algo=config_dict['routing_algo']

    # BENCH="FFT3D"
    # BENCH_PARAMS=" nx=100 ny=100 nz=100 npRow=12"
    # CPE=1
    BENCH=config_dict['benchmark']
    BENCH_PARAMS=config_dict['benchargs']
    CPE=config_dict['Cores_per_EP']

    gen_InterNIC_traffic_trace: bool = 'traffic_trace_file' in config_dict.keys()
    Traffic_trace_file=""
    if gen_InterNIC_traffic_trace:
        Traffic_trace_file=config_dict['traffic_trace_file']


    EXP_SUFFIX=""

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

    # Use standard firefly defaults - traffic tracing is now handled by endpointNIC plugins
    PlatformDefinition.setCurrentPlatform("firefly-defaults")  


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
    endpointNIC = EndpointNIC(use_reorderLinkControl=True, topo=topo)
    
    # Add source routing plugin
    endpointNIC.addPlugin("sourceRoutingPlugin", routing_table=routing_table)
    
    # Add traffic tracing plugin if enabled
    if gen_InterNIC_traffic_trace:
        endpointNIC.addPlugin("trafficTracingPlugin", 
                             csv_filename=Traffic_trace_file,
                             enable_tracing=True)
    
    # Configure network interface parameters
    endpointNIC.link_bw = f"{UNIFIED_ROUTER_LINK_BW}Gb/s"
    endpointNIC.input_buf_size = "32kB"
    endpointNIC.output_buf_size = "32kB"
    endpointNIC.vn_remap = [0]
    
    ep = EmberMPIJob(0,topo.getNumNodes(), numCores = CPE*EPR)
    ep.setEndpointNIC(endpointNIC)
    ep.addMotif("Init")
    ep.addMotif(BENCH+BENCH_PARAMS)
    ep.addMotif("Fini")
    ep.nic.nic2host_lat= "10ns"
        
    system = System()
    system.setTopology(topo)
    system.allocateNodes(ep,"linear")

    system.build()

    # sst.setStatisticLoadLevel(9)

    # sst.setStatisticOutput("sst.statOutputCSV");
    # sst.setStatisticOutputOptions({
    #     "filepath" : "stats.csv",
    #     "separator" : ", "
    # })

    # sst.setStatisticLoadLevel(10)

    # sst.setStatisticOutput("sst.statOutputCSV");
    # sst.setStatisticOutputOptions({
    #     "filepath" : f"statistics_BENCH_{BENCH+BENCH_PARAMS}_EPR_{EPR}_ROUTING_{PATHS}_V_{V}_D_{D}_TOPO_{topo_name}_SUFFIX_{EXP_SUFFIX}_.csv"                                                                                                                                                                                                                                                                                                     ,
    #     "separator" : ", "
    # })
    # sst.enableAllStatisticsForComponentType("merlin.linkcontrol", {"type":"sst.AccumulatorStatistic","rate":"0ns"})

    # sst.enableAllStatisticsForAllComponents({"type":"sst.AccumulatorStatistic","rate":"0ns"})

