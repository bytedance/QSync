# This is the final target for our predictor
# Given a graph of the maximized speed (e.g. in transformer, always full-fp16, but conv can have several layer that < 16). Since we provided the direct implementation to cutlass, the precision can be easily extended
# Given a graph of V100 (in fp32)
# We try to add bit to T4 that
    # 1. it satisfied the constraints
    # 2. it can approach the cost of V100
    # 3. Reduce the variance as much as possible

from cross_node_predict import (
    predict_cross_result
)

from utils import (
    load_pickle_graph, get_abs_cng_path,
    load_cng_data,
    get_abs_cng_data_path,
    load_node_data,
    read_prof_indicator,
    load_data_mappers, load_cast_mappers,
    set_gpu_maps_with_device_and_bit
)

from core import (
    change_dag_bit,
    get_available_bitwidth_change_node_in_dag, get_topo_available_bitwidth_change_node_order,
    init_graph_bit,
    change_bwd_cost,
    change_fwd_cost,
    check_graph_bit,
    add_cpu_gaps,
    set_dag_uniform_bitwidth,
    map_dag_bit_to_critical_graph
)

from algos import (
    greedy_mem_requirement_single_node
)

from config import config
from qsync.conf import config as qconfig

from fastest_plan import (
    get_latency_optimized_plan, set_optimal_bit
)


import argparse
parser = argparse.ArgumentParser(description=' mv profiled result scripts.')
parser.add_argument('--model-name', type=str, default='bert',
                    help='mode_name') 
parser.add_argument('--indicator_type', type=str, default='GVAR',
                    help='indicator type') 
parser.add_argument('--mem_limit', type=float, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    T4_file_name = f'{model_name}_fp16_comm_t4'
    T4_fp32_file_name = f'{model_name}_fp32_comm_t4'
    V100_file_name = f'{model_name}_fp32_comm_v100'

    V100_file_path = f'{V100_file_name}_cpeg.pkl'
    V100_file_path = get_abs_cng_path(V100_file_path)
    V100_graph = load_pickle_graph(V100_file_path)
    V100_data_pack = load_cng_data(V100_file_name)

    # _, V100_dag_2_nodes, V100_mapper_dict = V100_data_pack
    # get all bit relevant ops
    

    node_data_int8_file = f'{model_name}_int8_comm_t4'
    node_data_fp16_file = f'{model_name}_fp16_comm_t4'
    node_data_fp32_file = f'{model_name}_fp32_comm_t4'
    # find fastest way & update
    # load data maappers and cast mappers
    ava_bits = qconfig.available_bits
    load_data_mappers(ava_bits, [node_data_fp32_file, node_data_fp16_file, node_data_int8_file])
    load_cast_mappers(ava_bits, [node_data_fp32_file, node_data_fp16_file, node_data_int8_file])

    fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, OVERALL_TIME = load_cng_data(node_data_fp32_file)
    T4_file_path = f'{node_data_fp32_file}_cpeg.pkl'
    T4_file_path = get_abs_cng_path(T4_file_path)
    T4_graph = load_pickle_graph(T4_file_path)
    

    # read prof data, same for V100 or T4
    profile_data_path = f"/qsync_niti_based/profile_result/profile_data_{model_name}_75_{args.indicator_type}.npz"
    indicator, layer_order, M, model_context_size = read_prof_indicator(profile_data_path)
    # Set a Mem Limitation
    MAX_M = sum(list(M.values())) + model_context_size
    MEM_LIMIT = 16 * config.MB # summation used in KB; do some conversion
    if MAX_M < MEM_LIMIT: MEM_LIMIT = None 
    else:
        MEM_LIMIT = MEM_LIMIT
    # MEM_LIMIT = None
    MEM_LIMIT = args.mem_limit
    
    set_gpu_maps_with_device_and_bit('t4', 16)
    add_cpu_gaps(T4_graph)
    set_gpu_maps_with_device_and_bit('v100', 32)
    add_cpu_gaps(V100_graph)
    
    # subgraph_bitwidth_mapper = get_latency_optimized_plan(fwd_di_graph, T4_graph, ava_bits, T4_dag_2_nodes, T4_mapper_dict, M, MEM_LIMIT)
    # set_dag_uniform_bitwidth(fwd_di_graph, 32)
    # set_optimal_bit(fwd_di_graph, subgraph_bitwidth_mapper)
    set_dag_uniform_bitwidth(fwd_di_graph, 16)
    map_dag_bit_to_critical_graph(fwd_di_graph, T4_dag_2_nodes, T4_graph, T4_mapper_dict)
    # dag_available_dict = get_available_bitwidth_change_node_in_dag(fwd_di_graph)
    # topo_node_order = get_topo_available_bitwidth_change_node_order(fwd_di_graph)

    # len_ind = len(indicator) # don't consider last layer
    # len_dag_ava_layer = len(topo_node_order)
    # assert len_ind == len_dag_ava_layer, f"indicator number not correc {len_ind} {len_dag_ava_layer}"
    file_name = model_name
    greedy_mem_requirement_single_node(indicator, layer_order, fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, T4_graph, V100_graph, M, MEM_LIMIT, model_name, model_name, ind_name=args.indicator_type)








