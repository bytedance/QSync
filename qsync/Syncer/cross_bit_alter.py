# This is synchronizer for case that memory is not scarced
# Input: Profiler Result for T4, FP16 result for V100 
# Target:
    # synchronized the latency of T4 with V100, don't add additional latency. 
    # T4 recover some bit.

from qsync.Predictor.cross_node_predict import (
    predict_cross_result
)

from qsync.Predictor.utils import (
    load_pickle_graph, get_abs_cng_path,
    load_cng_data,
    get_abs_cng_data_path,
    load_node_data,
    read_prof_indicator,
    load_data_mappers, load_cast_mappers,
    set_gpu_maps_with_device_and_bit
)

from qsync.Predictor.core import (
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
    greedy
)

from qsync.Predictor.config import config
from qsync.LpTorch.conf import config as qconfig

from qsync.Predictor.fastest_plan import (
    get_latency_optimized_plan, set_optimal_bit
)


import argparse
parser = argparse.ArgumentParser(description='mv profiled result scripts.')
parser.add_argument('--model-name', type=str, default='bert',
                    help='mode_name') 
parser.add_argument('--indicator_type', type=str, default='GVAR',
                    help='indicator type') 
parser.add_argument('--arch', type=int, default=75, help='arch type')
parser.add_argument('--random', action="store_true", help="random indicator.")
parser.add_argument('--backbone', type=str, default=None,
                    help='backbone for change') 
parser.add_argument('--comparer', type=str, default=None,
                    help='comparer for bitwidth change, usually the faster one. e.g. T4') 

if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name

    # arch
    arch = args.arch
    qconfig.set_available_bit(arch)

    # load model mappers
    node_data_int8_file = f'{model_name}_int8_comm_t4'
    node_data_fp16_file = f'{model_name}_fp16_comm_t4'
    node_data_fp32_file = f'{model_name}_fp32_comm_t4'
    mapper_file_dict = {
        8: node_data_int8_file,
        16: node_data_fp16_file,
        32: node_data_fp32_file,
    }

    # find fastest way & update
    ava_bits = qconfig.available_bits
    load_data_mappers(ava_bits, mapper_file_dict)
    load_cast_mappers(ava_bits, mapper_file_dict)
    # backbone file
    backbone_file = node_data_fp16_file if args.backbone is None else args.backbone
    fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, OVERALL_TIME = load_cng_data(backbone_file)
    T4_file_path = f'{backbone_file}_cpeg.pkl'
    T4_file_path = get_abs_cng_path(T4_file_path)
    T4_graph = load_pickle_graph(T4_file_path)


    # load V100 fp16 result
    V100_file_name = f'{model_name}_fp32_comm_v100' if args.comparer is None else args.backbone
    V100_file_path = f'{V100_file_name}_cpeg.pkl'
    V100_file_path = get_abs_cng_path(V100_file_path)
    V100_graph = load_pickle_graph(V100_file_path)
    V100_data_pack = load_cng_data(V100_file_name)
    
    # Profiller Data, same for T4 or V100
    profile_data_path = f"/qsync_niti_based/profile_result/profile_data_{model_name}_75_{args.indicator_type}.npz"
    indicator, layer_order, M, model_context_size = read_prof_indicator(profile_data_path)
    '''
        shuffle the indicator result if necessary
    '''
    # collect all layes has values
    if args.random:
        print("Original indicator: ", indicator)
        print("*--------Doing random shuffle---------*")
        import numpy as np 
        keys, values = [], []
        for k, v in indicator.items():
            if len(v) > 0:
                keys.append(k)
                values.append(v)
        # set seed.
        np.random.seed(42)
        np.random.shuffle(values)
        for idx, key in enumerate(keys):
            indicator[key] = values[idx]
        print("Shuffled:", indicator)
        # import pdb; pdb.set_trace()
    # Set a Mem Limitation
    MAX_M = sum(list(M.values())) + model_context_size
    MEM_LIMIT = 6 * config.MB # summation used in KB; do some conversion
    if MAX_M < MEM_LIMIT: MEM_LIMIT = None 
    else:
        MEM_LIMIT = MEM_LIMIT
    MEM_LIMIT = None # add if you need
    
    # Setup cpu gaps for beter prediction
    set_gpu_maps_with_device_and_bit('t4', 16)
    add_cpu_gaps(T4_graph)
    set_gpu_maps_with_device_and_bit('v100', 16)
    add_cpu_gaps(V100_graph)

    # search optimality
    # subgraph_bitwidth_mapper = get_latency_optimized_plan(fwd_di_graph, T4_graph, ava_bits, T4_dag_2_nodes, T4_mapper_dict, M, MEM_LIMIT)
    # set_dag_uniform_bitwidth(fwd_di_graph, 32)
    # set_optimal_bit(fwd_di_graph, subgraph_bitwidth_mapper)

    # init with 16
    set_dag_uniform_bitwidth(fwd_di_graph, 16)
    map_dag_bit_to_critical_graph(fwd_di_graph, T4_dag_2_nodes, T4_graph, T4_mapper_dict)

    # start search
    file_name = model_name
    greedy(indicator, layer_order, fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, T4_graph, V100_graph, M, MEM_LIMIT, model_name, model_name)