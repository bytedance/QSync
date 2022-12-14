# test file
from qsync.Predictor.utils import (
    store_mapper_result_data, load_mapper_result_data,
    store_mapper_result_data_npy, load_mapper_result_data_npy,
    load_cng_data,
    load_data_mappers, load_cast_mappers,
    get_abs_cng_path, load_pickle_graph,
    set_gpu_maps_with_device_and_bit,
    percent_diff
)

from qsync.Predictor.core import (
    fetch_real_duration_from_traces,
    set_dag_uniform_bitwidth, set_layers_uniform_bitwidth,
    get_available_bitwidth_change_node_in_dag,
    change_relevant_node_bitwidth,
    check_dag_bit,
    map_dag_bit_to_critical_graph,
    fetch_cost_from_store,
    predict_latency
)

from qsync.Predictor.subgraph import (
    DFS_to_find_blocks,
    find_unique_blocks,
    is_isomorphic_graph_in_graphs
)

import copy 
from config import config
from qsync.LpTorch.conf import config as qconfig

# def get_baseline_critical_graph(baseline_name):
#     baseline_path = f'{baseline_name}_cpeg.pkl'
#     baseline_file_path = get_abs_cng_path(baseline_path)
#     baseline_critical_graph = load_pickle_graph(baseline_file_path)
#     return baseline_critical_graph


def test_bert_prediction():
    print("test bert result")
    model_name = 'bert'
    # load relevant profiling data
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

     # backbone
    T4_file_name = node_data_fp32_file
    T4_data_pack = load_cng_data(T4_file_name) 
    fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, OVERALL_TIME = T4_data_pack

    # initialize the graph
    set_dag_uniform_bitwidth(fwd_di_graph, 32) # init with 32 backbone
    dag_available_dict = get_available_bitwidth_change_node_in_dag(fwd_di_graph)

    # load critical path graph
    T4_file_path = f'{T4_file_name}_cpeg.pkl'
    T4_file_path = get_abs_cng_path(T4_file_path)
    T4_critical_graph = load_pickle_graph(T4_file_path)

    # test first case, convert all list
    embs = dag_available_dict['aten::embedding']
    lis = dag_available_dict['aten::linear']
    matmuls = dag_available_dict['aten::matmul']

    # import pdb; pdb.set_trace()

    DEVICE = 't4'
    
    bit = 16
    set_gpu_maps_with_device_and_bit(DEVICE, 32)
    baseline_name = 'fp16_li_comm_t4'
    base_time = load_cng_data(baseline_name)[-1] /1000

    # first case LI-HALF
    set_dag_uniform_bitwidth(fwd_di_graph, 32)
    set_layers_uniform_bitwidth(fwd_di_graph, lis, 16)
    change_relevant_node_bitwidth(fwd_di_graph)
    predicted = predict_latency(fwd_di_graph, T4_critical_graph, T4_dag_2_nodes, T4_mapper_dict, _debug_level=0)
    print("predicted vs base", predicted, base_time)
    print("%.3f" % percent_diff(base_time, predicted))
    # import pdb; pdb.set_trace()

    # second case LI_INT8
    # load baseline
    
    bit = 8
    baseline_name = 'int8_li_comm_t4'
    base_time = load_cng_data(baseline_name)[-1] /1000

    set_dag_uniform_bitwidth(fwd_di_graph, 32)
    set_layers_uniform_bitwidth(fwd_di_graph, lis, 8)
    change_relevant_node_bitwidth(fwd_di_graph)
    predicted = predict_latency(fwd_di_graph, T4_critical_graph, T4_dag_2_nodes, T4_mapper_dict, _debug_level=0)
    print("predicted vs base", predicted, base_time)
    print("%.3f" % percent_diff(base_time, predicted))

    # FIRST BERT LAYER
    bit = 16
    baseline_name = 'fp16_layer3_comm_t4'
    base_time = load_cng_data(baseline_name)[-1] /1000

    set_dag_uniform_bitwidth(fwd_di_graph, 32)
    graph_blocks = DFS_to_find_blocks(fwd_di_graph)
    unique_graphs = find_unique_blocks(graph_blocks, fwd_di_graph)
    emb_0 = unique_graphs[0][0]
    attention_0 = unique_graphs[1][0]
    intermedate_output = unique_graphs[2][0]

    attentions = []
    inter_outs = []
    for subg in unique_graphs:
        if is_isomorphic_graph_in_graphs(attention_0, subg):
            attentions.append(subg)
        elif is_isomorphic_graph_in_graphs(intermedate_output, subg):
            inter_outs.append(subg)
    att_1, att_3, att_5 = attentions[0][0], attentions[2][0], attentions[4][0]
    io_1, io_3, io_5 = inter_outs[0][0], inter_outs[2][0], inter_outs[4][0]

    set_layers_uniform_bitwidth(fwd_di_graph, att_1.nodes, 16)
    set_layers_uniform_bitwidth(fwd_di_graph, att_3.nodes, 16)
    set_layers_uniform_bitwidth(fwd_di_graph, att_5.nodes, 16)

    set_layers_uniform_bitwidth(fwd_di_graph, io_1.nodes, 16)
    set_layers_uniform_bitwidth(fwd_di_graph, io_3.nodes, 16)
    set_layers_uniform_bitwidth(fwd_di_graph, io_5.nodes, 16)

    change_relevant_node_bitwidth(fwd_di_graph)

    # check_dag_bit(fwd_di_graph)

    predicted = predict_latency(fwd_di_graph, T4_critical_graph, T4_dag_2_nodes, T4_mapper_dict, _debug_level=0)
    print("predicted vs base", predicted, base_time)
    print("%.3f" % percent_diff(base_time, predicted))


    # TODO: MultiNode Prediction here With T4.
    # V100_file_path = f'{V100_file_name}_cpeg.pkl'
    # V100_file_path = get_abs_cng_path(V100_file_path)
    # V100_critical_graph = load_pickle_graph(V100_file_path)
    # V100_data_pack = load_cng_data(V100_file_name)

def test_resnet_prediction():
    print("test resnet result")
    # load relevant profiling data
    model_name = 'resnet50'
    # load relevant profiling data
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

    T4_file_name = 'resnet50_fp32_comm_t4' # backbone

    T4_data_pack = load_cng_data(T4_file_name) # backbone
    fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, OVERALL_TIME = T4_data_pack

    # initialize the graph
    set_dag_uniform_bitwidth(fwd_di_graph, 32) # init with 32 backbone
    dag_available_dict = get_available_bitwidth_change_node_in_dag(fwd_di_graph)

    # load critical path graph
    T4_file_path = f'{T4_file_name}_cpeg.pkl'
    T4_file_path = get_abs_cng_path(T4_file_path)
    T4_critical_graph = load_pickle_graph(T4_file_path)
    
    convs = dag_available_dict['aten::conv2d']

    DEVICE = 't4'
    set_gpu_maps_with_device_and_bit(DEVICE, 32)

    bit = 16
    baseline_name = 'fp16_conv_comm_t4'
    base_time = load_cng_data(baseline_name)[-1] /1000
    # import pdb; pdb.set_trace()
    # first case LI-HALF
    set_dag_uniform_bitwidth(fwd_di_graph, 32)
    set_layers_uniform_bitwidth(fwd_di_graph, convs, 16)
    change_relevant_node_bitwidth(fwd_di_graph)
    predicted = predict_latency(fwd_di_graph, T4_critical_graph, T4_dag_2_nodes, T4_mapper_dict, _debug_level=2)
    print("predicted vs base", predicted, base_time)
    print("%.3f" % percent_diff(base_time, predicted))
    # check_dag_bit(fwd_di_graph)
    # # casting cost of the first op in conv should be calculated independently
    # # Add casting cost of first operator
    # # def predict_first_node_casting(fwd_di_graph):
    # #     from predictor import ana_predictor
    # #     import numpy as np
    # #     first_node = fwd_di_graph.nodes['aten::conv2d_1']
    # #     first_node_in_shapes = first_node['in_shapes'][0]
    # #     first_node_weight_shapes = first_node['weight_shape']
    # #     first_node_out_shapes = first_node['out_shapes']
    # #     res = ana_predictor.predict_with_bit(32, first_node['bit'], np.prod(first_node_in_shapes)+ np.prod(first_node_weight_shapes) + np.prod(first_node_out_shapes))
    # #     return res 
    
    # first_node_cost = predict_first_node_casting(fwd_di_graph)
    # predicted += first_node_cost

    # # import pdb;pdb.set_trace()
    # # second case LI_INT8
    # # load baseline

    bit = 8
    baseline_name = 'int8_conv_comm_t4'
    base_time = load_cng_data(baseline_name)[-1] /1000

    set_dag_uniform_bitwidth(fwd_di_graph, 32)
    set_layers_uniform_bitwidth(fwd_di_graph, convs, 8)
    change_relevant_node_bitwidth(fwd_di_graph)
    # first_node_cost = predict_first_node_casting(fwd_di_graph)
    predicted = predict_latency(fwd_di_graph, T4_critical_graph, T4_dag_2_nodes, T4_mapper_dict, _debug_level=0)
    # predicted += first_node_cost
    print("predicted vs base", predicted, base_time)
    print("%.3f" % percent_diff(base_time, predicted))
    # import pdb;pdb.set_trace()
    # Conv Layers

    bit = 16
    baseline_name = 'fp16_layer2_comm_t4'
    base_time = load_cng_data(baseline_name)[-1] /1000

    set_dag_uniform_bitwidth(fwd_di_graph, 32)
    graph_blocks = DFS_to_find_blocks(fwd_di_graph)
    unique_graphs = find_unique_blocks(graph_blocks, fwd_di_graph)

    layer1 = [unique_graphs[1][0], unique_graphs[2][0], unique_graphs[3][0]]
    layer2 = [unique_graphs[8][0], unique_graphs[9][0], unique_graphs[10][0], unique_graphs[11][0], unique_graphs[12][0]]

    def set_nodes_in_layers(layerx):
        for layer in layerx:
            set_layers_uniform_bitwidth(fwd_di_graph, layer.nodes, 16)

    set_nodes_in_layers(layer1)
    set_nodes_in_layers(layer2)
    change_relevant_node_bitwidth(fwd_di_graph)

    # check_dag_bit(fwd_di_graph)
    predicted = predict_latency(fwd_di_graph, T4_critical_graph, T4_dag_2_nodes, T4_mapper_dict, _debug_level=0)
    print("predicted vs base", predicted, base_time)
    print("%.3f" % percent_diff(base_time, predicted))
    

    # V100_file_name  = 'resnet50_fp32_comm_v100'
    # V100_data_pack = load_cng_data(V100_file_name)
    # V100_file_path = f'{V100_file_name}_cpeg.pkl'
    # V100_file_path = get_abs_cng_path(V100_file_path)
    # V100_critical_graph = load_pickle_graph(V100_file_path)

# whether the qantization is fused makes different result
config.quantize_fused = qconfig.fuse_int_output
config.act_quant = qconfig.act_quant
config.kernel_quant = qconfig.kernel_quant
# Test for the bitwidth change prediction
test_resnet_prediction()
test_bert_prediction()



# Test the cross device prediction 
