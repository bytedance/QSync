import argparse
import os 
import networkx as nx 

from qsync.Predictor.core import (
    change_gap_time, 
    get_mapper_and_critical_path_graph, 
    get_available_bitwidth_change_node_in_dag,
    change_dag_bit,
    init_graph_bit,
    get_available_bitwidth_change_node_in_dag,
    check_graph_bit,
    add_critical_edges,
    analyze_graph,
    predict
)

from qsync.Predictor.config import config 
from qsync.Predictor.utils import (
    store_critical_node_graph, store_node_data, load_node_data, load_node_data_to_graph,
    write_cng_data, 
    load_cng_data,
    set_gpu_maps_with_device_and_bit
)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--device_name",
        type=str,
        default='t4',
        help="device name",
    )

    parser.add_argument(
        "--bit",
        type=int,
        default=32,
        help="bit",
    )

    parser.add_argument(
        "--target_trace_idx",
        type=int,
        default=0,
        help="bit",
    )


    parser.add_argument(
        "--model_name",
        type=str,
        default='bert',
        help="model name",
    )

    parser.add_argument(
        "--generate_mapping",
        action="store_true",
        help="generate mapping for bit, ignore non-consitent latency.",
    )

    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="filename",
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # control flags
    store = True
    load_data = not store
    load_data = False
    test_bit_change = False
    prediction = True
    store_critical_graph = True
    
    DEVICE = args.device_name
    bit = args.bit
    suffix = f'fp{bit}' if bit >= 16 else f'int{bit}'

    # DEVICE='t4'
    # bit = 32

    # DEVICE='t4'
    # bit = 8

    # model_name = 'roberta'
    config.model_name = model_name = args.model_name
    # model_name = 'resnet50'
    # model_name = 'vgg16'
    if model_name == 'roberta' or model_name == 'bert':
        config.FIRST_ATEN_OP = config.FIRST_DAG_OP = "aten::embedding"
        config.OPTIMIZER_NAME = 'AdamW'
        config.CONSIDER_OPTIMIZER_TIME = True
    elif model_name == 'resnet50' or model_name == 'vgg16':
        config.FIRST_DAG_OP = "aten::conv2d" # not this one
        tag_suffix = f'FP{bit}' if bit >= 16 else f'Int{bit}'
        config.FIRST_ATEN_OP = f"Conv2d{tag_suffix}"
        config.OPTIMIZER_NAME = 'SGD'

        # TODO: cannot match up with the optimizer. 
        config.CONSIDER_OPTIMIZER_TIME = False
    ADD_COMM = True
    target_trace_idx = args.target_trace_idx
    
    if ADD_COMM:
        suffix += f'_comm_{DEVICE}'
    # BERT
    # 32 502 predict, 504 real
    # 16 258 predict, 261 real
    # ROBERTA
    # 32 1254 pred real 1248
    # 16 378 pred, Real 371

    # resnet50
    # 32 prd188-- real 769
    # 16 prd 188 real 192
    # 8 prd 510 real 517

    # vgg16
    # 32 prd 3809 real 3811
    # 16 prd 1088 real 1089
    # 8 prd 1493 real 1494
    # suffix = "fp16_comm_t4"
    model_trace = f"{model_name}_{suffix}"
    trace_name = os.path.abspath(os.path.join(config.pred_root, f"traces/exp/{model_name}/{suffix}.json"))

    if args.file_name is not None:
        trace_name = os.path.abspath(os.path.join(config.pred_root, f"traces/exp/{model_name}/{args.file_name}.json"))
        model_trace = f"{model_name}_{args.file_name}"

    CPU_COST = 0.05 # gap between precision conversions

    # Partial Quantization result
    # BERT
    # file_name_trace = 'li16_only' # 280 pre, 278 Actual %mixed pred 278
    # file_name_trace = 'emb16_only' # 513 pred, 508 actual %mixed pred 514
    # file_name_trace = 'liemb16' # 274.96 pred, 274 Actual %mixed pred 278
    # file_name_trace = 'liln16' # 260 pred, 259 Actual %mixed 250

    # ROBERTA
    # file_name_trace = 'li16_only' # 409 pre, 401 Actual %mixed pred 406
    # file_name_trace = 'liln16' # 376 Pre, 367 Actual %mixed pred 364

    # trace_name = f"/qsync_niti_based/analytical_predictor/traces/bps/{model_name}_{file_name_trace}.json"

    # Analysis Multi-Node Multi-GPU Training
    # file_name = 'T4_V100_comm_fp32_T4'
    # DEVICE = 't4'
    # file_name = 'T4_V100_comm_fp32_V100'
    # DEVICE = 'v100'
    # trace_name = f"/qsync_niti_based/analytical_predictor/traces/bps/{file_name}.json"

    # if DEVICE == 'v100':
    #     change_gap_time(p_gap_cpu=0.02, p_gap_gpu=0.001)
    # elif DEVICE == 't4':
    #     change_gap_time(p_gap_cpu=0.02, p_gap_gpu=0.001)
    
    config.set_cur_cpu_gaps(DEVICE)
    if args.generate_mapping:
        prediction = False
        # remove all gaps
        config_gap = {
            'fwd': 0.001, 
            'bwd': 0.001,
            'optimizer_step': 0.001,
            'optimizer_zero_grad': 0.001
        }
        config.set_gaps_with_config(config_gap)
    else:
        set_gpu_maps_with_device_and_bit(DEVICE, bit)

    critical_path_node_graph, fwd_di_graph, dag_2_nodes, (REMOVAL_TIME, pf_event), mapper_dict = get_mapper_and_critical_path_graph(trace_name, target_trace_idx=target_trace_idx)
    # fwd_graph_analysis(fwd_di_graph)    
    dag_available_dict = get_available_bitwidth_change_node_in_dag(fwd_di_graph)

    
    assert not store or store != load_data, "Can only store or load!"

    if store:
        store_node_data(critical_path_node_graph, model_trace)
        # exit()

    if load_data:
        file_fp16 = f"{model_trace}_bit16"
        file_fp32 = f"{model_trace}_bit32"
        bit32_mapping = load_node_data(file_fp32)
        bit16_mapping = load_node_data(file_fp16)
    

    assert not test_bit_change or test_bit_change != store, "Cannot store while changing bits for prediction"
    
    if test_bit_change:
        init_graph_bit(fwd_di_graph, bit)
        # dag_test_node = dag_available_dict['aten::embedding'][1]
        # key_dags = list(dag_2_nodes.keys())

        # # test case 1:
        for layer_norm in dag_available_dict['aten::layer_norm']:
            change_dag_bit(fwd_di_graph, dag_2_nodes, layer_norm, 16, critical_path_node_graph, mapper_dict)
        # # # test case 2
        # cnt = 0
        # for idx, emb in enumerate(dag_available_dict['aten::embedding']):
        #     change_dag_bit(fwd_di_graph, dag_2_nodes, emb, 16, critical_path_node_graph, mapper_dict) # 509 pred

        # test case 3
        for linear in dag_available_dict['aten::linear']:
            change_dag_bit(fwd_di_graph, dag_2_nodes, linear, 16, critical_path_node_graph, mapper_dict) # 278 pred
        
        for node_k in fwd_di_graph:
            change_bwd_cost(node_k, fwd_di_graph, dag_2_nodes, critical_path_node_graph, mapper_dict)
        
        for node_k in fwd_di_graph:
            change_fwd_cost(node_k, fwd_di_graph, dag_2_nodes, critical_path_node_graph, mapper_dict)

        check_graph_bit(fwd_di_graph, dag_2_nodes, critical_path_node_graph)

    OVERALL_TIME = pf_event['dur']
    RECORD_TIME = OVERALL_TIME - REMOVAL_TIME
    # store critical path graph
    if store_critical_graph:
        store_critical_node_graph(critical_path_node_graph, model_trace)
        cng_data = [fwd_di_graph, dag_2_nodes, mapper_dict, RECORD_TIME]
        write_cng_data(cng_data, model_trace)


    
    # prediction
    if prediction:
        print(OVERALL_TIME, REMOVAL_TIME)
        # import pdb; pdb.set_trace()
        predict(critical_path_node_graph, OVERALL_TIME, REMOVAL_TIME)
        