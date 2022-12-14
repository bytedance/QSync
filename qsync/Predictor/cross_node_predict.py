# when doing multinode prediction, the key is
# change the communication node by
# new_start_A = max(new_last_A, start_A), new_start_B = max(new_last_B, start_B)
# new_A_dur = maximize (new_start_A, new_start_B) - new_start_A + maximize (dur_A, dur_B) + gap
# new_B_dur = maximize (new_start_A, new_start_B) - new_start_B + maximize (dur_A, dur_B) + gap
# new_last_A = new_start_A + new_A_dur
# new_last_B = new_start_B + new_B_dur

# To make simple prediction (fp32), we need two traces of single device multi-gpu (with communication)
# To make mixed prediction, we further require DAG and predictor of casting latency.
# The prediction is made via following steps:
#   1. we convert the model to its mixed-precision form. Map kernel time, add conversion cost, map comm time
#   2. we recalculate the starting point of communication
#   3. Based on the new graph, we cross prediction

model_name = 'bert'

from .core import (
    is_comm_op, predict, reset_communication_start,
    add_cpu_gaps, rm_cpu_gaps, zero_cpu_gaps
)

from .utils import (
    load_pickle_graph, get_abs_cng_path,
    set_gpu_maps_with_device_and_bit
)

from .config import config
from .predictor import ana_predictor

import copy 


def cross_communication(di_graphA, di_graphB, model_name):
    # gap here is the extra cost required for multinode compared with single node
    # get all communication node
    nodes_A, nodes_B = di_graphA.nodes, di_graphB.nodes 
    # They should always have same number of keys
    keys = [node for node in nodes_A if is_comm_op(nodes_A, node)]
    new_last_A, new_last_B = 0, 0
    comm_pack = config.comm_pack_size[model_name]
    last_one = len(keys) - 1
    # print(len(keys))
    for idx, k in enumerate(keys):
        # import pdb; pdb.set_trace()

        comm_A = nodes_A[k]
        comm_B = nodes_B[k]

        start_A, start_B = comm_A['start'] + config.comm_bias , comm_B['start']
        dur_A, dur_B = comm_A['avg'], comm_B['avg']

        new_start_A = max(new_last_A, start_A)
        new_start_B = max(new_last_B, start_B)

        # tensor_size = comm_pack[idx]
        # add_dur = ana_predictor.predict_comm_add_cost(tensor_size)
        add_dur = 0
        # print(add_dur, tensor_size, idx)

        forced_comm_time = max(dur_A, dur_B) + add_dur # at least communicate so muchtime

        new_A_dur = max(new_start_A, new_start_B) - new_start_A + forced_comm_time 
        new_B_dur = max(new_start_A, new_start_B) - new_start_B  + forced_comm_time 

        comm_A['avg'] = new_A_dur
        comm_B['avg'] = new_B_dur

        # print(f"Update {k}, A start from {start_A} -> {new_start_A}, B start from {start_B}->{new_start_B}")
        # print(f"Update {k}, A cost from {dur_A} -> {new_A_dur}, B cost from {dur_B}->{new_B_dur}")
        


        new_last_A = new_start_A + new_A_dur
        new_last_B = new_start_B + new_B_dur

    return di_graphA, di_graphB

def predict_cross_result(A_graph, B_graph, model_name):
    # do cross prediction
    A_graph = copy.deepcopy(A_graph)
    B_graph = copy.deepcopy(B_graph) 

    # import pdb; pdb.set_trace()
    reset_communication_start(A_graph)
    reset_communication_start(B_graph)
    new_A_graph, new_B_graph = cross_communication(A_graph, B_graph, model_name)
    cpu_interference_cost=config.cpu_interference_cost

    # rm_cpu_gaps(new_A_graph)
    # A_preds_cross = predict(new_A_graph, _debug_level=0)
    A_preds_cross = predict(new_A_graph, _debug_level=0, cpu_interference_cost=10)
    # rm_cpu_gaps(new_B_graph)
    # B_preds_cross = predict(new_B_graph, _debug_level=0)
    B_preds_cross = predict(new_B_graph, _debug_level=0, cpu_interference_cost=40)
    # print(B_preds_cross)
    print("crossed", A_preds_cross, B_preds_cross)

    return A_preds_cross, B_preds_cross


def is_cross_change_B_graph(A_graph, B_graph, previous_cost=None, model_name=None):
    # B graph is the reference graph
    # B_preds = predict(B_graph, _debug_level=0)
    A_preds_cross, B_preds_cross = predict_cross_result(A_graph, B_graph, model_name)
    B_preds_cross = max(B_preds_cross, A_preds_cross)
    if previous_cost is not None and B_preds_cross > previous_cost:
        return False, A_preds_cross
    return True, A_preds_cross


def predict_cross_result_file(T4_file_path, V100_file_path, model_name=None):
    
    T4_file_path = get_abs_cng_path(T4_file_path)
    V100_file_path = get_abs_cng_path(V100_file_path)

    T4_graph = load_pickle_graph(T4_file_path)
    V100_graph = load_pickle_graph(V100_file_path)
    
    set_gpu_maps_with_device_and_bit('t4', 16)
    T4_preds = predict(T4_graph, _debug_level=0)
    set_gpu_maps_with_device_and_bit('v100', 32)
    V100_preds = predict(V100_graph, _debug_level=0)
    print("original", T4_preds, V100_preds)

    set_gpu_maps_with_device_and_bit('t4', 16)
    add_cpu_gaps(T4_graph)
    set_gpu_maps_with_device_and_bit('v100', 32)
    add_cpu_gaps(V100_graph)


    return predict_cross_result(T4_graph, V100_graph, model_name)
    

def test():
    # read backbone
    # T4_file_path = 'fp32_comm_t4.json_cpeg.pkl'
    # V100_file_path = 'fp32_comm_v100.json_cpeg.pkl'
    # predict_cross_result_file(T4_file_path, V100_file_path)

    # model_name = 'bert'
    # T4_file_path = f'{model_name}_fp16_comm_t4_cpeg.pkl'
    # V100_file_path = f'{model_name}_fp32_comm_v100_cpeg.pkl'
    # predict_cross_result_file(T4_file_path, V100_file_path, model_name=model_name)

    # model_name = 'roberta'
    # T4_file_path = f'{model_name}_fp16_comm_t4_cpeg.pkl'
    # V100_file_path = f'{model_name}_fp32_comm_v100_cpeg.pkl'
    # predict_cross_result_file(T4_file_path, V100_file_path, model_name=model_name)


    model_name = 'resnet50'
    T4_file_path = f'{model_name}_fp16_comm_t4_cpeg.pkl'
    V100_file_path = f'{model_name}_fp16_comm_v100_cpeg.pkl'
    predict_cross_result_file(T4_file_path, V100_file_path, model_name=model_name)


if __name__ == '__main__':
    
    test()





