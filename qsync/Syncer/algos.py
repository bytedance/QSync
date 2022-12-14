from qsync.Predictor.cross_node_predict import (
    is_cross_change_B_graph, predict_cross_result
)

from qsync.Predictor.core import (
    change_dag_bit,
    calculate_mem_occuupation,
    is_bitwidth_changeable_node,
    predict
)

from qsync.Predictor.utils import (
    store_mapper_result_data, load_mapper_result_data,
    store_mapper_result_data_npy, load_mapper_result_data_npy,
    map_layer_name_to_dag_node
)

import copy 
import networkx as nx 


from qsync.LpTorch.conf import config as qconfig
# The indicator list always follows a ascending order
# e.g. 8, 16, 32 and its variance -> getting larger


def calculate_decrement_gain(value_set, idx):
    if idx - 1 >= 0:
        return value_set[idx - 1]
    return None

def calculate_increment_gain(value_set, idx):
    len_bit = len(value_set)
    if idx < len_bit - 1:
        return value_set[idx] - value_set[idx - 1]
    return None

def layer_to_bit_from_node_to_bit(node_to_bit_mapper, node_to_layer_mapper):
    new_mapper = {}
    for k, v in node_to_bit_mapper.items():
        new_mapper[node_to_layer_mapper[k]] = v
    return new_mapper

def visualize_bit_change(init_mapper, after_mapper):
    ava_bits = qconfig.available_bits
    cnt = 0
    all_layer_cnt = len(list(init_mapper.keys()))
    for k, bit_a in init_mapper.items():
        bit_b = after_mapper[k]
        if bit_a != bit_b: 
            print(f"layer {k} change bit from {ava_bits[bit_a]} to {ava_bits[bit_b]}")
            cnt += 1
    print(f"Update layer {cnt} / {all_layer_cnt}")



def MEM_AVA_CHECK(indicator_mapper, node_to_layer_mapper, M, M_limit):
    cur_M = 0
    # calculate sum of mem 
    for node, idx in indicator_mapper.items():
        layer = node_to_layer_mapper[node]
        cur_M_lyaer = M[layer][idx]
        cur_M += cur_M_lyaer
    import pdb; pdb.set_trace()
    if cur_M < M_limit:
        return True
    else:
        return False
    
def convert_bitidx_mapping_to_bit_mapping(ava_bits, layer_mapper):
    new_mapper = {}
    for k, v in layer_mapper.items():
        new_mapper[k] = ava_bits[v]
    return new_mapper
    
# Algos
def SAA():
    print("Simulate Anneal Arithmetic")

def MCMC():
    print("Markov Chain Monte Carlo")

def greedy(indicator, layer_order, fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, T4_graph, V100_graph, M, M_limit=0, file_name="mapper", model_name=None, random=False):
    layer_to_node_mapper, node_to_layer_mapper = map_layer_name_to_dag_node(layer_order, fwd_di_graph)
    _, node_to_layer_mapper_full = map_layer_name_to_dag_node(layer_order, fwd_di_graph, bitwidth_only=False)

    indicator_mapper = {}
    for k, v in layer_to_node_mapper.items():
        indicator_mapper[v] =  indicator[k]

    import heapq # use heap here
    print("Apply Greedy Algorithm")
    # for greedy, the logic is very simple. everytime we find the most variance decrement term we can gain
    # and choose it as the target to change the bit
    possible_bitwidth_choices = len(qconfig.available_bits)
    layer_count = len(list(indicator_mapper.keys()))
    nodes = fwd_di_graph.nodes
    # we build a heap to record the time

    layer_to_idx_mapper = {}  # build a idx mapper to record bit idx for each layer
    gain_heap = [] # build a heap to record the gain for each change
    for layer, value in indicator_mapper.items():
        node_layer = nodes[layer]
        # print(node_layer, layer)
        if 'bit' not in node_layer: continue # usually the classifier layers.
        bit = node_layer['bit']
        idx_bit = qconfig.available_bits.index(bit)
        layer_to_idx_mapper[layer] = idx_bit
        # gain = calculate_increment_gain(value, idx_bit)
        gain = calculate_increment_gain(value, idx_bit)

        if gain is not None:
            heapq.heappush(gain_heap, (-gain, layer)) # use -gain to change min heap -> max heap
    
    initial_bit_mapping = copy.deepcopy(layer_to_idx_mapper)
    not_surpass = True
    # import pdb; pdb.set_trace()
    cross_result = predict_cross_result(T4_graph, V100_graph, model_name)
    origin_multi_node_cost = max(cross_result) # training time for multi - inf
    print("Initial multi-node cost", origin_multi_node_cost, cross_result)

    def try_bit_change(fwd_di_graph, T4_dag_2_nodes, layer, bit, T4_graph, T4_mapper_dict, multi_node_cost):
        print("Try", layer, "to bit->", bit)
        T4_graph = copy.deepcopy(T4_graph)
        change_dag_bit(fwd_di_graph, T4_dag_2_nodes, layer, bit, T4_graph, T4_mapper_dict) 
        # check memory
        current_m = calculate_mem_occuupation(fwd_di_graph, (node_to_layer_mapper_full, M))
        if M_limit is not None and M_limit !=0 and current_m > M_limit:
            return False, None 
        # change_dag_bit()
        not_surpass, multi_node_cost = is_cross_change_B_graph(T4_graph, V100_graph, multi_node_cost, model_name)
        return not_surpass, multi_node_cost
    

    node_change_records = [] # used to create window
    
    while len(gain_heap) != 0: # when all possible is gone, loop over
        minus_gain, layer = heapq.heappop(gain_heap) 
        new_bit_idx = layer_to_idx_mapper[layer] + 1
        bit =  qconfig.available_bits[new_bit_idx]
        not_surpass, multi_node_cost = try_bit_change(fwd_di_graph, T4_dag_2_nodes, layer, bit, T4_graph, T4_mapper_dict, origin_multi_node_cost)
        if not_surpass: 
            print(f"Maintain the bit change! {layer} to {bit}, cost {multi_node_cost}")
            node_change_records.append((layer, qconfig.available_bits[new_bit_idx-1]))
            # maintain the change to the graph
            change_dag_bit(fwd_di_graph, T4_dag_2_nodes, layer, bit, T4_graph, T4_mapper_dict) 
            layer_to_idx_mapper[layer] += 1 # reduce one bit
            # add possible change to graph
            gain = calculate_increment_gain(indicator_mapper[layer], new_bit_idx)
            if gain is not None:
                heapq.heappush(gain_heap, (-gain, layer))
        else:
            # surpassed. The change is abandoned.
            continue

       

    # update compete
    print("Search END")
    # model layer_bit mapper
    initial_layer_bit_mapping = layer_to_bit_from_node_to_bit(initial_bit_mapping, node_to_layer_mapper)
    final_layer_bit_mapping = layer_to_bit_from_node_to_bit(layer_to_idx_mapper, node_to_layer_mapper)
        
    visualize_bit_change(initial_layer_bit_mapping, final_layer_bit_mapping)

    # file_name = f"greedy_{file_name}.pkl"
    # store_mapper_result_data(final_layer_bit_mapping, file_name)
    # load_mapper_result_data(file_name)

    # import pdb; pdb.set_trace()
    # the prediction still have some very small error
    # however, can culmulate to big dfference 
    # introdcuess some recovery here 
    # import pdb; pdb.set_trace()
    # recover = 7 
    # recover_result = []
    # for idx, node in enumerate(list(reversed(node_change_records))):
    #     if idx == recover - 1:
    #         break 
    #     node_name, node_bit = node 
    #     final_layer_bit_mapping[node_to_layer_mapper_full[node_name]] = node_bit
    #     recover_result.append(node)
    # print("Recoverd due to boundary errr", recover_result)
    print(node_change_records)
    
    final_layer_bit_mapping = convert_bitidx_mapping_to_bit_mapping([8, 16, 32], final_layer_bit_mapping)
    file_name = f"greedy_{file_name}.npy" if not random else f"greedy_{file_name}_random.npy"
    store_mapper_result_data_npy(final_layer_bit_mapping, file_name)
    print(final_layer_bit_mapping)
    # load_mapper_result_data_npy(file_name)


def greedy_mem_requirement_single_node(indicator, layer_order, fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, T4_graph, V100_graph, M, M_limit=0, file_name="mapper", model_name=None, ind_name='GVAR'):
    layer_to_node_mapper, node_to_layer_mapper = map_layer_name_to_dag_node(layer_order, fwd_di_graph)
    _, node_to_layer_mapper_full = map_layer_name_to_dag_node(layer_order, fwd_di_graph, bitwidth_only=False)

    indicator_mapper = {}
    # import pdb; pdb.set_trace()
    for k, v in layer_to_node_mapper.items():
        indicator_mapper[v] =  indicator[k]

    import heapq # use heap here
    print("Apply Greedy Algorithm")
    # comparing with the former one, this is much easier since we greedily select bitwidth with lowest sensitivity 
    # then reduce it to lower bit and check the memory constraint, nothing else. 

    possible_bitwidth_choices = len(qconfig.available_bits)
    layer_count = len(list(indicator_mapper.keys()))
    nodes = fwd_di_graph.nodes
    # we build a heap to record the time

    layer_to_idx_mapper = {}  # build a idx mapper to record bit idx for each layer
    gain_heap = [] # build a heap to record the gain for each change
    for layer, value in indicator_mapper.items():
        if len(value) == 0: continue
        node_layer = nodes[layer]
        # print(node_layer, layer)
        if 'bit' not in node_layer: continue # usually the classifier layers.
        bit = node_layer['bit']
        idx_bit = qconfig.available_bits.index(bit)

        layer_to_idx_mapper[layer] = idx_bit
        gain = calculate_decrement_gain(value, idx_bit)
        if gain is not None:
            heapq.heappush(gain_heap, (-gain, layer)) # use -gain to change min heap -> max heap

    # import pdb; pdb.set_trace()
    initial_bit_mapping = copy.deepcopy(layer_to_idx_mapper)


    def try_bit_change(fwd_di_graph, T4_dag_2_nodes, layer, bit, T4_graph, T4_mapper_dict):
        print("Try", layer, "to bit->", bit)
        # check memory
        current_m = calculate_mem_occuupation(fwd_di_graph, (node_to_layer_mapper_full, M))
        print(current_m)
        if current_m <= M_limit: # end condition
            return False, None 
        else:
            return True, None
    

    node_change_records = [] # used to create window
    
    # import pdb; pdb.set_trace()
    while len(gain_heap) != 0: # when all possible is gone, loop over
        minus_gain, layer = heapq.heappop(gain_heap) 
        new_bit_idx = layer_to_idx_mapper[layer] - 1
        # print("new bit idx", new_bit_idx)
        if new_bit_idx == -1:
            continue
        bit =  qconfig.available_bits[new_bit_idx]
        not_surpass, multi_node_cost = try_bit_change(fwd_di_graph, T4_dag_2_nodes, layer, bit, T4_graph, T4_mapper_dict)
        if not_surpass: 
            print(f"Maintain the bit change! {layer} to {bit}, cost {multi_node_cost}")
            node_change_records.append((layer, qconfig.available_bits[new_bit_idx-1]))
            # maintain the change to the graph
            change_dag_bit(fwd_di_graph, T4_dag_2_nodes, layer, bit, T4_graph, T4_mapper_dict) 
            layer_to_idx_mapper[layer] -= 1 # add one bit
            # add possible change to graph
            gain = calculate_increment_gain(indicator_mapper[layer], new_bit_idx)
            if gain is not None:
                heapq.heappush(gain_heap, (-gain, layer))
        else:
            # surpassed. The change is abandoned.
            continue

       

    # update compete
    print("Search END")
    # model layer_bit mapper
    initial_layer_bit_mapping = layer_to_bit_from_node_to_bit(initial_bit_mapping, node_to_layer_mapper)
    final_layer_bit_mapping = layer_to_bit_from_node_to_bit(layer_to_idx_mapper, node_to_layer_mapper)
        
    visualize_bit_change(initial_layer_bit_mapping, final_layer_bit_mapping)

    file_name = f"greedy_{file_name}_int8_{ind_name}"
    # store_mapper_result_data(final_layer_bit_mapping, file_name)
    # res = load_mapper_result_data(file_name)
    # import pdb; pdb.set_trace()


    # import pdb; pdb.set_trace()
    # the prediction still have some very small error
    # however, can culmulate to big dfference 
    # introdcuess some recovery here 
    # import pdb; pdb.set_trace()
    # recover = 7 
    # recover_result = []
    # for idx, node in enumerate(list(reversed(node_change_records))):
    #     if idx == recover - 1:
    #         break 
    #     node_name, node_bit = node 
    #     bit_idx = qconfig.available_bits.index(node_bit)
    #     final_layer_bit_mapping[node_to_layer_mapper_full[node_name]] = bit_idx
    #     recover_result.append(node)
    final_layer_bit_mapping = convert_bitidx_mapping_to_bit_mapping([8, 16, 32], final_layer_bit_mapping)
    file_name = f"greedy_{file_name}.npy"
    store_mapper_result_data_npy(final_layer_bit_mapping, file_name)
    print(node_change_records)
    # print("Recoverd due to boundary errr", recover_result)
    

def multi_contraints_pack():
    print("multi contraint pack")



