import numpy as np
import networkx as nx 
import copy 

from .utils import (
    load_pickle_graph, get_abs_cng_path,
    load_cng_data,
    get_abs_cng_data_path,
    load_node_data,
    read_prof_indicator,
    load_data_mappers, load_cast_mappers,
    map_layer_name_to_dag_node,
    is_mem_relevant_ops, is_implement_limit_ops,
    set_gpu_maps_with_device_and_bit,
    map_layer_name_to_dag_node,
    store_mapper_result_data_npy
)

from .core import (
    is_bitwidth_changeable_node,
    dag_2_nodes_to_fwd_to_bwd,
    change_dag_bit, change_relevant_node_bitwidth,
    init_graph_bit,
    change_graph_bit_single_node,
    get_aten_cuda_node,
    remove_all_node_with_type,
    get_correct_name,
    get_dag_bit_widths, set_dag_bit_widths,
    calculate_mem_occuupation,
    get_all_changeable_nodes_in_graph,
    predict_latency, predict_latency_cuda_only
    
)


from .subgraph import (
    DFS_to_find_blocks,
    find_unique_blocks,
    is_isomorphic_within_graphs,
    get_value_from_iso_g
)

from .predictor import (
    ana_predictor
)
from .config import config 
from qsync.LpTorch.conf import config as qconfig




def fetch_cost_from_graph(k, value_dict, node_data, mapper_dict, fwd_di_graph, bit):
    # cuda mapper
    fwd_mapper = mapper_dict['fwd']
    bwd_mapper = mapper_dict['bwd']
    # cpu
    fwd_name = value_dict['fwd']
    bwd_name = value_dict['bwd'][0]

    node_in_fwd_graph = fwd_di_graph.nodes[k]
    if 'weight_shape' in node_in_fwd_graph:
        weight_numel = np.prod(node_in_fwd_graph['weight_shape'])
        if bit == 8:
            casting_cost = ana_predictor.predict_float_to_int(weight_numel)
        elif bit == 16:
            casting_cost = ana_predictor.predict(weight_numel)
        else:
            casting_cost = 0
    else:
        casting_cost = 0

    fwd_time = node_data[fwd_name]['avg']
    bwd_time = node_data[bwd_name]['avg']

    fwd_cuda_name = fwd_mapper[fwd_name][0]
    bwd_cuda_name = bwd_mapper[bwd_name][0]

    fwd_cuda_time = node_data[fwd_cuda_name]['avg']
    bwd_cuda_time = node_data[bwd_cuda_name]['avg']

    

    return fwd_cuda_time + casting_cost, max(bwd_time, bwd_cuda_time)


def get_fastest_bit_setup():
    node_data_int8 =  load_node_data(node_data_int8_file)
    node_data_fp16 =  load_node_data(node_data_fp16_file)
    node_data_fp32 =  load_node_data(node_data_fp32_file)

    node_8_changable = [i for i in node_data_int8 if is_bitwidth_changeable_node(i)]
    node_16_changable = [i for i in node_data_fp16 if is_bitwidth_changeable_node(i)]
    node_32_changable = [i for i in node_data_fp32 if is_bitwidth_changeable_node(i)]

    fwd_di_graph, int8_dag_2_nodes, int8_mapper_dict = load_cng_data(node_data_int8_file)
    fwd_di_graph, fp16_dag_2_nodes, fp16_mapper_dict = load_cng_data(node_data_fp16_file)
    fwd_di_graph, fp32_dag_2_nodes, fp32_mapper_dict = load_cng_data(node_data_fp32_file)

    keys = fp32_dag_2_nodes.keys()
    value_list = []
    selections = []
    mapper_key_select = {}

    for k in keys:
        value_8 = int8_dag_2_nodes[k]
        value_16 = fp16_dag_2_nodes[k]
        value_32 = fp32_dag_2_nodes[k]

        if not is_bitwidth_changeable_node(k): continue

        fwd_8, bwd_8 = fetch_cost_from_graph(k, value_8, node_data_int8, int8_mapper_dict, fwd_di_graph, 8)
        fwd_16, bwd_16 = fetch_cost_from_graph(k, value_16, node_data_fp16, fp16_mapper_dict, fwd_di_graph, 16)
        fwd_32, bwd_32 = fetch_cost_from_graph(k, value_32, node_data_fp32, fp32_mapper_dict, fwd_di_graph, 32)

        cost_8 = fwd_8 + bwd_8
        cost_16 = fwd_16 + bwd_16
        cost_32 = fwd_32 + bwd_32

        if k == 'aten::linear_2':
            import pdb; pdb.set_trace()

        cost_list = [cost_8, cost_16, cost_32]
        value_list.append(cost_list)
        selection = np.argmin(cost_list)
        selections.append(selection)
        mapper_key_select[k] = selection
    
    # 对所有选择了 int8 的节点，判断下一个节点是否是 fp16, 判断增加了 cost 之后

    return mapper_key_select


# fastest_selection = get_fastest_bit_setup()

# print(fastest_selection)


    # for idx in range(len(node_32_changable)):
    #     node_8_name = node_8_changable[idx]
    #     node_16_name = node_16_changable[idx]
    #     node_32_name = node_32_changable[idx]

    #     node_8_avg = node_data_int8[node_8_name]['avg']
    #     node_16_avg = node_data_fp16[node_16_name]['avg']
    #     node_32_avg = node_data_fp32[node_32_name]['avg']

    #     bwd_node_8 = int8_dag_2_nodes[node_8_name]

    #     data = [node_8_avg, node_16_avg, node_32_avg]
    #     max_data = max(data)
    #     arg_idx = data.index(max_data)


# DAG is constructed from the basic blocks
# Partition DAG to basic blocks
# Detect repeating DAG graph and marks
    # same structure and op name: import networkx.algorithms.isomorphism as iso
# Bit Plan -> DAG change bitwidth -> predict speed (tologocial sort -> sum up)






                


def generate_all_possible_cases(L, ava_bits):
    poss_com_l = len(ava_bits)
    values = [[i] for i in range(poss_com_l)]
    def gen_recursive(values, L, poss_com_l):
        new_values = []
        while len(values) != 0:
            value = values.pop()
            for i in range(poss_com_l):
                new_value = value + [i]
                new_values.append(new_value)
        [values.append(new_value) for new_value in new_values]
        if L - 1 != 0:
            gen_recursive(values, L-1, poss_com_l)
    if L != 1:
        gen_recursive(values, L-1, poss_com_l)
    
    return values

# can only got cpu's ancestor and GPU

def change_bit_width_with_config(di_graph, nodes, bit_plan, ava_bit_set):
    for idx, bit_idx in enumerate(bit_plan):
        node = nodes[idx]
        bit = ava_bit_set[bit_idx]
        change_dag_bit()


def independent_latency(sub_graph, dag_2_nodes, mapper_dict):
    fwd_to_bwd_mapper = mapper_dict['fwd_to_bwd']
    fwd_cuda_mapper = mapper_dict['fwd']
    bwd_cuda_mapper = mapper_dict['bwd']

    sub_graph_nodes = sub_graph.nodes
    cost = 0
    for node_k in sub_graph.nodes:
        value = dag_2_nodes[node_k]
        fwd_node_k = value['fwd']
        # bwd_node_k = value['bwd'][0]
        cuda_fwd_node = fwd_cuda_mapper[fwd_node_k][0]
        bit = sub_graph_nodes[node_k]['bit']
        bit_mapping, name_mapping = config.get_mapper_with_bit(bit)
        cost += bit_mapping[get_correct_name(cuda_fwd_node, name_mapping)]['avg']
    
    return cost


def latency_for_subgraph(sub_graph, dag_2_nodes, mapper_dict):
    fwd_to_bwd_mapper = mapper_dict['fwd_to_bwd']
    fwd_cuda_mapper = mapper_dict['fwd']
    bwd_cuda_mapper = mapper_dict['bwd']

    
    
    output_bit_mapping = config.output_bit_mapping
    output_bit_mapping_bwd = config.output_bit_mapping_bwd

    cpu_cost = 0
    cuda_cost = 0
    sub_graph_nodes = sub_graph.nodes
    for node_k in sub_graph:
        value = dag_2_nodes[node_k]
        fwd_node_k = value['fwd']
        bwd_node_k = value['bwd'][0]

        if node_k not in fwd_to_bwd_mapper: continue
        bit = sub_graph_nodes[node_k]['bit']
        bit_mapping, name_mapping = config.get_mapper_with_bit(bit)

        node_in_fwd_graph = sub_graph.nodes[node_k]
        if 'weight_shape' in node_in_fwd_graph:
            weight_numel = np.prod(node_in_fwd_graph['weight_shape'])
            if bit == 8:
                casting_cost = ana_predictor.predict_float_to_int(weight_numel)
            elif bit == 16:
                casting_cost = ana_predictor.predict(weight_numel)
            else:
                casting_cost = 0
        else:
            casting_cost = 0

        # print(bwd_node_k, bit, get_correct_name(bwd_node_k, name_mapping))
        # print(fwd_node_k, bit,get_correct_name(fwd_node_k, name_mapping) )
        cpu_cost += (bit_mapping[ get_correct_name(fwd_node_k, name_mapping) ]['avg'] + bit_mapping[get_correct_name(bwd_node_k, name_mapping)]['avg'])

        cuda_fwd_node = fwd_cuda_mapper[fwd_node_k][0]
        cuda_bwd_node = bwd_cuda_mapper[bwd_node_k][0]


        cuda_cost += (bit_mapping[get_correct_name(cuda_fwd_node, name_mapping)]['avg'] + bit_mapping [get_correct_name(cuda_bwd_node, name_mapping)]['avg'])

        # further add all casting cost   
        preds = sub_graph.pred[node_k]
        succs = sub_graph.succ[node_k]
        cast_fwd, cast_bwd = 0, 0
        if len(preds) > 0:
            for idx, pred in enumerate(preds):
                pred_bit = sub_graph_nodes[pred]['bit']
                in_bit = output_bit_mapping[pred_bit]
                if in_bit != bit:
                    in_shapes = sub_graph_nodes[pred]['in_shapes']
                    in_shapes = [i for i in in_shapes if i is not None] 
                    in_shapes = np.squeeze(in_shapes)
                    if len(in_shapes) > 0:
                        if len(in_shapes.shape) > 1:
                            in_shape = in_shapes[idx]
                        else:
                            in_shape = in_shapes
                        if in_shape is not None:
                            cast_fwd = ana_predictor.predict_with_bit(in_bit, bit, np.prod(in_shape))
        if len(succs) > 0:
            for succ in succs:
                succ_bit = sub_graph_nodes[succ]['bit']
                out_bit = output_bit_mapping_bwd[succ_bit]
                if out_bit != bit:
                    out_shapes = sub_graph_nodes[succ]['out_shapes'][0][0]
                    cast_bwd = ana_predictor.predict_with_bit(out_bit, bit, np.prod(out_shapes))
        
        cuda_cost += (cast_fwd + cast_bwd)
    
    return cpu_cost, cuda_cost




def exhaustive_for_single_subgraph(sub_graph, combined_critical_subgraph, ava_bits, dag_2_nodes, mapper_dict, mem_constraints=0, Mappers=None):
    # exhaustive cases
    nodes = get_all_changeable_nodes_in_graph(sub_graph)
    L = len(nodes)
    # print("Generate plan for nodes: ", L)
    all_cases = generate_all_possible_cases(L, ava_bits)


    smallest_cuda_cost = 9e9
    optimal_plan = None
    dag_bit_widths_plan = None
    # print(sub_graph.nodes)
    # print(combined_critical_subgraph.nodes)
    for case in all_cases:
        init_graph_bit(sub_graph, 32)
        # change a case
        for idx, bit in enumerate(case):
            layer = nodes[idx]
            change_graph_bit_single_node(sub_graph, layer, ava_bits[bit])
            change_relevant_node_bitwidth(sub_graph)
        get_dag_bit_widths(sub_graph)

        cuda_cost = predict_latency_cuda_only(sub_graph, combined_critical_subgraph, dag_2_nodes, mapper_dict, _debug_level=0)
        # cuda_cost = independent_latency(sub_graph, dag_2_nodes, mapper_dict)

        # cuda_cost = predict_latency(fwd_di_graph, T4_critical_graph, dag_2_nodes, mapper_dict, _debug_level=0)
        # print(cuda_cost, case)
        # cpu_cost, cuda_cost = latency_for_subgraph(sub_graph, dag_2_nodes, mapper_dict)
        if mem_constraints != 0: # add memory constraints here
            case_mem = calculate_mem_occuupation(sub_graph, Mappers)
            # print(case_mem, mem_constraints)
            if case_mem > mem_constraints: continue
        # print(case, cpu_cost, cuda_cost)
        if cuda_cost < smallest_cuda_cost:
            optimal_plan = case 
            dag_bit_widths_plan = get_dag_bit_widths(sub_graph)
            smallest_cuda_cost = cuda_cost
    # set the optimal plan
    # get the reversed result
    # import pdb; pdb.set_trace()
    # print(dag_bit_widths_plan)
    set_dag_bit_widths(sub_graph, dag_bit_widths_plan)
    last_node = next(reversed(list(nx.topological_sort(sub_graph))))
    l_node_value = sub_graph.nodes[last_node]
    l_node_bit = l_node_value['bit']
    # print(l_node_bit)
    return optimal_plan, smallest_cuda_cost
    
def generate_optimal_map(optimal_plan, sub_graph, ava_bits):
    nodes = get_all_changeable_nodes_in_graph(sub_graph)
    optimal_map = {}
    for idx, bit in enumerate(optimal_plan):
        layer = nodes[idx]
        optimal_map[layer] = ava_bits[bit]
    
    return optimal_map



def get_critical_subgraph(critical_start_node, critical_end_node, ascending_list, critical_path_graph):
    if critical_start_node is not None:
        idx = ascending_list.index(critical_start_node)
        cur_ascending_list = ascending_list[idx:]
    else:
        cur_ascending_list = ascending_list
    
        
    subg_nodes = []

    def append_nodes(node):
        if node not in subg_nodes:
            subg_nodes.append(node)
        cuda_node = f'{node}_kernel'
        if cuda_node not in subg_nodes:
            subg_nodes.append(cuda_node)
    for node in cur_ascending_list:
        if critical_end_node != node:
            append_nodes(node)
        else:
            append_nodes(node)
            break # iter until end
    
    sub_critical_graph = nx.subgraph(critical_path_graph, subg_nodes)
    return sub_critical_graph

def get_fwd_and_bwd_subgraph(start_node, end_node, ascending_list, critical_path_graph, dag_2_nodes): 
    critical_start_node_fwd = dag_2_nodes[start_node]['fwd']
    critical_start_node_bwd = dag_2_nodes[start_node]['bwd'][0]

    # print(dag_2_nodes)
    critical_end_node_fwd = dag_2_nodes[end_node]['fwd']
    critical_end_node_bwd = dag_2_nodes[end_node]['bwd'][0]

    
    fwd_critical_graph = get_critical_subgraph(critical_start_node_fwd, critical_end_node_fwd, ascending_list, critical_path_graph)
    # import pdb; pdb.set_trace()
    bwd_critical_graph = get_critical_subgraph(critical_start_node_bwd, critical_end_node_bwd, list(reversed(ascending_list)), critical_path_graph)

    combined_critical_subgraph = nx.compose(fwd_critical_graph, bwd_critical_graph)
    
    return fwd_critical_graph, bwd_critical_graph, combined_critical_subgraph



def exhaustive_all_subgraphs(fwd_di_graph, unique_graphs, critical_path_graph, ava_bits, dag_2_nodes, mapper_dict, subgraph_mem_constraints=None, Mappers=None, isomorphism=False):
    mapper_all = {}
    ascending_list_all_graph = copy.deepcopy(list(nx.topological_sort(critical_path_graph)))
    start_node = list(nx.topological_sort(fwd_di_graph))[0]

    unique_graph_to_its_plan = {}

    for idx_graph, (sub_graph, partition_node) in enumerate(unique_graphs):
        if subgraph_mem_constraints is not None:
            mem_constraints = subgraph_mem_constraints[sub_graph]
        else:
            mem_constraints = 0
        end_node = partition_node
        if isomorphism: # check same structure
            optimal_plan = get_value_from_iso_g(sub_graph, unique_graph_to_its_plan)
        else:
            optimal_plan = None 
        if idx_graph == len(unique_graphs) - 1:
            break 
        if optimal_plan is None:
            # corspd_fwd_node = dag_2_nodes[partition_node]['fwd']
            # print(start_node, end_node)
            '''
                subgraph generation / some case the topo order is wiered 
            '''
            # fwd_cgraph, bwd_cgraph, combined_critical_subgraph = get_fwd_and_bwd_subgraph(start_node, end_node, ascending_list_all_graph, critical_path_graph, dag_2_nodes)
            # sub_cgraph_pack = (fwd_cgraph, bwd_cgraph, combined_critical_subgraph)

            # fwd_cpu_node = corspd_fwd_node
            # fwd_cuda_node = get_aten_cuda_node(partition_node, critical_graph)

            # des_graph_nodes = nx.descendants(des_graph, fwd_cpu_node)
            # cur_fwd_graph_nodes = nx.ancestors(des_graph, fwd_cuda_node)
            # des_graph = critical_graph.subgraph(des_graph_nodes)
            # cur_fwd_graph = critical_graph.subgraph(cur_fwd_graph_nodes)
            # only need fwd graph, since we can use mapper to get the corresponding bwd time
            optimal_plan, smallest_cuda_cost = exhaustive_for_single_subgraph(sub_graph, critical_path_graph, ava_bits, dag_2_nodes, mapper_dict, mem_constraints, Mappers)
        unique_graph_to_its_plan[sub_graph] = optimal_plan
        optimal_map = generate_optimal_map(optimal_plan, sub_graph, ava_bits)
        mapper_all.update(optimal_map)
        print("find plan for", sub_graph, optimal_map)
        start_node = end_node # for next subgraph
    return mapper_all

def set_optimal_bit(dag_graph, graph_bitwidth_mapper):
    for k, v in graph_bitwidth_mapper.items():
        dag_graph.nodes[k]['bit'] = v
        # print(k, v)

def parition_memory_budgets(unique_graphs, node_to_layer_mapper, M, memory_limit, ava_bits):
    unique_graph_memory_budget = {}
    for (unique_graph, end_node) in unique_graphs:
        graph_nodes = unique_graph.nodes
        sum_layer_M = 0
        for node in graph_nodes:
            if is_mem_relevant_ops(node):
                # fetch memory
                # print(node_to_layer_mapper)
                if '+' in node: continue # later add nodes
                layer = node_to_layer_mapper[node]
                layer_M = M[layer]
                # mem gap
                layer_M_gap = ava_bits[0] / ava_bits[-1] * layer_M
                # calculate the memory
                if is_implement_limit_ops(node): # at most half
                    layer_M_gap = layer_M / 2
                sum_layer_M += layer_M_gap
        unique_graph_memory_budget[unique_graph] = sum_layer_M
    
    values = np.array(list(unique_graph_memory_budget.values()))
    overall_mem = sum(values)
    ratio = values / overall_mem
    assigned_budget = ratio * memory_limit


    subgraph_mem_constraints = {}
    ks = list(unique_graph_memory_budget.keys())
    for idx, sub_graph in enumerate(ks):
        subgraph_mem_constraints[sub_graph] = assigned_budget[idx]
    return subgraph_mem_constraints
        

def get_latency_optimized_plan(fwd_di_graph, critical_path_graph, ava_bits, dag_2_nodes, mapper_dict,  M=None, M_limit=None):
    graph_blocks = DFS_to_find_blocks(fwd_di_graph)
    unique_graphs = find_unique_blocks(graph_blocks, fwd_di_graph)
    subgraph_mem_constraints = None 
    node_to_layer_mapper = None 
    if M_limit is not None:
        layer_to_node_mapper, node_to_layer_mapper = map_layer_name_to_dag_node(M, fwd_di_graph, bitwidth_only=False)
        subgraph_mem_constraints = parition_memory_budgets(unique_graphs, node_to_layer_mapper, M, M_limit, ava_bits)
    # import pdb; pdb.set_trace()
    mappers = (node_to_layer_mapper, M)
    subgraph_bitwidth_mapper = exhaustive_all_subgraphs(fwd_di_graph, unique_graphs, critical_path_graph, ava_bits, dag_2_nodes, mapper_dict, subgraph_mem_constraints, mappers, isomorphism=config.fastest_plan_iso)
    return subgraph_bitwidth_mapper



import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=' mv profiled result scripts.')
    parser.add_argument('--model-name', type=str, default='bert',
                        help='mode_name') 
    parser.add_argument('--indicator_type', type=str, default='GVAR',
                        help='indicator type') 
    parser.add_argument('--mem_limit', type=float, default=0)
                        
    args = parser.parse_args()
    model_name = args.model_name
    # set mappers
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

    fwd_di_graph, T4_dag_2_nodes, T4_mapper_dict, OVERALL_TIME = load_cng_data(node_data_fp32_file)
    T4_file_path = f'{node_data_fp32_file}_cpeg.pkl'
    T4_file_path = get_abs_cng_path(T4_file_path)
    T4_graph = load_pickle_graph(T4_file_path)



    set_gpu_maps_with_device_and_bit('T4', 32)

    # get mem limitation
    profile_data_path = f"/qsync_niti_based/profile_result/profile_data_{args.model_name}_75_{args.indicator_type}.npz"
    indicator, layer_order, M, model_context_size = read_prof_indicator(profile_data_path)
    # Set a Mem Limitation
    MAX_M = sum(list(M.values())) + model_context_size
    MEM_LIMIT = 6 * config.MB # summation used in KB; do some conversion
    if MAX_M < MEM_LIMIT: MEM_LIMIT = None 
    else:
        MEM_LIMIT = MEM_LIMIT
    # MEM_LIMIT = None
    MEM_LIMIT = args.mem_limit
    
    # get available bit
    ava_bits = qconfig.available_bits
    fwd_di_graph, fp32_dag_2_nodes, fp32_mapper_dict, OVERALL_TIME = load_cng_data(node_data_fp32_file)
    subgraph_bitwidth_mapper = get_latency_optimized_plan(fwd_di_graph, T4_graph, ava_bits, fp32_dag_2_nodes, fp32_mapper_dict, M, MEM_LIMIT)

    _, node_to_layer_mapper_full = map_layer_name_to_dag_node(layer_order, fwd_di_graph, bitwidth_only=False)

    print(subgraph_bitwidth_mapper)

    final_layer_bit_mapping = {}
    for k, bit in subgraph_bitwidth_mapper.items():
        bit_idx = ava_bits.index(bit)
        final_layer_bit_mapping[node_to_layer_mapper_full[k]] = bit_idx
    # import pdb; pdb.set_trace()

    file_name = f"fastest_{model_name}.npy"
    store_mapper_result_data_npy(final_layer_bit_mapping, file_name)

    set_optimal_bit(fwd_di_graph, subgraph_bitwidth_mapper)


