import os
import networkx as nx 
import pickle
import numpy as np 
from qsync.Predictor.config import config

def create_folder(folder_path):
    folder = os.path.exists(folder_path)
    if not folder:
        os.makedirs(folder_path)
        print(folder_path+"---create OK---")

def check_file_exits(file_path):
    file_exist = os.path.exists(file_path)
    assert file_exist, f"file not exists {file_path}"

def get_abs_node_data_path(file_name):
    return os.path.abspath(os.path.join(config.node_data_folder, file_name))

def get_abs_mapper_result_path(file_name):
    return os.path.abspath(os.path.join(config.mapper_result_folder, file_name))


# store the graph & load graph using nodes
def store_node_data(graph, file_name="node_data"):
    data_path = get_abs_node_data_path(file_name)
    create_folder(os.path.abspath(os.path.join(config.node_data_folder)))
    node_data = list(graph.nodes.data())
    with open(data_path, 'wb') as f:
        pickle.dump(node_data, f)
        print("Saved node data to ->", data_path)

def load_node_data_to_graph(graph, file_name="node_data"):
    data_path = get_abs_node_data_path(file_name)
    check_file_exits(data_path)
    node_data = None
    with open(data_path, 'rb') as f:
        node_data = pickle.load(f)
    attr_dict = {}
    for (node_k, node_v) in node_data:
        attr_dict[node_k] = node_v
    nx.set_node_attributes(graph, attr_dict)

def load_node_data(file_name="node_data"):
    data_path = get_abs_node_data_path(file_name)
    check_file_exits(data_path)
    node_data = None
    with open(data_path, 'rb') as f:
        node_data = pickle.load(f)
    attr_dict = {}
    for (node_k, node_v) in node_data:
        attr_dict[node_k] = node_v
    return attr_dict


# store the graph & load graph using nodes
def store_mapper_result_data_npy(mapper_result, file_name="node_data"):
    data_path = get_abs_mapper_result_path(file_name)
    create_folder(os.path.abspath(os.path.join(config.mapper_result_folder)))
    np.save(data_path, mapper_result)
    print("Saved nmapper_result data to ->", data_path)
    
def load_mapper_result_data_npy(file_name="node_data"):
    data_path = get_abs_mapper_result_path(file_name)
    check_file_exits(data_path)
    return np.load(data_path, allow_pickle=True)


# store the graph & load graph using nodes
def store_mapper_result_data(mapper_result, file_name="node_data"):
    data_path = get_abs_mapper_result_path(file_name)
    create_folder(os.path.abspath(os.path.join(config.mapper_result_folder)))
    with open(data_path, 'wb') as f:
        pickle.dump(mapper_result, f)
        print("Saved nmapper_result data to ->", data_path)
    
def load_mapper_result_data(file_name="node_data"):
    data_path = get_abs_mapper_result_path(file_name)
    check_file_exits(data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_pickle_graph(file_path="graph.pkl"):
    check_file_exits(file_path)
    graph = nx.read_gpickle(file_path)
    return graph

def store_critical_node_graph(di_graph, model_trace):
    assert di_graph and model_trace, "require param"
    file_path = os.path.abspath(os.path.join(config.cng_folder, f'{model_trace}_cpeg.pkl'))
    create_folder(os.path.abspath(config.cng_folder))
    nx.write_gpickle(di_graph, file_path)
    print(f"{model_trace} information saved to {file_path}")


def load_dag_graph(model_name):
    dag_graph_name  = f"{model_name}_fwd_graph.pkl"
    file_name = os.path.abspath(os.path.join(config.dag_folder, dag_graph_name))
    fwd_di_graph = nx.read_gpickle(file_name)
    return fwd_di_graph

def store_dag_graph(dag_graph, model_name):
    create_folder(os.path.abspath(config.dag_folder))
    dag_graph_name  = f"{model_name}_fwd_graph.pkl"
    file_name = os.path.abspath(os.path.join(config.dag_folder, dag_graph_name))
    nx.write_gpickle(dag_graph, file_name)
    print("Exported", file_name)

    

def get_abs_cng_path(file_name, model_name=None):
    if model_name is None:
        return os.path.abspath(os.path.join(config.cng_folder, file_name))
    else:
        return os.path.abspath(os.path.join(config.cng_folder, model_name, file_name))

def get_abs_cng_data_path(file_name, model_name=None):
    if model_name is None:
        return os.path.abspath(os.path.join(config.cng_data_folder,f"{file_name}.pkl"))
    else:
        return os.path.abspath(os.path.join(config.cng_data_folder, model_name, f"{file_name}.pkl"))


# rw with pickle
def write_data_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(file_path):
    check_file_exits(file_path)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_cng_data(data, file_name):
    file_path = get_abs_cng_data_path(file_name)
    create_folder(os.path.abspath(config.cng_data_folder))
    write_data_pickle(data, file_path)

def load_cng_data(file_name):
    file_path = get_abs_cng_data_path(file_name)
    return load_data_pickle(file_path)


def recover_dict(item):
    if type(item) is not dict:
        return item.item()

def read_prof_indicator(file_path):
    # "/qsync_niti_based/profile_result/profile_data_bert_75.npz"
    data = np.load(file_path, allow_pickle=True)
    M, E, RD, omega_matrix, min_M, min_E, smax_M, max_E, ava_idx, layer_order, model_context_size  = \
                data['M'], data['E'], data['RD'], data['omega'], data['minM'], data['minE'], data['maxM'], data['maxE'], data['ava_idx'],\
                    data['layer_order'], data['model_context_size']
    omega_matrix = recover_dict(omega_matrix)
    layer_order = recover_dict(layer_order)
    M = recover_dict(M)

    model_context_size = model_context_size.item()
    return omega_matrix, layer_order, M, model_context_size

def load_data_mappers(ava_bits, mapper_file_names):
    bit8_mapping = bit16_mapping = bit19_mapping = bit32_mapping = None 
    for bit, mapper_file_name in mapper_file_names.items():
        if bit == 8:
            bit8_mapping = load_node_data(mapper_file_name)
        elif bit == 16:
            bit16_mapping = load_node_data(mapper_file_name)
        elif bit == 19:
            bit19_mapping = load_node_data(mapper_file_name)
        elif bit == 32:
            bit32_mapping = load_node_data(mapper_file_name)
    
    config.set_bitwidth_mapping(bit8_mapping=bit8_mapping, bit16_mapping=bit16_mapping, bit19_mapping=bit19_mapping, bit32_mapping=bit32_mapping)

def load_cast_mappers(ava_bits, mapper_file_names):
    bit8_cast_mapper = bit16_cast_mapper = bit19_cast_mapper = bit32_cast_mapper = None 
    for bit, mapper_file_name in mapper_file_names.items():
        if bit == 8:
            _,_, bit8_cast_mapper_dict,_ = load_cng_data(mapper_file_name)
            bit8_cast_mapper = bit8_cast_mapper_dict['fwd_to_cast']
        elif bit == 16:
            _,_, bit16_cast_mapper_dict,_ = load_cng_data(mapper_file_name)
            bit16_cast_mapper = bit16_cast_mapper_dict['fwd_to_cast']
        elif bit == 19:
            _,_, bit19_cast_mapper_dict,_ = load_cng_data(mapper_file_name)
            bit19_cast_mapper = bit19_cast_mapper_dict['fwd_to_cast']
        elif bit == 32:
            _,_, bit32_cast_mapper_dict,_ = load_cng_data(mapper_file_name)
            bit32_cast_mapper = bit32_cast_mapper_dict['fwd_to_cast']

    
    config.set_bitwidth_cast_mapper(bit8_cast_mapper=bit8_cast_mapper, bit16_cast_mapper=bit16_cast_mapper, bit19_cast_mapper=bit19_cast_mapper,\
     bit32_cast_mapper=bit32_cast_mapper)


def is_mem_relevant_ops(op_name):
    for mem_op in config.mem_relevant_op:
        if mem_op in op_name:
            return True
    return False

def is_implement_limit_ops(op_name):
    for limit_op in config.implement_limit_ops:
        if limit_op in op_name:
            return True
    return False

def is_appended_criterion_ops(op_name):
    if '+' in op_name:
        return True
    else:
        return False
    
def all_equal(iterator):
    return len(set(iterator)) <= 1
        
# create another mapper here to map layer to correct indicator
# profiler result always follows the real execution order. so we just match them
def map_layer_name_to_dag_node(layer_order, fwd_di_graph, bitwidth_only=True):
    def do_mapping(list_A, list_B, mapper):
        if len(list_A) == 0 or len(list_B) == 0: 
            # print("no op for mapping")
            return 
        assert len(list_A) == len(list_B) or abs(len(list_A) - len(list_B)) == 1, f"not correct {len(list_A)} / {len(list_B)}"
        # map A to B 
        for idx, A in enumerate(list_A):
            mapper[A] = list_B[idx]
    def do_reverse_mapping(mapper):
        reverse_mapper = {}
        for k, v in mapper.items():
            reverse_mapper[v] = k
        return reverse_mapper
    node_execute_order = list(nx.topological_sort(fwd_di_graph))
    embs = []; lns = []; lis = []; convs = []; bns = [] ;matmuls = []
    if not bitwidth_only:
        dropouts = []; softmaxs = []; gelus = []; bns = []; relus = []; adds = []
    for node_name in node_execute_order:
        if '+' in node_name:
            continue
        if 'embedding' in node_name:
            embs.append(node_name)
        elif 'linear' in node_name:
            lis.append(node_name)
        elif 'matmul' in node_name:
            matmuls.append(node_name)
        elif 'conv2d' in node_name:
            convs.append(node_name)

        if not bitwidth_only:
            if 'dropout' in node_name:
                dropouts.append(node_name)
            elif 'softmax' in node_name:
                softmaxs.append(node_name)
            elif 'layer_norm' in node_name:
                lns.append(node_name)
            elif 'gelu' in node_name:
                gelus.append(node_name)
            elif 'batch_norm' in node_name or 'batchnorm' in node_name:
                bns.append(node_name)
            elif 'relu' in node_name:
                relus.append(node_name)
            elif 'add' in node_name:
                adds.append(node_name)
        
    
    
    embs_l = []; lns_l = []; lis_l = []; convs_l = []; bns_l = []; matmuls_l = []
    if not bitwidth_only:
        dropouts_l = []; softmaxs_l= []; gelus_l = [];  bns_l = []; relus_l = []; adds_l = []
    for layer_name, v in layer_order.items():
        if 'embedding' in layer_name:
            embs_l.append(layer_name)
        elif 'linear' in layer_name:
            lis_l.append(layer_name)
        elif 'matmul' in layer_name:
            matmuls_l.append(layer_name)
        elif 'conv' in layer_name:
            convs_l.append(layer_name)

        if not bitwidth_only:
            if 'dropout' in layer_name:
                dropouts_l.append(layer_name)
            elif 'softmax' in layer_name:
                softmaxs_l.append(layer_name)
            elif 'gelu' in layer_name:
                gelus_l.append(layer_name)
            elif 'layernorm' in layer_name:
                lns_l.append(layer_name)
            
            elif 'batch_norm' in layer_name or 'batchnorm' in layer_name or 'bn' in layer_name:
                bns_l.append(layer_name)
            elif 'relu' in layer_name:
                relus_l.append(layer_name)
            # elif 'add' in layer_name:
            #     adds_l.append(layer_name)

    # import pdb; pdb.set_trace()
    # do mapping
    layer_to_node_mapper = {}
    do_mapping(embs_l, embs, layer_to_node_mapper)
    do_mapping(lns_l, lns, layer_to_node_mapper)
    do_mapping(lis_l, lis, layer_to_node_mapper)
    do_mapping(matmuls_l, matmuls, layer_to_node_mapper)
    do_mapping(convs_l, convs, layer_to_node_mapper)
    do_mapping(bns_l, bns, layer_to_node_mapper)
    # do_mapping(relus_l, relus, layer_to_node_mapper)

    if not bitwidth_only:
        do_mapping(dropouts_l, dropouts, layer_to_node_mapper)
        do_mapping(softmaxs_l, softmaxs, layer_to_node_mapper)
        do_mapping(gelus_l, gelus, layer_to_node_mapper)
   

    return layer_to_node_mapper, do_reverse_mapping(layer_to_node_mapper) # from node to layer


def set_gpu_maps_with_device_and_bit(DEVICE, bit):
    DEVICE = DEVICE.lower()
    if config.enable_test_gap:
        config.cpu_time_gaps = config.cpu_test_gap
    config.set_cur_cpu_gaps(DEVICE)
    if DEVICE == 't4':
        # add gaps for profile data prediction
        if bit == 16:
            config.set_fwd_time(0.07)
        elif bit == 8:
            config.set_fwd_time(0.05)
        elif bit == 32 or bit == 19:
            config.set_fwd_time(0.02) # fp32 gap
        else:
            config.set_fwd_time(0.001)
    elif DEVICE == 'v100':
        if bit == 32:
            config.set_fwd_time(0.06)


def percent_diff(target, pred):
    return abs(target - pred) / target


def bfs_layers(G, sources):
    if sources in G:
        sources = [sources]

    current_layer = list(sources)
    visited = set(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    while current_layer:
        yield current_layer
        next_layer = list()
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.append(child)
        current_layer = next_layer



def get_depth_of_nodes(fwd_di_graph):
    ascending_list = list(nx.topological_sort(fwd_di_graph))
    first_node = ascending_list[0]
    depth_result = dict(enumerate(bfs_layers(fwd_di_graph, first_node)))
    name_first_result = {}
    for depth, v in depth_result.items():
        for v_name in v:
            for ava_name in config.bit_width_changeable_nodes:
                if ava_name in v_name:
                    name_first_result[v_name] = depth
    return name_first_result

def save_depth_graph(fwd_di_graph, model_name):
    path = os.environ['HOME']
    path = os.path.join(path, "configs", "model_depth", f"{model_name}_depth_data")
    abs_path = os.path.abspath(path)
    depth_result = get_depth_of_nodes(fwd_di_graph)
    write_data_pickle(depth_result, abs_path)

