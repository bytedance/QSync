# add the profiled data of pytorch into torch vision
import networkx as nx 
import json
import torch 
import tensorboard
import pickle
from collections import defaultdict
import torch.nn.functional as F

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from utils import (
    get_depth_of_nodes,
    save_depth_graph,
    store_dag_graph
)

# from QSync import QModule
def load_networkx_graph(gml_path:str):
    mygraph = nx.read_gml(gml_path)
    return mygraph

def load_bwd_data(my_graph:nx.DiGraph, bwd_trace_file:str):
    print("load trace data")
    events = None 
    with open(bwd_trace_file, "r") as f:
        trace_file = json.load(f)
        events = trace_file['traceEvents']
    event_len = len(events)
    event_idx = 0
    # print(my_graph)
    ascending_list = nx.topological_sort(my_graph)
    nodes = my_graph.nodes
    for name in reversed(list(ascending_list)):
        node = nodes[name]
        if '()' in node['name']:
            # root
            continue
        while event_idx <= event_len -1:
            cur_event = events[event_idx]
            cur_event_name = cur_event["name"]
            event_idx += 1
            if 'dur' in cur_event:
                cur_event_time = cur_event["dur"]
                if node["name"] in cur_event_name:
                    # print(f"node name{node['name']}, duration: {cur_event_time}")
                    node["avg"] = cur_event_time
                    break
            else:
                node["avg"] = 0
        # print(node)

def tensorboard_graph_to_fwd_graph(model_graph, fwd_trace:str):
    nx_di_graph = nx.DiGraph()
    seen = set()
    for node in model_graph.node:
        node_name = node.name
        node_op = node.op
        if node_name not in seen:
            seen.add(node_name)
            nx_di_graph.add_node(node_name, name=node_name, op=node_op)
        else:
            nx_di_graph.nodes[node_name]['op'] = node_op
        if hasattr(node, 'input'):
            node_inputs = node.input 
            for inp in node_inputs:
                if inp not in seen: # havent created input node yet
                    nx_di_graph.add_node(inp, name=inp)
                nx_di_graph.add_edge(inp, node_name)
    ascending_list = nx.topological_sort(nx_di_graph)
    nodes = nx_di_graph.nodes
    nx.write_gml(nx_di_graph, "fwd.gml")
    for name in ascending_list:
        cur_node = nodes[name]
        # print(cur_node) if 'op' in cur_node and 'aten' in cur_node['op'] else None
# def save_model_onnx(model, model_name_or_path="bert"):
#     # from torch_ort import DebugOptions
#     # import torch_ort 
#     # torch_ort.ORTModule(bert, DebugOptions(save_onnx=True, onnx_prefix=model_name))
#     from transformers import (
#         AutoTokenizer
#     )
#     from transformers.onnx import FeaturesManager
#     import transformers
#     from pathlib import Path
#     feature = "question-answering"
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     # load config
#     model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
#     onnx_config = model_onnx_config(model.config)
#     # print(onnx_config)
#     output_path = Path(f"{model_name_or_path}.onnx")
#     # export
#     onnx_inputs, onnx_outputs = transformers.onnx.export(
#             preprocessor=tokenizer,
#             model=model,
#             config=onnx_config,
#             opset=13,
#             output=output_path,
#     )
#     return output_path

# def onnx_to_networkx(onx_path):
#     import onnx
#     onx_model = onnx.load(onx_path)
#     for node in onx_model.graph.node:
#         print(node.input, node.name)

#     print("Conert onnx model to networkx representation")

# create node for u and
def create_node_and_edges(id_u, id_v, seen_node, di_graph):

    if id_u not in seen_node:
        # di_graph.add_node(id_u, attr=u)
        di_graph.add_node(id_u)
        seen_node.add(id_u)
    if id_v not in seen_node:
        # di_graph.add_node(id_v, attr=v)
        di_graph.add_node(id_v)
        seen_node.add(id_v)
    # make edge from u to v
    di_graph.add_edge(id_u, id_v)

def remove_node_preserve_edges(node_k, di_graph):
    preds = di_graph.predecessors(node_k)
    succs = di_graph.successors(node_k)
    di_graph.remove_node(node_k)
    for pred in preds:
        for suc in succs:
            if pred != suc: # loop
                di_graph.add_edge(pred, suc)



import re 
def get_shape(node_desc):
    node_str = str(node_desc)
    def strip_to_shape(shape_str):
        res_list = shape_str.strip().split('(')
        shape_str_part2 = res_list[1]
        sizes_num = shape_str_part2.split(',')
        sizes_num = [num.strip() for num in sizes_num]
        sizes_num = [int(num) for num in sizes_num if num.isnumeric()]
        return sizes_num

    def got_shapes_from_res(res):
        shapes = [strip_to_shape(i) for i in res]
        return shapes
        
    res = re.findall(r"[Long|Float|Half]\([?\s\d+\,]+", node_str)
    if res is not None and len(res) != 0:
        shapes = got_shapes_from_res(res)
    else:
        shapes = None
    return shapes

# for conv, we got kernel_shape in the node_desc
def get_conv_weight(node_desc):
    node_str = str(node_desc)
    res = re.search(r"kernel_shape=\[[?\s\d+\,]+]", node_str)[0]
    shape_kernel = res.split('[')[1].split(']')[0].split(',')
    shape_kernel = [int(i.strip()) for i in shape_kernel]
    return shape_kernel


bit_width_changeable_nodes = [
    'aten::embedding', 'aten::linear', 'aten::layer_norm', 'aten::conv2d', 'aten::batch_norm'
]


def pruinning_fwd_graph(fwd_di_graph):
    
    nodes = fwd_di_graph.nodes
    try:
        res = list(nx.find_cycle(fwd_di_graph, orientation="ignore"))
        # remove cycles
        for edge in res:
            fwd_di_graph.remove_edge(edge)
    except:
        print("no cycle found")
    # import pdb; pdb.set_trace()

    ascending_list = list(nx.topological_sort(fwd_di_graph))
    neglect_ops = ['aten::Int', 'aten::rsub', 'aten::expand', 'aten::mul', 'aten::slice', 'aten::size', 'aten::to', 'aten::select', 'aten::unsqueeze']
    free_ops = ['batch_norm']
    # neglect_times = 7 # some trace is different from the training grap
    for name in ascending_list:
        node = nodes[name]
        if 'op' not in node or 'aten' not in node['op'] or node['op'] in neglect_ops:
            if 'op' in node and node['op'] in free_ops:
                continue

            in_edges = fwd_di_graph.in_edges(name)
            out_edges = fwd_di_graph.out_edges(name)
            if len(in_edges) == 0 or len(out_edges) == 0:
                fwd_di_graph.remove_node(name)
                continue
            # print(in_edges, out_edges)
            remove_node_preserve_edges(name, fwd_di_graph)
    
    ascending_list = list(nx.topological_sort(fwd_di_graph))
    nodes = fwd_di_graph.nodes

    while nodes[ascending_list[0]]['op'] != first_node:
        for name in ascending_list:
            node = nodes[name]
            node_op = node['op']

            if node_op != first_node:
                fwd_di_graph.remove_node(name)
                continue
            else:
                break
        ascending_list = list(nx.topological_sort(fwd_di_graph))
        nodes = fwd_di_graph.nodes

def print_fwd_graph(fwd_di_graph):
    ascending_list = nx.topological_sort(fwd_di_graph)
    nodes = fwd_di_graph.nodes

    for name in ascending_list:
        node = nodes[name]
        node_op = node['op']

        in_edges = fwd_di_graph.in_edges(name, data=True)
        out_edges = fwd_di_graph.out_edges(name, data=True)
        # print(name, len(in_edges), len(out_edges))
        # print(node['op'])
        # if 'relu' in node['op']:
        #     print(node, len(in_edges), len(out_edges))
        #     for edge in out_edges:
        #         print(fwd_di_graph.nodes[edge[1]])
        # if "clamp" in node['op']:
        #     print(node, len(in_edges), len(out_edges))
        #     for edge in out_edges:
        #         print(fwd_di_graph.nodes[edge[1]])


def gen_fwd_graph(traced_model: torch.jit._trace.TopLevelTracedModule, model_name):
    
    name_dict = defaultdict(int)
    fwd_di_graph = nx.DiGraph()
    torch_graph = traced_model.inlined_graph 
    seen = set()
    # first_op='aten::embedding'
    # first_op_got = False
    # the following part follows the hidden layer
    for torch_node in torch_graph.nodes():
        # Op
        op = torch_node.kind()
        
        unique_inp_ids = [i.unique() for i in torch_node.inputs()]
        unique_output_ids = [i.unique() for i in torch_node.outputs()]

        inp_shapes = []
        out_shapes = []
        weight_shape = []
        if 'aten' in op:
            # import pdb; pdb.set_trace()
            inp_shapes = [get_shape(i) for i in torch_node.inputs()]
            out_shapes = [get_shape(i) for i in torch_node.outputs()]
            if op in bit_width_changeable_nodes:
                if 'layer_norm' in op: 
                    weight_shape = [out_shapes[0][0][-1]]
                elif 'linear' in op:
                    weight_shape = [inp_shapes[0][0][-1], out_shapes[0][0][-1]]
                elif 'embedding' in op:
                    available_inputs = [i for i in inp_shapes if i is not None]
                    if 'word_embedding' in str(torch_node):
                        weight_shape = [vocab_size, hidden_size]
                    elif 'position' in str(torch_node):
                        weight_shape = [2, hidden_size]
                    elif 'token_type' in str(torch_node):
                        weight_shape = [max_pos_embedding, hidden_size]

        op_record = op
        node_id = f"{op_record}_{name_dict[op_record]}"
        name_dict[op_record] += 1
        seen.add(node_id)
        
        fwd_di_graph.add_node(node_id, op=op, weight_shape=weight_shape, in_shapes=inp_shapes, out_shapes=out_shapes)
        # params = {k: torch_node[k] for k in torch_node.attributeNames()} 
        # add node for input values and output values, make edges
        [create_node_and_edges(i, node_id, seen, fwd_di_graph) for i in unique_inp_ids]
        [create_node_and_edges(node_id, i, seen, fwd_di_graph) for i in unique_output_ids]

        in_edges = fwd_di_graph.in_edges(node_id)
        out_edges = fwd_di_graph.out_edges(node_id)
        
    pruinning_fwd_graph(fwd_di_graph)
    print_fwd_graph(fwd_di_graph)
    store_dag_graph(fwd_di_graph, model_name)
    return fwd_di_graph

# code here follows hidden_layer.

def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))

def get_fwd_graph_conv(torch_graph, model_name):
    fwd_di_graph = nx.DiGraph()
    name_dict = defaultdict(int)
    op_mapping = {
        'onnx::Conv': 'aten::conv2d',
        'onnx::BatchNormalization': 'batch_norm', # my implementation, may be wrong and require modification
        'onnx::Add': 'aten::add',
        'onnx::Relu': 'aten::relu',
        'onnx::MaxPool':'aten::max_pool2d',
        'onnx::GlobalAveragePool': 'aten::adaptive_avg_pool2d',
        'onnx::AveragePool': 'aten::adaptive_avg_pool2d',
        'onnx::Flatten': 'aten::flatten',
        'onnx::Gemm': 'aten::linear'
    }
    def compute_out_shape_for_conv(inp_shape, params):
        # the easiest way, compute it
        out = F.conv2d(torch.randn(inp_shape), torch.randn(params['kernel_shape']), None, stride=tuple(params['strides']), \
         padding=tuple([params['pads'][0], params['pads'][2]]), dilation=tuple(params['dilations']))
        return out.shape
    
    # debug
    # dump_pytorch_graph(torch_graph)
    # import pdb; pdb.set_trace()
    torch_node_to_id = {}
    input_maps = {}

    def pytorch_id(torch_node, op):
        if torch_node in torch_node_to_id:
            return torch_node_to_id[torch_node], False, None
        op_record = op_mapping[op] # map onnx to aten op
        node_id = f"{op_record}_{name_dict[op_record]}"
        torch_node_to_id[torch_node] = node_id
        name_dict[op_record] += 1
        return node_id, True, op_record

    for torch_node in torch_graph.nodes():
        # Op
        op = torch_node.kind()
        # print(op)

        # Parameters
        params = {k: torch_node[k] for k in torch_node.attributeNames()} 
        # Inputs/outputs
        # TODO: inputs = [i.unique() for i in node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]
        # Get output shape
        out_shapes = get_shape(torch_node)
        # Get input & weight shape
        if len(input_maps) == 0:
            all_inputs = [i for i in torch_node.inputs()][0]
            input_list = str(all_inputs).split(', %')
            input_sub_list = [i.split(':') for i in input_list]
            for element in input_sub_list:
                key = element[0].strip()
                if key.isnumeric(): key = int(key)
                else: key = 0
                val = element[1]
                value_shape = get_shape(val)
                input_maps[key] = value_shape
        
        unique_inp_ids =  [i.unique() for i in torch_node.inputs()]
        unique_output_ids = [i.unique() for i in torch_node.outputs()]
        try:
            unique_inp_shapes = [input_maps[i] for i in unique_inp_ids]
        except:
            import pdb; pdb.set_trace()
        for idx, val in enumerate(unique_output_ids):
            input_maps[val] = out_shapes[idx]

        inp_shape = unique_inp_shapes[0]
        if len(unique_inp_shapes) > 1: 
            weight_shape = unique_inp_shapes[1]
        else:
            weight_shape = []   
        
        node_id, add_node_flag, op_record = pytorch_id(torch_node, op)
        if add_node_flag:
            # print("add node", node_id)
            fwd_di_graph.add_node(node_id, op=op_record, in_shapes=inp_shape, weight_shape=weight_shape, out_shapes=out_shapes)
        else:
            fwd_di_graph.nodes[node_id].update(in_shapes=inp_shape, weight_shape=weight_shape, out_shapes=out_shapes)
        
        # iterate later
        for target_torch_node in torch_graph.nodes():
            target_inputs = [i.unique() for i in target_torch_node.inputs()]
            target_torch_node_op = target_torch_node.kind()
            if set(outputs) & set(target_inputs):

                target_node_id, add_node_flag, op_record = pytorch_id(target_torch_node, target_torch_node_op)
                if add_node_flag:
                    # print("add target node", target_node_id)
                    fwd_di_graph.add_node(target_node_id, op=op_record)
                fwd_di_graph.add_edge(node_id, target_node_id)


    pruinning_fwd_graph(fwd_di_graph)
    print_fwd_graph(fwd_di_graph)
    store_dag_graph(fwd_di_graph, model_name)
    return fwd_di_graph
    

def make_graph(gr):
    import graphviz
    dot = graphviz.Digraph(format='svg', graph_attr={'labelloc': 't'})

    nodes = {}
    for i in gr.inputs():
        nname = i.debugName()
        label = nname.split('.')[0]
        nodes[nname] = (nname, dot)
        dot.node(nname, label, color='blue')

    unseen_ops = {'prim::ListConstruct', 'aten::index', 
                  'aten::size', 'aten::slice', 'aten::unsqueeze', 'aten::squeeze',
                  'aten::to', 'aten::view', 'aten::permute', 'aten::transpose', 'aten::contiguous',
                  'aten::permute', 'aten::Int', 'prim::TupleUnpack', 'prim::ListUnpack', 'aten::unbind',
                  'aten::select', 'aten::detach', 'aten::stack', 'aten::reshape', 'aten::split_with_sizes',
                  'aten::cat', 'aten::expand', 'aten::expand_as', 'aten::_shape_as_tensor',
                  'aten::_size_if_not_equal', 'prim::BroadcastSizes',
                  'prim::Constant',
                  }

    def process_block(nodeit, dot):
        firstnode = None
        lastnode = None
        for n in nodeit:
            k = n.kind()
            outs = list(n.outputs())
            inps = list(n.inputs())
            type_outs = [o.type().kind() for o in outs]
            type_inps = [o.type().kind() for o in inps]
            if k == 'prim::If':
                label = 'If'
                nname = outs[0].debugName()
                for i in inps:
                    src, srcdot = nodes.get(i.debugName(), (None, None))
                    if src is not None:
                        srcdot.edge(src, nname + '_in')
                dot.node(nname + '_in', 'If', shape='diamond')
                dot.node(nname, '', width='0.1', height='0.1')
                dot.edge(nname + '_in', nname, style='invis')
                nodes[nname] = (nname, dot)
                bl = list(n.blocks())
                for i, b in enumerate(bl):
                    with dot.subgraph(name=f"cluster_{nname}_{i}", graph_attr={'label':''}) as sub_dot:
                        firstnode, lastnode = process_block(b.nodes(), sub_dot)
                    dot.edge(nname + '_in', firstnode, label="yn"[i])
                    dot.edge(lastnode, nname)
                if firstnode is None:
                    firstnode = nname + '_in'
                lastnode = nname
            elif k == 'prim::DifferentiableGraph':
                label = 'DifferentiableGraph'
                nname = outs[0].debugName()
                nodes[nname] = (nname, dot)
                sg = n.g('Subgraph')
                nis = list(n.inputs())
                sgis = list(sg.inputs())
                assert len(nis) == len(sgis)
                for ni, sgi in zip(nis, sgis):
                    if ni.debugName() in nodes:
                        nodes[sgi.debugName()] = nodes[ni.debugName()]
                with dot.subgraph(name=f"cluster_{nname}", graph_attr={
                    'label': 'DifferentiableGraph', 'labelloc':'b', 'labeljust':'r'}) as sub_dot:
                    firstnode, lastnode = process_block(sg.nodes(), sub_dot)
                nos = list(n.outputs())
                sgos = list(sg.outputs())
                assert len(nos) <= len(sgos)
                for no, sgo in zip(nos, sgos):
                    if sgo.debugName() in nodes:
                        nodes[no.debugName()] = (nodes[sgo.debugName()][0], dot)
            elif k not in unseen_ops:
                if k == 'prim::CallFunction':
                    label = 'call ' + next(n.inputs()).node().s("name")
                else:
                    label = k.replace('aten::', '').replace('prim::', '')
                nname = outs[0].debugName()
                dot.node(nname, label, shape='box', style='rounded')
                for o in outs:
                    nodes[o.debugName()] = (nname, dot)
                for i in inps:
                    src, srcdot = nodes.get(i.debugName(), (None, None))
                    if src is not None:
                        srcdot.edge(src, nname)
                if firstnode is None:
                    firstnode = nname
                lastnode = nname
        return firstnode, lastnode

    process_block(gr.nodes(), dot)
    dot.node('.outputs', 'outputs', color='blue')
    for i, o in enumerate(gr.outputs()):
        src, srcdot = nodes.get(o.debugName(), (None, None))
        if src is not None:
            dot.edge(src, '.outputs')

    return dot

    
def load_sample_data(file_name):
    data = None
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def mv_to_cpu(data):
    if type(data) is dict:
        new_data = {}
        for k,v in data.items():
            new_data[k] = v.cpu()
    if type(data) is list:
        new_data = []
        for i in data:
            new_data.append(i.cpu())
    if type(data) is torch.Tensor:
        new_data = data.cpu()
    return new_data

vocab_size, hidden_size, max_pos_embedding = 0, 0, 0
def generate_model_graph_bert():
    global vocab_size, hidden_size, max_pos_embedding
    model_name_or_path = "bert-base-uncased"
    model_name = 'bert'
    data = load_sample_data(f'./input_data/{model_name}_input.pkl')
    cpu_data = mv_to_cpu(dict(data))
    tokens_tensor, segments_tensors = cpu_data['input_ids'], cpu_data['token_type_ids']
    dummy_input = [tokens_tensor, segments_tensors]
    # import pdb; pdb.set_trace()

    config = AutoConfig.from_pretrained(model_name_or_path, torchscript=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )
    model.eval()
    # model = QModule(model)

    vocab_size = config.vocab_size 
    hidden_size = config.hidden_size
    max_pos_embedding = config.max_position_embeddings

    traced_model = torch.jit.trace(model, dummy_input)
    fwd_di_graph = gen_fwd_graph(traced_model, model_name)
    
    save_depth_graph(fwd_di_graph, model_name)

def generate_model_graph_roberta():
    global vocab_size, hidden_size, max_pos_embedding
    model_name = 'roberta'
    model_name_or_path = 'roberta-base'
    

    data = load_sample_data(f'./input_data/{model_name}_input.pkl')
    cpu_data = mv_to_cpu(dict(data))
    input_ids, attention_mask, labels = cpu_data['input_ids'], cpu_data['attention_mask'], cpu_data['labels']
    # token_type_ids = torch.zeros(input_ids.shape)
    dummy_input = [input_ids]

    # download model & vocab.
    config = AutoConfig.from_pretrained(model_name_or_path, torchscript=True)
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config
    )
    model.resize_token_embeddings(50265)
    model.eval()
    # model = QModule(model)
    vocab_size = config.vocab_size 
    hidden_size = config.hidden_size
    max_pos_embedding = config.max_position_embeddings
    traced_model = torch.jit.trace(model, dummy_input)
    # print(traced_model)
    fwd_di_graph = gen_fwd_graph(traced_model, model_name)

    save_depth_graph(fwd_di_graph, model_name)

from torchvision import models
def generate_model_graph_resnet50():
    model_name = 'resnet50'
    cpu_data = torch.randn(64, 3, 224, 224)

    model = models.resnet50()
    model.eval()
    trace, out = torch.jit._get_trace_graph(model, cpu_data)
    torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    fwd_di_graph = get_fwd_graph_conv(torch_graph, model_name)

    save_depth_graph(fwd_di_graph, model_name)

    return fwd_di_graph

def generate_model_graph_vgg16():
    model_name = 'vgg16'
    cpu_data = torch.randn(64, 3, 224, 224)

    model = models.vgg16()
    model.eval()
    trace, out = torch.jit._get_trace_graph(model, cpu_data)
    torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    fwd_di_graph = get_fwd_graph_conv(torch_graph, model_name)
    save_depth_graph(fwd_di_graph, model_name)
    return fwd_di_graph

def generate_model_graph_vgg16bn():
    model_name = 'vgg16bn'
    cpu_data = torch.randn(64, 3, 224, 224)

    model = models.vgg16_bn()
    model.eval()
    trace, out = torch.jit._get_trace_graph(model, cpu_data)
    torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    fwd_di_graph = get_fwd_graph_conv(torch_graph, model_name)
    save_depth_graph(fwd_di_graph, model_name)
    return fwd_di_graph


import argparse
parser = argparse.ArgumentParser(description=' mv profiled result scripts.')
parser.add_argument('--model-name', type=str, default=None,
                    help='mode_name') 

if __name__ == '__main__':
    args = parser.parse_args()
    # my_graph = load_networkx_graph("./bwd.gml")
    # load_bwd_data(my_graph, "/home/ubuntu/torchvizx/test/bwd_trace.json")
    first_node = 'aten::embedding'
    if args.model_name is None or args.model_name == 'bert':
        generate_model_graph_bert()
    if args.model_name is None or args.model_name == 'roberta':
        generate_model_graph_roberta()
    first_node = 'aten::conv2d'
    if args.model_name is None or args.model_name == 'vgg16':
        generate_model_graph_vgg16()
    if args.model_name is None or args.model_name == 'vgg16bn':
        generate_model_graph_vgg16bn()
    if args.model_name is None or args.model_name == 'resnet50':
        generate_model_graph_resnet50()



