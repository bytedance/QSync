# read trace, get all kernel traces
# find its corresponding
import json
from collections import defaultdict 
import networkx as nx
import pickle
import numpy as np

from .config import config
from .predictor import ana_predictor

from .utils import (
    is_mem_relevant_ops, is_implement_limit_ops,
    is_appended_criterion_ops,
    all_equal,
    load_dag_graph
)


LASTEST_TORCH = config.LASTEST_TORCH


def is_cpu_op(nodes, node_k):
    if 'type' in nodes[node_k] and nodes[node_k]['type'] == 'cpu_op':
        return True
    return False

def is_cuda_op(nodes, node_k):
    if 'type' in nodes[node_k] and nodes[node_k]['type'] == 'cuda_op':
        return True
    return False

def is_comm_op(nodes, node_k):
    if 'type' in nodes[node_k] and nodes[node_k]['type'] == 'comm_op':
        return True
    return False

def is_root_op(nodes, node_k):
    if 'type' in nodes[node_k] and nodes[node_k]['type'] == 'root':
        return True
    return False

def pred_first_comm(node_k, nodes, di_graph):
    preds = di_graph.pred[node_k]
    for pred in preds:
        if is_comm_op(nodes, pred):
            return pred
    return None 

def succ_first_comm(node_k, nodes, di_graph):
    succs = di_graph.succ[node_k]
    for succ in succs:
        if is_comm_op(nodes, succ):
            return succ
    return None 

def pred_first_cuda(node_k, nodes, di_graph):
    preds = di_graph.pred[node_k]
    for pred in preds:
        if is_cuda_op(nodes, pred):
            return pred
    return None 

def succ_first_cuda(node_k, nodes, di_graph):
    succs = di_graph.succ[node_k]
    for succ in succs:
        if is_cuda_op(nodes, succ):
            return succ
    return None 

def succ_first_cpu(node_k, nodes, di_graph):
    succs = di_graph.succ[node_k]
    for succ in succs:
        if is_cpu_op(nodes, succ):
            return succ
    return None 

def succ_first_comm(node_k, nodes, di_graph):
    succs = di_graph.succ[node_k]
    for succ in succs:
        if is_comm_op(nodes, succ):
            return succ
    return None 

def pred_first_cpu(node_k, nodes, di_graph):
    preds = di_graph.pred[node_k]
    for pred in preds:
        if is_cpu_op(nodes, pred):
            return pred
    return None 

def former_cpu_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_cpu_op(nodes, node_k), "not a cpu_node"
    return pred_first_cpu(node_k, nodes, di_graph)

def former_cuda_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_cuda_op(nodes, node_k), "not a cuda_node"
    return pred_first_cuda(node_k, nodes, di_graph)

def later_cuda_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_cuda_op(nodes, node_k), "not a cuda_node"
    return succ_first_cuda(node_k, nodes, di_graph)

def later_cpu_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_cpu_op(nodes, node_k), "not a cpu_node"
    return succ_first_cpu(node_k, nodes, di_graph)

def get_aten_cuda_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_cpu_op(nodes, node_k), "not a cpu node"
    return succ_first_cuda(node_k, nodes, di_graph)

def get_cuda_cpu_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_cuda_op(nodes, node_k), "not a cuda node"
    return pred_first_cpu(node_k, nodes, di_graph)

def get_comm_cuda_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_comm_op(nodes, node_k), "not a comm node"
    return pred_first_cuda(node_k, nodes, di_graph)

def get_cuda_comm_node(node_k, di_graph):
    nodes = di_graph.nodes
    assert is_cuda_op(nodes, node_k), "not a cuda node"
    return succ_first_comm(node_k, nodes, di_graph)


def remove_all_node_with_type(di_graph, node_type=None):
    assert node_type, "require a type"
    nodes = di_graph.nodes
    node_k_list = list(nodes.keys())
    for node_k in node_k_list:
        node = nodes[node_k]
        if 'type' in node and node['type'] == node_type:
            di_graph.remove_node(node_k)


    
# debug
def sum_duration_from_events(events):
    values = [i['dur'] for i in events]
    return sum(values)

# def clear_node_name(self, di_graph):
    # # define some node clearing logic here.
    # # since we want to use mapping to do kernel time exchange. if the mapped name is not correct, can hardly made it.
    # nodes = di_graph.nodes
    # for node in nodes:
    #     if 'qsync::linear' in node: # contrains fp32, fp16 or int8
    #         node_name_parts = node_name.split("_")
    #         new_name = "_".join(node_name_parts[0], node_name_parts[-1])
    #         mapping = {node: new_name}
    #         di_graph = nx.relabel_nodes(di_graph, mapping)
    
    # return di_graph
    

# Replayer Related.
import dpro
from dpro.dag_utils import dag_longest_path
def predict_dag_latency(critical_path_graph, logging_level="INFO", _debug_level=2):
    # latency prediction!
    logger = dpro.logger_utils.SingleLogger(".", "ana.log", logging_level=logging_level)
    res = dag_longest_path(critical_path_graph, weight='avg', default_weight=None, _debug_level=_debug_level)
    path_length = sum([i[1] for i in res])
    return path_length


# requires
def predict_latency_to_cuda_node(node_k, di_graph):
    copied_graph = copy.deepcopy(di_graph)
    remove_all_node_with_type(copied_graph, "comm_op")
    # break GPU edges
    copied_graph.remove_edges_from(list(copied_graph.out_edges(node_k)))
    # break corresponding cpu edges
    nodes = copied_graph.nodes
    cpu_node = get_cuda_cpu_node(node_k, copied_graph)
    target_remove_edges = [edge for edge in copied_graph.out_edges(cpu_node) if is_cpu_op(nodes, edge[1])]
    copied_graph.remove_edges_from(target_remove_edges)
    
    edgelist = copied_graph.edges()
    un_directed_graph = nx.Graph(edgelist)
    length = len(list(nx.connected_components(un_directed_graph)))
    if length == 1: import pdb; pdb.set_trace()
    for comp in nx.connected_components(un_directed_graph):
        sub_G = copied_graph.subgraph(comp)
        break
    # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
    # start from node_k, remove all succeeing nodes
    # remove_target_list = [node_k]
    # remove_target = remove_target_list.pop(0)
    # while remove_target:
    #     connected_nodes = [i[1] for i in copied_graph.out_edges(node_k)]
    #     for c_node in connected_nodes:
    #         remove_target_list.append(c_node)
    #     copied_graph.remove_node(remove_target)
    #     if len(remove_target_list) > 0:
    #         remove_target = remove_target_list.pop(0)
    #     else:
    #         remove_target = None
    # do prediction
    sub_G = add_critical_edges(sub_G)
    path_length = predict_dag_latency(sub_G, _debug_level=0)
    return path_length



def analyze_graph(critical_path_graph, OVERALL_TIME=None, REMOVAL_TIME=None, _debug_level=2):
    node_keys = list(critical_path_graph.nodes.keys())
    # get all forward node
    # root = critical_path_graph.nodes[root_node_name]
    fwd_nodes = []; bwd_nodes = []; opt_nodes = []; opt_step_nodes = []
    for key in node_keys:
        node = critical_path_graph.nodes[key]
        if node['pth'] == 'fwd':
            fwd_nodes.append(node)
        if node['pth'] == 'bwd':
            bwd_nodes.append(node)
        if node['pth'] == 'opt_comm':
            opt_nodes.append(node) # optimizer only a DtoD time. very small, then is the update time
        if node['pth'] == 'opt_step':
            opt_step_nodes.append(node) # optimizer only a DtoD time. very small, then is the update time
    
    path_length = predict_dag_latency(critical_path_graph, _debug_level=_debug_level)
    if REMOVAL_TIME is not None and OVERALL_TIME is not None:
        REMOVAL_TIME /= 1000
        ACT_TIME = OVERALL_TIME / 1000 - REMOVAL_TIME
        print("Actual Time", ACT_TIME)
        print("Error Rate :%.3f%%" % (abs(path_length - ACT_TIME) / ACT_TIME * 100))
    return path_length


def read_trace(file_name):
    trace_file = None
    with open(file_name, 'r') as f:
        trace_file = json.load(f)
    return trace_file

id_mapper = {}
name_cnt = defaultdict(int)

def get_new_name(event_name):
    cnt = name_cnt[event_name]
    name_cnt[event_name] += 1
    new_name = f"{event_name}_{cnt}" 
    return new_name

# very important in customization
def get_simplified_event_name(event):
    event_name = event['name']
    if "_kernel" in event_name or "Kernel" in event_name:
        if 'nccl' in event_name:
            event_name = 'ncclKernel'
        else:
            event_name = event_name.split("::")[1].split("<")[0]
    elif "indexSelectLargeIndex" in event_name:
        event_name = "indexSelectLargeIndex"
    elif "autograd::engine::evaluate_function:" in event_name:
        event_name = event_name.split("autograd::engine::evaluate_function:")[1].strip()
    # customization
    return event_name

def get_unique_id_for_node(event):
    id_event = id(event)
    event_name = get_simplified_event_name(event)
    if id_event in id_mapper:
        return id_mapper[id_event]
    else:
        # cnt = name_cnt[event_name]
        # name_cnt[event_name] += 1
        new_name = get_new_name(event_name)
        id_mapper[id_event] = new_name
        return new_name

# runtime ops that is useless
useless_runtime_name = config.useless_runtime_name

# operators to be tracked
# since cuda luanch takes most of the time, the operators that incurs cuda launch is the one used for tracing
ops_to_track = config.ops_to_track

other_fwd_op = [
    "DataLoader", 
]

def in_other_fwd_op(event):
    event_name = event['name']
    for op in other_fwd_op:
        if op in event_name:
            return True
    return False

aten_op_to_bwd_mapper = config.aten_op_to_bwd_mapper
# correspodnign fwd & bwd bitiwidth relevant kernel
bit_width_changeable_nodes = config.bit_width_changeable_nodes
deny_nodes = config.deny_nodes 
# in bert, only these ops requires grad
grad_required_bwds = config.grad_required_bwds
next_mm_target = config.next_mm_target

if LASTEST_TORCH:
    grad_required_bwds.pop('addbackward0', None)
    grad_required_bwds.pop('mmbackward0', None)


def get_bwd_lwcase_name(title):
    title_splited = title.split("autograd::engine::evaluate_function:")
    bwd_name = title_splited[-1].strip().lower()
    return bwd_name

def get_bwd_lwcase_name_event(bwd_event):
    title = bwd_event['name']
    bwd_name = get_bwd_lwcase_name(title)
    return bwd_name

def is_bitwidth_changeable_node(name):
    for poss_name in bit_width_changeable_nodes:
        if poss_name in name:
            return poss_name
    return False

def is_bitwidth_changeable_dag_node(name):
    for poss_name in config.bit_width_changeable_dag_nodes:
        if poss_name in name:
            return poss_name
    return False

def is_cast_node_required_node(name):
    for poss_name in config.cast_required_nodes:
        if poss_name in name:
            return poss_name
    return False

def is_deny_list_node(name):
    for poss_name in deny_nodes:
        if poss_name in name:
            return poss_name
    return False

def node_requires_grad(name):
    for poss_name in grad_required_bwds:
        if poss_name == name:
            return poss_name
    return False

# tool for modify critical graph


def update_cast_node_former(node_k, critical_path_graph, cpu_time, cuda_time):
    if not config.enable_cast_cost_modeling: return 
    former_cpu = former_cpu_node(node_k, critical_path_graph)
    cuda_node = get_aten_cuda_node(node_k, critical_path_graph)
    former_cuda = former_cuda_node(cuda_node, critical_path_graph)
    nodes = critical_path_graph.nodes 
    cpu_cast_node_name = f'added_cast_cpu_former_{node_k}'
    cuda_cast_node_name = f'added_cast_cuda_former_{node_k}'
    if former_cpu is not None: # else did nothing
        if cpu_cast_node_name != former_cpu:
            critical_path_graph.add_node(cpu_cast_node_name, name=cpu_cast_node_name, avg=cpu_time, type='cpu_op', sync=False, pth='fwd', gap=config.cur_cpu_gaps['fwd'])
            critical_path_graph.add_node(cuda_cast_node_name, name=cuda_cast_node_name, avg=cuda_time, type='cuda_op', sync=False, pth='bwd')
            # remove edges
            critical_path_graph.remove_edge(former_cpu, node_k)
            critical_path_graph.remove_edge(former_cuda, cuda_node)
            # create edegs
            # CPU to gpu
            critical_path_graph.add_edge(cpu_cast_node_name, cuda_cast_node_name, type="cpu_gpu_gap", avg=0)
            # link together
            critical_path_graph.add_edge(former_cpu, cpu_cast_node_name, type="cast_gap", avg=0)
            critical_path_graph.add_edge(cpu_cast_node_name, node_k, type="cast_gap", avg=0)

            critical_path_graph.add_edge(former_cuda, cuda_cast_node_name, type="cast_gap", avg=0)
            critical_path_graph.add_edge(cuda_cast_node_name, cuda_node, type="cast_gap", avg=0)
            # if cpu_cast_node_name == 'added_cast_cpu_qsync::linear_fp32_0':
            #     import pdb; pdb.set_trace()
            
            # print(former_cpu, cpu_cast_node_name,node_k, cpu_time)
            # print(former_cuda, cuda_cast_node_name,cuda_node, cuda_time)
        else:
            nodes[former_cpu]['avg'] = cpu_time
            nodes[former_cuda]['avg'] = cuda_time


def update_cast_node_later(node_k, critical_path_graph, cpu_time, cuda_time):
    if not config.enable_cast_cost_modeling: return 
    later_cpu = later_cpu_node(node_k, critical_path_graph)
    cuda_node = get_aten_cuda_node(node_k, critical_path_graph)
    later_cuda = later_cuda_node(cuda_node, critical_path_graph)
    nodes = critical_path_graph.nodes 
    cpu_cast_node_name = f'added_cast_cpu_later_{node_k}'
    cuda_cast_node_name = f'added_cast_cuda_later_{node_k}'
    if cpu_cast_node_name != later_cpu:
        critical_path_graph.add_node(cpu_cast_node_name, name=cpu_cast_node_name, avg=cpu_time, type='cpu_op', sync=False, pth='fwd', gap=config.cur_cpu_gaps['bwd'])
        critical_path_graph.add_node(cuda_cast_node_name, name=cuda_cast_node_name, avg=cuda_time, type='cuda_op', sync=False, pth='bwd')
        # remove edges
        critical_path_graph.remove_edge(node_k, later_cpu)
        critical_path_graph.remove_edge(cuda_node, later_cuda)
        # create edegs
        # CPU to gpu
        critical_path_graph.add_edge(cpu_cast_node_name, cuda_cast_node_name, type="cpu_gpu_gap", avg=0)
        # link together
        critical_path_graph.add_edge(node_k, cpu_cast_node_name, type="cast_gap", avg=0)
        critical_path_graph.add_edge(cpu_cast_node_name, later_cpu,  type="cast_gap", avg=0)

        critical_path_graph.add_edge(cuda_node, cuda_cast_node_name,  type="cast_gap", avg=0)
        critical_path_graph.add_edge(cuda_cast_node_name, later_cuda,  type="cast_gap", avg=0)

        # if cpu_cast_node_name == 'added_cast_cpu_qsync::linear_fp32_0':
        #     import pdb; pdb.set_trace()
        
        # print(former_cpu, cpu_cast_node_name,node_k, cpu_time)
        # print(former_cuda, cuda_cast_node_name,cuda_node, cuda_time)
    else:
        nodes[later_cpu]['avg'] = cpu_time
        nodes[later_cuda]['avg'] = cuda_time
    

def sort_with_ts(events):
    new_events = sorted(events, key=lambda d: d['ts']) 
    return new_events


# first deal with forward

def map_external_id_to_kernel(launch_kernel, cuda_events):
    if 'args' in launch_kernel and 'correlation' in launch_kernel['args']:
        extern_id = launch_kernel['args']['correlation']
    else:
        print("wrong match!!!")
        print(launch_kernel)
        exit()
    try:
        corresponding_kernel = [event for event in cuda_events if event['args']['correlation'] == extern_id][0]
    except:
        print(launch_kernel)
    return corresponding_kernel

def get_sorted_catogrized_events(trace_name):
    trace_file = read_trace(trace_name)
    trace_events = trace_file['traceEvents']
    cpu_events, runtime_events, cuda_events, python_function_events = [], [], [], []
    user_annotations = []
    MemsetEvents, MemcpyEvents = [], []
    for event in trace_events:
        if 'cat' in event:
            if event['cat'] == 'Kernel':
                cuda_events.append(event)
            elif event['cat'] == 'cpu_op':
                cpu_events.append(event)
            elif event['cat'] == 'cuda_runtime' or event['cat'] == 'Runtime': # latest / old version
                runtime_events.append(event)
            elif event['cat'] == 'python_function':
                python_function_events.append(event)
            elif event['cat'] == "user_annotation":
                user_annotations.append(event)
            elif event['cat'] == "Memset":
                MemsetEvents.append(event)
            elif event['cat'] == "Memcpy":
                MemcpyEvents.append(event)
    cpu_events, runtime_events, cuda_events = sort_with_ts(cpu_events), sort_with_ts(runtime_events), sort_with_ts(cuda_events)
    python_function_events = sort_with_ts(python_function_events)
    user_annotations = sort_with_ts(user_annotations)
    MemsetEvents, MemcpyEvents = sort_with_ts(MemsetEvents), sort_with_ts(MemcpyEvents)
    return cpu_events, runtime_events, cuda_events, python_function_events, user_annotations, MemsetEvents, MemcpyEvents


def get_sorted_comm_events(trace_name):
    trace_file = read_trace(trace_name)
    trace_events = trace_file['traceEvents']
    comm_events = []
    for event in trace_events:
        if 'cat' in event:
            if event['cat'] == 'Kernel' and 'nccl' in event['name']:
                comm_events.append(event)
    comm_events = sort_with_ts(comm_events)
    return comm_events

def filter_comm_events(cuda_events):
    check_text = 'nccl' # according to backend
    # since communication events also in cuda events. Thus we need to filter it out.
    cuda_wo_comm_events = [event for event in cuda_events if check_text not in event['name']]
    comm_events = [event for event in cuda_events if check_text in event['name']]
    return cuda_wo_comm_events, comm_events

def get_profiler_step(user_annotations):
    profiler_step_events = [event for event in user_annotations if 'ProfilerStep' in event['name']]
    profiler_step_events = sort_with_ts(profiler_step_events)
    return profiler_step_events

def get_optimizer_step(user_annotations):
    optimizer_step_events = [event for event in user_annotations if 'Optimizer.step' in event['name']]
    optimizer_step_events = sort_with_ts(optimizer_step_events)
    return optimizer_step_events

def get_optimizer_zero_grad_step(user_annotations):
    optimizer_zrg_events = [event for event in user_annotations if 'Optimizer.zero_grad' in event['name']]
    optimizer_zrg_events = sort_with_ts(optimizer_zrg_events)
    return optimizer_zrg_events

def get_scheduler_time(python_functions):
    scheduler_events = [event for event in python_functions if 'scheduler' in event['name']]
    sorted_scheduler_events = sort_with_ts(scheduler_events)
    first_scheduler_event = sorted_scheduler_events[0]
    return first_scheduler_event['dur'] / 1000




def restrict_events_within_profiler_step(profiler_events, events_records:list, idx=0):
    # remove all events outside the profiler out-most profiler event
    profiler_event = profiler_events[idx]
    start_ts = profiler_event['ts']
    end_ts = start_ts + profiler_event['dur']

    empty_list = [[] for _ in range(len(events_records))]
    for idx, events_record in enumerate(events_records):
        empty_list[idx] = [event for event in events_record if event['ts'] >= start_ts and event['ts'] < end_ts]
    return empty_list

def map_cpu_op_to_kernel(cpu_op, maping_dict):
    return maping_dict[get_unique_id_for_node(cpu_op)]

def map_cpu_events_to_kernels(cpu_events, launch_kernel_events, mapping_events):
    event_idx, launch_kernel_idx = 0, 0
    event_to_kernel = defaultdict(list) # save result
    # deal with forward events
    # print(len(launch_kernel_events))
    kernel_event_length = len(launch_kernel_events)
    end_ts = 0
    for event in cpu_events:
        op_name = event['name'] 
        start_ts = event['ts']
        if start_ts < end_ts: continue # the previous available aten result is larger
        dur = event['dur']
        end_ts = start_ts + dur
        event_to_kernel[get_unique_id_for_node(event)] = []
        while launch_kernel_idx < kernel_event_length and launch_kernel_events[launch_kernel_idx]['ts'] < start_ts:
            launch_kernel_idx += 1
        while launch_kernel_idx < kernel_event_length and launch_kernel_events[launch_kernel_idx]['ts'] <= end_ts:
            launch_kernel = launch_kernel_events[launch_kernel_idx]
            event_to_kernel[get_unique_id_for_node(event)].append(map_external_id_to_kernel(launch_kernel, mapping_events))
            launch_kernel_idx += 1
        event_idx += 1
        last_end = end_ts if len(event_to_kernel) > 0 else 0
        # print(op_name, event_idx, launch_kernel_idx)
    return event_to_kernel


def sort_sheet_with_top_most_op(eventsheet):
    sorted_event_sheet = sort_with_ts(eventsheet)
    new_event_sheet = []
    new_ts = 0
    for event in sorted_event_sheet:
        ts_event = event['ts']
        ts_dur = event['dur']
        if ts_event > new_ts:
            new_event_sheet.append(event)
            new_ts = ts_event + ts_dur
    return new_event_sheet # will only last the top most available ops now.

# deal with the events before the first model forward events
# the time always a constant for any bitwidth setup
def get_pre_aten_to_kernel(cpu_events_before, runtime_events, mapping_events):
    launch_kernel_events = [event for event in runtime_events if event['name'] not in useless_runtime_name]
    aten_events = sort_sheet_with_top_most_op(cpu_events_before)
    aten_to_kernel = map_cpu_events_to_kernels(aten_events, launch_kernel_events, mapping_events)
    return aten_events, aten_to_kernel

def get_fwd_aten_to_kernel(cpu_events, cuda_events, runtime_events, python_function_events, mapping_events):
    # first model Aten OP
    first_forward_event = [event for event in cpu_events if config.FIRST_ATEN_OP in event['name']][0] # should be only one
    # first backward op
    if LASTEST_TORCH:
        first_backward_event = [event for event in python_function_events if backward_title in event['name']][0] # should be only one
    else:
        first_backward_event = [event for event in cpu_events if "autograd" in event['name']][0] # should be only one
    
    fwd_start_ts = first_forward_event['ts']
    fwd_end_ts = first_backward_event['ts']
    fwd_dur = first_forward_event['dur']

    # get necessary aten events we required
    aten_events = [event for event in cpu_events if (event['name'] in ops_to_track or in_other_fwd_op(event)) and event['ts'] < fwd_end_ts and event['ts'] >= fwd_start_ts]
    aten_events_before = [event for event in cpu_events if (event['name'] in ops_to_track or in_other_fwd_op(event)) and event['ts'] < fwd_start_ts]
    launch_kernel_events = [event for event in runtime_events if event['name'] not in useless_runtime_name]
    # print(len(aten_events), len(launch_kernel_events))
    aten_events = sort_sheet_with_top_most_op(aten_events)
    aten_to_kernel = map_cpu_events_to_kernels(aten_events, launch_kernel_events, mapping_events)
    print("Got FWD ops: ", len(aten_events))
    return aten_events, aten_to_kernel, aten_events_before


def get_bwd_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events):
    ops_to_track = 'autograd::engine::evaluate_function' # trace all autograd functions
    bwd_events = [event for event in cpu_events if ops_to_track in event['name']]
    launch_kernel_events = [event for event in runtime_events if event['name'] not in useless_runtime_name]
    # deal with forward events
    # print(len(launch_kernel_events))
    # bwd_events = sort_sheet_with_top_most_op(bwd_events)
    bwd_to_kernel = map_cpu_events_to_kernels(bwd_events, launch_kernel_events, mapping_events)
    print("Got BWD ops:", len(bwd_events))
    return bwd_events, bwd_to_kernel

# get the communication mem copy events
def get_comm_mem_copy_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events):
    if LASTEST_TORCH:
        ops_to_track = 'torch.distributed.ddp.reducer::copy_bucket_to_grad'
        comm_mem_events = [event for event in cpu_events if ops_to_track in event['name']]
    else:
        # old torch has no user annotation for the communication, collect them between optimizer & backward
        last_backward_op = [event for event in cpu_events if 'autograd::engine::evaluate_function' in event['name']][-1]
        optimizer_step = [event for event in cpu_events if 'Optimizer.step' in event['name']][0]
        last_backward_op_end = last_backward_op['ts'] + last_backward_op['dur']
        start_optimizer_step = optimizer_step['ts']
        comm_mem_events = [event for event in cpu_events if event['ts'] > last_backward_op_end and event['ts'] < start_optimizer_step]

    launch_kernel_events = [event for event in runtime_events if event['name'] not in useless_runtime_name]
    # deal with forward events
    # print(len(launch_kernel_events))
    comm_mem_to_kernel = map_cpu_events_to_kernels(comm_mem_events, launch_kernel_events, mapping_events)
    print("Got Comm MemCpy ops:",len(comm_mem_events))
    return comm_mem_events, comm_mem_to_kernel

# get the optimizer step time
def get_optimizer_step_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events, optimizer_event):
    ops_to_track = 'aten::'
    start_ts = optimizer_event['ts']
    end_ts = start_ts + optimizer_event['dur']
    optimizer_events = [event for event in cpu_events if ops_to_track in event['name'] and event['ts'] < end_ts and event['ts'] > start_ts]
    launch_kernel_events = [event for event in runtime_events if event['name'] not in useless_runtime_name and event['ts'] < end_ts and event['ts'] > start_ts]
    # deal with forward events
    optimizer_to_kernel = map_cpu_events_to_kernels(optimizer_events, launch_kernel_events, mapping_events)
    print("Got Optimizer Ops:",len(optimizer_events))
    return optimizer_events, optimizer_to_kernel


def get_non_empty_cpu_ops(cpu_ops, mapping_dict):
    new_ops = [cpu_op for cpu_op in cpu_ops if len(map_cpu_op_to_kernel(cpu_op, mapping_dict)) != 0]
    print("Non Empty Ops:", len(new_ops))
    return new_ops


'''
    Fetch Attrs
'''

def get_input_infos_from_event(event):
    input_dims, input_type = None, None
    if "args" in event:
        if "Input Dims" in event["args"]:
            input_dims = event["args"]['Input Dims']
            input_type = event["args"]["Input type"]
    return input_dims, input_type



'''
    OP Adders.
    Since the tracing result can be different with respect to the true training graph at end. Add necessary node.
'''
log_softmax_op = 'aten::log_softmax'
nll_loss_nd_op = 'aten::nll_loss_nd'
clamp_op = 'aten::clamp'
div_op = 'aten::div'
div_underscore_op = 'aten::div_'
add_op = 'aten::add'
split_op = 'aten::split'
squeeze_op = 'aten::squeeze'
contiguous_op = 'aten::contiguous'

op_cnt_dict = defaultdict(int)

def append_one_node_at_tail(node_op, last_op_name, di_graph):
    op_cnt = op_cnt_dict[node_op]
    node_name = f'{node_op}+{op_cnt}'
    di_graph.add_node(node_name, op=node_op)
    di_graph.add_edge(last_op_name, node_name)
    op_cnt_dict[node_op] += 1
    return node_name

def rm_last_node(di_graph):
    descending_list = reversed(list(nx.topological_sort(di_graph)))
    last_node_k = next(descending_list)
    di_graph.remove_node(last_node_k)


def add_several_ops_at_end(op_append_list, di_graph):
    descending_list = reversed(list(nx.topological_sort(di_graph)))
    last_node_k = next(descending_list)
    for op in op_append_list:
        last_node_k = append_one_node_at_tail(op, last_node_k, di_graph)

# the fwd graph is extracted from tracer, thus the loss function should be add later
def add_cross_entropy_tails(di_graph):
    op_append_list = [log_softmax_op, nll_loss_nd_op]
    add_several_ops_at_end(op_append_list, di_graph)

# transformers tails
def add_div_tails(di_graph):
    op_append_list = [add_op, div_op, div_op]
    add_several_ops_at_end(op_append_list, di_graph)

# def maintain_last_idx(idx_list, new_idx):
#     idx_list.append(new_idx)
#     idx_list = sorted(idx_list)
#     if idx_list == [*range(idx_list[0], idx_list[-1] + 1)]: 
#         idx_list = [idx_list[-1]]
#         return idx_list[-1]
#     else:
#         len_list = len(idx_list)
#         for idx in range(1, len_list):
#             gap = idx_list[idx] - idx_list[idx-1]
#             if gap != 1:
#                 return idx_list[idx]

def get_bound_start(set_, new_num):
    set_.add(new_num)
    length_set = len(set_)
    first_element = list(set_)[0]
    if list(range(first_element, length_set)) == list(set_):
        set_.clear()
        return True
    return False

last_appear_idx = defaultdict(int)
special_aten_mapping = config.special_aten_mapping
def map_aten_event_to_relevant_kernel(fwd_di_graph, aten_events, aten_to_kernel):
    # the fwd DAG
    ascending_list = nx.topological_sort(fwd_di_graph)
    nodes = fwd_di_graph.nodes
    # bwd trace is not necessary needed, since fwd trace can map to the bwd kernel.

    dag_2_aten = {}
    aten_idx = 0
    aten_length = len(aten_events)

    op_set = set()
    idx_set = set()
    idx_set.add(0)

    start = False
    quick_forward=False
    first_op_aten_idx = None 
    for node_k in ascending_list:
        node = nodes[node_k]
        node_op = node['op']
        # print(node_op)
        op_set.add(node_op)
        if start: 
            pass
        else:
            if node_op == config.FIRST_DAG_OP:
                start = True
            else:
                continue
        # degbug
        # if node_k == 140620447463152:
        #     import pdb; pdb.set_trace()
        # map the point on DAG to the aten_event
        # print(node_op)
        # find the op of last appeared idx
        aten_idx = last_appear_idx[node_op] if first_op_aten_idx is None or (first_op_aten_idx is not None and last_appear_idx[node_op] >= first_op_aten_idx) else first_op_aten_idx
        
        if quick_forward: 
            # import pdb;pdb.set_trace()
            aten_idx = new_idx
            quick_forward=False
        while aten_idx <= aten_length - 1:
            aten_event = aten_events[aten_idx]
            aten_name = aten_event['name']
            # if 'pool' in aten_name:
            # print(aten_name)
            if aten_name == node_op or (node_op in special_aten_mapping and aten_name in special_aten_mapping[node_op]): # exactly equals to (pytorch version) or cutomized ops
                new_idx =  aten_idx + 1
                quick_forward = get_bound_start(idx_set, new_idx)
                last_appear_idx[node_op] = new_idx
                if first_op_aten_idx is None:
                    first_op_aten_idx = new_idx
                # print("Matched", aten_name, aten_idx)
                dag_2_aten[node_k] = aten_event # map a node to the aten event
                break
            aten_idx += 1
    # print(op_set)
    # import pdb;pdb.set_trace()
    return dag_2_aten

def get_bwd_to_acc_map(bwd_events):
    bwd_to_acc_map = {}
    acc_grad_events = []
    for bwd_event in bwd_events:
        bwd_name = get_bwd_lwcase_name_event(bwd_event)
        bwd_id = get_unique_id_for_node(bwd_event)
        # deal with accumulate
        if 'accumulategrad' in bwd_name: 
            acc_id = bwd_id
            bwd_to_acc_map[cur_grad_required_bwd].append(acc_id) # link the accumulation event to correct bwd event
            acc_grad_events.append(bwd_event)
            continue # TODO: link it later
        if bwd_name in grad_required_bwds:
            cur_grad_required_bwd = get_unique_id_for_node(bwd_event)
            bwd_to_acc_map[cur_grad_required_bwd] = []
    return acc_grad_events, bwd_to_acc_map

import copy
# these op are included in a larger op,   
last_appear_idx_bwd = defaultdict(int)
def construct_runtime_mapping(dag_2_aten, bwd_events):
    len_bwd_events = len(bwd_events)
    bwd_idx = len_bwd_events - 1
    dag_2_all = {}
    for aten_k, aten_v in dag_2_aten.items():
        aten_name = aten_v['name']
        # try:
        required_bwds = copy.deepcopy(aten_op_to_bwd_mapper[aten_name])

       
        dag_2_all[aten_k] = {
            'fwd': aten_v,
            'bwd': []
        }

        if len(required_bwds) == 0: 
            # print(aten_v)
            continue # operations like clamp

        # print(aten_k, type(aten_k))
        # if type(aten_k) is str and 'clamp' in aten_k:
        #     import pdb; pdb.set_trace()
        jump_once = False
        for bwds in required_bwds:
            if bwds not in last_appear_idx_bwd: last_appear_idx_bwd[bwds] = len_bwd_events - 1
            bwd_idx = last_appear_idx_bwd[bwds]
            # print(bwd_idx, bwds)
            if jump_once:
                jump_once = False
                continue
            while bwd_idx >= 0:
                bwd_event = bwd_events[bwd_idx]
                bwd_name = get_bwd_lwcase_name_event(bwd_event)
                # deal with accumulate
                if 'accumulategrad' in bwd_name: 
                    bwd_idx -= 1
                    continue # TODO: link it later
                # print(aten_k, bwd_name, bwds)
                # import pdb; pdb.set_trace()

                if "admmbackward" in bwd_name:
                    if aten_name == 'aten::linear':
                        dag_2_all[aten_k]['bwd'].append(bwd_event)
                        last_appear_idx_bwd[bwds] = bwd_idx - 1
                        jump_once = True
                        break
                if bwd_name == bwds:
                    dag_2_all[aten_k]['bwd'].append(bwd_event)
                    last_appear_idx_bwd[bwds] = bwd_idx - 1
                    break
                bwd_idx -= 1
        # print(aten_name, len(dag_2_all[aten_k]['bwd']) )
    return dag_2_all

def convert_dag_event_to_g_nodes(dag_2_all_events, critical_path_graph):
    # import pdb; pdb.set_trace()
    dag_2_nodes = {}
    for k, v in dag_2_all_events.items():
        fwd_event = v['fwd']
        new_fwd_event = get_unique_id_for_node(fwd_event) 
        bwd_events = v['bwd']
        new_bwd_events = [get_unique_id_for_node(event) for event in bwd_events]
        dag_2_nodes[k] = {}
        dag_2_nodes[k]['fwd'] = new_fwd_event
        dag_2_nodes[k]['bwd'] = new_bwd_events

    return dag_2_nodes

def dag_2_nodes_to_fwd_to_bwd(dag_2_nodes):
    fwd_to_bwd_mapper = {}
    for k, v in dag_2_nodes.items():
        fwd_node = v['fwd']
        bwd_nodes = v['bwd']
        fwd_to_bwd_mapper[fwd_node] = bwd_nodes
    return fwd_to_bwd_mapper



# pytorch all-reduce implementation,
# comm happened in the first next relevant operator
def find_next_commbwd(event, sorted_bwd_event, optimizer_events):
    target = copy.deepcopy(next_mm_target)
    if not LASTEST_TORCH: target += ['addbackward0']
    len_bwd_event = len(sorted_bwd_event)
    idx_event = sorted_bwd_event.index(event)

    while idx_event < len_bwd_event-1:
        idx_event += 1
        new_event = sorted_bwd_event[idx_event]
        new_event_name = new_event['name'].strip().lower()
        for name in target:
            if name in new_event_name:
                # print("new_event id", id(new_event))
                return new_event
    return optimizer_events[0] # first 



gap_cpu = 0.001
gap_gpu = 0.001
gap_comm = 0.004

# different version of PyTorch and different part has different density of cuda functions
def change_gap_time(p_gap_cpu = 0.001, p_gap_gpu = 0.001, p_gap_comm = 0.004):
    global gap_comm, gap_cpu, gap_gpu
    gap_comm = p_gap_comm
    gap_cpu = p_gap_cpu
    gap_gpu = p_gap_gpu

# three possible implementation to conenct costs
def merge_kernel_cost_to_aten(event, events_mapper, pth_name):
    id_event = get_unique_id_for_node(event)
    # input_dims, input_type = get_input_infos_from_event(event)
    kernels = map_cpu_op_to_kernel(event, events_mapper)
    avg = 0
    for kernel in kernels:
        if pth_name == 'bwd': 
            if 'nccl' in kernel['name']: continue # don't relate the gradient comm event here
        dur = kernel['dur'] / 1000 + gap_gpu
        avg += dur 
    return id_event, avg


def link_kernel_cost_merged(event, events_mapper, critical_path_graph, previous_kernel_node, pth_name):
    id_event = get_unique_id_for_node(event)
    kernels = map_cpu_op_to_kernel(event, events_mapper)
    dur = 0
    id_kernel = f"{id_event}_kernel"
    

    for kernel in kernels:
        if pth_name == 'bwd': 
            if 'nccl' in kernel['name']: continue # don't relate the gradient comm event here
        dur += (kernel['dur'] / 1000)
    
    critical_path_graph.add_node(id_kernel, name=id_kernel, avg=dur, type='cuda_op', gap=gap_gpu, pth=pth_name)
    # critical_path_graph.add_edge(id_event, id_kernel, type="cpu_to_cuda_gap", avg=0)
    if previous_kernel_node is not None:
        critical_path_graph.add_edge(previous_kernel_node, id_kernel, type="cuda_gap", avg=0)

    return id_kernel, [id_kernel]

def link_kernel_cost_to_aten(event, events_mapper, critical_path_graph, previous_kernel_node, pth_name):
    id_event = get_unique_id_for_node(event)
    kernels = map_cpu_op_to_kernel(event, events_mapper)
    first_kernel = True
    kernel_node_list = []
    dur = 0
    for kernel in kernels:
        id_kernel = get_unique_id_for_node(kernel)
        if first_kernel:
            critical_path_graph.add_edge(id_event, id_kernel, type="cpu_to_cuda_gap", avg=0)
            first_kernel = False # link to first kernel

        dur += (kernel['dur'] / 1000 + gap_gpu)

        critical_path_graph.add_node(id_kernel, name=id_kernel, avg=dur, type='cuda_op', pth=pth_name)
        dur = 0
        kernel_node_list.append(id_kernel)
        if previous_kernel_node is not None:
            critical_path_graph.add_edge(previous_kernel_node, id_kernel, type="cuda_gap", avg=0)
        if pth_name == 'bwd': 
            if 'nccl' in kernel['name']: continue # don't relate the gradient comm event here
        previous_kernel_node = id_kernel
    return previous_kernel_node, kernel_node_list

# there exitst cuda synchronize that force synchronization of the CPU and GPU time. This greatly affected the final result we got
# to approximate the result, we manually remove the cuda synchronization time in the operation and only left the remaining time as cpu time
# Don't do it in pre_root nodes (not relevant, what we want is more accurate)
IS_PRE_ROOT = False
def remove_cuda_synchronize_time(event, runtime_events):
    if IS_PRE_ROOT:
        return 0
    # find the cuda synchronize events
    # sync_events = [event for event in runtime_events if 'Synchronize' in event['name'] or 'cudaMemcpyAsync' in event['name']]
    sync_events = [event for event in runtime_events if 'Synchronize' in event['name']]

    # find the corresponding runtime sync events for the target event 
    event_start = event['ts']
    event_end = event_start + event['dur']
    sync_event_list = []
    dur_sync = 0
    for sync_event in sync_events:
        start = sync_event['ts']
        end = sync_event['dur'] + start
        if event_start <= start and end <= event_end:
            sync_event_list.append(sync_event)
            dur_sync += sync_event['dur']
    
    return dur_sync


def construct_graph_with_event(critical_path_graph, events, events_mapper, previous_cpu_node, previous_kernel_node, runtime_events, pth_name='fwd'):
    # print(events)
    cpu_nodes = []
    cpu_cuda_mapping = {}
    for event in events:
        id_event = get_unique_id_for_node(event)
        # id_event, avg = merge_kernel_cost_to_aten(event, events_mapper, pth_name)
        # critical_path_graph.add_node(id_event, name=id_event, avg=avg, type='cpu_op', pth=pth_name)
        dur_aten = event['dur'] / 1000 
        dur_sync = remove_cuda_synchronize_time(event, runtime_events)
        dur_aten -= dur_sync / 1000
        # if dur_sync > 0:
        #     import pdb; pdb.set_trace()
        # debug
        # if 'embedding' in id_event:
        #     print(event)
        #     print(dur_aten)
        be_sync = dur_sync > 0
        critical_path_graph.add_node(id_event, name=id_event, avg=dur_aten, type='cpu_op', sync=be_sync, pth=pth_name)
        # previous_kernel_node, kernel_node_list = link_kernel_cost_to_aten(event, events_mapper, critical_path_graph, previous_kernel_node, pth_name)
        id_kernel, kernel_node_list = link_kernel_cost_merged(event, events_mapper, critical_path_graph, previous_kernel_node, pth_name)
        
        cpu_nodes.append(id_event)
        cpu_cuda_mapping[id_event] = kernel_node_list

        # create necessary edges
        critical_path_graph.add_edge(previous_cpu_node, id_event, type="cpu_gap", avg=0) # Link between CPU events
        critical_path_graph.add_edge(id_event, id_kernel, type="cpu_to_cuda_gap", avg=0) # GPU event relies on its CPU event
        # update old pointer
        previous_cpu_node = id_event
        previous_kernel_node = id_kernel

    return previous_cpu_node, previous_kernel_node, cpu_nodes, cpu_cuda_mapping

optimizer_ops_mapper = {
    'SGD': ['aten::add_', 'aten::mul_', 'aten::add_','aten::add_'],
    'AdamW': ['aten::mul_', 'aten::mul_', 'aten::add_', 'aten::mul_', 'aten::addcmul_', 'aten::sqrt', 'aten::div', 'aten::add_']
}
def construct_optimizer_tensor_block(critical_path_graph, optimizer_step_events, optimizer_step_to_kernel, previous_cpu_node, last_comm_node, runtime_events, pth_name='opt_step'):
    # an optimizer is endup with addcmul (which is the gradient update operation)
    # different optimizer will have different updating strategy
    # for example. Decay requires an aten::add_, momentum requires an aten::mul_ and a aten::add_
    # and SGD update requires an add_
    # thus, for SGD that use momentun and decay, the step ops should be add_, mul_, add_, add_ 
    # which consistent to the source code :https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
    # adamw: mul_, mul_, add_, mul_, addcmul_, sqrt, div, add_
    # this requires to partition optimizer to correct blocks to corresponding tensor
    
    indicators =  copy.deepcopy(optimizer_ops_mapper[config.OPTIMIZER_NAME])
    block_name =  'optimizer_tensor_block'
    # in old version,
    packed_event = []
    previous_kernel_node = last_comm_node

    newly_added_nodes = []
    block_event_list = []
    for event in optimizer_step_events:
        packed_event.append(event)
        event_name = event['name']
        if event_name in indicators:
            if event_name == indicators[0]:
                indicators.pop(0)
        if len(indicators) == 0:
            indicators =  copy.deepcopy(optimizer_ops_mapper[config.OPTIMIZER_NAME])
            new_block_name = get_new_name(block_name)
            # calculate the avg cost
            previous_cpu_node, previous_kernel_node, cpu_nodes, cpu_cuda_mapping = construct_graph_with_event(critical_path_graph, packed_event, optimizer_step_to_kernel, previous_cpu_node, previous_kernel_node, runtime_events, pth_name)
            # avg_all = 0
            # for event in packed_event:
            #     _, avg = merge_kernel_cost_to_aten(event, optimizer_step_to_kernel, pth_name)
            #     # don't add node here
            #     avg_all += avg
            # # add to graph
            # critical_path_graph.add_node(new_block_name, name=new_block_name, avg=avg_all, type='opt_step', pth=pth_name)
            # newly_added_nodes.append(new_block_name)
            # critical_path_graph.add_edge(last_node, new_block_name)
            block_event_list.append((block_name, cpu_nodes, cpu_cuda_mapping))
            packed_event = []
    return  previous_cpu_node, previous_kernel_node, block_event_list


def get_bwd_node_to_opt_step_mapper(bwd_nodes, block_event_list):
    # do bwd_nodes to opt nodes here
    cnt = 0
    opt_step_idx = 0
    opt_nums = len(block_event_list)
    bwd_to_step_mapper = {}
    accumulate_op_count = 0
    # debug
    records_debugging = defaultdict(int)
    for idx, node_k in enumerate(bwd_nodes):
        bwd_name = get_bwd_lwcase_name(node_k)
        bare_bwd_name = bwd_name.split('_')[0]
        if 'accumulate' in bwd_name:
            accumulate_op_count += 1
            continue
        key_ = node_requires_grad(bare_bwd_name)
        if key_:
            grads_num = grad_required_bwds[key_]
            # deal with special add
            if key_ == 'addbackward0':
                if idx + 1 <= len(bwd_nodes) - 1 and ('accumulate' not in get_bwd_lwcase_name(bwd_nodes[idx + 1])):
                    # remove the situation that it is a simple add function not bias add in linear
                    bwd_next_name = get_bwd_lwcase_name(bwd_nodes[idx + 1])
                    bare_next_bwd_name = bwd_next_name.split('_')[0]
                    if bare_next_bwd_name in config.bias_handling: pass
                    else: continue
            bwd_to_step_mapper[node_k] = []
            bound_this_term = opt_step_idx + grads_num
            while opt_step_idx < opt_nums and opt_step_idx < bound_this_term:
                opt_step_node = block_event_list[opt_step_idx]
                bwd_to_step_mapper[node_k].append(opt_step_node)
                opt_step_idx += 1
            # print(key_, grads_num, accumulate_op_count)
            cnt += grads_num
            records_debugging[bare_bwd_name] += grads_num
    # import pdb; pdb.set_trace()
    assert cnt == accumulate_op_count, f'bwd gradient update != accumulator ops {cnt}-{accumulate_op_count}'
    if config.CONSIDER_OPTIMIZER_TIME:
        assert cnt == opt_nums, f"bwd gradient update requirement != optimizer step tims {cnt}-{opt_nums}"   
    return bwd_to_step_mapper

# for the cpu event with synchronization, we need to point its corresponding cuda node to 
# its cpu successor
def sync_dealing(di_graph):
    nodes = di_graph.nodes 
    # Create CPU lookup table for quick search
    sync_cpu_nodes = [node for node in nodes if is_cpu_op(nodes, node) and nodes[node]['sync']]
    for sync_node in sync_cpu_nodes:
        cuda_node = get_aten_cuda_node(sync_node, di_graph)
        next_cpu_node = later_cpu_node(sync_node, di_graph)
        # create sync edge
        di_graph.add_edge(cuda_node, next_cpu_node, type="sync_edge", avg=0)

# ABANDON, not correct.
def comm_start_calculation(di_graph):
    print("Add start point for communication, may takes some time")
    nodes = di_graph.nodes 
    comm_nodes = [node for node in nodes if is_comm_op(nodes, node)]
    last_comm_end = 0
    for comm_node in comm_nodes:
        cuda_node = get_comm_cuda_node(comm_node, di_graph)
        lat = predict_latency_to_cuda_node(cuda_node, di_graph)
        comm_time = nodes[comm_node]['avg']
        if lat > last_comm_end:
            start_time = lat 
        else:
            start_time = last_comm_end
        nodes[comm_node]['start'] = start_time 
        last_comm_end = start_time + comm_time
        # print(comm_node, start_time)
    print("Start Point Added.")
    # import pdb; pdb.set_trace()

def map_communication_to_graph_nccl(acc_grad_events, critical_path_graph, bwd_events, bwd_to_kernel, bwd_cuda_mapper, comm_mem_events, comm_cuda_mapper, previous_kernel_node, previous_cpu_node):
    last_comm_node = None
    acc_to_comm_mapper = defaultdict(list)
    for acc_grad_event in acc_grad_events:
        acc_node = get_unique_id_for_node(acc_grad_event)
        kernels = map_cpu_op_to_kernel(acc_grad_event, bwd_to_kernel)
        kernels = [kernel for kernel in kernels if 'nccl' in kernel['name']] # only related to communication
        if len(kernels) != 0:
            # pytorch nccl implementation logic, may be different in other framework
            event = find_next_commbwd(acc_grad_event, bwd_events, comm_mem_events) # point comm kernel to the head of the event
            id_event = get_unique_id_for_node(event)
            
            for kernel in kernels:
                id_kernel = get_unique_id_for_node(kernel) # communication kernel
                
                avg_time = kernel['dur'] / 1000

                first_bwd_node = get_aten_cuda_node(id_event, critical_path_graph) # the cuda node that run simultaneously with comm
                cuda_node_before = former_cuda_node(first_bwd_node, critical_path_graph)
                
                critical_path_graph.add_node(id_kernel, name=id_kernel, avg=avg_time, type='comm_op', pth='comm_k')
                acc_to_comm_mapper[acc_node].append(id_kernel)
                # critical_path_graph.add_edge(id_event, id_kernel, avg=avg_time) 
                critical_path_graph.add_edge(cuda_node_before, id_kernel, type="gpu_comm_gap", avg=0) 
                
                if last_comm_node:
                    critical_path_graph.add_edge(last_comm_node, id_kernel, type="comm_comm_gap", avg=0) 
                last_comm_node = id_kernel
            # print(critical_path_graph.edges(id_event))
    # import pdb; pdb.set_trace()
    if last_comm_node is not None:
        end_node_cpu_name = "end_train_cpu"
        end_node_kernel_name = "end_train_kernel"
        critical_path_graph.add_node(end_node_cpu_name, type='cpu_op', gap=0, sync=False, pth='end_of_main_training', avg=0)
        critical_path_graph.add_node(end_node_kernel_name, type='cuda_op', gap=0, pth='end_of_main_training', avg=0) # for later use, define it as cuda op

        critical_path_graph.add_edge(previous_cpu_node, end_node_cpu_name, type='cpu_to_cpu', avg=0)
        critical_path_graph.add_edge(end_node_cpu_name, end_node_kernel_name, type='cpu_to_cuda', avg=0)
        # combine the previous kernel cost to this special node to be new kernel end (optimizer happens after comm ends)
        critical_path_graph.add_edge(previous_kernel_node, end_node_kernel_name, type='kernel_to_train_end', avg=0)
        critical_path_graph.add_edge(last_comm_node, end_node_kernel_name, type='comm_to_train_end', avg=0)
        last_comm_node = end_node_kernel_name
        previous_cpu_node = end_node_cpu_name
    return previous_cpu_node, last_comm_node, acc_to_comm_mapper

# core function in construct critical path graph
def construct_critical_path(aten_events, bwd_events, aten_to_kernel_fwd, bwd_to_kernel, acc_grad_events, \
 comm_mem_events, comm_mem_to_kernel, optimizer_step_events, optimizer_step_to_kernel, optimizer_zrg_events, optimizer_zrg_to_kernel, runtime_events):
    critical_path_graph = nx.DiGraph()
    root_node = critical_path_graph.add_node(config.root_node_name, type='root', avg=0, pth='root')
    previous_cpu_node, previous_kernel_node = config.root_node_name, None
    # add fwd, bwd, optimizer node
    # cpu_gap = config.cur_cpu_gaps['fwd']
    # change_gap_time(p_gap_cpu=cpu_gap)
    # import pdb; pdb.set_trace()

    previous_cpu_node, previous_kernel_node, fwd_nodes, fwd_cuda_mapper = construct_graph_with_event(critical_path_graph, aten_events, aten_to_kernel_fwd, previous_cpu_node, previous_kernel_node, runtime_events, pth_name='fwd')
    
    # cpu_gap = config.cur_cpu_gaps['bwd']
    # change_gap_time(p_gap_cpu=cpu_gap)

    previous_cpu_node, previous_kernel_node, bwd_nodes, bwd_cuda_mapper = construct_graph_with_event(critical_path_graph, bwd_events, bwd_to_kernel, previous_cpu_node, previous_kernel_node, runtime_events, pth_name='bwd')

    # comm mem copy
    previous_cpu_node, previous_kernel_node, opt_comm_nodes, comm_cuda_mapper = construct_graph_with_event(critical_path_graph, comm_mem_events, comm_mem_to_kernel, previous_cpu_node, previous_kernel_node, runtime_events, pth_name='opt_comm')
    # TODO: add comm node
    # everytime, add communication, find the node that overlap the communication time
    # start from the point of the beginging of the first kernel
    # last_comm_node = None
    previous_cpu_node, last_comm_node, acc_to_comm_mapper = map_communication_to_graph_nccl(acc_grad_events, critical_path_graph, bwd_events, bwd_to_kernel, bwd_cuda_mapper, comm_mem_events, comm_cuda_mapper, previous_kernel_node, previous_cpu_node)
    
    last_comm_node = previous_kernel_node if last_comm_node is None else last_comm_node
    # import pdb; pdb.set_trace()
    # debug
    # optimizer update happends after the communication. After Communication
    if config.CONSIDER_OPTIMIZER_TIME:
        # cpu_gap = config.cur_cpu_gaps['optimizer_step']
        # change_gap_time(p_gap_cpu=cpu_gap)
        previous_cpu_node, previous_kernel_node, block_event_list = construct_optimizer_tensor_block(critical_path_graph, optimizer_step_events, optimizer_step_to_kernel, previous_cpu_node, last_comm_node, runtime_events, pth_name='opt_step')
        # add zerograd
        # cpu_gap = config.cur_cpu_gaps['optimizer_zero_grad']
        # change_gap_time(p_gap_cpu=cpu_gap)
        previous_cpu_node, previous_kernel_node, zrg_nodes, zrg_cuda_mapper = construct_graph_with_event(critical_path_graph, optimizer_zrg_events, optimizer_zrg_to_kernel, previous_cpu_node, previous_kernel_node, runtime_events, pth_name='opt_zrg')
        # nx.write_gml(critical_path_graph, "runtime.gml")
        bwd_to_step_mapper = get_bwd_node_to_opt_step_mapper(bwd_nodes, block_event_list)
    else:
        block_event_list = []
        zrg_cuda_mapper = {}
        bwd_to_step_mapper = {}
    
    
    # manually add edges to avoid
    # Synchronization
    sync_dealing(critical_path_graph)
    # add communication time (start point)
    # comm_start_calculation(critical_path_graph)


    mapper_dict = {
        'fwd': fwd_cuda_mapper,
        'bwd': bwd_cuda_mapper,
        'opt_dist': comm_cuda_mapper,
        'opt_step': block_event_list,
        'opt_zrg': zrg_cuda_mapper,
        'bwd_to_step': bwd_to_step_mapper,
        'acc_to_comm': acc_to_comm_mapper
    }
    print("Export the critical path graph")
    return critical_path_graph, mapper_dict



def add_critical_edges(critical_graph):
    # edges = critical_graph.edges
    nodes = critical_graph.nodes
    for k, v in critical_graph.nodes.items():
        in_edges = critical_graph.in_edges(k, data=True)
        if len(in_edges) == 0: continue
        cur_node = nodes[k]
        for u, v, data in in_edges:
            # print(u, v, data)
            try:
                data['avg'] = cur_node['avg']
            except:
                import pdb; pdb.set_trace()
    return critical_graph


    
# bit relevant operations
def init_graph_bit(dag_graph, bit=32):
    dag_nodes = dag_graph.nodes 
    dag_node_keys = dag_nodes.keys()
    for k in dag_node_keys:
        node = dag_nodes[k]
        node['bit'] = bit

def change_graph_bit_single_node(dag_graph, k, bit=32):
    dag_nodes = dag_graph.nodes 
    dag_node_keys = dag_nodes.keys()
    node = dag_nodes[k]
    node['bit'] = bit

def change_relevant_node_bitwidth(sub_graph):
    output_bit_mapping = config.output_bit_mapping
    sub_graph_nodes = sub_graph.nodes
    topo_order = nx.topological_sort(sub_graph)
    for node_k in topo_order:
        if is_bitwidth_changeable_node(node_k): continue
        if is_deny_list_node(node_k): continue
        preds = sub_graph.pred[node_k]
        in_bits = [output_bit_mapping[sub_graph_nodes[pred]['bit']] for pred in preds]
        first_bits = in_bits[0]
        if all_equal(in_bits):
            sub_graph_nodes[node_k]['bit'] = first_bits
            # print("change", node_k, first_bits)

def check_graph_bit(dag_graph, dag_2_nodes, critical_path_node_graph):
    dag_nodes = dag_graph.nodes 
    dag_node_keys = dag_nodes.keys()
    for k in dag_node_keys:
        node = dag_nodes[k]
        bit = node['bit']

        corspd_fwd_node = dag_2_nodes[k]['fwd']
        corspd_bwd_node = dag_2_nodes[k]['bwd']

        mapping = config.bit32_mapping
        name_mapping = config.fp32_name_mapping
        if bit == 16:
            mapping = config.bit16_mapping
            name_mapping = config.fp16_name_mapping
        elif bit == 8:
            cur_mapping = config.bit8_mapping
            name_mapping = config.int8_name_mapping
        
        if node_k not in cur_mapping:
            correct_fetch_name = get_correct_name(corspd_fwd_node, name_mapping)
        else:
            correct_fetch_name = corspd_fwd_node

        fwd_kernel = mapping[correct_fetch_name]
        bwd_kernels = [mapping[get_correct_name(bwd_node, name_mapping)] for bwd_node in corspd_bwd_node]
        fwd_cost = fwd_kernel['avg']
        bwd_cost = sum([kernel['avg'] for kernel in bwd_kernels])

        fwd_node_in_graph = critical_path_node_graph.nodes[corspd_fwd_node]
        in_graph_avg = fwd_node_in_graph['avg']
        in_gragh_bwd_avg = sum([critical_path_node_graph.nodes[c_bwd_node]['avg'] for c_bwd_node in corspd_bwd_node])

        succs = dag_graph.successors(k)
        succ_bits = [output_bit_mapping[dag_nodes[succ]['bit']] for succ in succs]
        print(node, bit, f"In record cost{(fwd_cost, bwd_cost)}", f"In graph cost{in_graph_avg, in_gragh_bwd_avg}")


def map_new_cost(preds_bits, node_bit):
    result = [bit != node_bit for bit in pred_bits]
    if sum(result) == 0: return # all bit are same
    # conversion is required, caculated how many conversion is required
    # find bit and input dims

available_bit_width_change_nodes = []
output_bit_mapping = config.output_bit_mapping
output_bit_mapping_bwd = config.output_bit_mapping_bwd




default_store_cost = {
    'memset': 0.001
}
def get_default_store_cost(node_k):
    if node_k in default_store_cost:
        return default_store_cost[node_k]
    return 0.0

def get_correct_name(node_k, name_mapping):
    for k, v in name_mapping.items():
        if k in node_k: 
            fetch_k = node_k.replace(k, v)
            return fetch_k
    return node_k

def fetch_cost_from_store(node_k, bit):
    cur_mapping = config.bit32_mapping
    name_mapping = config.fp32_name_mapping
    if bit == 16:
        cur_mapping = config.bit16_mapping
        name_mapping = config.fp16_name_mapping
    elif bit == 8:
        cur_mapping = config.bit8_mapping
        name_mapping = config.int8_name_mapping

    if node_k not in cur_mapping:
        correct_fetch_name = get_correct_name(node_k, name_mapping)
    else:
        correct_fetch_name = node_k 
    
    


    if correct_fetch_name in cur_mapping:
        node_store = cur_mapping[correct_fetch_name]
        node_cost = node_store['avg']
    else:
        # no reference in mapping (lookup table) for the case. ops like memset will encounter such problem
        print("Dont find", node_k)
        node_cost = get_default_store_cost(node_k)
        # TODO: similar reaons
        return node_cost
        import pdb; pdb.set_trace()
    return node_cost

def fetch_cpu_casting_time(node_k, bit, bwd=False):
    if bit == 16 or bit == 32:
        cur_mapping = config.bit16_mapping
        name_mapping = config.fp16_name_mapping
        cast_mapper = config.bit16_cast_mapper
    elif bit == 8:
        cur_mapping = config.bit8_mapping
        name_mapping = config.int8_name_mapping
        cast_mapper = config.bit8_cast_mapper

    if node_k not in cur_mapping:
        correct_fetch_name = get_correct_name(node_k, name_mapping)
    else:
        correct_fetch_name = node_k 
    
    # if'batch_norm' in node_k:
    #     import pdb; pdb.set_trace()

    if correct_fetch_name not in cast_mapper:
        return 0 
    cast_node = cast_mapper[correct_fetch_name]
    if correct_fetch_name in cur_mapping:
        node_store = cur_mapping[cast_node]
        node_cost = node_store['avg']
    else:
        # no reference in mapping (lookup table) for the case. ops like memset will encounter such problem
        print("Dont find", cast_node)
        import pdb; pdb.set_trace()
        
        node_cost = get_default_store_cost(cast_node)
    if bwd:
        if bit == 8:
            node_cost = 0.3 # This is an profiler-based estimated cost for cpu
        

    return node_cost

def add_cost_for_node(node_k, di_graph, cost_add):
    node = di_graph.nodes[node_k]
    node['avg'] += cost_add

def update_cost_for_node(node_k, di_graph, cost):
    try:
        node = di_graph.nodes[node_k]
    except:
        import pdb; pdb.set_trace()
    node['avg'] = cost

def update_single_node_cost(node_k, di_graph, bit, cuda_mapper):
    cuda_nodes = cuda_mapper[node_k]
    # Get New Cost
    # TODO: change it to prediction model
    cpu_cost = fetch_cost_from_store(node_k, bit)
    if type(cuda_nodes) is not list:
        cuda_cost = fetch_cost_from_store(cuda_nodes, bit)
    else:
        cuda_costs = [fetch_cost_from_store(cuda_node, bit) for cuda_node in cuda_nodes]

    # Update CPU cost
    update_cost_for_node(node_k, di_graph, cpu_cost)
    # if 'linear' in node_k: # debug
    #     print(node_k, cpu_cost, cuda_nodes) 
    # Update GPU cost
    if type(cuda_nodes) is not list:
        update_cost_for_node(cuda_nodes, di_graph, cuda_cost)
    else:
        for idx, cost in enumerate(cuda_costs):
            cuda_node = cuda_nodes[idx]
            update_cost_for_node(cuda_node, di_graph, cost)
            # if 'linear' in cuda_node: # debug
            #     print(cuda_node, cost, bit)

    
    return cuda_nodes

def get_inshapes_and_weight_shape_from_node(cur_node):
    if 'in_shapes' not in cur_node:
        input_shapes = [0]
    else:
        cur_node_shapes = [shape for shape in cur_node['in_shapes'] if shape is not None]
        input_shapes = np.squeeze([shape for shape in cur_node_shapes if shape is not None])
        input_shapes = np.array(input_shapes)

    if 'weight_shape' in cur_node:
        weight_size = cur_node['weight_shape']
    else:
        weight_size = [0]
    return input_shapes, weight_size


def change_fwd_cost(dag_node_k, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict):

    # get current node
    dag_nodes = dag_graph.nodes
    cur_node = dag_nodes[dag_node_k]
    bit = cur_node['bit']
    # get fwd cpu nodes
    corspd_fwd_node = dag_2_nodes[dag_node_k]['fwd'] # get the DAG mapped node in critical path graph
    preds = dag_graph.pred[dag_node_k]
    pred_bits = [output_bit_mapping[dag_nodes[pred]['bit']] for pred in preds]

    # get fwd cuda nodes
    fwd_cuda_mapper = mapper_dict['fwd']
    cuda_nodes = update_single_node_cost(corspd_fwd_node, critical_path_node_graph, bit, fwd_cuda_mapper) # update the node data on critical graph
                                                                                            # including both CPU and CUDA nodes
    # determine different predictors
    input_cast_predictor = ana_predictor.predict_with_bit if config.act_quant == 'scale' else ana_predictor.predict_with_bit_channel
    kernel_cast_predictor = ana_predictor.predict_with_bit if config.kernel_quant == 'scale' else ana_predictor.predict_with_bit_channel
    dequant_cast_predictor = ana_predictor.predict_float_scale if config.kernel_quant == config.act_quant else ana_predictor.predict_float_channel # only scale + channel or scale + scale is possible

    # add cuda conversion cost
    conv_cost_fwd = 0
    if 'in_shapes' not in cur_node:
        pass
    else:
        input_shapes, weight_size = get_inshapes_and_weight_shape_from_node(cur_node)
        conv_cost_fwd = 0

        if len(pred_bits) != 0:
            for idx, pred_bit in enumerate(pred_bits):
                if pred_bit != bit: 
                    if len(input_shapes.shape) > 1:
                        shape = input_shapes[idx]
                    else:
                        shape = input_shapes
                    # prediction conversion cost
                    # conv_cost_fwd += ana_predictor.predict(np.prod(shape))
                    conv_cost_fwd += input_cast_predictor(pred_bit, bit, np.prod(shape))
                    # conv_cost_fwd += 0
        else:
            # first node cast
            if 'embedding' not in corspd_fwd_node:
                conv_cost_fwd += input_cast_predictor(32, bit, np.prod(input_shapes))
        
        # add cost for weight conversion
        conv_cost_fwd += kernel_cast_predictor(32, bit, np.prod(weight_size))
        # add cost for output scale calculation
        if bit <= 8:
            try:
                scale_shape = 1 if config.act_quant == 'scale' else input_shapes[0]
            except:
                import pdb; pdb.set_trace()
            conv_cost_fwd += dequant_cast_predictor(scale_shape)
        
        cpu_dq_fwd = 0
        if 'out_shapes' in cur_node:
            out_shapes = np.squeeze(cur_node['out_shapes'])
            if out_shapes is not None and  np.any(out_shapes != None):
                if len(out_shapes.shape) == 1: # always have only one output for any op
                    out_shape = out_shapes
                else:
                    out_shape = out_shapes[0]
                # only when quantzation fusion is not did here, we require extra scaling prediction
                if bit <= 8 and not config.quantize_fused:
                    dq_fwd = dequant_cast_predictor(np.prod(out_shapes))
                    cpu_dq_fwd = 0.03 + config.cur_cpu_gaps['fwd'] # TODO: estimate it using data
                    dq_fwd += 0.22 #  wrong estmation of the quantizer of int8, always smaller with a constant
                                    # TODO: deal with it
                    # print(dq_fwd)
                    update_cast_node_later(corspd_fwd_node, critical_path_node_graph, cpu_dq_fwd, dq_fwd)
    # if 'batch_norm' in corspd_fwd_node:
    #     import pdb; pdb.set_trace()
    # print(cur_node)
    # add cuda time
    if len(cuda_nodes) != 0:
        if conv_cost_fwd != 0:
            cpu_cast_cost = fetch_cpu_casting_time(corspd_fwd_node, bit)
            if bit == 8:
                conv_cost_fwd += 0.22 # wrong estmation of the quantizer of int8, always smaller with a constant
            update_cast_node_former(corspd_fwd_node, critical_path_node_graph, cpu_cast_cost, conv_cost_fwd)
            # if 'batch_norm' in dag_node_k:
            #     import pdb; pdb.set_trace()
            
            # print(corspd_fwd_node, cpu_cast_cost, conv_cost_fwd)
            # print(cpu_cast_cost, conv_cost_fwd, corspd_fwd_node)
            # add_cost_for_node(cuda_nodes[0], critical_path_node_graph, conv_cost_fwd)
    # add cpu cast time 
    # if conv_cost_fwd != 0: # means the cpu cost exists
        # must be either FP32->FP16, FP16->FP32 or FP32->INT8, FP16->INT8
        # for FP16/FP32->INT8, mapped cpu result is same 
        # cpu_cast_cost = fetch_cpu_casting_time(corspd_fwd_node, bit)
        # cpu_cast_cost += config.cur_cpu_gaps['fwd']
        # add_cost_for_node(corspd_fwd_node, critical_path_node_graph, cpu_cast_cost)
    

def change_bwd_cost(dag_node_k, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict):
    # get fwd nodes
    dag_nodes = dag_graph.nodes
    cur_node = dag_nodes[dag_node_k]
    bit = cur_node['bit']
    # corspd bwd nodes 
    corspd_bwd_nodes = dag_2_nodes[dag_node_k]['bwd'] # e.g. linear has 3 bp nodes
    succs = dag_graph.succ[dag_node_k]

    # bwd cuda node
    bwd_cuda_mapper = mapper_dict['bwd']
    cuda_nodes_list = [update_single_node_cost(corspd_bwd_node, critical_path_node_graph, bit, bwd_cuda_mapper) for corspd_bwd_node in corspd_bwd_nodes]

    input_cast_predictor = ana_predictor.predict_with_bit if config.act_quant == 'scale' else ana_predictor.predict_with_bit_channel
    kernel_cast_predictor = ana_predictor.predict_with_bit if config.kernel_quant == 'scale' else ana_predictor.predict_with_bit_channel

    # added bwd cuda cost
    conv_cost_bwd = 0
    if 'out_shapes' not in cur_node:
        pass
    else:
        # print(succs)
        if len(succs) > 0:
            try:
                succ_bits = [output_bit_mapping_bwd[dag_nodes[succ]['bit']] for succ in succs]
            except:
                import pdb; pdb.set_trace()
            if cur_node['out_shapes'][0] is not None:
                output_shapes = [shape for shape in cur_node['out_shapes'][0] if shape is not None]
                shape = output_shapes[0]
            else:
                # operators like split don't have a output, since it is a middle node, mixed-precision won't be affected.
                output_shapes = []
                shape = []
            if len(succ_bits) != 0:
                for idx, succ_bit in enumerate(succ_bits):
                    if succ_bit != bit: 
                        # print(np.prod(shape))
                        # prediction conversion cost
                        # conv_cost_bwd += ana_predictor.predict(np.prod(shape))
                        # print(succ_bit, output_bit_mapping_bwd[bit])
                        conv_cost_bwd += ana_predictor.predict_with_bit(succ_bit, output_bit_mapping_bwd[bit], np.prod(shape)) 
                        # conv_cost_bwd += 0
                        # import pdb;pdb.set_trace()
        
    
    # deal with cast node
    # we need to add cast node first
    # print(cuda_nodes_list)
    if len(cuda_nodes_list) != 0 and len(cuda_nodes_list[0]) != 0:
        # Add conversion cost to a cuda node
        cuda_node = cuda_nodes_list[0][0]
        corspd_fwd_node = dag_2_nodes[dag_node_k]['fwd']
        cur_node = dag_nodes[dag_node_k]
        in_shapes, weight_shape = get_inshapes_and_weight_shape_from_node(cur_node)
        # add dequantization cost here for int8 operators
        # dq_cst = 0
        # if bit == 8:
        #     dq_input_cst = input_cast_predictor(bit, 16, np.prod(in_shapes))
        #     dq_kernel_cst = kernel_cast_predictor(bit, 16, np.prod(weight_shape))
        #     dq_cst = dq_input_cst + dq_kernel_cst
        #     conv_cost_bwd += dq_cst
        cpu_cast_cost = 0
        # cpu_cast_cost = fetch_cpu_casting_time(corspd_fwd_node, bit, bwd=True) # cpu time is similar
        corspd_bwd_node = corspd_bwd_nodes[0]

        # if dq_cst != 0:
        # print(corspd_bwd_node, cpu_cast_cost, conv_cost_bwd, dq_cst)

        update_cast_node_former(corspd_bwd_node, critical_path_node_graph, cpu_cast_cost, conv_cost_bwd)

        # add_cost_for_node(cuda_node, critical_path_node_graph, conv_cost_bwd)
    # if len(corspd_bwd_nodes) != 0 and conv_cost_bwd != 0:
        # The conversion cost is just the controversial of the forward
        # corspd_fwd_node = dag_2_nodes[dag_node_k]['fwd']
        # cpu_cast_cost = fetch_cpu_casting_time(corspd_fwd_node, bit)
        # add_cost_for_node(corspd_bwd_nodes[0], critical_path_node_graph, cpu_cast_cost)
        # neglect cuda cost (since is dense )
        # return 

# calculate cost
def calculate_cost_with_conversion(dag_node_k, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict, consider_neighbors=True):
    if consider_neighbors: 
        # get current node again
        dag_nodes = dag_graph.nodes
        cur_node = dag_nodes[dag_node_k]
        bit = cur_node['bit']
        try:
            corspd_fwd_node = dag_2_nodes[dag_node_k]['fwd']
        except:
            # TODO: some of the fwd_digraph generation will add extra nodes like contiguous
            # is not compute-intensive and can be neglect
            return 
            import pdb; pdb.set_trace()
        # only successors will be affected to another new bit
        succs = dag_graph.succ[dag_node_k]
        # iterate succs to check successor's inputs
        for succ in succs:
            preds = dag_graph.pred[succ]
            pred_bits = [output_bit_mapping[dag_nodes[pred]['bit']] for pred in preds]
            succ_op = dag_nodes[succ]['op']

            # import pdb;pdb.set_trace()
            if all_equal(pred_bits) and not is_bitwidth_changeable_node(succ_op) and not is_deny_list_node(succ_op):
                # print(dag_node_k, succ)
                change_dag_bit(dag_graph, dag_2_nodes, succ, pred_bits[0], critical_path_node_graph, mapper_dict, consider_neighbors)
                
    # change fwd cost and bwd cost of existing node 
    change_fwd_cost(dag_node_k, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict)
    change_bwd_cost(dag_node_k, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict)
    




def change_dag_bit(dag_graph, dag_2_nodes, dag_node_k, bit, critical_path_node_graph, mapper_dict, consider_neighbors=True):
    if is_deny_list_node(dag_node_k): return # if is node in deny_list, don't change it
    # get current node 
    dag_nodes = dag_graph.nodes 
    dag_node_keys = list(dag_2_nodes.keys())
    cur_node = dag_nodes[dag_node_k]
    if dag_node_k not in dag_node_keys:
        # TODO: exists some case that dag node is not in dag node keys
        # This problem is caused by the alias, for the moment, ignores the case when it is not main components
        if is_bitwidth_changeable_dag_node(dag_node_k):
            import pdb; pdb.set_trace()
        else:
            return
        # import pdb; pdb.set_trace()
    assert dag_node_k in dag_node_keys, f"{dag_node_k}, {cur_node}"
    # set current node bit to certain bit 
    # it could be either node in allowlist or inferlist
    cur_node['bit'] = bit
    succs = dag_graph.succ[dag_node_k]
    preds = dag_graph.pred[dag_node_k]
    # update the result 
    calculate_cost_with_conversion(dag_node_k, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict, consider_neighbors)
    # update succs's cost (fwd casting)
    # update preds's cost (bwd casting)
    for suc in succs:
        calculate_cost_with_conversion(suc, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict, consider_neighbors)
    
    # for pred in preds:
    #     calculate_cost_with_conversion(pred, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict, consider_neighbors)
    
    # bwd_to_acc_mapper = mapper_dict['bwd_to_acc']
    # bwd_to_step_mapper = mapper_dict['bwd_to_step']
    # bwd_cuda_mapper = mapper_dict['bwd'] # accumulategrad is also backward node
    # acc_to_comm_mapper = mapper_dict['acc_to_comm']

    # Gradient Relevant Change.
    # Not used in ourcase since we force fp32 weight, fp32 gradient.
    # if cur_node['op'] in bit_width_changeable_nodes:
    #     weight_shape = cur_node['weight_shape'] 
    #     # TODO: Weight_shape is required in communication
    #     corspd_bwd_nodes = dag_2_nodes[dag_node_k]['bwd']
    #     for bwd_node in corspd_bwd_nodes:
    #         # TODO: Communication
    #         # change accumulation time 
    #         if bwd_node in bwd_to_acc_mapper:
    #             acc_nodes = bwd_to_acc_mapper[bwd_node] # have multiple acc nodes. 
    #             [update_single_node_cost(node_k, critical_path_node_graph, bit, bwd_cuda_mapper) for node_k in acc_nodes]
    #             # fetch communication for acc nodes, update it accordingly
    #             # import pdb; pdb.set_trace()
    #             for acc_node in acc_nodes: # update communication
    #                 comm_nodes = acc_to_comm_mapper[acc_node]
    #                 for comm_node in comm_nodes:
    #                     new_comm_cost = fetch_cost_from_store(comm_node, bit)
    #                     update_cost_for_node(comm_node, critical_path_node_graph, new_comm_cost)

    #         # change optimizer time
    #         if bwd_node in bwd_to_step_mapper:
    #             step_nodes = bwd_to_step_mapper[bwd_node] # may have multiple step nodes for a bwd op   
    #                                                       # each step node is made of (blockid, cpu_nodes, cpu_to_cuda_node_mapper)                          
    #             for (blockid, cpu_nodes, cuda_mapper) in step_nodes:
    #                 # update each using record
    #                 [update_single_node_cost(node_k, critical_path_node_graph, bit, cuda_mapper) for node_k in cpu_nodes]
           

    # import pdb; pdb.set_trace()

# iterate a dag graph, mapped it to the critical path_node_graph
def map_dag_bit_to_critical_graph(dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict):
    consider_neighbors = False
    for dag_node_k in dag_graph:
        if dag_node_k in dag_2_nodes:
            calculate_cost_with_conversion(dag_node_k, dag_graph, dag_2_nodes, critical_path_node_graph, mapper_dict, consider_neighbors)
def get_available_bitwidth_change_node_in_dag(dag_graph):
    op_dict = defaultdict(list)
    for k, v in dag_graph.nodes.data():
        op = v['op']
        if op in bit_width_changeable_nodes:  
            op_dict[op].append(k)
    return op_dict

def get_topo_available_bitwidth_change_node_order(dag_graph):
    ascend_order = nx.topological_sort(dag_graph)
    nodes = dag_graph.nodes
    topo_node_order = []
    for node_k in ascend_order:
        v = nodes[node_k]
        op = v['op']
        if op in bit_width_changeable_nodes:
            topo_node_order.append(node_k)
    return topo_node_order


def get_first_cpu_and_cuda(root_node, di_graph):
    nodes = di_graph.nodes 
    # assert is_root_op(nodes, root_node), "not a root"
    cpu_node = succ_first_cpu(root_node, nodes, di_graph)
    cuda_node = succ_first_cuda(cpu_node, nodes, di_graph)
    return cpu_node, cuda_node

def is_cast_op(op_name):
    for cast_op_name in config.casting_ops:
        if cast_op_name in op_name:
            return True
    return False

def generate_cast_mapper(aten_events):
    length = len(aten_events)
    idx = 0
    fwd_to_cast_mapper = {}
    while idx <= length - 1:
        aten_event = aten_events[idx]
        aten_op = get_unique_id_for_node(aten_event)
        if is_cast_op(aten_op):
            cast_op = aten_op
            # print(cast_op)
            idx += 1
            op_event = aten_events[idx]
            op_name = get_unique_id_for_node(op_event)
            while idx <= length - 1:
                op_event = aten_events[idx]
                op_name = get_unique_id_for_node(op_event)
                if is_cast_op(op_name): # int8 profile result
                    cast_op = op_name
                if is_cast_node_required_node(op_name):
                    fwd_to_cast_mapper[op_name] = cast_op
                    break
                idx += 1
        idx += 1
    # import pdb; pdb.set_trace()
    return fwd_to_cast_mapper



def get_mapper_and_critical_path_graph(trace_name, target_trace_idx=0, bit=None):

    cpu_events, runtime_events, cuda_events, python_function_events, \
     user_annotations, MemsetEvents, MemcpyEvents = get_sorted_catogrized_events(trace_name)
    
    if LASTEST_TORCH:
        # latest torch use user_annotations for profilers
        profiler_step_events = get_profiler_step(user_annotations)
        optimizer_step_events = get_optimizer_step(user_annotations)
        optimizer_zrg_events = get_optimizer_zero_grad_step(user_annotations)
    else:
        # old torch use cpuops
        profiler_step_events = get_profiler_step(cpu_events) # in torch version < 1.10.1. 
        optimizer_step_events = get_optimizer_step(cpu_events)
        optimizer_zrg_events = get_optimizer_zero_grad_step(cpu_events)


    all_cuda_events = cuda_events
    # target_trace_idx
    # get comm
    comm_events = get_sorted_comm_events(trace_name)
    # filter cuda events
    # cuda_events, comm_events = filter_comm_events(cuda_events)
    # restrict trace within a profiler result
    cpu_events, runtime_events, cuda_events, python_function_events, \
     user_annotations, MemsetEvents, MemcpyEvents, comm_events, optimizer_step_events, optimizer_zrg_events = restrict_events_within_profiler_step(profiler_step_events, \
    events_records = [cpu_events, runtime_events, cuda_events, python_function_events, user_annotations, MemsetEvents, MemcpyEvents, comm_events, \
    optimizer_step_events, optimizer_zrg_events], idx=target_trace_idx)

    profiler_ts_end = profiler_step_events[target_trace_idx]['ts'] + profiler_step_events[target_trace_idx]['dur']
    if not config.CONSIDER_OPTIMIZER_TIME:
        # collect time for zero grad and 
        optimizer_step_start = optimizer_step_events[0]['ts']
        # check last comm event
        last_comm_event = comm_events[-1]
        last_end_ts = last_comm_event['ts'] + last_comm_event['dur']
        if last_end_ts > optimizer_step_start:
            gap = last_end_ts - optimizer_step_start
        else:
            gap = 0
        # dur_optimizer_zero_grad = optimizer_zrg_events[0]['dur']
        dur_optimizer_relevant = profiler_ts_end - optimizer_step_start - gap

    # scheduler_time = get_scheduler_time(python_function_events)

    # create mapping events
    mapping_events = cuda_events + MemsetEvents + MemcpyEvents 
    all_mapping_events = all_cuda_events + MemsetEvents + MemcpyEvents
    launch_kernel_events = [event for event in runtime_events if event['name'] not in useless_runtime_name]


    aten_events, aten_to_kernel_fwd, aten_events_before = get_fwd_aten_to_kernel(cpu_events, cuda_events, runtime_events, python_function_events, mapping_events)

    bwd_events, bwd_to_kernel = get_bwd_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events)
    comm_mem_events, comm_mem_to_kernel =  get_comm_mem_copy_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events)
    optimizer_step_events, optimizer_step_to_kernel = get_optimizer_step_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events, optimizer_step_events[0])
    optimizer_zrg_events, optimizer_zrg_to_kernel = get_optimizer_step_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, all_mapping_events, optimizer_zrg_events[0])
    
    # CPU TIME
    aten_time = sum_duration_from_events(aten_events)
    bwd_time = sum_duration_from_events(bwd_events)
    gap = bwd_events[-1]['ts'] - aten_events[0]['ts']
    # print("dur", gap, aten_time + bwd_time, gap - aten_time - bwd_time)

    # remove kernel-irrelevant cpu_ops
    # aten_events = get_non_empty_cpu_ops(aten_events, aten_to_kernel_fwd)
    # bwd_events = get_non_empty_cpu_ops(bwd_events, bwd_to_kernel)
    # optimizer_events = get_non_empty_cpu_ops(optimizer_events, optimizer_to_kernel)
    # optimizer_step_events = get_non_empty_cpu_ops(optimizer_step_events, optimizer_step_to_kernel)

    # Do OP -> Kernel mapping
    fwd_di_graph = load_dag_graph(config.model_name)

    # MODIFY the trace graph to satisfy our need. (sequential end)
    if config.model_name == 'bert':
        # rm_last_node(fwd_di_graph)
        op_append_list = [clamp_op, clamp_op]
        add_several_ops_at_end(op_append_list, fwd_di_graph)
        add_cross_entropy_tails(fwd_di_graph)
        add_cross_entropy_tails(fwd_di_graph)
        op_append_list = [add_op, div_op, div_op, div_underscore_op]
        add_several_ops_at_end(op_append_list, fwd_di_graph)

    elif config.model_name == 'roberta':
        tails_to_add = [log_softmax_op, nll_loss_nd_op, div_op]
        add_several_ops_at_end(tails_to_add, fwd_di_graph)
    elif config.model_name == 'resnet50':
        tails_to_add = [log_softmax_op, nll_loss_nd_op]
        add_several_ops_at_end(tails_to_add, fwd_di_graph)
    

    dag_2_aten = map_aten_event_to_relevant_kernel(fwd_di_graph, aten_events, aten_to_kernel_fwd)
    acc_grad_events, bwd_to_acc_map = get_bwd_to_acc_map(bwd_events)
    # # add mapping from OP -> bwd
    dag_2_all_events = construct_runtime_mapping(dag_2_aten, bwd_events)
    # don't need to consider optimizer's connection since optimizer is added after the communication and has no overalapping.


    # create node graph
    # the node data than can be pickled into a json file. (profile result)
    critical_path_node_graph, mapper_dict = construct_critical_path(aten_events, bwd_events, aten_to_kernel_fwd, bwd_to_kernel, acc_grad_events, \
    comm_mem_events, comm_mem_to_kernel, optimizer_step_events, optimizer_step_to_kernel, optimizer_zrg_events, optimizer_zrg_to_kernel, runtime_events)
    
    dag_2_nodes = convert_dag_event_to_g_nodes(dag_2_all_events, critical_path_node_graph)

    pf_event = profiler_step_events[target_trace_idx]

    # create casting cost mapper
    cast_mapper = generate_cast_mapper(aten_events)

    mapper_dict['bwd_to_acc'] = bwd_to_acc_map
    mapper_dict['fwd_to_cast'] = cast_mapper
    mapper_dict['fwd_to_bwd'] = dag_2_nodes_to_fwd_to_bwd(dag_2_nodes)

    # deal with the communication before the main training process
    # this part of time are always constant but not neglectable
    events_before, events_before_mapper = get_pre_aten_to_kernel(aten_events_before, runtime_events, mapping_events)
    global IS_PRE_ROOT
    IS_PRE_ROOT = True
    pre_root = config.pre_root_node_name 
    root_node = critical_path_node_graph.add_node(pre_root, avg=0, type='root', pth='root')
    previous_cpu_node, previous_kernel_node, nodes, mapper = construct_graph_with_event(critical_path_node_graph, events_before, events_before_mapper, pre_root, None, runtime_events, pth_name='pre')
    cpu_node, cuda_node = get_first_cpu_and_cuda(config.root_node_name, critical_path_node_graph)
    critical_path_node_graph.remove_node(config.root_node_name)
    critical_path_node_graph.add_edge(previous_cpu_node, cpu_node, type="cpu_gap", avg=gap_cpu)
    critical_path_node_graph.add_edge(previous_kernel_node, cuda_node, type="gpu_gap", avg=gap_gpu)

    # import pdb; pdb.set_trace()
    if not config.CONSIDER_OPTIMIZER_TIME:
        REMOVAL_TIME = dur_optimizer_relevant
    else:
        REMOVAL_TIME = 0

    return critical_path_node_graph, fwd_di_graph, dag_2_nodes, (REMOVAL_TIME, pf_event), mapper_dict


def fwd_graph_analysis(fwd_di_graph):
    nodes = fwd_di_graph.nodes
    for node, v in nodes.data():
        in_edges = fwd_di_graph.in_edges(node)
        print(node, len(in_edges))
# for debug
def get_in_edges_out_edges(node_k, di_graph):
    in_edges = di_graph.in_edges(node_k)
    out_edges = di_graph.out_edges(node_k)
    print(f"In edges {len(in_edges)} {in_edges}, Out Edges {len(out_edges)} {out_edges}")

# we need to add proper cpu gaps in prediction
def add_cpu_gaps(critical_path_node_graph):

    cpu_gap_fwd = config.cur_cpu_gaps['fwd']
    cpu_gap_bwd = config.cur_cpu_gaps['bwd']
    cpu_gap_opt_step = config.cur_cpu_gaps['optimizer_step']
    cpu_gap_zero_grad = config.cur_cpu_gaps['optimizer_zero_grad']

    nodes = critical_path_node_graph.nodes

    for node_k, node_v in critical_path_node_graph.nodes(data=True):
        cpu_gap = 0
        if 'pth' not in node_v: continue
        if not is_cpu_op(nodes, node_k): continue
        node_v_pth = node_v['pth']
        if node_v_pth == 'fwd':
            cpu_gap = cpu_gap_fwd
        elif node_v_pth == 'bwd' or node_v_pth == 'opt_comm':
            cpu_gap = cpu_gap_bwd 
        elif node_v_pth == 'opt_step':
            cpu_gap = cpu_gap_opt_step
        elif node_v_pth == 'opt_zrg':
            cpu_gap = cpu_gap_zero_grad
        
        node_v['avg'] += cpu_gap
        node_v['gap'] = cpu_gap

def rm_cpu_gaps(critical_path_node_graph):
    cpu_gap_fwd = config.cur_cpu_gaps['fwd']
    cpu_gap_bwd = config.cur_cpu_gaps['bwd']
    cpu_gap_opt_step = config.cur_cpu_gaps['optimizer_step']
    cpu_gap_zero_grad = config.cur_cpu_gaps['optimizer_zero_grad']

    nodes = critical_path_node_graph.nodes

    for node_k, node_v in critical_path_node_graph.nodes(data=True):
        if 'gap' in node_v:
            node_v['avg'] -= node_v['gap']
            node_v['gap'] = 0

def zero_cpu_gaps(critical_path_node_graph):
    cpu_gap_fwd = config.cur_cpu_gaps['fwd']
    cpu_gap_bwd = config.cur_cpu_gaps['bwd']
    cpu_gap_opt_step = config.cur_cpu_gaps['optimizer_step']
    cpu_gap_zero_grad = config.cur_cpu_gaps['optimizer_zero_grad']

    nodes = critical_path_node_graph.nodes

    for node_k, node_v in critical_path_node_graph.nodes(data=True):
        if 'pth' not in node_v: continue
        if not is_cpu_op(nodes, node_k): continue
        node_v_pth = node_v['pth']
        if node_v_pth == 'fwd':
            cpu_gap = cpu_gap_fwd
        elif node_v_pth == 'bwd' or node_v_pth == 'opt_comm':
            cpu_gap = cpu_gap_bwd 
        elif node_v_pth == 'opt_step':
            
            cpu_gap = cpu_gap_opt_step
        elif node_v_pth == 'opt_zrg':
            cpu_gap = cpu_gap_zero_grad
        
        node_v['gap'] = 0

# there is a case for the graph change from CPU bound -> GPU bound
# we detect these case, and add lines
def add_assistant_lines(critical_path_node_graph):
    cpu_node = next(nx.topological_sort(critical_path_node_graph))
    cpu_node, cuda_node = get_first_cpu_and_cuda(cpu_node, critical_path_node_graph)
    # cpu_node, cuda_node = get_first_cpu_and_cuda(config.pre_root_node_name, critical_path_node_graph)
    cpu_lookup_table, cuda_lookup_table = defaultdict(int), defaultdict(int)
    comm_lookup_table = defaultdict(int)
    def get_previous_cost(node):
        cpu_node = pred_first_cpu(node, nodes, critical_path_node_graph)
        cuda_node = pred_first_cuda(node, nodes, critical_path_node_graph)
        comm_node = pred_first_comm(node, nodes, critical_path_node_graph)
        cpu_cost = cpu_lookup_table[cpu_node] if cpu_node is not None else 0
        cuda_cost = cuda_lookup_table[cuda_node] if cuda_node is not None else 0
        comm_cost = comm_lookup_table[comm_node] if comm_node is not None else 0
        return (cpu_cost, cuda_cost, comm_cost)
    nodes = critical_path_node_graph.nodes
    cnt = 0
    while cpu_node is not None:
        new_cpu_cost = max(get_previous_cost(cpu_node)) + nodes[cpu_node]['avg']
        cpu_lookup_table[cpu_node] = new_cpu_cost
        cuda_start = max(get_previous_cost(cuda_node))
        new_cuda_cost = cuda_start + nodes[cuda_node]['avg']
        cuda_lookup_table[cuda_node] = new_cuda_cost

        comm_node = get_cuda_comm_node(cuda_node, critical_path_node_graph)
        if comm_node is not None:
            prev_comm_end = cuda_start
            # get previous communication cost
            prev_node = pred_first_comm(comm_node, nodes, critical_path_node_graph)
            if prev_node is not None: 
                prev_comm_end = max(prev_comm_end, comm_lookup_table[prev_node])
            nodes[comm_node]['start'] = prev_comm_end
            # print("Start point", comm_node, prev_comm_end)
            end_comm = prev_comm_end + nodes[comm_node]['avg'] # end of the communication
            comm_lookup_table[comm_node] = end_comm
        
        
        l_cpu = later_cpu_node(cpu_node, critical_path_node_graph)
        if l_cpu is not None and new_cuda_cost!=0: 
            if new_cuda_cost < new_cpu_cost:
                # point cuda node to later_cpu node
                critical_path_node_graph.add_edge(cuda_node, l_cpu, type="assist", avg=0)
                # print(cuda_node, l_cpu, new_cuda_cost, new_cpu_cost)
            else:
                if new_cuda_cost < new_cpu_cost + nodes[l_cpu]['gap']:
                    critical_path_node_graph.add_edge(cuda_node, l_cpu, type="assist", avg=0)
                    # print("add lines", cuda_node, l_cpu)
        cpu_node = l_cpu
        if l_cpu is not None: 
            cuda_node = get_aten_cuda_node(cpu_node, critical_path_node_graph)
    # debug
    comm_nodes = [node for node in nodes if is_comm_op(nodes, node)]
    # import pdb; pdb.set_trace()
    # empty_72_kernel        
    # get_in_edges_out_edges('aten::add__0', critical_path_node_graph)
    # get_in_edges_out_edges('end_train_cpu', critical_path_node_graph)
    # 
    # get_in_edges_out_edges('aten::empty_72_kernel', critical_path_node_graph)
    return cpu_lookup_table, cuda_lookup_table

def reset_communication_start(di_graph):
    di_graph_nodes = di_graph.nodes 
    comm_nodes = [node for node in di_graph.nodes if is_comm_op(di_graph_nodes, node)]
    copied_graph = copy.deepcopy(di_graph)
    add_assistant_lines(copied_graph)
    for comm_node in comm_nodes:
        # print(f"Set {comm_node} start -> {copied_graph.nodes[comm_node]['start']}")
        di_graph_nodes[comm_node]['start'] = copied_graph.nodes[comm_node]['start']



def predict(critical_path_node_graph, OVERALL_TIME=None, REMOVAL_TIME=None, _debug_level=2, cpu_interference_cost=0):
    critical_path_node_graph = copy.deepcopy(critical_path_node_graph)
    if cpu_interference_cost != 0:
        critical_path_node_graph.nodes['end_train_cpu']['avg'] += cpu_interference_cost
    add_cpu_gaps(critical_path_node_graph)
    cpu_lookup_table, cuda_lookup_table = add_assistant_lines(critical_path_node_graph)

    critical_path_edge_graph = add_critical_edges(critical_path_node_graph)

    return analyze_graph(critical_path_edge_graph, OVERALL_TIME, REMOVAL_TIME, _debug_level)


# subgraph or prediction
def predict_latency(fwd_di_graph, critical_graph, dag_2_nodes, mapper_dict, _debug_level=2):
    critical_graph = copy.deepcopy(critical_graph)
    map_dag_bit_to_critical_graph(fwd_di_graph, dag_2_nodes, critical_graph, mapper_dict)
    # we don't set CPU gap here to make a pure comparison on the data
    res = predict(critical_graph, _debug_level=_debug_level)
    return res 

def predict_latency_cuda_only(fwd_di_graph, critical_graph, dag_2_nodes, mapper_dict, _debug_level=2):
    critical_graph = copy.deepcopy(critical_graph)
    map_dag_bit_to_critical_graph(fwd_di_graph, dag_2_nodes, critical_graph, mapper_dict)
    all_cost = 0
    fwd_cuda_mapper = mapper_dict['fwd']
    bwd_cuda_mapper = mapper_dict['bwd']

    for dag_node_k in fwd_di_graph.nodes:
        if dag_node_k in dag_2_nodes:
            _node_fwd = dag_2_nodes[dag_node_k]['fwd']
            cuda_fwd = fwd_cuda_mapper[_node_fwd][0]
            # print(dag_2_nodes[dag_node_k]['fwd'], dag_2_nodes[dag_node_k]['bwd'])
            cost = critical_graph.nodes[cuda_fwd]['avg'] 
            if len(dag_2_nodes[dag_node_k]['bwd']) > 0:
                _node_bwd = dag_2_nodes[dag_node_k]['bwd'][0]
                cuda_bwd = bwd_cuda_mapper[_node_bwd][0]
                cost += critical_graph.nodes[cuda_bwd]['avg']
            all_cost += cost
    # for node, v in critical_graph.nodes(data=True):
    #     if is_cuda_op(critical_graph.nodes, node):
    #         # print(node, v['avg'])
    #         all_cost += v['avg']
    return all_cost

# fetch communication time
def fetch_comm_information_from_profile_data(trace_name, target_trace_idx=0, pass_nums = 1):
    cpu_events, runtime_events, cuda_events, python_function_events, \
     user_annotations, MemsetEvents, MemcpyEvents = get_sorted_catogrized_events(trace_name)
    
    if LASTEST_TORCH:
        # latest torch use user_annotations for profilers
        profiler_step_events = get_profiler_step(user_annotations)
        optimizer_step_events = get_optimizer_step(user_annotations)
        optimizer_zrg_events = get_optimizer_zero_grad_step(user_annotations)
    else:
        # old torch use cpuops
        profiler_step_events = get_profiler_step(cpu_events) # in torch version < 1.10.1. 
        optimizer_step_events = get_optimizer_step(cpu_events)
        optimizer_zrg_events = get_optimizer_zero_grad_step(cpu_events)


    all_cuda_events = cuda_events
    # target_trace_idx
    # get comm
    comm_events = get_sorted_comm_events(trace_name)
    # filter cuda events
    # cuda_events, comm_events = filter_comm_events(cuda_events)
    # restrict trace within a profiler result
    cpu_events, runtime_events, cuda_events, python_function_events, \
     user_annotations, MemsetEvents, MemcpyEvents, comm_events, optimizer_step_events, optimizer_zrg_events = restrict_events_within_profiler_step(profiler_step_events, \
    events_records = [cpu_events, runtime_events, cuda_events, python_function_events, user_annotations, MemsetEvents, MemcpyEvents, comm_events, \
    optimizer_step_events, optimizer_zrg_events], idx=target_trace_idx)

    profiler_ts_end = profiler_step_events[target_trace_idx]['ts'] + profiler_step_events[target_trace_idx]['dur']
    if not config.CONSIDER_OPTIMIZER_TIME:
        # collect time for zero grad and 
        optimizer_step_start = optimizer_step_events[0]['ts']
        # check last comm event
        last_comm_event = comm_events[-1]
        last_end_ts = last_comm_event['ts'] + last_comm_event['dur']
        if last_end_ts > optimizer_step_start:
            gap = last_end_ts - optimizer_step_start
        else:
            gap = 0
        # dur_optimizer_zero_grad = optimizer_zrg_events[0]['dur']
        dur_optimizer_relevant = profiler_ts_end - optimizer_step_start - gap

    # scheduler_time = get_scheduler_time(python_function_events)

    # create mapping events
    mapping_events = cuda_events + MemsetEvents + MemcpyEvents 
    all_mapping_events = all_cuda_events + MemsetEvents + MemcpyEvents
    launch_kernel_events = [event for event in runtime_events if event['name'] not in useless_runtime_name]


    aten_events, aten_to_kernel_fwd, aten_events_before = get_fwd_aten_to_kernel(cpu_events, cuda_events, runtime_events, python_function_events, mapping_events)

    bwd_events, bwd_to_kernel = get_bwd_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events)
    comm_mem_events, comm_mem_to_kernel =  get_comm_mem_copy_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events)
    optimizer_step_events, optimizer_step_to_kernel = get_optimizer_step_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, mapping_events, optimizer_step_events[0])
    optimizer_zrg_events, optimizer_zrg_to_kernel = get_optimizer_step_aten_to_kernel(cpu_events, runtime_events, python_function_events, launch_kernel_events, all_mapping_events, optimizer_zrg_events[0])
    
    # CPU TIME
    aten_time = sum_duration_from_events(aten_events)
    bwd_start_dur = bwd_events[0]['ts'] - aten_events[0]['ts']
    gap_dur = bwd_start_dur - aten_time
    print("dur gap", gap_dur)


    comm_events = get_sorted_comm_events(trace_name)
    comm_events, cpu_events = restrict_events_within_profiler_step(profiler_step_events, \
    events_records = [comm_events, cpu_events], idx=target_trace_idx)

    first_aten_start = aten_events[0]['ts'] + gap_dur # real start time


    comm_data_pack = {}
    cnt = 0
    for comm_event in comm_events:
        cnt += 1
        if cnt == pass_nums: continue
        unique_id = get_unique_id_for_node(comm_event)
        avg = comm_event['dur']
        ts = comm_event['ts']
        relative_start_ts = ts - first_aten_start

        comm_data_pack[unique_id] = {
            'avg': avg /1000,
            'start': relative_start_ts /1000
        }
    
    id_mapper.clear()
    name_cnt.clear()

    return comm_data_pack

def calculate_real_duration_from_comm_data(comm_data_pack_1, comm_data_pack_2):
    keys = comm_data_pack_1.keys()
    for k in keys:
        data_1 = comm_data_pack_1[k]
        data_2 = comm_data_pack_2[k]
        start_1, start_2 = data_1['start'], data_2['start']
        real_start = max(start_1, start_2)
        gap_start_1 = real_start - start_1
        gap_start_2 = real_start - start_2
        dur_A = data_1['avg'] - gap_start_1
        dur_B = data_2['avg'] - gap_start_2
        comm_data_pack_1[k]['real_dur'] = dur_A
        comm_data_pack_2[k]['real_dur'] = dur_B

def fetch_real_duration_from_traces(traces_1, traces_2, target_trace_idx=0):
    
    # comm_file_name = '/qsync_niti_based/analytical_predictor/traces/bps/bert/fp16_fp32_comm_t4.json'
    comm_pack_1 = fetch_comm_information_from_profile_data(traces_1, target_trace_idx=target_trace_idx)
    comm_pack_2 = fetch_comm_information_from_profile_data(traces_2, target_trace_idx=target_trace_idx)
    calculate_real_duration_from_comm_data(comm_pack_1, comm_pack_2)
    return comm_pack_1, comm_pack_2

# 
def check_dag_bit(fwd_di_graph):
    for node_k, node_v in fwd_di_graph.nodes(data=True):
        print(node_k, node_v['bit'])
# dag relevant
def set_dag_uniform_bitwidth(dag_graph, bit):
    for node_k, node_v in dag_graph.nodes(data=True):
        node_v['bit'] = bit

def set_layers_uniform_bitwidth(dag_graph, layers, bit):
    for layer in layers:
        if not is_bitwidth_changeable_node(layer): continue 
        dag_graph.nodes[layer]['bit'] = bit 

def get_dag_bit_widths(dag_graph):
    dag_bit_mapper = {}
    for node_k, node_v in dag_graph.nodes(data=True):
        dag_bit_mapper[node_k] = node_v['bit']
    return dag_bit_mapper

def set_dag_bit_widths(dag_graph, dag_bit_mapper):
    for node_k, node_v in dag_graph.nodes(data=True):
        node_v['bit'] = dag_bit_mapper[node_k]




def calculate_mem_occuupation(subgraph, Mappers):
    sg_nodes = subgraph.nodes 
    node_to_layer_mapper, M = Mappers
    mem_sum = 0
    for node_k, node_v in sg_nodes.items():
        if is_appended_criterion_ops(node_k): continue
        if is_mem_relevant_ops(node_k):
            # TODO: fixed it later
            if node_k not in node_to_layer_mapper:
                mem_occupied = 1
            else:
                layer = node_to_layer_mapper[node_k]
                layer_M = M[layer]
                node_bit = node_v['bit']
                # print(node_k, is_implement_limit_ops(node_k), node_v['bit'])
                if is_implement_limit_ops(node_k) and node_bit < 16:
                    node_bit = 16
                mem_occupied = (node_bit / 32) * layer_M
            mem_sum += mem_occupied
    return mem_sum


def get_all_changeable_nodes_in_graph(subgraph):
    changable_nodes_list = []
    nodes = subgraph.nodes 
    for node in nodes:
        if is_bitwidth_changeable_node(node):
            changable_nodes_list.append(node)
    return changable_nodes_list