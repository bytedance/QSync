import networkx as nx 

from .core import (
    get_all_changeable_nodes_in_graph,
    map_dag_bit_to_critical_graph
)


def is_isomorphic_graph_in_graphs(G1, subg):
    G2, partition_node = subg
    if nx.is_isomorphic(G1, G2):
        return True
    return False

def is_isomorphic_graph_in_graphs_sole(G1, subg):
    G2 = subg
    if nx.is_isomorphic(G1, G2):
        return True
    return False

def is_isomorphic_within_graphs(G1, subgs):
    for subg in subgs:
        if is_isomorphic_graph_in_graphs(G1, subg):
            return True
    return False


def get_value_from_iso_g(G1, subgs_dict):
    for subg in subgs_dict:
        if is_isomorphic_graph_in_graphs_sole(G1, subg):
            return subgs_dict[subg]
    return None

def DFS_to_find_blocks(fwd_di_graph):
    graph_blocks = []
    parititoned_second_graph = fwd_di_graph
    bfs_next_node = []
    current_graph_nodes = []
    ascending_list = nx.topological_sort(fwd_di_graph)
    for node in ascending_list:
        current_graph_nodes.append(node)
        if len(fwd_di_graph.edges(node)) == 0: 
            sub_graph = fwd_di_graph.subgraph(current_graph_nodes)
            graph_blocks.append((sub_graph, node))
            return graph_blocks # end of graph
        if node in bfs_next_node:
            node_idx = bfs_next_node.index(node)
            bfs_next_node.pop(node_idx)
            if len(bfs_next_node) == 0:
                # print("do partition", node)
                sub_graph = fwd_di_graph.subgraph(current_graph_nodes)
                
                graph_blocks.append((sub_graph, node))
                current_graph_nodes = []
        succs = fwd_di_graph.succ[node]
        [bfs_next_node.append(i) for i in succs if i not in bfs_next_node]
        
    return graph_blocks


def find_unique_blocks(graph_blocks, fwd_di_graph):
    # print(len(graph_blocks))
    # Merge graph with only one node and not bitwidth relevant to the previous node
    merged_graphs = []
    for (sub_graph, partition_node) in graph_blocks:
        nodes = sub_graph.nodes 
        len_node = len(nodes)
        if len(get_all_changeable_nodes_in_graph(sub_graph)) == 0:
            # print("merged", nodes)
            last_graph = merged_graphs.pop(-1)
            l_nodes = last_graph[0].nodes
            new_nodes = list(l_nodes) + list(nodes)
            new_sub_graph = fwd_di_graph.subgraph(new_nodes)
            merged_graphs.append((new_sub_graph, list(nodes)[0]))
        else:
            merged_graphs.append((sub_graph, partition_node))
    
    # unique_graphs = []
    # # import networkx.algorithms.isomorphism as iso
    # for (sub_graph, partition_node) in merged_graphs:
    #     if len(unique_graphs) == 0:
    #         unique_graphs.append((sub_graph, partition_node))
    #         continue
    #     if not is_isomorphic_within_graphs(sub_graph, unique_graphs):
    #         unique_graphs.append((sub_graph, partition_node))
    # return unique_graphs

    return merged_graphs


# subgraph or prediction
def predict_sub_graph_latency(fwd_di_graph, start_node, end_node, critical_graph, dag_2_nodes, mapper_dict, _debug_level=2):
    critical_graph = copy.deepcopy(critical_graph)
    map_dag_bit_to_critical_graph(fwd_di_graph, dag_2_nodes, critical_graph, mapper_dict)
    # we don't set CPU gap here to make a pure comparison on the data
    res = predict(critical_graph, _debug_level=_debug_level)
    return res 