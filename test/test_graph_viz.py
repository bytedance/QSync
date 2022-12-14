import torch.nn as nn
import torch 
from torchviz import make_dot
from torchvision import models
import numpy as np 


# use torchviz got the bp graph
# use
# convert the 

model = models.resnet50()
x = torch.randn(10, 3, 224, 224)

y = model(x)
vis_graph_fwd = make_dot(model(x), params=dict(model.named_parameters()))
vis_graph_bwd = make_dot(y.mean(), params=dict(model.named_parameters()))

vis_graph_fwd.view()
vis_graph_bwd.view()

# import pydotplus
# import networkx

# def cvt_to_digraph(digraph):
#     dotplus = pydotplus.graph_from_dot_data(digraph.source)
#     # print(type(dotplus)) # prints <class 'pydotplus.graphviz.Dot'>
#     # if graph doesn't have multiedges, use dotplus.set_strict(true)
#     nx_graph = networkx.nx_pydot.from_pydot(dotplus)
#     # print(type(nx_graph))
#     return nx_graph

# cvt_to_digraph(vis_graph_fwd)