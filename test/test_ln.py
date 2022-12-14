from torch import nn 
import torch 
from qsync.LpTorch.layers import construct_qln_by_ln
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module
layer_norm(embedding)
# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W).cuda()
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W]).cuda()
output = layer_norm(input)

qln_ins = construct_qln_by_ln(layer_norm)
output2 = qln_ins(input)
print(torch.max(output2 - output))


fake_out = torch.randn_like(output)
loss_fn = nn.MSELoss()
loss_ori = loss_fn(fake_out, output)
loss_ori.backward()
loss = loss_fn(fake_out, output2)
loss.backward()
print(torch.max(layer_norm.weight.grad - qln_ins.weight.grad))

