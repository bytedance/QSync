import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from qsync.LpTorch.layers import construct_qmaxpool_by_maxpool, construct_qbn_by_bn
# pool of square window of size=3, stride=2
m = nn.MaxPool2d(3, stride=2)
# pool of non-square window
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m(input)
# print(output)

qm = construct_qmaxpool_by_maxpool(m)
output2 = qm(input.to(memory_format = torch.channels_last))
print(torch.max(output2 - output))


# With Learnable Parameters
bn = nn.BatchNorm2d(100)
# Without Learnable Parameters
input = torch.randn(20, 100, 35, 45)
output1 = bn(input)

# With Learnable Parameters
qbn = construct_qbn_by_bn(bn)
output2 = qbn(input.to(memory_format = torch.channels_last))
print(torch.max(output2 - output1))
# print(output2)
# bn = bn.to(memory_format=torch.channels_last)
# # print(bn.running_mean.shape, bn.running_var.shape, bn.weight.shape, bn.bias.shape)
# output3 = F.batch_norm(input.to(memory_format = torch.channels_last),
#             # If buffers are not to be tracked, ensure that they won't be updated
#             bn.running_mean,
#             bn.running_var,
#             bn.weight, bn.bias, True, bn.momentum, bn.eps)

# print(torch.max(output3 - output1))

# target output size of 5x7
m = nn.AdaptiveAvgPool2d((5,7))
input = torch.randn(1, 64, 8, 9)
output1 = m(input)
# target output size of 7x7 (square)
output2 = m(input.to(memory_format = torch.channels_last))
print(torch.max(output2 - output1))
m = m.to(memory_format = torch.channels_last) 
# output3 = m(input.to(memory_format = torch.channels_last))
# print(torch.max(output3 - output1))