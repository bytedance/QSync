import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from qsync.LpTorch.utils import compute_tensor_bytes
from qsync.LpTorch.layers import construct_qlinear_by_linear
from qsync.LpTorch.layers import QAdd, QMatmul, QDropout
from qsync import QModule
from qsync.LpTorch.env import TransGELU

qadd = QAdd()
# a = torch.rand(1024, 1024//32, requires_grad=True).cuda()
# b = torch.rand(1024, 1024//32, requires_grad=True).cuda()
# print(torch.cuda.memory_allocated())
# c = a + b 
# print(torch.cuda.memory_allocated())
# c_fake = torch.rand_like(c)

# loss_fn = nn.MSELoss()
# loss = loss_fn(c, c_fake)
# print(torch.cuda.memory_allocated())

# loss.backward()
# inp1 = torch.rand(10, 768, 768, requires_grad=True).cuda()
# inp2 = torch.rand(10, 768, 768, requires_grad=True).cuda()


# trans_gelu_ins = TransGELU()
# trans_gelu_ins2 = TransGELU()

# QMat1 = QMatmul()
# QMat2 = QMatmul()

# weight1 = torch.rand(768, 768, requires_grad=True).cuda()
# weight2 = torch.rand(768, 768, requires_grad=True).cuda()


# mem_1 = torch.cuda.memory_allocated()
# # res3 = F.relu(inp1)
# # res3 = F.dropout(inp1, p = 0.1)
# # res3 = trans_gelu_ins(inp1)
# # res3 = F.gelu(inp1)
# # res3 = F.softmax(inp1)

# res3 = QMat1(inp1, weight1)

# mem_2 = torch.cuda.memory_allocated()
# gap_activation1 = mem_2 - mem_1

# res4 = QMat2(inp2, weight2)
# # res4 = F.dropout(inp2, p = 0.1)
# # res4 = trans_gelu_ins2(inp2)
# # res4 = F.relu(inp2)
# # res4 = F.gelu(inp2)
# # res4 = F.softmax(inp2)
# mem_3 = torch.cuda.memory_allocated()
# # res3 = qadd(inp1 , inp2)
# gap_activation2 = mem_3 - mem_2
# print(gap_activation1, gap_activation2)
# print((gap_activation2 - compute_tensor_bytes([res3])) / gap_activation2)

# test_li = nn.Linear(768,768).cuda()

# class test_mod(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.name = "fuck"
#         self.l1 = nn.Linear(768,768).cuda()
#         self.l2 = nn.Linear(768,768).cuda()
#         self.l3 = nn.Linear(768,768).cuda()

#     def forward(self, inp):
#         res = 0
#         x = self.l1(inp)
#         res += compute_tensor_bytes([inp, self.l1.weight])
#         x = self.l2(x)
#         res += compute_tensor_bytes([x, self.l2.weight])
#         x = self.l3(x)
#         res += compute_tensor_bytes([x, self.l3.weight])
#         print(res)
#         return x 

# inp = torch.rand(10, 768, 768).cuda()
# test_mod = test_mod()
# print(torch.cuda.memory_allocated())
# q_test_mod = QModule(test_mod)
# print(torch.cuda.memory_allocated())
# last_allocate = torch.cuda.memory_allocated()
# res = q_test_mod(inp)
# cur_allocate = torch.cuda.memory_allocated()
# print(cur_allocate - last_allocate)


#  construct_qlinear_by_linear(test_li)
# res = test_li(inp)
# cur_allocate = torch.cuda.memory_allocated()
# print((cur_allocate - last_allocate) / (1024 ** 3))
# print(cur_allocate)
# print(compute_tensor_bytes([test_li.weight, test_li.weight, test_li.weight, inp, inp, res]))
# print("compute tensors")


# q_li = construct_qlinear_by_linear(test_li)
# res_2 = q_li(inp)
# new_allocate = torch.cuda.memory_allocated()
# print((new_allocate - cur_allocate) / (1024 ** 3))

# res3 = inp1 + inp2



class test_mod_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "fuck"
        self.QMat1 = QMatmul()
        self.QAdd1 = QAdd()
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax()
        self.GELU = TransGELU()
        self.RELU = nn.ReLU()
        
        self.l2 = nn.Linear(768,768).cuda()
        self.l1 = nn.Linear(768,768).cuda()

    def forward(self, inp1):
        res = 0
        x = self.l1(inp)
        # res += compute_tensor_bytes([self.l1.weight])
        x2 = self.l2(inp)
        # res += compute_tensor_bytes([self.l2.weight])
        # x3 = self.RELU(x)
        # x3 = self.softmax(x)
        # x3 = self.GELU(x)
        # x3 = self.QMat1(x, x2)
        x3 = self.QAdd1(x, x2)
        # x3 = self.drop(x)
        res += compute_tensor_bytes([x3])
        # res += compute_tensor_bytes([x3, x, x2])
        # res += compute_tensor_bytes([x3]) + compute_tensor_bytes([x]) * 0.25
        # res += compute_tensor_bytes([x3, x]) 
        print(res)
        return x3


inp = torch.rand(768, 768).cuda()
test_mod = test_mod_2()
q_test_mod = QModule(test_mod)
print("start", torch.cuda.memory_allocated())
last_allocate = torch.cuda.memory_allocated()
res = q_test_mod(inp)
cur_allocate = torch.cuda.memory_allocated()
print(cur_allocate - last_allocate)
import pdb; pdb.set_trace()
