import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models
from qsync.LpTorch.conf import config
from qsync.LpTorch.layers import construct_qconv_by_conv
from utils import test_qconv_speed, test_qconv_speed_splited

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
device = torch.device('cuda') # necessary, cutlass only allows gpu device
torch.manual_seed(1234)
# vgg second last result
BS = 512
inp = torch.rand(BS, 512, 14, 14, device=device)
model = models.vgg16().cuda()
op = model.features[28]
config.simu = True
op = construct_qconv_by_conv(op)
qconv_weight = op.weight

torch.tensor(0).cuda()

conversion_cost, computation_cost = [], []
bp_conversion_cost = []
bit_width_column = []
cvt_Duration = 0
bp_cost = 0

warmup_times = 10
iter_times = 10
# warmup
for idx in range(warmup_times):
    op(inp)

start.record()
for idx in range(iter_times):
    op(inp)
end.record()
torch.cuda.synchronize()
Duration = start.elapsed_time(end)
print("fp32 Cost", Duration / iter_times) # fp32 result here is not correct due to the cachine of torch

conversion_cost.append(cvt_Duration / iter_times)
bp_conversion_cost.append(bp_cost / iter_times)
computation_cost.append(Duration / iter_times)
bit_width_column.append(32)


start.record()
for _ in range(iter_times):
    fp_qconv_inp = inp.half()
    fp_qconv_weight = qconv_weight.half()

end.record()
torch.cuda.synchronize()
cvt_Duration = start.elapsed_time(end)
print("fp16 conversion", cvt_Duration / iter_times)

fp_qconv_inp_cp = fp_qconv_inp.detach().clone()
start.record()
for _ in range(iter_times):
    fp_qconv_inp_cp.float()

end.record()
torch.cuda.synchronize()
bp_cost = start.elapsed_time(end)
print("fp16 bp conversion", bp_cost)
    
f_op = op.half()
start.record()
for _ in range(iter_times):
    f_op(fp_qconv_inp)
end.record()
torch.cuda.synchronize()
Duration = start.elapsed_time(end)
print("fp16 Cost", Duration / iter_times)

conversion_cost.append(cvt_Duration / iter_times)
bp_conversion_cost.append(bp_cost / iter_times)
computation_cost.append(Duration / iter_times)
bit_width_column.append(16)




BS, M, K = 12, 512, 768
in_channels = out_channels = K
inp = torch.rand(BS, M, K).cuda()


op = nn.Linear(in_channels, out_channels).cuda()


start.record()
for idx in range(iter_times):
    op(inp)
end.record()
torch.cuda.synchronize()
Duration = start.elapsed_time(end)
print("fp32 Cost Linear", Duration / iter_times) # fp32 result here is not correct due to the cachine of torch

conversion_cost.append(cvt_Duration / iter_times)
bp_conversion_cost.append(bp_cost / iter_times)
computation_cost.append(Duration / iter_times)
bit_width_column.append(32)


start.record()
for _ in range(iter_times):
    fp_qli_inp = inp.half()
    fp_qli_weight = op.weight.half()

end.record()
torch.cuda.synchronize()
cvt_Duration = start.elapsed_time(end)
print("fp16 conversion Linear", cvt_Duration / iter_times)

fp_qli_inp_cp = fp_qli_inp.detach().clone()
start.record()
for _ in range(iter_times):
    fp_qli_inp_cp.float()

end.record()
torch.cuda.synchronize()
bp_cost = start.elapsed_time(end)
print("fp16 bp conversion Linear", bp_cost)
    
f_op = op.half()
start.record()
for _ in range(iter_times):
    f_op(fp_qli_inp)
end.record()
torch.cuda.synchronize()
Duration = start.elapsed_time(end)
print("fp16 Cost Linear", Duration / iter_times)
