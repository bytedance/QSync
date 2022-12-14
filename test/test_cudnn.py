import torch
import torch.nn as nn 
from torchvision import models
from qsync.LpTorch.layers import construct_qconv_by_conv
from utils import test_qconv_speed, test_qconv_speed_splited
from qsync.LpTorch.conf import config
from qsync.LpTorch.utils import convert_from_nchw_to_nhwc, convert_from_nhwc_to_nchw, close_metric
from qsync.LpTorch.QModule import QModule
from qsync.LpTorch.quant import qdq_int8, quantize_int8
 

import cudnn_conv

device = torch.device('cuda') # necessary, cutlass only allows gpu device
torch.manual_seed(1234)
# vgg second last result
BS = 512
inp = torch.rand(BS, 512, 14, 14, device=device)
model = models.vgg16().cuda()
op = model.features[28]


res0 = op(inp)
print(op.weight.shape)
print(res0.shape)

qweight, w_s = quantize_int8(op.weight)
qinp, inp_s = quantize_int8(inp)
padding_h, padding_w = op.padding 
stride_h, stride_w = op.stride 
dilation_h, dilation_w = op.dilation 
print(stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w)
qinp_nhwc = convert_from_nchw_to_nhwc(qinp)
qweight_nhwc = convert_from_nchw_to_nhwc(qweight)
res1 = convert_from_nchw_to_nhwc(res0)
nhwc_weight =  convert_from_nchw_to_nhwc(op.weight)
nhwc_inp = convert_from_nchw_to_nhwc(inp)

res = cudnn_conv.int8_conv(qinp_nhwc, qweight_nhwc, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, w_s * inp_s)
# res = cudnn_conv.int8_conv(inp, op.weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, 1)
res2 = cudnn_conv.fp32_conv_nhwc(nhwc_inp, nhwc_weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w)
print(torch.max(torch.abs(res1 - res)))
print(torch.max(torch.abs(res1 - res2)))

inp_cl = qinp.to(memory_format=torch.channels_last)
weight_cl = qweight.contiguous(memory_format=torch.channels_last)
res_cl = cudnn_conv.int8_conv_cl(inp_cl, weight_cl, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, w_s * inp_s)
print(res_cl.shape, res0.shape)
print(torch.max(torch.abs(res_cl - res0.to(memory_format=torch.channels_last))))
# Conversion from NCHW -> NCHW_VEC_T hasn't been debugged
# res = cudnn_conv.int8_conv_nchw(qinp, qweight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, w_s * inp_s)

