from qsync.LpTorch.quant import quantize_int4_optimized, quantize_int4, dequantize_int4, get_amax, get_scale
import torch
import numpy as np
from qsync.LpTorch.conf import config
from torchvision import models
from qsync.LpTorch.utils import convert_from_nchw_to_nhwc, convert_from_nhwc_to_nchw, close_metric
from qsync.LpTorch.functions import *

def simple_test():
    config.optimized_int = False
    # A = torch.rand(512, 14, 14, 512).cuda()
    A = torch.load("q_input.pt").cuda()
    A_shape = A.shape
    PACTED, scale = quantize_int4(A)
    # print(PACTED)
    DE_A = dequantize_int4(PACTED, A_shape, scale)
    # print(DE_A, scale)


    # torch qdq
    amax_A = get_amax(A)
    scale = get_scale(amax_A, 4)
    q_A = torch.div(A, scale, rounding_mode='floor')
    # print(q_A)
    dq_A = q_A * scale
    # print(dq_A, scale)
    print(torch.max(dq_A - DE_A))

def test_op():
    BS = 256
    inp = torch.rand(BS, 512, 14, 14).cuda()
    model = models.vgg16().cuda()
    op = model.features[28]
    from qsync.LpTorch.layers import construct_qconv_by_conv
    q_conv = construct_qconv_by_conv(op)
    q_conv.reset_bits(4)
    res_qconv_4 = q_conv(inp)
    print(res_qconv_4)
    inp = torch.rand(512, 512, 14, 14).cuda()
    res_qconv_4 = q_conv(inp)
    print(res_qconv_4)

    # stride_h, stride_w = op.stride
    # padding_h, padding_w = op.padding


    # (output, packed_input, packed_weight), scale_inp, scale_w, shape_inp, shape_w = conv2d_int4(inp, op.weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0)
    # print(output, packed_input, packed_weight)
    # A = torch.load("q_input.pt").cuda()
    # PACTED, scale = quantize_int4(A)
    # W = torch.load("q_weight.pt").cuda()
    # PACTED_W, scale = quantize_int4(W)
    # print(packed_input - PACTED)
    # print(packed_weight - PACTED_W)
    # config.simu = True
    # q_conv.convert_to_nchw()
    # res_qconv_4_simu = q_conv(inp)
    # diff = res_qconv_4 - res_qconv_4_simu
    # print('output diff int4', close_metric(res_qconv_4, res_qconv_4_simu))
# simple_test()
# simple_test()
test_op()
# exit()
# A_shape = torch.load("q_input_shape.pt")
# # print(PACTED)
# DE_A = dequantize_int4(PACTED, A_shape, scale)
# # print(DE_A)
# A = torch.rand(512, 14, 14, 512).cuda()
# PACTED, scale = quantize_int4(A)
# print(PACTED)
# print(A)
# PACTED, scale = quantize_int4(A)
# scale = 0.0008
# W = torch.load("q_weight.pt").cuda()
# print(A_shape)
# N = 1
# for i in A_shape:
#     N *= i
# print(N // 2 + 1)
# print(PACTED_LOAD.numel())
# DE_A = dequantize_int4(PACTED_LOAD, A_shape, scale)
# print(DE_A)
