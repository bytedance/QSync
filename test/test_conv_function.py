'''
    Test fp/bp functionality for cutlass conv functions
'''

import torch
import torch.nn as nn 
from torchvision import models

from utils import test_qconv_speed, test_qconv_speed_splited
from qsync import QModule
from qsync.LpTorch.conf import config 
from qsync.LpTorch.layers import construct_qconv_by_conv
from qsync.LpTorch.utils import convert_from_nchw_to_nhwc, convert_from_nhwc_to_nchw, close_metric
from qsync.LpTorch.quant import qdq_int8
 

device = torch.device('cuda') # necessary, cutlass only allows gpu device
torch.manual_seed(1234)
# vgg second last result
BS = 64
inp = torch.rand(BS, 512, 14, 14, device=device)
model = models.vgg16().cuda()
op = model.features[28]

# inp = torch.rand(128, 64, 56, 56,  device=device)
# model = models.resnet18().cuda()
# op = model.layer2[0].downsample[0]
# print(op.padding, op.dilation)
# BS = 64
# inp = torch.rand(BS, 64, 56, 56, device=device)
# model = models.resnet50().to(device)
# op = model.layer1[0].conv1

# print(op.weight.shape)


# import pdb; pdb.set_trace()

# inp = torch.rand(BS, 3, 224, 224, device=device)
# model = models.alexnet().cuda()
# op = model.features[0]
# inp = torch.rand(BS, 3, 224, 224, device=device)
# model = models.resnet50().cuda()
# op = model.conv1
# print(op.weight.shape)

# channel last and cudnn
config.channels_last = True
if config.channels_last:
    inp = inp.to(memory_format=torch.channels_last)
    op = op.to(memory_format=torch.channels_last)

def test_qconv_fp_bp():
    fp_res = op(inp) # fp32 result in python

    # construct a new qconv op based on the op
    q_conv = construct_qconv_by_conv(op)

    available_bits = config.available_bits

    print(f"Test {available_bits} qconv fp functionality & result")

    # define loss function
    loss_fn = nn.MSELoss()
    
    # random qconv output with same shape
    rand_out = torch.randn_like(fp_res)
    fp_test_loss = loss_fn(rand_out, fp_res)
    fp_test_loss.backward()
    grad_fp32 = op.weight.grad # fp32 gradient

    # 32bit test
    if 32 in available_bits:
        res_qconv_32 = q_conv(inp)
        print('output diff fp32', close_metric(res_qconv_32, fp_res))
        # import pdb; pdb.set_trace()
        cutlass_fp32_loss = loss_fn(rand_out, res_qconv_32)
        cutlass_fp32_loss.backward()
        print("here")
        cutlass_fp32_grad = q_conv.weight.grad
        if not config.channels_last:
            cutlass_fp32_grad = convert_from_nhwc_to_nchw(cutlass_fp32_grad)
        print("gradient diff fp32", torch.max(cutlass_fp32_grad - grad_fp32))

    # tf32
    if 19 in available_bits:
        q_conv.reset_bits(19)
        res_qconv_19 = q_conv(inp)
        diff = res_qconv_19 - fp_res
        print('output diff tf32', close_metric(res_qconv_19, fp_res))

        cutlass_fp19_loss = loss_fn(rand_out, res_qconv_19)
        cutlass_fp19_loss.backward()
        cutlass_fp19_grad = q_conv.weight.grad
        if not config.channels_last:
            cutlass_fp19_grad = convert_from_nhwc_to_nchw(cutlass_fp19_grad)
        print("gradient diff tf32", torch.max(cutlass_fp19_grad - grad_fp32))

    if 16 in available_bits:
        q_conv.reset_bits(16)
        res_qconv_16 = q_conv(inp)
        diff = res_qconv_16 - fp_res
        print('output diff fp16', close_metric(res_qconv_16, fp_res))

        if config.half_to_half and config.channels_last:
            rand_out_half = rand_out.half()
        else:
            rand_out_half = rand_out
        
        cutlass_fp16_loss = loss_fn(rand_out_half, res_qconv_16)
        cutlass_fp16_loss.backward()
        cutlass_fp16_grad = q_conv.weight.grad
        if not config.channels_last:
            cutlass_fp16_grad = convert_from_nhwc_to_nchw(cutlass_fp16_grad)
        print("gradient diff fp16", torch.max(cutlass_fp16_grad - grad_fp32))

    if 8 in available_bits:
        q_conv.reset_bits(8)
        res_qconv_8 = q_conv(inp)
        diff = res_qconv_8 - fp_res
        print('output diff int8', close_metric(res_qconv_8, fp_res))
        cutlass_int8_loss = loss_fn(rand_out, res_qconv_8)
        cutlass_int8_loss.backward()
        cutlass_int8_grad = q_conv.weight.grad
        if not config.channels_last:
            cutlass_int8_grad = convert_from_nhwc_to_nchw(cutlass_int8_grad)
        print("gradient diff int8", torch.max(cutlass_int8_grad - grad_fp32))
    
    if 4 in available_bits:
        q_conv.reset_bits(4)
        res_qconv_4 = q_conv(inp)
        diff = res_qconv_4 - fp_res
        print('output diff int4', close_metric(res_qconv_4, fp_res))

        if config.half_to_half and config.channels_last:
            rand_out = rand_out.half()


        cutlass_int4_loss = loss_fn(rand_out, res_qconv_4)
        cutlass_int4_loss.backward()
        cutlass_int4_grad = q_conv.weight.grad
        if not config.channels_last:
            cutlass_int4_grad = convert_from_nhwc_to_nchw(cutlass_int4_grad)
        print("gradient diff int4", torch.max(cutlass_int4_grad - grad_fp32))





    
        

def test_bp_full_model():
    cp_model = models.vgg16()
    model_ = QModule(cp_model)
    model_.cuda()
    inp_ = torch.rand(BS, 3, 224, 224, device=device)
    res = model_(inp_)
    
    rand_out = torch.rand(res.shape).cuda()
    loss_fn = nn.MSELoss()
    loss = loss_fn(res, rand_out)
    loss.backward()

# check quantized kernel is right
import copy
def test_qconv_dqd_conv():
    fp_res = op(inp)

    q_conv = construct_qconv_by_conv(op)
    q_conv.reset_bits(8)
    q_conv.cuda()
    q_out = q_conv(inp)

    op_new = copy.deepcopy(op)
    op_new.weight = nn.Parameter(qdq_int8(op_new.weight))
    qdq_inp = qdq_int8(inp)

    qdq_out = op_new(qdq_inp)
    print("int kernel vs.", torch.max(torch.abs(q_out - qdq_out)))
    print("torch kernel vs torch out vs.", torch.max(torch.abs(q_out - fp_res)))
    print("cutlass kernel vs torch out vs.", torch.max(torch.abs(fp_res - qdq_out)))

def test_conv_function():
    # test_qconv_dqd_conv() # V100 cannot run this part
    test_qconv_fp_bp()
    # test_bp_full_model()
    # test_qconv_speed(q_conv, inp)
    test_qconv_speed_splited(op, inp)

if __name__ == '__main__':
    test_conv_function()