import torch
import torch.nn as nn
import cutlasslinear_cuda
import time
from qsync.LpTorch.conf import config
from qsync.LpTorch.quant import quantize_int8, dequantize_int8
from qsync.LpTorch.layers import QLinear, construct_qlinear_by_linear
from utils import test_li_speed, test_qlinear_speed_splited

device = torch.device('cuda') # necessary, cutlass only allows gpu device
# BS = 12
# # A: NHW * RSC 512 * 14 * 14 * 3 * 3 * 512
# # B: RSC * K 3 * 3 * 512 * 512
# in_channels = 512
# out_channels = 12 # here, must be multiple of 16

# IN BERT
# torch.Size([12, 384, 3072]) torch.Size([768, 3072])
# torch.Size([12, 384, 768]) torch.Size([768, 768])

# INP_DIM = (768, 4608)
# WEIGHT_DIM = (3072, 4608)

INP_DIM = (12, 384, 768)
WEIGHT_DIM = (768, 768)

# in roberta
# torch.Size([64, 61, 768]) torch.Size([3072, 768])
# torch.Size([64, 61, 3072]) torch.Size([768, 3072])

# INP_DIM = (64, 61, 768)
# WEIGHT_DIM = (3072, 768)

# INP_DIM = (64, 61, 3072)
# WEIGHT_DIM = (768, 3072)

if len(INP_DIM) == 3:
    BS, M, K = INP_DIM
    out_channels, in_channels = WEIGHT_DIM
    inp = torch.rand(BS, M, K).cuda()
else:
    M, K = INP_DIM
    out_channels, in_channels = WEIGHT_DIM
    inp = torch.rand(M, K).cuda()

op = nn.Linear(in_channels, out_channels).cuda()

def diff(cut_lass_out, ori_out):
    print(torch.max((cut_lass_out - ori_out) / ori_out))
    print(torch.max(cut_lass_out - ori_out))


def test_functionality():

    # create data
    fp_res = op(inp) # fp32 result in python
    # construct a new qconv op based on the op
    qli = construct_qlinear_by_linear(op)
    
    available_bits = config.available_bits
    if 32 in available_bits:
        qli.reset_bits(32)
        fp32_out = qli(inp)

        print("fp32 max gap", torch.max(fp_res - fp32_out))
    
    if 19 in available_bits:
        qli.reset_bits(19)
        tf32_out = qli(inp)

        print("tf32 max gap", torch.max(fp_res - tf32_out))
    
    if 16 in available_bits:
        qli.reset_bits(16)
        half_out = qli(inp)
        print("half max gap", torch.max(fp_res - half_out))

    if 8 in available_bits:
        qli.reset_bits(8)
        int8_out = qli(inp)
        print("int8 max gap", torch.max(fp_res - int8_out))

    

    
    # print(fp_res.shape)
    
    
    # test bp
    loss_fn = nn.MSELoss()
    rand_out = torch.randn_like(fp_res)
    fp_test_loss = loss_fn(rand_out, fp_res)
    fp_test_loss.backward()
    grad_fp32 = op.weight.grad # fp32 gradient

    if 8 in available_bits:
        qli.reset_bits(8)
        loss = loss_fn(rand_out, int8_out)
        loss.backward()
        grad_int8 = qli.weight.grad
        # grad_int8 = torch.t(grad_int8)
        print("int8 grad max gap", torch.max(grad_int8 - grad_fp32))

    if 16 in available_bits:
        qli.reset_bits(16)
        if config.half_to_half:
            rand_out_half = rand_out.half()
        else:
            rand_out_half = rand_out
        loss = loss_fn(rand_out_half, half_out)
        loss.backward()
        grad_half = qli.weight.grad
        # grad_half = torch.t(grad_half)
        print("half grad max gap", torch.max(grad_half - grad_fp32))

    #     torch_res = torch.matmul(inp.half(), qli.weight.half())
    #     print(torch.max(ori_out - torch_res))

    # if 19 in available_bits:
    #     qli.reset_bits(19)
    #     tf32_out = qli(inp)

    #     print("tf32 max gap", torch.max(ori_out - tf32_out))

    # if 32 in available_bits:
    #     qli.reset_bits(32)
    #     fp32_out = qli(inp)

    #     print("fp32 max gap", torch.max(ori_out - fp32_out))





    

def test_op_speed():
    # create data
    inp = torch.rand(4, 32, 512).cuda()
    weight = torch.rand(512,512).cuda()
    bias = torch.tensor([0]).cuda()
    ori_out = torch.matmul(inp, weight)

    # weight transpose
    weight_transposed = weight.transpose(1, 0).contiguous() # usually not costful since in Transformer, K = N
    print(weight_transposed.shape)
    # int8
    inp_q, inp_s = quantize_int8(inp)
    weight_q, weight_s = quantize_int8(weight_transposed)
    inp_half = inp.half()

    available_bits = config.available_bits
    
    T = 100
    # float
    if 32 in available_bits:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(T):
            weight_transposed = weight.transpose(1, 0).contiguous()
            out_fp = cutlasslinear_cuda.gemm_float(inp, weight_transposed, bias)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print('float op', Duration)

    weight_half = weight_transposed.half()
    # half
    if 16 in available_bits:
        start.record()
        for _ in range(T):
            out_half = cutlasslinear_cuda.gemm_half(inp_half, weight_half, bias)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print('half op', Duration)

    if 8 in available_bits:
        start.record()
        for _ in range(T):
            out_int8 = cutlasslinear_cuda.gemm_int8(inp_q, weight_q, bias)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print('INT op', Duration)

    
    

if __name__ == '__main__':
    test_functionality()
    # test_op_speed()
    # test_li_speed(qli, inp)
    qli = construct_qlinear_by_linear(op)
    test_qlinear_speed_splited(qli, inp)