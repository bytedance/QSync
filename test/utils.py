import time, torch
from qsync.LpTorch.conf import config
import pandas as pd
from qsync.LpTorch.layers import construct_qconv_by_conv
from qsync.LpTorch.quant import quantize_int8, dequantize_int8, quant_with_scale
from qsync.LpTorch.functions import *
from qsync.LpTorch.utils import convert_from_nchw_to_nhwc
import os
import torch.nn.functional as F 
import cuda_quantization.quant as cuda_q 

def export_to_csv(file_name, df):
    return 
    data_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "DataAnalysis", "data"))
    file_path = f'{data_path}/{file_name}.csv'
    df.to_csv(file_path, index=False)

print_assert = False
# int8 requires tensor-core, when i<4, we can only use F
def use_torch_for_case(funct, input, weight, bias, stride_h, stride_w, padding_h, padding_w, dq_scale=1):
    global print_assert
    # # tensore core cannot be used
    # if sum([input.shape[0] % 4 + input.shape[1] % 4 if config.channels_last else input.shape[3] % 4]):
    #     if not print_assert:
    #         print("*******ALERT - Use torch functional due to constraints i>4, bit < 16 is not available*****")
    #         print_assert = True
    #     if input.dtype == torch.int8:
    #         return None # no implementation
    #     res = F.conv2d(input, weight, bias, stride=(stride_h, stride_w), padding=(padding_h, padding_w))
    # else:
    res = funct(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=dq_scale)
    return res 
    
iter_times = 100
warm_times = 10
def test_qconv_speed(q_conv, inp):
    config.nchw = False
    inp = convert_from_nchw_to_nhwc(inp)

    torch.tensor(0).cuda()
    print("Test execution speed for each conv")
    available_bits = config.available_bits

    if 32 in available_bits:
        q_conv.reset_bits(32)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # warm up
        for _ in range(warm_times):
            q_conv(inp)

        start.record()
        for _ in range(iter_times):
            q_conv(inp)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("fp32 Cost", Duration)

    if 19 in available_bits:
        q_conv.reset_bits(19)

        # warm up
        for _ in range(warm_times):
            q_conv(inp)

        start.record()
        for _ in range(iter_times):
            q_conv(inp)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("tf32 Cost", Duration)

    if 16 in available_bits:
        q_conv.reset_bits(16)

        # warm up
        for _ in range(warm_times):
            q_conv(inp)

        start.record()
        for _ in range(iter_times):
            q_conv(inp)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("fp16 Cost", Duration)

    if 8 in available_bits:
        q_conv.reset_bits(8)

        # warm up
        for _ in range(warm_times):
            q_conv(inp)

        start.record()
        for _ in range(iter_times):
            q_conv(inp)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("int8 Cost", Duration)



import copy 
# test splited conv speed
def test_qconv_speed_splited(p_op, p_inp, csv_file_name="conv_cost", dq_fusion=True):

    op = p_op
    inp = p_inp.detach().clone()
    # construct a new qconv op based on the op
    qconv = construct_qconv_by_conv(op)
    qconv.to(inp.device) # move it to device

    available_bits = config.available_bits

    config.nchw = False
    if not config.channels_last:
        inp = convert_from_nchw_to_nhwc(inp)
    else:
        inp = inp.to(memory_format=torch.channels_last)
    torch.tensor(0).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    print("Test splited execution speed for each conv")
    qconv_weight, qconv_stride, qconv_padding = qconv.weight, qconv.stride, qconv.padding
    qconv_stride_h, qconv_stride_w = qconv_stride
    qconv_padding_h, qconv_padding_w = qconv_padding

    conversion_cost, computation_cost = [], []
    bp_conversion_cost = []
    bit_width_column = []
    cvt_Duration = 0
    bp_cost = 0

    # print(qconv_weight.shape, inp.shape)


    # warmup
    for idx in range(warm_times):
        use_torch_for_case(conv2d_fp32, inp, qconv_weight, None, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, 1)
        # conv2d_fp32(inp, qconv_weight, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, 1)

    if 32 in available_bits:
        start.record()
        for _ in range(iter_times):
            use_torch_for_case(conv2d_fp32, inp, qconv_weight, None, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, 1)
            # conv2d_fp32(inp, qconv_weight, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, 1)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("fp32 Cost", Duration / iter_times)

        conversion_cost.append(cvt_Duration / iter_times)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(32)


    if 19 in available_bits:
        start.record()
        for _ in range(iter_times):
            conv2d_tf32(inp, qconv_weight, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, 1)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("tf32 Cost", Duration)

        conversion_cost.append(cvt_Duration / iter_times)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(19)


    if 16 in available_bits:
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
        print("fp16 bp conversion", bp_cost / iter_times)
    
    
        start.record()
        for _ in range(iter_times):
            use_torch_for_case(conv2d_fp16, fp_qconv_inp, fp_qconv_weight, None, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, 1)
            # conv2d_fp16(fp_qconv_inp, fp_qconv_weight, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, 1)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("fp16 Cost", Duration / iter_times)

        conversion_cost.append(cvt_Duration / iter_times)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(16)

    if 8 in available_bits:
        start.record()
        int_qconv_inp, s_1 = quantize_int8(inp)
        int_qconv_weight, s_2 = quantize_int8(qconv_weight)
        s_o = s_1 * s_2
        for _ in range(iter_times):
            int_qconv_inp = quant_with_scale(inp, s_1)
            int_qconv_weight =  quant_with_scale(qconv_weight, s_2)

        end.record()
        torch.cuda.synchronize()
        cvt_Duration = start.elapsed_time(end)
        print("int8 conversion", cvt_Duration / iter_times)

        start.record()
        for _ in range(iter_times):
            dequantize_int8(int_qconv_inp, s_1)
        end.record()
        torch.cuda.synchronize()
        bp_cost = start.elapsed_time(end)
        print("int8 bp conversion", bp_cost / iter_times)
        
        if not dq_fusion:
            s_o = 1
        start.record()
        for _ in range(iter_times):
            res = use_torch_for_case(conv2d_int8, int_qconv_inp, int_qconv_weight, None, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, s_o)
            # res = conv2d_int8(int_qconv_inp, int_qconv_weight, qconv_stride_h, qconv_stride_w, qconv_padding_h, qconv_padding_w, s_o)
            if not dq_fusion and res is not None:
                dequantize_int8(res, s_o)

        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("int8 Cost", Duration / iter_times)

        


        conversion_cost.append(cvt_Duration / iter_times)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(8)

    df = pd.DataFrame(data={"bit":bit_width_column,"cvt_cost":conversion_cost, "cpt_cost":computation_cost, 'bp_cost': bp_conversion_cost})
    export_to_csv(csv_file_name, df)
    print(df)




def test_li_speed(qli, inp):
    print("Test execution speed for each linear")
    qli.reset_bits(32)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        qli(inp)
    end.record()
    torch.cuda.synchronize()
    Duration = start.elapsed_time(end)
    print("fp32 Cost", Duration)

    qli.reset_bits(19)
    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        qli(inp)
    end.record()
    torch.cuda.synchronize()
    Duration = start.elapsed_time(end)
    print("tf32 Cost", Duration)


    qli.reset_bits(16)
    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        qli(inp)
    end.record()
    torch.cuda.synchronize()
    Duration = start.elapsed_time(end)
    print("fp16 Cost", Duration)


    qli.reset_bits(8)
    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        qli(inp)
    end.record()
    torch.cuda.synchronize()
    Duration = start.elapsed_time(end)
    print("int8 Cost", Duration)

def test_qlinear_speed_splited(qlinear, inp, csv_file_name="linear_cost"):
    config.nchw = False
    torch.tensor(0).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    conversion_cost, computation_cost = [], []
    bp_conversion_cost = []
    bit_width_column = []

    bp_cost = 0

    available_bits =  config.available_bits

    print("Test splited execution speed for each qlinear")
    if 32 in available_bits:
        start.record()
        qlinear(inp)
        qli_weight = qlinear.weight
        # qli_weight = torch.t(qli_weight).contiguous()
        bias = torch.tensor(0).cuda()
        for _ in range(iter_times):
            linear_fp32(inp, qli_weight,  bias, 1)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("fp32 Cost", Duration)

        conversion_cost.append(0)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(32)

    if 19 in available_bits:
        start.record()
        for _ in range(iter_times):
            linear_tf32(inp, qli_weight,  bias, 1)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("tf32 Cost", Duration)

        conversion_cost.append(0)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(19)


    if 16 in available_bits:
        start.record()
        for _ in range(iter_times):
            fp_qli_inp = inp.half()
            fp_qweight = qli_weight.half()

        end.record()
        torch.cuda.synchronize()
        cvt_Duration = start.elapsed_time(end)
        print("fp16 conversion", cvt_Duration)

        fp_qli_inp_cp = fp_qli_inp.detach().clone()
        start.record()
        for _ in range(iter_times):
            fp_qli_inp_cp.float()
        end.record()
        torch.cuda.synchronize()
        bp_cost = start.elapsed_time(end)
        print("fp16 bp conversion", bp_cost)

    
        start.record()
        for _ in range(iter_times):
            linear_fp16(fp_qli_inp, fp_qweight, bias, 1)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("fp16 Cost", Duration)

        conversion_cost.append(cvt_Duration / iter_times)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(16)

    if 8 in available_bits:
        start.record()
        for _ in range(iter_times):
            int_qli_inp, s_1 = quantize_int8(inp)
            int_qli_weight, s_2 = quantize_int8(qli_weight)
            s_o = s_1 * s_2

        end.record()
        torch.cuda.synchronize()
        cvt_Duration = start.elapsed_time(end)
        print("int8 conversion", cvt_Duration)

        inp_cp = inp.detach().clone()
        start.record()
        for _ in range(iter_times):
            dequantize_int8(inp_cp, s_1)

        end.record()
        torch.cuda.synchronize()
        bp_cost = start.elapsed_time(end)
        print("int8 bp conversion", bp_cost)
        
        
        start.record()
        for _ in range(iter_times):
            linear_int8(int_qli_inp, int_qli_weight, bias, s_o)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        print("int8 Cost", Duration)

        conversion_cost.append(cvt_Duration / iter_times)
        bp_conversion_cost.append(bp_cost / iter_times)
        computation_cost.append(Duration / iter_times)
        bit_width_column.append(8)

    df = pd.DataFrame(data={"bit":bit_width_column,"cvt_cost":conversion_cost, "cpt_cost":computation_cost, 'bp_cost': bp_conversion_cost})
    export_to_csv(csv_file_name, df)
    print(df)


