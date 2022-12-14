'''
    Casting Cost Prediction Model
'''

import torch 
import random 
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt 
import os 
import pickle

from qsync.LpTorch.quant import quantize_int8, dequantize_int8, quantize_int16, quantize_int32
from qsync.Predictor.utils import create_folder

SAMPLE_NUM = (50, 50)
# define sample tensorsize to generate data for casting 
def create_samples_conv(samples_num_c, dtype, samples):
    sample_shapes_convs = [
        [3, 224, 224],
        [3, 112, 112],
        [64, 112, 112],
        [56, 56, 64],
    ]

    BS_size_conv = [32, 64, 96, 128]

    for _ in range(samples_num_c):
        BS_C = random.choice(BS_size_conv)
        shape = random.choice(sample_shapes_convs)
        samples.append(torch.rand(BS_C, *shape, dtype=dtype).cuda())

def create_samples_trans(samples_num_t, dtype, samples):
    sample_shapes_trans = [
        [768, 768],
        [1024, 1024]
    ]

    BS_size_trans = [3, 6, 8, 12]
    for _ in range(samples_num_t):
        BS_t = random.choice(BS_size_trans)
        shape = random.choice(sample_shapes_trans)
        samples.append(torch.rand(BS_t, *shape, dtype=dtype).cuda())

def create_samples(dtype=torch.float32):
    # create samples
    samples_num_c, samples_num_t = SAMPLE_NUM
    samples = []
    create_samples_conv(samples_num_c, dtype, samples)
    create_samples_trans(samples_num_t, dtype, samples)
    
    return samples 

repeat_times = 5
# cuda
def fp32_to_fp16(sample_tensor):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        sample_tensor.half()
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

def fp16_to_fp32(sample_tensor):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        sample_tensor.float()
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

def fp32_to_int8(sample_tensor):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        res, scale = quantize_int8(sample_tensor)
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

def fp16_to_int8(sample_tensor):
    sample_tensor = sample_tensor.half()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        res, scale = quantize_int8(sample_tensor)
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times



def fp32_to_int8_channel(sample_tensor):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        res, scale = quantize_int8(sample_tensor, "channel")
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

def fp16_to_int8_channel(sample_tensor):
    sample_tensor = sample_tensor.half()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        res, scale = quantize_int8(sample_tensor, "channel")
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

# scaling in fp32 to fp32
def scale_fp32_to_fp32(sample_tensor):
    int8_tensor, scale = quantize_int8(sample_tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        dequantize_int8(sample_tensor, scale)
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

# scaling in fp32 to fp32
def scale_int8_to_fp32(sample_tensor):
    sample_int, scale = quantize_int8(sample_tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        float_sample = dequantize_int8(sample_tensor,scale)
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

# scaling in fp32 to fp32
def scale_int8_to_fp16(sample_tensor):
    sample_int, scale = quantize_int8(sample_tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        float_sample = dequantize_int8(sample_tensor,scale)
        float_sample.half()
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times


# channel-wise
def channel_fp32_to_fp32(sample_tensor):
    int8_tensor, scale = quantize_int8(sample_tensor, "channel")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        dequantize_int8(sample_tensor, scale)
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

# channel-wise
def channel_int8_to_fp32(sample_tensor):
    sample_int, scale = quantize_int8(sample_tensor, "channel")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        float_sample = dequantize_int8(sample_tensor,scale)
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times

# channel-wise
def channel_int8_to_fp16(sample_tensor):
    sample_int, scale = quantize_int8(sample_tensor, "channel")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        float_sample = dequantize_int8(sample_tensor,scale)
        float_sample.half()
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times


def int16_to_int8(sample_tensor):
    int16_tens, scale = quantize_int16(sample_tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        (int16_tens / 2 ** 8 ).char()
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times


def int32_to_int8(sample_tensor):
    int32_tens, scale = quantize_int32(sample_tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat_times):
        (int32_tens / 2 ** 24).char()
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    return cvt_Duration / repeat_times


# cpu
import time 
def fp32_to_fp16_cpu(sample_tensor):
    start = time.time()
    for _ in range(repeat_times):
        sample_tensor.half()
    end = time.time()
    cvt_Duration = end - start
    return cvt_Duration / repeat_times

def fp16_to_fp32_cpu(sample_tensor):
    start = time.time()
    for _ in range(repeat_times):
        sample_tensor.float()
    end = time.time()
    cvt_Duration = end - start
    return cvt_Duration / repeat_times

def fp32_to_int8_cpu(sample_tensor):
    start = time.time()
    for _ in range(repeat_times):
        quantize_int8(sample_tensor)
    end = time.time()
    cvt_Duration = end - start
    return cvt_Duration / repeat_times

def fp16_to_int8_cpu(sample_tensor):
    start = time.time()
    for _ in range(repeat_times):
        quantize_int8(sample_tensor)
    end = time.time()
    cvt_Duration = end - start
    return cvt_Duration / repeat_times


def calculate_cuda_cost():
    float_samples = create_samples()
    half_samples = create_samples(dtype=torch.float16)
    # fp32 -> fp16
    fp322fp16_record_points = []
    for idx, sample in enumerate(float_samples):
        fp322fp16_record_points.append([sample.numel(), fp32_to_fp16(sample)])
    # fp16 -> fp32
    fp162fp32_record_points = []
    for idx, sample in enumerate(half_samples):
        fp162fp32_record_points.append([sample.numel(), fp16_to_fp32(sample)])
    # fp32 -> int8
    fp322int8_record_points = []
    for idx, sample in enumerate(float_samples):
        fp322int8_record_points.append([sample.numel(), fp32_to_int8(sample)])
    # half -> int8
    fp162int8_record_points = []
    for idx, sample in enumerate(half_samples):
        fp162int8_record_points.append([sample.numel(), fp16_to_int8(sample)])
    
    # float -> float
    dq_float_record_points = []
    for idx, sample in enumerate(half_samples):
        dq_float_record_points.append([sample.numel(), scale_fp32_to_fp32(sample)])
    
    # int8, scalar -> float
    int8_scale_float_record_points = []
    for idx, sample in enumerate(float_samples):
        int8_scale_float_record_points.append([sample.numel(), scale_int8_to_fp32(sample)])

    # int8, scalar -> half
    int8_scale_half_record_points = []
    for idx, sample in enumerate(float_samples):
        int8_scale_half_record_points.append([sample.numel(), scale_int8_to_fp16(sample)])
    
    # channel-wise results

    # fp32 -> int8
    fp322int8_record_points_channel = []
    for idx, sample in enumerate(float_samples):
        fp322int8_record_points_channel.append([sample.numel(), fp32_to_int8_channel(sample)])
    # half -> int8
    fp162int8_record_points_channel = []
    for idx, sample in enumerate(half_samples):
        fp162int8_record_points_channel.append([sample.numel(), fp16_to_int8_channel(sample)])
    
     # float -> float
    dq_float_record_points_channel = []
    for idx, sample in enumerate(half_samples):
        dq_float_record_points_channel.append([sample.numel(), channel_fp32_to_fp32(sample)])
    
    # int8, scalar -> float
    int8_channel_float_record_points = []
    for idx, sample in enumerate(float_samples):
        int8_channel_float_record_points.append([sample.numel(), channel_int8_to_fp32(sample)])

    # int8, scalar -> half
    int8_channel_half_record_points = []
    for idx, sample in enumerate(float_samples):
        int8_channel_half_record_points.append([sample.numel(), channel_int8_to_fp16(sample)])
    
    # inference
    # int32 -> int8
    # int16 -> int8
    int32_2_int8_record_points = []
    for idx, sample in enumerate(float_samples):
        int32_2_int8_record_points.append([sample.numel(), int32_to_int8(sample)])
    
    int16_2_int8_record_points = []
    for idx, sample in enumerate(float_samples):
        int16_2_int8_record_points.append([sample.numel(), int16_to_int8(sample)])


    # repeat 5 times for each cost
    return fp322fp16_record_points, fp162fp32_record_points, fp322int8_record_points, fp162int8_record_points, \
            dq_float_record_points, int8_scale_float_record_points, int8_scale_half_record_points,\
            fp322int8_record_points_channel, fp162int8_record_points_channel, \
            dq_float_record_points_channel, int8_channel_float_record_points, int8_channel_half_record_points, \
            int32_2_int8_record_points, int16_2_int8_record_points


def calculate_cpu_cost():
    float_samples = create_samples()
    half_samples = create_samples(dtype=torch.float16)
    # fp32 -> fp16
    fp322fp16_record_points = []
    for idx, sample in enumerate(float_samples):
        fp322fp16_record_points.append([sample.numel(), fp32_to_fp16_cpu(sample)])
    # fp16 -> fp32
    fp162fp32_record_points = []
    for idx, sample in enumerate(half_samples):
        fp162fp32_record_points.append([sample.numel(), fp16_to_fp32_cpu(sample)])
    # fp32 -> int8
    fp322int8_record_points = []
    for idx, sample in enumerate(float_samples):
        fp322int8_record_points.append([sample.numel(), fp32_to_int8_cpu(sample)])
    # half -> int8
    fp162int8_record_points = []
    for idx, sample in enumerate(half_samples):
        fp162int8_record_points.append([sample.numel(), fp16_to_int8_cpu(sample)])
    # repeat 5 times for each cost
    return fp322fp16_record_points, fp162fp32_record_points, fp322int8_record_points, fp162int8_record_points

import torch.nn.functional as F 
def padding_test(sample):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    mod_num = 16 # for int8
    for _ in range(repeat_times):
        if sample.size(1) % mod_num != 0:
            padding_channels = mod_num - sample.size(1) % mod_num
            input_padded = F.pad(sample, (0,0,0,0,0,padding_channels),"constant", 0)
    end.record()
    torch.cuda.synchronize()
    cvt_Duration = start.elapsed_time(end)
    padded_nums = input_padded.numel() - sample.numel()
    return padded_nums, cvt_Duration / repeat_times
# padding cost for int8
def padding_cost():
    samples = []
    create_samples_conv(100, torch.float32, samples)
    samples = [quantize_int8(sample)[0] for sample in samples]
    # import pdb; pdb.set_trace()
    record_points = []
    for idx, sample in enumerate(samples):
        record_points.append(padding_test(sample))
    return record_points


    

def linear_func(x, a, b):
    return a*x + b


def curve_fitting(func, x_y_points):
    x_y_points = np.array(x_y_points)
    x = x_y_points[:,0]
    y = x_y_points[:,1]
    params = curve_fit(func, x, y)

    [a, b] = params[0]
    return (a, b)


def plotting_curve(func, params, xy_data):
    x = xy_data[:,0]
    y = xy_data[:,1]
    pred_y = func(x, *params)

    plt.scatter(x, y)
    plt.plot(x, pred_y, '--', color='red')
    plt.show()



def store_data(record_points, file_name="data.pkl"):
    with open(file_name, 'wb') as f:
        pickle.dump(record_points, f)

def load_data(file_name="data.pkl"):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


cur_file = 'curve_data'
def FILEPATH(file_name):
    create_folder(os.path.abspath(os.path.join(__file__, os.pardir, cur_file)))# ensures folder exists
    file_path_name = os.path.abspath(os.path.join(__file__, os.pardir, cur_file, file_name))
    return file_path_name



def cast_modeling():
    fp322fp16_record_points, fp162fp32_record_points, fp322int8_record_points, fp162int8_record_points, \
    dq_float_record_points, int8_scale_float_record_points, int8_scale_half_record_points, \
    fp322int8_record_points_channel, fp162int8_record_points_channel, \
    dq_float_record_points_channel, int8_channel_float_record_points, int8_channel_half_record_points, \
    int32_2_int8_record_points, int16_2_int8_record_points = calculate_cuda_cost()
    # not correct 
    # fp322fp16_record_points_cpu, fp162fp32_record_points_cpu, fp322int8_record_points_cpu, fp162int8_record_points_cpu = calculate_cpu_cost()
    # print(len(fp322fp16_record_points), len(fp162fp32_record_points))
    store_data(fp322fp16_record_points, FILEPATH('fp322fp16.pkl'))
    store_data(fp162fp32_record_points, FILEPATH('fp162fp32.pkl'))
    store_data(fp322int8_record_points, FILEPATH('fp322int8.pkl'))
    store_data(fp162int8_record_points, FILEPATH('fp162int8.pkl'))

    store_data(dq_float_record_points, FILEPATH('scale_fp32.pkl'))
    store_data(int8_scale_float_record_points, FILEPATH('scale_int8_float.pkl'))
    store_data(int8_scale_half_record_points, FILEPATH('scale_int8_half.pkl'))

    store_data(fp322int8_record_points_channel, FILEPATH('fp322int8_c.pkl'))
    store_data(fp162int8_record_points_channel, FILEPATH('fp162int8_c.pkl'))
    store_data(dq_float_record_points_channel, FILEPATH('scale_fp32_c.pkl'))
    store_data(int8_channel_float_record_points, FILEPATH('channel_int8_float.pkl'))
    store_data(int8_channel_half_record_points, FILEPATH('channel_int8_half.pkl'))
    # inference
    store_data(int32_2_int8_record_points, FILEPATH('int32_2_int8.pkl'))
    store_data(int16_2_int8_record_points, FILEPATH('int16_2_int8.pkl'))

    # padded_data_points = padding_cost()
    # store_data(padded_data_points, FILEPATH('pad.pkl'))


    # fp322fp16_record_points = load_data(FILEPATH('fp322fp16.pkl'))
    # fp162fp32_record_points = load_data(FILEPATH('fp162fp32.pkl'))
    params_a = curve_fitting(linear_func, fp322fp16_record_points)
    params_b = curve_fitting(linear_func, fp162fp32_record_points)
    print("fp32 -> fp16", params_a, params_b)


    params_c = curve_fitting(linear_func, fp322int8_record_points)
    params_d = curve_fitting(linear_func, fp162int8_record_points)
    print(f"fp32 quantized int8 {params_c}. FP16 quantized int8 {params_d}")

    params_e = curve_fitting(linear_func, dq_float_record_points)
    params_f = curve_fitting(linear_func, int8_scale_float_record_points)
    params_g = curve_fitting(linear_func, int8_scale_half_record_points)
    print(f"dq scale quantized int8 {params_e}. int8 dq float {params_f} int8 dq half {params_g}")


    params_h = curve_fitting(linear_func, fp322int8_record_points_channel)
    params_i = curve_fitting(linear_func, fp162int8_record_points_channel)
    params_j = curve_fitting(linear_func, dq_float_record_points_channel)
    params_k = curve_fitting(linear_func, int8_channel_float_record_points)
    params_l = curve_fitting(linear_func, int8_channel_half_record_points)
    print(f"Channel-wise FP32 -> Int8 {params_h}. FP16->INT8 {params_i},  dq float {params_j}, int8->FP32 {params_k} int8->half {params_l}")

    params_m = curve_fitting(linear_func, int32_2_int8_record_points)
    params_n = curve_fitting(linear_func, int16_2_int8_record_points)
    print(f"INT32 -> INT8 {params_m}, INT16 -> INT8 {params_n}")


def comm_modeling():
    # SINGLE_NODE_GPUS = 4
    # OVERALL_GPUS = 8
    # NODE_NUM = 2

    bert_bucket_result = [2363138, 7087872, 28563456]

    # single_node_function = a_single + beta_single * (2 * data_size ) / Bandwidth * (SINGLE_NODE_GPUS - 1)/SINGLE_NODE_GPUS
    # multi_node = a_multi + (2 * data_size ) / Bandwidth * (OVERALL_GPUS - NODE_NUM)/SINGLE_NODE_GPUS  + (2 * data_size ) / Bandwidth * (OVERALL_GPUS - NODE_NUM)/SINGLE_NODE_GPUS

    # ring all-reduce 2(N-1)k/N. where N is the number of gpus, k is the data size
    # tree all-reduce introduces link between nodes and can be achieved by different ways (balanced tree, split tree)
    # but since the data-size is not changed, we can regard it as introducing deficiencies on the previous communication result
    # for simplicity, we explicitly model the communication result prediction with respect to single node multi-communication

    # remove outliers
    comm_part1 = [12.50, 11.81]
    comm_part2 = [20.96, 21.63, 21.817, 21.218, 20.943, 19.43]
    comm_part3 = [41.55, 41.632]

    comm_part2_tree = [28.199, 28.405, 29.011, 29.011, 28.187, 28.715]
    comm_part3_tree = [84.787, 85.728]

    comm_part_2_len = len(comm_part2)
    comm_part_3_len = len(comm_part3)

    Xs = [bert_bucket_result[1] for _ in range(comm_part_2_len)]
    Xs += [bert_bucket_result[2] for _ in range(comm_part_3_len)]

    Ys = [comm_part2_tree[i] - comm_part2[i] for i in range(comm_part_2_len)]
    Ys += [comm_part3_tree[i] - comm_part3[i] for i in range(comm_part_3_len)]
    
    # import pdb; pdb.set_trace()
    print(Xs, Ys)
    print(len(Ys) , len(Xs))
    params_ys = curve_fit(linear_func,  Xs, Ys)
    a,b = params_ys[0]
    print("fitting result", a, b)
    res_est = [linear_func(i, a, b) for i in Xs]
    print(res_est)
    # store_data(fp322int8_record_points_channel, FILEPATH('fp322int8_c.pkl'))


if __name__ == '__main__':
    cast_modeling()
    # comm_modeling()


