
import torch
import numpy as np
from .conf import config 
import pickle
def convert_from_nchw_to_nhwc(old_t):
    if config.channels_last:
        return old_t.to(memory_format=torch.channels_last)
    else:
        new_t = old_t.permute(0,2,3,1).contiguous()
    return new_t

def convert_from_nhwc_to_nchw(old_t):
    if config.channels_last:
        return old_t.to(memory_format=torch.contiguous_format)
    else:
        new_t = old_t.permute(0,3,1,2).contiguous()
    return new_t

def transpose_dim(p_input):
    if p_input.dim() < 3:
        return p_input.transpose(1,0)
    else:
        return p_input.transpose(2,1)

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        x_size = x.element_size() * x.nelement()
        ret += x_size

    return ret

import os
def disable_cache_allocator():
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'


def enable_cache_allocator():
    del os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']

import pandas as pd
def interpret_ILP_result(csv_file, available_bits=[8,16,32]):
    df = pd.read_csv(csv_file)
    res = df.index[df['solution_value'] == 1].tolist()

    for i in res:
        layer = df['column_i'][i]
        bit = available_bits[df['column_j'][i]]
        print("Layer {} choose bit {}".format(layer, bit))
    return df 


# wrapper function
def wrap_function(f, *args, **parmas):
    def run_function():
        return f(*args, **parmas)
    return run_function

def caculate_latency_ratio(E):
    lat_matrix = E / E[:,-1][:,np.newaxis]
    print(E.sum(axis=0))
    print(lat_matrix)

# read profile data
def load_profile_file(data_path):
    data = np.load(data_path, allow_pickle=True)
    M, E, RD, omega_matrix, min_M, min_E, max_M, max_E = \
        data['M'], data['E'], data['RD'], data['omega'], data['minM'], data['minE'], data['maxM'], data['maxE']
    print("Profiled Data Loaded")
    # print(M)
    # caculate_latency_ratio(E)
    # print(len(E), len(M), len(RD), len(omega_matrix))
    # print(RD)
    # print(omega_matrix)
    # np.savetxt("E.csv", E, delimiter=",")
    return M, E, RD, omega_matrix, min_M, min_E, max_M, max_E

# metric use to check whether implementation is correct
def close_metric(a, b):
    return torch.max(torch.abs(a - b))


def get_RD_matrix(profiled_configs, available_bits):
    layer_r_D_array = {}
    for layer_unique_id, values in profiled_configs.items():
        layer_r_D = values['act']['data']
        layer_r_D_array[layer_unique_id] = layer_r_D # R & D
    return layer_r_D_array




import subprocess as sp
from threading import Thread , Timer
import sched, time

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

class MaxValue(object):
    def __init__(self, name):
        self.name = name
        self.max_value = 0
        self.n = 0
    def update(self, val):
        self.max_value = max(val, self.max_value)
        self.n += 1

    @property
    def max(self):
        return self.max_value


class AveragedValue(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.n = 0
    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def check_file_exits(file_path):
    file_exist = os.path.exists(file_path)
    assert file_exist, f"file not exists {file_path}"

def load_data_pickle(file_path):
    check_file_exits(file_path)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# def nvidia_info():
#     # pip install nvidia-ml-py
#     nvidia_dict = {
#         "state": True,
#         "nvidia_version": "",
#         "nvidia_count": 0,
#         "gpus": []
#     }
#     try:
#         nvmlInit()
#         nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
#         nvidia_dict["nvidia_count"] = nvmlDeviceGetCount()
#         for i in range(nvidia_dict["nvidia_count"]):
#             handle = nvmlDeviceGetHandleByIndex(i)
#             memory_info = nvmlDeviceGetMemoryInfo(handle)
#             gpu = {
#                 "gpu_name": nvmlDeviceGetName(handle),
#                 "total": memory_info.total,
#                 "free": memory_info.free,
#                 "used": memory_info.used,
#                 "temperature": f"{nvmlDeviceGetTemperature(handle, 0)}℃",
#                 "powerStatus": nvmlDeviceGetPowerState(handle)
#             }
#             nvidia_dict['gpus'].append(gpu)
#     except NVMLError as _:
#         nvidia_dict["state"] = False
#     except Exception as _:
#         nvidia_dict["state"] = False
#     finally:
#         try:
#             nvmlShutdown()
#         except:
#             pass
#     return nvidia_dict

# def get_gpu_usage():
#     max_rate = 0.0
#     info = nvidia_info()
#     # print(info)
#     used = info['gpus'][0]['used']
#     return used

