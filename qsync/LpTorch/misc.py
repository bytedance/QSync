import torch
import numpy as np 
def inp_to_device(inp, device):
    if isinstance(inp, dict):
        for k, v in inp.items():
            inp[k] = inp_to_device(v, device)
    elif isinstance(inp, list):
        for ii, v in enumerate(inp):
            inp[ii] = inp_to_device(v, device)
    elif isinstance(inp, torch.Tensor):
        # import pdb; pdb.set_trace()
        inp = inp.to(torch.device(device))
        return inp
    else:
        print("warning: tried to move un-handled input to some device")
        return inp
    return inp

def inp_to_device_half(inp, device):
    if isinstance(inp, dict):
        for k, v in inp.items():
            inp[k] = inp_to_device(v, device)
    elif isinstance(inp, list):
        for ii, v in enumerate(inp):
            inp[ii] = inp_to_device(v, device)
    elif isinstance(inp, torch.Tensor):
        # import pdb; pdb.set_trace()
        inp = inp.to(torch.device(device))
        return inp.half()
    else:
        print("warning: tried to move un-handled input to some device")
        return inp
    return inp

def inp_to_model(model, inp):
    if isinstance(inp, dict):
        model(**inp)
    elif isinstance(inp, list):
        model(*inp)
    elif isinstance(inp, torch.Tensor):
        model(inp)
    else:
        print("warning: tried to move un-handled input to some device")
    return inp

def get_device(model):
    try:
        return next(model.parameters()).get_device()
    except StopIteration:
        return torch.device('cpu')  # Model w/o parameters assume CPU

import pandas as pd
def interpret_ILP_result(df, available_bits=[8,16,32]):
    res = df.index[df['solution_value'] == 1].tolist()
    for i in res:
        layer = df['column_i'][i]
        bit = available_bits[df['column_j'][i]]
        print("Layer {} choose bit {}".format(layer, bit))


from .ILP_solver import select_bit_with_ILP
from .utils import load_profile_file
# calculate the maximum memory sacings available
def calculate_compression(data_path, available_bits = [8, 16, 20, 32], step=-50):
    M, E, RD, omega_matrix, min_M, min_E, max_M, max_E = load_profile_file(data_path)
    fp32_lat = np.sum(E, axis=0)[-1]
    MEM_FP32 = np.sum(M, axis=0)[-1]
    MEM_LOWEST = np.sum(M, axis=0)[0]
    ava_m = 0
    print(MEM_FP32, MEM_LOWEST)
    for mem_test in range(int(MEM_FP32), int(MEM_LOWEST), step):
        try:
            opt_df, out_value = select_bit_with_ILP(M_B=mem_test, T_B=fp32_lat, M_mem=M, M_L=E, M_sens=omega_matrix, layers=len(M), available_bits = available_bits)
            ava_m = mem_test
            print(f"Try MEM {ava_m}", end = "\r")
        except:
            print("Minimum M", ava_m, f"Compression Rate {1 - ava_m / int(MEM_FP32)}")
            interpret_ILP_result(opt_df, available_bits)
            break
    print("Maximum Speed Compression Rate", 1 - min_E / fp32_lat)
    return [ava_m, MEM_FP32], [min_E.item(), fp32_lat]

