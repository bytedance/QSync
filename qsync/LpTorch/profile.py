'''
    Profile the current model on the device and get statistical information (mem & speed)
'''


# test the function of conv op
import copy 
import numpy as np 
from pynvml import *
import torch
import torch.nn as nn 
import torchvision 
from .conf import config
from .misc import inp_to_device, inp_to_device_half, inp_to_model
from torch.profiler import profile, record_function, ProfilerActivity
from .utils import get_gpu_memory

def profile_sta(qmodel, calib_input, available_bits=[8, 16, 19, 32]):
    config.enable_mem_profile()
    for bit in available_bits:
        for idx, data in enumerate(calib_input):
            inp_to_model(qmodel, data)
            break
            # config.reset_profile_cnt()
    print("Collected the memory information")
    torch.cuda.empty_cache() # clean to save memory
    # config.enable_time_profile()
    # for bit in available_bits:
    #     qmodel.set_bits(bit)
    #     for idx, data in enumerate(calib_input):
    #         inp_to_model(qmodel, data)
    #         config.reset_profile_cnt()
    # print("Collected the computational information")
    # torch.cuda.empty_cache() # clean to save memory
    qmodel.set_bits(32)
    config.enable_act_profile()
    for idx, data in enumerate(calib_input):
        inp_to_model(qmodel, data)
        # config.reset_profile_cnt()
    print("Collected the activation information")
    
    config.disable_profile()
    torch.cuda.empty_cache() # clean to save memory
    return config.profiled_data

def calculate_mem_for_different_bits(bit, profiled_configs):
    mem_all = 0
    for layer_idx, values in profiled_configs.items():
        bit_value = values[bit]
        mem = bit_value['mem']
        # time_cost = bit_value['time']
        mem_all += mem
    return mem_all

def calculate_time_for_different_bits(bit, profiled_configs):
    time_all = 0
    for layer_idx, values in profiled_configs.items():
        bit_value = values[bit]
        time = bit_value['time']
        # time_cost = bit_value['time']
        time_all += time
    return time_all

def get_constraints_matrix(profiled_configs, available_bits):
    mem_array = {}
    latency_array = {}
    layer_r_D_array = {}
    for layer_unique_id, values in profiled_configs.items():
        try:
            layer_mem = values['mem']
        except:
            # print(layer_idx, values)
            print('N')
        # layer_latency = [values[bit]['time'] for bit in available_bits]
        layer_r_D = values['act']['data']
        mem_array[layer_unique_id] = layer_mem # M
        # latency_array.append(layer_latency) # E
        layer_r_D_array[layer_unique_id] = layer_r_D # R & D
        # import pdb; pdb.set_trace()
    return mem_array, latency_array, layer_r_D_array



# get system information among all gpus.
def get_straggler_info():
    assert torch.cuda.is_available(), 'no cuda found'
    num_ = torch.cuda.device_count()
    nvmlInit()
    info_dict = {}
    for gpu_idx in range(num_):
        h = nvmlDeviceGetHandleByIndex(gpu_idx)
        info = nvmlDeviceGetMemoryInfo(h)
        # print(f'total    : {info.total}')
        # print(f'free     : {info.free}')
        # print(f'used     : {info.used}')
        print(f'used gpu{gpu_idx}: {info.used // config.unit}')
        info_dict[gpu_idx] = {'mem': info.used // config.unit}
    return info_dict


def get_M_E_max(M=None,E=None):
    assert M is not None and E is not None, "M and E are both required!"
    M_keys = list(M.keys())
    row_length = len(M_keys)
    assert row_length > 0, "No Quantized Kernel Detected"
    min_M_agg, max_M_agg = 0, 0
    min_E_agg, max_E_agg = 0, 0
    for idx in range(row_length):
        max_M, min_M =  max(M[M_keys[idx]]), min(M[M_keys[idx]])
        # max_E, min_E = max(E[idx]), min(E[idx])

        min_M_agg += min_M
        max_M_agg += max_M

        # min_E_agg += min_E
        # max_E_agg += max_E



    min_M, min_E = min_M_agg, min_E_agg
    max_M, max_E = max_M_agg, max_E_agg
    # F_E = E.sum(axis=0)[-1] # time for floating point
    F_E = 0
    print(f"Get Constraints for Operator Budget M(memory) between {min_M:.4f}~{max_M:.4f}, minimum E(latency) {min_E:.4f}~{max_E:.4f}, F_E {F_E}")
    return min_M, max_M, min_E, max_E, F_E
    

# Profile Model
# profile iter is the maximum profiling rounds required
def profile_conv(model, data_loader, criterion=None, rank=None, profile_iter=10, available_bits=None, \
     profiled_data_path='profiled_data.npz'):
    assert rank is not None, "Require the device to do profiling"
    assert data_loader and criterion, "Profiling need criterion and data loader"

    # import pdb; pdb.set_trace()

    config.training = False
    device = torch.device(rank)
    num_data = 0
    calib_input = []
    calib_backward = []

    model_context_size = get_gpu_memory()[0] * 1024 # record in MB
    
    for image, target in data_loader:
        data = inp_to_device(image, device)
        data = data.to(memory_format=torch.channels_last)
        calib_input.append(data)
        # import pdb; pdb.set_trace()
        bwd_data = inp_to_device(image, device).to(memory_format=torch.channels_last)
        calib_backward.append([bwd_data, target])
        num_data += 1
        if num_data >= profile_iter: break 
    profiled_data = profile_sta(model, calib_input, available_bits)
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_profile"):
    #         profiled_data = profile_sta(model, calib_input, available_bits)
    # prof.export_chrome_trace("trace.json")
    # print("here")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    M, E, RD = get_constraints_matrix(profiled_data, available_bits)
    model.RD = RD
    # min_M, max_M, min_E, max_E, F_E = get_M_E_max(M, E)
    min_M, max_M, min_E, max_E, F_E = 0, 0, 0, 0, 0

    torch.cuda.empty_cache() # clean to save memory
    memory_available_in_profile = True
    # Profile the latency gap using BP fp16 flow
    model.set_bits(32)
    """
        Calculate Half-based BP speed.
        Not necessary be able to profile (Due To the possible memory consumtion)
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for idx in range(len(calib_backward)):
        image, target = calib_backward[idx]
        try:
            output = model(inp_to_device(image, device))
            # import pdb; pdb.set_trace()
            if type(output) is torchvision.models.inception.InceptionOutputs:
                loss1 = criterion(output.logits, inp_to_device(target, device))
                loss2 = criterion(output.aux_logits, inp_to_device(target, device))
                loss = loss1 + 0.4*loss2
            else:
                loss = criterion(output, inp_to_device(target, device))
            # loss.backward() # for debug
            loss /= len(calib_backward)
            loss.backward()
        except:
            memory_available_in_profile = False
            print("Memory is not available for profiling gradient. Unified with 1 and the backward time profiling is not available")
    end.record()
    torch.cuda.synchronize()
    Duration = start.elapsed_time(end)
    # bp_time = Duration // len(calib_backward)
    # print('backward time fp32', bp_time)
    # model.set_bits(16)
    # if memory_available_in_profile:
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     for idx in range(len(calib_backward)):
    #         try:
    #             image, target = calib_backward[idx]
    #             output = model(inp_to_device(image, device))
    #             # import pdb; pdb.set_trace()
    #             if type(output) is torchvision.models.inception.InceptionOutputs:
    #                 loss1 = criterion(output.logits, inp_to_device(target, device))
    #                 loss2 = criterion(output.aux_logits, inp_to_device(target, device))
    #                 loss = loss1 + 0.4*loss2
    #             else:
    #                 loss = criterion(output, inp_to_device(target, device))
    #             loss.backward() # for debug
    #         except:
    #             print("Memory is not available for profiling gradient. Unified with 1 and the backward time profiling is not available")
    #     end.record()
    #     torch.cuda.synchronize()
    #     Duration = start.elapsed_time(end)
    #     opt_bp_time = Duration // len(calib_backward)
    #     print('backward time half', opt_bp_time)
    # else:
    #     opt_bp_time = 0 # cannot profile
    # import pdb; pdb.set_trace()
    ava_idx = model.get_available_list()
    omega_dict = model.compute_omega_matrix()
    # print(self.omega_matrix)
    # save M, E, RD and omega_matrix
    memory_reserved_full = torch.cuda.max_memory_allocated() // config.unit

    np.savez(profiled_data_path, M=M, E=E, RD=RD, omega=omega_dict, \
    minM=min_M, minE=min_E, maxM=max_M, maxE=max_E, ava_idx=ava_idx, mem=memory_reserved_full, F_E=F_E, \
    layer_order=config.mapping_dict, model_context_size=model_context_size)
    print(f"Save profiled_data to {profiled_data_path}")
    print(f"Shape, M: {len(M)}, E:{len(E)}, RD:{len(RD)}, omega:{len(omega_dict)}, mem:{memory_reserved_full}, model_context:{model_context_size}")

def debug_hook(mod, inp, out):
    print("debuggging")
    print(torch.cuda.memory_allocated())
    # import pdb; pdb.set_trace()

def profile_transformer(model, data_loader, rank=None, profile_iter=10, available_bits=None, \
     profiled_data_path='profiled_data.npz', optimizer=None):
    
    assert rank is not None, "Require the device to do profiling"
    assert data_loader , "Profiling need criterion and data loader"

    config.training = False
    device = torch.device(rank)
    num_data = 0
    calib_input = []
    calib_backward = []

    model_context_size = get_gpu_memory()[0] * 1024 # record in MB
    # embeddings = model.model.bert.embeddings
    # # embeddings.register_forward_hook(debug_hook)

    # mod = model.model.bert.encoder.layer[0]
    # mod.register_forward_hook(debug_hook)

    for step, batch in enumerate(data_loader):
        calib_input.append(inp_to_device(batch, device))
        calib_backward.append(inp_to_device(batch, device))
        num_data += 1
        if num_data >= profile_iter: break 
    
    # import pdb; pdb.set_trace()
    
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_profile"):
    profiled_data = profile_sta(model, calib_input, available_bits)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")
    # print("here")
    # profiled_data = profile_sta(model, calib_input, available_bits)
    M, E, RD = get_constraints_matrix(profiled_data, available_bits)
    model.RD = RD
    # min_M, max_M, min_E, max_E, F_E = get_M_E_max(M, E)
    min_M, max_M, min_E, max_E, F_E = 0, 0, 0, 0, 0

    torch.cuda.empty_cache() # clean to save memory
    memory_available_in_profile = True
    # Profile the latency gap using BP fp16 flow
    model.set_bits(32)
    # config.enable_bwd_time_profile()
    if memory_available_in_profile:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for idx, batch in enumerate(calib_backward):
            try:
                output = model(**batch)
                loss = output.loss / len(calib_backward)
                # loss.backward() # for debug
                
                loss.backward()
                optimizer.step()
            except:
                memory_available_in_profile = False
                print("Memory is not available for profiling gradient. Unified with 1 and the backward time profiling is not available")
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        opt_bp_time = Duration // len(calib_backward)
        print('backward time half', opt_bp_time)
    else:
        opt_bp_time = 0 # cannot profile
    # config.disable_profile()
    ava_idx = model.get_available_list()
    omega_dict = model.compute_omega_matrix()

    # print(omega_matrix)
    # save M, E, RD and omega_matrix
    memory_reserved_full = torch.cuda.max_memory_allocated() // config.unit

    np.savez(profiled_data_path, M=M, E=E, RD=RD, omega=omega_dict, \
    minM=min_M, minE=min_E, maxM=max_M, maxE=max_E, ava_idx=ava_idx, mem=memory_reserved_full, F_E=F_E, \
    layer_order=config.mapping_dict, model_context_size=model_context_size)
    print(f"Save profiled_data to {profiled_data_path}")
    print(f"Shape, M: {len(M)}, E:{len(E)}, RD:{len(RD)}, omega:{len(omega_dict)}, mem:{memory_reserved_full}, model_context:{model_context_size}")

# TODO: fixed later
# for deteron2.
def profile_wo_criteria(self, data_loader, comm, storage, optimizer):
    assert self.device is not None, "Require the device to do profiling"
    # T_B_i and M_B_i varies according to other training nodes. Thus here we only demonstrate the profile of other matrix.
    assert data_loader, "Profiling need criterion and data loader"
    config.training = False
    calib_input = []
    for i, data in enumerate(data_loader):
        # import pdb; pdb.set_trace()
        # calib_input.append({'img_metas': inp_to_device(data['img_metas'], self.device),'img': inp_to_device(data['img'], self.device)})
        calib_input.append(data)
        if i >= self.profile_iter: break 
    profiled_data = profile_sta(self.model, calib_input, self.available_bits)
    self.M, self.E, self.RD = get_constraints_matrix(profiled_data, self.available_bits)
    self.get_M_E_max()

    loss_dict = self.model(data)
    losses = sum(loss_dict.values())
    assert torch.isfinite(losses).all(), loss_dict
    loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    if comm.is_main_process():
        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    # print(len(self.M), len(self.RD), len([conv for conv in qconv_list if conv.used]))
    self.get_available_list() # for 
    self.compute_omega_matrix()
    # print(self.omega_matrix)
    # save M, E, RD and omega_matrix
    if comm.is_main_process():
        np.savez(self.profiled_data_path, M=self.M, E=self.E, RD=self.RD, omega=self.omega_matrix, \
        minM=self.min_M, minE=self.min_E, maxM=self.max_M, maxE=self.max_E, ava_idx=self.ava_idx)
        print(f"Save profiled_data to {self.profiled_data_path}")
        print(f"Shape, M: {self.M.shape}, E:{self.E.shape}, RD:{self.RD.shape}, omega:{self.omega_matrix.shape}")
