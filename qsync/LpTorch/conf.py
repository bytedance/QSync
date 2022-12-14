import torch
import torch.nn as nn 
from collections import defaultdict
from qsync.Utils import get_capability
import json
import os

from .env import *

def calculate_running_avg(old_sta, cnt, new_d):
    m = cnt / (cnt + 1)
    return old_sta * m + new_d * (1-m)

KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3


config_json_path = os.path.abspath(os.path.join(__file__, os.pardir, "config.json"))

# TODO: In the current implementation, time profiling is not important at all.
# In the profiling stage, we only need to obtain the variance indicator
# Thus we can remove the time profiling for fwd and bwd

class QuantizationConfig:
    def __init__(self):
        self.pred_root = os.path.join(__file__, os.pardir)
        self.config_json_path = config_json_path
        
        self.profile_act = False
        self.profile_mem = False
        self.profile_time = False # recommend not to record time and memory at the same time. else time will be delayed.
        self.profile_bwd_time = False
        self.profiled_data = defaultdict(lambda: defaultdict(dict)) # key 1 -> layer key2 -> bit key3-> mem or time cost
        self.profile_cnt = 0 # record the layer
        self.half_bn = True # enable bn 2 half

        self.KB = KB
        self.MB = MB
        self.GB = GB

        self.unit = KB

        # quantization method: scale for activation and kernel channel to speed
        # self.act_quant = 'channel'
        self.act_quant = 'scale'
        self.kernel_quant = 'scale'
        # self.kernel_quant = 'channel'

        self.config_ava_bits = { # the current available bit for qsync
            80: [8, 16, 19, 32], # A100，A10, add tf32 support
            75: [8, 16, 32], #'T4'
            70: [16, 32] #'V100'
        }

        self.reject_name_list = [] # contains the name list to reject conversion, normally, will be the classifier (last layer).
                                                                   # QSync will add a fp32 conversion node before it to make sure the input of node is fp32
        self.reject_name_list.append('qa_outputs') # bert
        self.reject_name_list.append('fc') # resnet50
        self.reject_name_list.append('classifier') # roberta, remove it when testing convoltution
        self.reject_name_list.append('pooler') 
        
        self.available_archs = list(self.config_ava_bits.keys())
        self.target_device = get_capability()
        self.available_bits = self.config_ava_bits[self.target_device]

        
        self.nchw = True # the model is train under a nchw flow
        self.save_q_weight = True # whehter save q weight or simply save the full precision weight

        self.mapping_dict = {} # find cases that single conv used twice, use the mapping dict for (id of the weight (id(weight)): profile_cnt)
        '''
            Quantization Layer
        '''

        # default setup, will be updated by config.json
        self.QTarget_Module = (nn.Conv2d, nn.ReLU, nn.BatchNorm2d) # we dont quantize classifier for conv models
        self.DynamicModule = (nn.Conv2d, nn.BatchNorm2d) 

        self.LIMITED_OPS = [nn.BatchNorm2d, nn.LayerNorm] # ops like layer norm / batchnorm don't have int8 / smaller implementation.
                              # the speed up can be only achieved by limited bits

        if os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                print("Config loaded.")
                data = json.load(f)
                self.optimized_int = data['optimized_int']
                print("Optimized Fixed Point Quantization?", self.optimized_int)
                if 'ops' in data:
                    ops = data['ops']
                    self.QTarget_Module, self.DynamicModule = self.map_str_to_op_list(ops)
                    print("Target Module", self.QTarget_Module, "Dynamic Module", self.DynamicModule)
        else:
            ops = ["nn.Conv2d", "nn.ReLU", "nn.BatchNorm2d", "nn.LayerNorm", "nn.ReLU", "nn.Linear", "nn.Embedding", "nn.MaxPool2d", "nn.Softmax", "nn.Dropout", "TransGELU"]
            self.optimized_int = False

        self.simu = 1 if 'QSYNC_SIMU' in os.environ and os.environ['QSYNC_SIMU'] != '0' else 0 # use simulation in training. Since our implementation in bp is much slower than torch(haven't use any parallel reduction).
        # print(os.environ['QSYNC_SIMU'], type(os.environ['QSYNC_SIMU']))
        if self.simu:
            self.optimized_int = False
            print(">>>>>>>>>>>>>>>>. Simulated Result ON. <<<<<<<<<<<<<<<<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>. Real Run ON. <<<<<<<<<<<<<<<<<<<<<<<")

        self.training = True 
        # Notice: You can test this function for latency, however, *it is not available now*
        # TODO: For the moment due to the wrong conversion from CUTLASS_HALF -> CUDNN_HALF. The result produced here may be wrong

        self.enable_half_conversion = False # when half conversion is enabled, the weight is stored in the half format, and each half kernel output will be half
        self.channels_last = True # Torch Beta Test.
        self.cudnn = False # once true, use CUDNN else CUTLASS
        self.half_to_half = True # we implemented half x half -> half and half x half -> float two version of half data flow
                                 # CUTLASS HALF has problem in converting to CUDNN Half. This problem should be fixed later
                                 # SINCE cutlass half has similar performance comparing with pytorch, we replace it when require half x half -> half
        self.disable_padding = False
        
        self.fwd_covs_shape, self.bwd_covs_shape = {}, {}
        self.pytorch_profiler = True # create profiler notations for operations
        self.enable_period_collect = False

        self.turnon_period_collect = False # turn on period collect
        self.enable_sta_collect = True # int8's problem
        self.sta_collect_perod = 10

        # self.optimized_int = True
        self.fuse_int_output = False
        self.optimized_fp16_bp = True
        self.fixed_save_fp16_context = False # proofed to be useless, and introduce extra memory, abandoned.

        # test method
        self.swapping = False
        self.layer_cpu_storage = {}

        # indicator chosen
        self.ava_indicator_types = ['GVAR', 'WVAR', 'HESS']
        self.indicator_type = self.ava_indicator_types[0]
        print(f"Default to use {self.indicator_type} indicator")
        print(">>>>>>>>>>>. end of qsync config <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    
    # memory relevant two methods
    def store_in_cpu(self, q_input, q_weight, bias, layer_id):
        cpu_input = q_input.detach().cpu()
        cpu_weight = q_weight.detach().cpu()
        cpu_bias = bias.detach().cpu()
        self.layer_cpu_storage[layer_id] = [cpu_input, cpu_weight, cpu_bias]

    def fetch_from_cpu(self, layer_id):
        return self.layer_cpu_storage[layer_id] 

    
    def map_available_bit(self, layer, bit):
        if isinstance(layer, (nn.Embedding, nn.LayerNorm, nn.BatchNorm2d, nn.MaxPool2d)):
            if bit < 16:
                bit = 16 # these layers haven't implement the int8 ops
        return bit 

    
    def map_str_to_op_list(self, str_list):
        op_set = set()
        for str_item in str_list:
            if str_item == 'nn.BatchNorm2d':
                op_set.add(nn.BatchNorm2d)
            elif str_item == 'nn.Conv2d':
                op_set.add(nn.Conv2d)
            elif str_item == 'nn.Linear':
                op_set.add(nn.Linear)
            elif str_item == 'nn.ReLU':
                op_set.add(nn.ReLU)
            elif str_item == 'nn.Dropout':
                op_set.add(nn.Dropout)
            elif str_item == 'nn.LayerNorm':
                op_set.add(nn.LayerNorm)
            elif str_item == 'nn.Embedding':
                op_set.add(nn.Embedding)
            elif str_item == 'nn.MaxPool2d':
                op_set.add(nn.MaxPool2d)
            elif str_item == 'nn.Softmax':
                op_set.add(nn.Softmax)
            elif str_item == 'nn.GELU':
                op_set.add(nn.GELU)
            elif str_item == 'TransGELU':
                op_set.add(TransGELU)
            else:
                continue
        op_list = tuple(op_set)
        no_dynamic_module = [nn.ReLU, nn.Dropout, nn.Softmax, TransGELU, nn.LayerNorm, nn.BatchNorm2d]
        op_dynamic = tuple([i for i in op_list if i not in no_dynamic_module])
        return op_list, op_dynamic
    

    def enable_simu(self, simu_arch):
        assert simu_arch in self.available_archs, "The target simu structure is not supported"
        self.simu = True 
        self.target_device = simu_arch
        self.available_bits = self.config_ava_bits[self.target_device]

    def disable_profile(self):
        self.profile_mem = False
        self.profile_time = False
        self.profile_act = False
        self.profile_bwd_time = False
        self.profile_cnt = 0

    def enable_mem_profile(self):
        self.disable_profile()
        self.profile_mem = True
    
    def enable_time_profile(self):
        self.disable_profile()
        self.profile_time = True
    
    def enable_act_profile(self):
        self.disable_profile()
        self.profile_act = True
    
    def enable_bwd_time_profile(self):
        self.disable_profile()
        self.profile_bwd_time = True

    
    def reset_profile_cnt(self):
        self.profile_cnt = 0
        self.mapping_dict = {}
    
    # CNT used for layer execution time indication
    def update_cnt(self, id_layer):
        if id_layer in self.mapping_dict and id_layer is not None: # appeared before. fetch the result
            return self.mapping_dict[id_layer] # profiled_cnt 
        else:
            # not appeared before. record as new one
            cur_profile_cnt = self.profile_cnt 
            self.mapping_dict[id_layer] = cur_profile_cnt # record the current layer using id(weight):profile_cnt
            self.profile_cnt += 1
            return cur_profile_cnt
    
    def store_sta_act(self, data_pack, id_layer=None, tag='fwd'):
        assert id_layer, "no id input"
        self.update_cnt(id_layer)
        # nor found layer, not found tag, not found result under tag
        if not self.profiled_data[id_layer] or not self.profiled_data[id_layer][tag]:
            self.profiled_data[id_layer][tag]['data'] = [i for i in data_pack]
            self.profiled_data[id_layer][tag]['cnt'] = 0 # average
        else:
            # import pdb; pdb.set_trace()
            old_value, old_cnt = self.profiled_data[id_layer][tag]['data'], self.profiled_data[id_layer][tag]['cnt']
            coming_values = data_pack
            for idx in range(len(coming_values)):
                e = old_value[idx]; n = coming_values[idx]
                old_value[idx] = calculate_running_avg(e, old_cnt, n)
            self.profiled_data[id_layer][tag]['cnt'] += 1
        
    def store_mem(self, bit, cost, id_layer=None):
        self.profiled_data[id_layer]['mem'] = cost 
    
    def store_sta(self, bit, cost, id_layer=None):
        if self.profile_time:
            tag = 'time'
        elif self.profile_bwd_time:
            tag = 'bwd_time'
        
        self.update_cnt(id_layer)
        # nor found layer, not found bit, not found tag under bit
        if not self.profiled_data[id_layer] or not self.profiled_data[id_layer][bit] or tag not in self.profiled_data[id_layer][bit]:
            self.profiled_data[id_layer][bit][tag] = cost 
            self.profiled_data[id_layer][bit]['cnt'] = 0 # average
        else:
            # import pdb; pdb.set_trace()
            old_cost, old_cnt = self.profiled_data[id_layer][bit][tag], self.profiled_data[id_layer][bit]['cnt']
            new_cost = calculate_running_avg(old_cost, old_cnt, cost)
            self.profiled_data[id_layer][bit][tag], self.profiled_data[id_layer][bit]['cnt'] = new_cost, old_cnt + 1

    def store_current_config(self, shape, flag='fwd', id_op=None):
        cur_profile_cnt = self.mapping_dict[id_op]
        if flag == 'fwd':
            self.fwd_covs_shape[cur_profile_cnt] = shape
        elif flag == 'bwd':
            self.bwd_covs_shape[cur_profile_cnt] = shape
config = QuantizationConfig()


def set_cfg(cfg_dict: dict):
    for k, v in cfg_dict.items():
        config.k = v

