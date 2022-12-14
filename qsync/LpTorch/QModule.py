import torch
import torch.nn as nn 
import torch.distributed as dist
import numpy as np 
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from torch import Tensor, device, dtype
from collections import OrderedDict

from .conf import config
from .layers import QConv2d, QLinear, QReLU, QMaxPool2d, QBatchNorm2d, convert_layers, QMatmul
from .ILP_solver import select_bit_with_ILP
import os 
import math
from .prof_sta import sta_result_cifar, sta_result_imagenet, sta_result_transformers
from .utils import get_RD_matrix
import pickle

from .hooks import register_bwd_collection_hook
import heapq as hq 
# last_layer_inp = {"key": None}
# def classifier_tracker(module, inp, output):
#     last_layer_inp["key"] = inp[0].detach().clone()

from qsync.Predictor.indicator import compute_with_variance, get_ind_with_value, reduce_with_step, increase_with_step, map_bit_setup_to_heap

class QModule(nn.Module):
    def __init__(self, model, rank=0, local_rank=0, _world_size=0, gamma=4, profile_iter=10, \
                profiled_data_path='profile_data.npz', \
                enable_wag=False, enable_ILP=True, enable_simu=False, optimized_fp16_bp=False, args=None, enable_db=False, simu_arch=0, \
                re_ILP=6, warmup_iters=5, enable_cudnn=False, is_channel_last=True, enable_runtime=False, is_adjustment_node=False, runtime_period=1000, \
                indicator_type=0, turnon_period_collect=False):
        super().__init__()

        config.DynamicModule = tuple(list(config.DynamicModule) + [QMatmul])
        config.indicator_type = config.ava_indicator_types[indicator_type]
        print("use indiator", config.indicator_type)
        
        print("---  Enable QSync ---")
        if enable_simu:
            config.enable_simu(simu_arch)
            print("Simulated QSync Performance")
        
        self.enable_db = enable_db # turn on DB
        assert re_ILP > warmup_iters, "reoptimized ILP should after the warmup"
        self.re_ILP = re_ILP

        
        self.model = model
        self.L = None # model length, which is required in indicator
        # for the moment, for conv model we only have channel last implementation
        if is_channel_last: config.channels_last = True ; config.nchw = False
        if enable_cudnn: config.cudnn = True

        [self.qconv_list, self.qli_list, self.qbn_list, self.qln_list, self.qemb_list, self.qgelu_list, self.qmatmul_list], self.name_to_module_mapper = convert_layers(self.model)
        # import pdb; pdb.set_trace()
        # model.cuda()
        if hasattr(model, 'device'):
            self.device = model.device
        else:
            self.device = torch.device(local_rank)
        self.step = 5 # control the batch updating step, fast first
        self.observation_window_size = 5
        self.cur_window = 0

        self.all_list = self.qconv_list + self.qli_list + self.qbn_list + self.qln_list + self.qemb_list + self.qmatmul_list
        # import pdb; pdb.set_trace()
        print("Converted kernel numbers:", len(self.all_list))
        # converted layers
        self.qconv_length = len(self.qconv_list)
        self.qli_length = len(self.qli_list)
        self.qbn_len = len(self.qbn_list)
        self.qln_len = len(self.qln_list)
        self.qemb_len = len(self.qemb_list)
        self.qmatmul_len = len(self.qmatmul_list)
        self.all_len = len(self.all_list)

        self.available_bits = config.available_bits


        # QSync Control Params
        self.w_star = 1 # the optimal weight agg 
        self.w_m_i = 1 # The gradient update for all fp nodes
        self.g_scalar = 1

        # # BS
        if args is not None and hasattr(args, 'dy_bs'):
            self.bs = args.dy_bs
            self.initial_bs = args.dy_bs
            self.l_iter_bs = self.bs

        # # distributed related blocks
        self.rank = rank
        self.local_rank = local_rank
        self._world_size = _world_size
        self.warmup_iters = warmup_iters

        self.total_memory = torch.cuda.get_device_properties(self.local_rank).total_memory
        self.available_mem = 0.95 * self.total_memory // config.unit
        self.threshold = 0.95 * self.total_memory // config.unit  # the case that not allowed

        # if _world_size > 1:
        #     worker_id = int(os.environ["DMLC_WORKER_ID"])
        #     gpus_per_node = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(','))
        #     num_workers = _world_size // gpus_per_node
        #     self.comm_port = f'tcp://{os.environ["WORKER_COMMUNICATION_URI"]}:{os.environ["WORKER_COMMUNICATION_PORT"]}'
        #     # init communication for dynamic batching
        #     dist.init_process_group(backend='nccl', init_method=self.comm_port, rank=rank, world_size=_world_size)
        #     # create internal group
        #     internal_group_ranks = [i for i in range(worker_id * gpus_per_node, \
        #      (worker_id+1) * gpus_per_node)]
        #     print(f"rank {self.rank}", internal_group_ranks)
        #     self.internal_group = dist.new_group(ranks=internal_group_ranks, backend='nccl')
        #     # self.masters_group = dist.new_group(ranks=[i * gpus_per_node for i in range(num_workers)],backend='nccl')
        #     self.num_workers = num_workers
        #     self.worker_id = worker_id
        #     self.gpus_per_node = gpus_per_node

        # if _world_size > 1:
        #     # create another channel for time sync
        #     internal_group_ranks = [i for i in self._world_size]
        #     dist.new_group(ranks=internal_group_ranks, backend='nccl')
        #     self.comm_port = f'tcp://{os.environ["WORKER_COMMUNICATION_URI"]}:{os.environ["WORKER_COMMUNICATION_PORT"]}'
        #     # init communication for dynamic batching
        #     dist.init_process_group(backend='gloo', init_method=self.comm_port, rank=rank, world_size=_world_size)

        self.ava_idx = []

        # profile result
        self.profile_iter = profile_iter # profiling iterations
        self.FPSigma = 1 * (self.all_len)
        self.Sigma = 1 * (self.all_len) # the total variance for this node
        self.AggSigma = 0 # the agg variance among nodes
        self.gamma = gamma # determine the weight contribution of bp or fp
        self.omega_matrix = None # store the sensitivity matrix
        self.RD = None # (max-min)^2, numel() matrix for all conv layer
        

        # control
        self.enable_ILP = enable_ILP
        self.enable_wag = enable_wag
        print(f"quant activation use {config.act_quant}, kernel use {config.kernel_quant}")

        # count
        self.cnt = 0

        # runtime migration
        self.l_bs_process_time = 0
        self.cur_bs_process_time = 0


        # self.comm_list = [tensor.zeros(comm_num, dtype=torch.float32) for _ in range(_world_size)]
        self.is_pioneer, self.is_straggler = False, False   


        self.determined_strategy = False
        self.enable_runtime_db = False

        if args is not None:
            self.ds_name = args.ds_name
            self.model_name = args.model_name 
            if self.ds_name == 'cifar10':
                self.memory_bs_scale = sta_result_cifar[self.model_name]['scale_mem_bs'] # number of memory increased with bs, should be same and known in previous
                self.model_size = sta_result_cifar[self.model_name]['model_size']
            elif self.ds_name == 'imagenet':
                self.memory_bs_scale = sta_result_imagenet[self.model_name]['scale_mem_bs'] # number of memory increased with bs, should be same and known in previous
                self.model_size = sta_result_imagenet[self.model_name]['model_size']
            elif self.model_name == 'BERT':
                self.memory_bs_scale = sta_result_transformers[self.model_name]['scale_mem_bs'] 
                self.model_size = sta_result_transformers[self.model_name]['model_size']

        self.failed_once = False
        self.mem_init = False
        self.is_quantized_node = False

        # self.track_last_layer()
        # self.last_layer_inp = last_layer_inp # point to the address

        # runtime adaption related
        # the runtimem adaption will affected the latecy and memory
        # we greedily maintain a heap with desceding variance gain 
        # each time update the layer at the top to its higher bits INT8 -> FP16 or FP16->FP32
        # once the latency or memory is exceeded, stop the process and turn off the future check
        # once the period is exceeded, stop the process and turn off the future check.

        # deal with slower is much simple, just move back / 现在不考虑
        self.is_adjustment_node = is_adjustment_node
        self.enable_runtime = enable_runtime
        self.mem_bound = None 
        self.cur_mem = 0
        self.collection_hooks = [] # hooks for period collection of omega data、
        comm_num = 3 # number of communication data
        self.comm_list = [torch.zeros(comm_num, dtype=torch.int64).cuda() for _ in range(_world_size)]
        self.runtime_period = runtime_period # how many iteratons do a start
        self.last_change_layer = None
        self.threshold_change = 1


        # s
        self.step = 4
        self.qsync_record = []

        config.turnon_period_collect = turnon_period_collect
    
    # turn on, off with need.
    def period_sta_collection_run(self, cur_iter):
        if config.turnon_period_collect:
            if cur_iter % config.sta_collect_perod == 0: 
                # print("collect")
                # start period collection
                config.enable_period_collect = True
            if cur_iter % config.sta_collect_perod == 1:
                config.enable_period_collect = False
    
    def record_bitwidth_change(self, records):
        np.save(f"rank_{self.rank}_bitwidth_change", records)
    

    def period_dynamic_batching(self, cur_iter, process_time, args):
        if cur_iter % self.runtime_period == 1:
            self.comm_vector = torch.tensor([self.rank, round(process_time), self.is_adjustment_node]).cuda()

            # communicate batchsize
            if self._world_size > 1:
                dist.all_gather(self.comm_list, self.comm_vector)
            
            # print(self.comm_list)
            
            np_comm_list = np.array([i.cpu().numpy() for i in self.comm_list])
            np_comm_list = np_comm_list[np.argsort(np_comm_list[:,0])] # sort by rank

            pioneer_rank = np_comm_list[np.argmax(np_comm_list[:, 1]), 0]
            straggler_rank = np_comm_list[np.argmin(np_comm_list[:, 1]), 0]
            if self.rank == pioneer_rank:
                args.dy_bs += self.step
            if self.rank == straggler_rank:
                args.dy_bs -= self.step
            
            print(np_comm_list)
            print(pioneer_rank, straggler_rank)
            print(args.dy_bs)
            dist.barrier()
    
    def period_bitwidth_adjustment(self, cur_iter, process_time):
        if cur_iter % self.runtime_period == 1:
            self.comm_vector = torch.tensor([self.rank, round(process_time), self.is_adjustment_node]).cuda()

            # communicate batchsize
            if self._world_size > 1:
                dist.all_gather(self.comm_list, self.comm_vector)
            
            # print(self.comm_list)
            
            np_comm_list = np.array([i.cpu().numpy() for i in self.comm_list])
            fp32_nodes = np_comm_list[np_comm_list[:,-1] != 1] # V100 in our trails
            lat_normal = np.mean(fp32_nodes[:,1])
            # print(fp32_nodes)
            print(lat_normal, process_time)



            if self.enable_runtime:
                if self.is_adjustment_node and abs( process_time - lat_normal) < self.threshold and process_time < lat_normal:
                    self.runtime_bitwidth_adaption()
                else:
                    self.enable_runtime = False
                    if self.last_change_layer is not None:
                        self.last_change_layer[0].reset_bits[last_change_layer[1]] # reset bit
                        self.qsync_record.pop(-1)
                        self.record_bitwidth_change(self.qsync_record)
        else:
            pass

    def runtime_bitwidth_adaption(self):
        omega_dict = self.compute_omega_matrix()
        bound_idx = len(self.available_bits) - 1
        heap_val = []
        # compute the top most layer for bitwidth change
        for layer, var_list in omega_dict.items():
            if layer not in self.name_to_module_mapper:
                continue
            layer_mod = self.name_to_module_mapper[layer]
            bit = layer_mod.bit 
            bit_idx = self.available_bits.index(bit)
            if bit_idx == bound_idx:
                continue
            neighbor_idx = bit_idx + 1
            var_gain = var_list[bit_idx] - var_list[neighbor_idx]
            # store this 
            val_pair = (var_gain, layer, bit_idx)
            hq.heappush(heap_val, val_pair)
        
        if len(heap_val) == 0: self.enable_runtime = False
        # reset bit
        iter_node = hq.nlargest(1, heap_val)[0]
        _, layer, bit_idx = iter_node
        target_layer_mod = self.name_to_module_mapper[layer]
        former_bit, new_bit = self.available_bits[bit_idx], self.available_bits[bit_idx + 1]
        # check if memory is ok
        if self.mem_bound is not None:
            mem = self.M[layer]
            mem_gain = (new_bit / 32) * mem - (former_bit / 32) * mem
            if self.cur_mem + mem_gain > mem_bound: 
                self.enable_runtime = False
                return
        target_layer_mod.reset_bits(new_bit)

        print(f"layer {layer} set {new_bit}")
        self.last_change_layer = (iter_node[1], former_bit) # stop if not ok
        self.qsync_record.append((layer, new_bit))

    

    def register_bwd_collection_hooks_for_layers(self, cur_iter):
        if cur_iter % self.runtime_period == 0:
            config.enable_act_profile()
            # register hook and collect result
            for mod in self.all_list:
                if isinstance(mod, config.DynamicModule):
                    hook = register_bwd_collection_hook(mod)
                    self.collection_hooks.append(hook)
    
    def remove_bwd_collection_hooks_for_layers(self):
        if len(self.collection_hooks) == 0: return
        for hook in self.collection_hooks:
            hook.remove()
        self.collection_hooks = []
        self.RD = get_RD_matrix(config.profiled_data, self.available_bits)
        # print(self.RD)
        config.disable_profile()
    
    def _get_RD_matrix(self):
        self.RD = get_RD_matrix(config.profiled_data, self.available_bits)
        return self.RD

    # load depth data
    def load_and_set_depth(self, model_name):
        from .utils import load_data_pickle
        path = os.environ['HOME']
        path = os.path.join(path, "configs", "model_depth", f"{model_name.lower()}_depth_data")
        abs_path = os.path.abspath(path)

        depth_data = load_data_pickle(abs_path)
        lis_depth = []
        convs_depth = []
        matmuls_depth = []
        # embs_depth = []
        max_v = 0
        for k, v in depth_data.items():
            k_lower = k.lower()
            if 'linear' in k_lower:
                lis_depth.append(v)
            elif 'conv' in k_lower:
                convs_depth.append(v)
            elif 'matmul' in k_lower:
                matmuls_depth.append(v)
            # elif 'embedding' in k_lower:
            #     print(k_lower, v)
            #     embs_depth.append(v)
            max_v = max(max_v, v)
        
        for idx, qli in enumerate(self.qli_list):
            qli.depth = lis_depth[idx]
        
        for idx, qconv in enumerate(self.qconv_list):
            qconv.depth = convs_depth[idx]
        
        for idx, qmatmul in enumerate(self.qmatmul_list):
            qmatmul.depth = matmuls_depth[idx]
        
        for idx, emb in enumerate(self.qemb_list):
            emb.depth = 0

        self.L = max_v 


    # store weight quantization result in advance
    def advance_q_weight_calculate(self):
        for conv in self.qconv_list:
            conv.quantize_weight()
    # def track_last_layer(self):
    #     classifier = list(self.model.children())[-1]
    #     classifier.register_forward_hook(classifier_tracker)
    
    # def get_last_layer_result(self):
    #     # print(self.last_layer_inp["key"])
    #     return self.last_layer_inp["key"]
    
    # def communicate_last_layer(self):
    #     sum_tensor = self.last_layer_inp["key"].detach().clone()
    #     dist.all_reduce(sum_tensor, group=self.internal_group)
    #     other_tensor = torch.empty_like(sum_tensor)
    #     # buffer = [torch.empty_like(sum_tensor) for _ in range(self.num_workers)] 
    #     if self.local_rank == 0:
    #         print("rank", self.rank)
    #         dst = (1 - self.worker_id) * self.gpus_per_node
    #         dist.isend(sum_tensor, dst)
    #         handle = dist.irecv(other_tensor, dst)
    #         handle.wait()
    #         # dist.all_gather(buffer, sum_tensor, group=self.masters_group)
    #     dist.barrier()
    #     # in our case, we only have two groups, buffer suppose to be two
    #     assert len(buffer) == 2, 'wrong number of clusters'
            
    '''
        State Control function for QModel
    '''
    
    # weight conversion, for debuging
    def convert_to_nhwc(self):
        for conv in self.qconv_list:
            conv.convert_to_nhwc() 
        
    def convert_to_nchw(self): 
        for conv in self.qconv_list:
            conv.convert_to_nchw()

    # get optimal bit selection given constraints
    # def find_optimal_bit(self):
    #     if not self.is_quantized_node: return 
    #     assert self.M_B and self.T_B, "Require constraints for ILP problem"
    #     print("Do ILP to find the optimal bits")
    #     if not self.enable_ILP: 
    #         omega_matrix = np.random.shuffle(self.omega_matrix)  # random selection / ablation to scheme
    #     else:
    #         omega_matrix = self.omega_matrix
    #     df, opt_val = select_bit_with_ILP(self.M_B, self.T_B, self.M, self.E, omega_matrix, len(self.E), available_bits = self.available_bits)
    #     print("ILP DONE, new node variance", opt_val)
    #     self.Sigma = opt_val
    #     self.set_ILP_result(df)
    #     self.bit_df = df

    # INIT the QSync, DO ILP
    def init_mem(self, mem_ocp_rate, T_B):
        rate = 1 - mem_ocp_rate
        self.threshold = round(self.available_mem * rate)
        # print("rate", rate, self.threshold, self.available_mem)
        # print(self.model_size, self.memory_bs_scale)
        if not (self.memory_bs_scale * self.bs + self.model_size >= self.threshold) and T_B >= self.F_E:
            print(f"Rank {self.rank}. Memroy ok, no initial bs change.")
            self.is_quantized_node = False
        else:
            thresh = 10 # negelectable difference. system vibration
            # Constraints
            LAT = T_B
            delta = (self.memory_bs_scale * self.bs + self.model_size - self.threshold)
            delta = 0 if delta < 0 else delta
            M_B = self.max_M - delta
            print("M_B required:", M_B)
            LAT_MAX = self.max_E
            self.M_B = M_B
            self.T_B = T_B
            self.find_optimal_bit()
            self.is_quantized_node = True
        self.mem_init = True
        if self._world_size > 1:
            dist.barrier()
            

    
    def record_last_iter_info(self, cur_bs_process_time):
        self.l_iter_bs = self.bs
        self.l_bs_process_time = self.cur_bs_process_time
        self.cur_bs_process_time = cur_bs_process_time

    def get_current_gpu_memory(self):
        # get current gpu memory here
        self.cur_mem = torch.cuda.memory_allocated(self.local_rank) // config.unit
    
    def check_mem_availability(self):
        # since we emulate the performance, this is disabled.
        self.get_current_gpu_memory()
        # self.cur_mem = self.model_size + self.bs * self.memory_bs_scale
        if self.cur_mem + self.step * self.memory_bs_scale >= self.threshold:
            return False
        return True
    
    # def convert_weight_to_half(self):
    #     print("Converting all weight in module to half")
    #     for mod in self.all_list:
    #         if isinstance(mod, config.DynamicModule):
    #             mod.half() # convert to half
    
    def set_bits(self, bit):
        print(f"Set bit with {bit}")
        print("lenghth:", len(self.qconv_list), len(self.qli_list), len(self.qbn_list), len(self.qln_list), len(self.qemb_list))
        for mod in self.all_list:
            if isinstance(mod, config.DynamicModule):
                # print(mod, bit)
                mod.reset_bits(bit)
        self.is_quantized_node = self.is_quantized_model()
    
    def set_bits_in_layers(self, bit, layer):
        for name, mod in layer.named_modules():
            if isinstance(mod, config.DynamicModule):
                print(mod, bit)
                mod.reset_bits(bit)
        self.is_quantized_node = self.is_quantized_model()
    
    def set_bs(self, n):
        self.bs = n

    # profile
    def get_available_list(self):
        
        if len(self.ava_idx) == 0:
            ava_list = []
            ava_list_idx = []
            for i, conv in enumerate(self.all_list):
                if conv.used:
                    ava_list.append(conv)
                    ava_list_idx.append(i)
            self.ava_list = ava_list
            self.ava_idx = ava_list_idx
        else:
            # print(len(self.ava_idx))
            # print(len(self.all_list))
            self.ava_list = [self.all_list[i] for i in self.ava_idx]
        
        self.FPSigma = 1 * len(self.ava_list)
        self.Sigma = 1 * len(self.ava_list)
        return self.ava_idx


     # Got indicator matrix
    def compute_omega_matrix(self):
        omega_matrix, omega_dict = compute_with_variance(self.name_to_module_mapper, self.available_bits, self.gamma, self.RD, self.L)
        self.omega_matrix = np.array(omega_matrix)
        # print(omega_matrix)
        # import pdb; pdb.set_trace()
        # self.Sigma_max = 1 / self.omega_matrix.sum(axis=0)[0]
        print("Get New Omega Matrix")
        return omega_dict
    

    # load the profiled data
    def load_profile_file(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.M, self.E, self.RD, self.omega_matrix, self.min_M, self.min_E, self.max_M, self.max_E, self.ava_idx, self.F_E  = \
            data['M'], data['E'], data['RD'], data['omega'], data['minM'], data['minE'], data['maxM'], data['maxE'], data['ava_idx'], data['F_E']
        print("Profiled Data Loaded")
        # self.Sigma_max = 1 / data['omega'].sum(axis=0)[0]
        self.get_available_list()
        
        
    '''
       ILP related
    '''
    def find_optimal_bit(self):
        M_B, T_B = self.M_B, self.T_B
        print("Do ILP to find the optimal bits")
        if not self.enable_ILP: 
            omega_matrix = np.random.shuffle(self.omega_matrix)  # random selection / ablation to scheme
        else:
            omega_matrix = self.omega_matrix
        LAT = T_B
        # Do iterative optimization 
        # Since 8bit allows 4x compression. And it should be always enough to solve the below problem
        failed_once = False # record whether it can be solved once
        print("Try ILP with", M_B, LAT)

        df, opt_val = select_bit_with_ILP(M_B, LAT, self.M, self.E, omega_matrix, available_bits = self.available_bits)
        if df is None:
            failed_once = True
            for LAT_i in np.arange(LAT, self.max_E, 10):
                print(M_B, LAT_i)
                df, opt_val = select_bit_with_ILP(M_B, LAT_i, self.M, self.E, omega_matrix, available_bits = self.available_bits)
                if df is None:
                    continue
                else:
                    break
            if df is not None:
                print("ILP DONE, new node variance", opt_val)
                self.M_B, self.T_B = M_B, LAT_i
            else:
                print("Exceed the ability")
        else:
            self.M_B, self.T_B = M_B, LAT

        self.Sigma = opt_val
        self.set_ILP_result(df)
        self.bit_df = df

        self.failed_once = failed_once

    # 
    def set_mapper_result(self, data_path):
        # with open(data_path, 'rb') as f:
        #     data = pickle.load(f)
        data = np.load(data_path, allow_pickle=True).item()
        for k, bit in data.items():
            if k not in self.name_to_module_mapper:
                continue
            layer = self.name_to_module_mapper[k]
            if isinstance(layer, config.DynamicModule):
                layer.reset_bits(bit)
                print(f"layer {layer} set bits {bit}")
        # import pdb; pdb.set_trace()
        

        
    
    # set ILP result to layers
    def set_ILP_result(self, df):
        available_bits = self.available_bits
        res = df.index[df['solution_value'] == 1].tolist()
        for idx, i in enumerate(res):
            layer = df['column_i'][i]
            bit = available_bits[df['column_j'][i]]
            # Calculation here get the selection for each layer in sequential order.
            self.ava_list[layer].reset_bits(bit)
            # print(self.ava_list[layer], f'set bit {bit}')
    
    '''
       Variance Indicated: Weighted Averaged Gradient
    '''

    
    def adjust_g_scalar(self):
        # new_g_scalar = round(self.bs / self.all_bs, 5)
        # new_g_scalar = math.ceil(self.initial_bs / self.bs)
        new_g_scalar = round(self.initial_bs / self.bs, 10)
        # new_g_scalar = self.initial_bs
        self.g_scalar = new_g_scalar

    # communicate to get the agg sigma
    def get_sum_sigma_term(self):
        if not self.enable_wag: return 
        div_Sigma = 1 / self.Sigma
        self.AggSigma = div_Sigma
        if self._world_size > 1:
            work = dist.all_reduce(self.AggSigma)  # SUM
        
    # compute new weight for gradient
    def compute_new_w_star(self):
        if not self.enable_wag: return 
        if self._world_size > 1:
            self.w_star = self.Sigma * (1 / self.AggSigma) * self._world_size # since torch ddp has a 1/world_size agg
        else:
            self.w_star = 1
        # print("Get new w_star", self.w_star)
        # inverse sigma here
        minimum = self._world_size * self.Sigma_max
        maximum = self._world_size * 1 / self.FPSigma
        QLEVEL = (maximum - self.AggSigma) / (maximum - minimum)
        self.QLevel = QLEVEL
        print("QLEVEL -------- :", self.QLevel)
        # print(self.AggSigma, minimum, maximum)

    def is_quantized_model(self):
        for n, mod in self.model.named_modules():
            if isinstance(mod, (QConv2d, QLinear)):
                if mod.bit != 32:
                    return True
        return False
    
    def reduce_with_h(self):
        bits_updating_map = reduce_with_step(self.reduce_h, self.gap_column, self.step, self.incre_h, self.temp_incre_h, self.temp_decre_h)
        for bit_map in bits_updating_map:
            self.ava_list[self.ava_idx[bit_map[0]]].reset_bits(self.available_bits[bit_map[1]])  # update the bit
        if len(bits_updating_map) != 0:
            print(f"Straggler Rank {self.rank} reduce bit") 
            print(bits_updating_map)
        else:
            self.runtime_method_name = 'db'
            self.enable_runtime_db = True
        
    
    def increase_with_h(self):
        bits_updating_map = increase_with_step(self.incre_h, self.gap_column, self.step, self.reduce_h, self.temp_incre_h, self.temp_decre_h)
        for bit_map in bits_updating_map:
            self.ava_list[self.ava_idx[bit_map[0]]].reset_bits(self.available_bits[bit_map[1]]) # update the bit
        if len(bits_updating_map) != 0:
            print(f"Pioneer Rank {self.rank} add bit") 
            print(bits_updating_map)

    def find_pioneer_qsync(self):
        # sort
        np_comm_list = np.array(self.comm_list)
        np_comm_list = np_comm_list[np.argsort(np_comm_list[:,0])] # sort by rank
        # Different from otther methods. Calculate the mean of the non-quantized node
        # (which is our quantization target)
        normal_nodes = np_comm_list[np_comm_list[:,5] == 7] # V100 in our trails
        lat_normal = np.mean(normal_nodes[:1])
        # there are two cases
        # 1. lat of the current node is <= lat_normal, which means the quant level is too high
        # 2. lat of the current node is >= lat-

    def find_pioneer(self):
        # sort
        np_comm_list = np.array(self.comm_list)
        np_comm_list = np_comm_list[np.argsort(np_comm_list[:,0])] # sort by rank

        self.np_comm_list = np_comm_list
        ava_list = np_comm_list[np_comm_list[:,5] == 1]
        if len(ava_list!=0):
            pioneer_idx = ava_list[np.argmin(ava_list[:, 1]), 0]
        else:
            pioneer_idx = None
        ava_straggler_list = np_comm_list[np_comm_list[:,4] > self.step]
        straggler_idx = ava_straggler_list[np.argmax(ava_straggler_list[:, 1]), 0]
        if pioneer_idx is not None and self.rank == pioneer_idx: self.is_pioneer = True
        if self.rank == straggler_idx: self.is_straggler = True
        self.all_bs = np.sum(np_comm_list[:, 4])



        return int(pioneer_idx), int(straggler_idx)
    

    def create_heap_for_adjustment(self):
        if self.mem_saving_first:
            self.gap_column = get_ind_with_value(self.M)
        else:
            self.gap_column = get_ind_with_value(self.E)
        self.incre_h, self.reduce_h = [],[]
        self.temp_incre_h, self.temp_decre_h = [], []
            
    def adjust_heap(self):
        if self.is_quantized_node:
            self.create_heap_for_adjustment()
            map_bit_setup_to_heap(self.bit_df, self.gap_column, self.reduce_h, self.incre_h, self.available_bits)  # reset the heap for adjustment
    
    # base on the information to choose the subsequent strategy
    def strategy_switcher(self):
        if self.determined_strategy: return  # this function will be only run once
        self.determined_strategy = True
        if not self.mem_init:
            self.mem_saving_first = False 
            self.adjust_heap()
            self.runtime_method_name = 'qsync' 
        mean_pioneer_lat = np.mean(self.np_comm_list[self.np_comm_list[:, -1] != True][:, 1])
        cur_lat = self.np_comm_list[self.rank, 1]
        if cur_lat < mean_pioneer_lat: # quantization speed up is enough
            # memory as priority to reduce the oscilation
            self.mem_saving_first = True
            self.adjust_heap()
            self.runtime_method_name = 'qsync'     
        else: # quantization is not enough for the case
            if self.failed_once: # means ILP failed at least once that lead to the delay
                # for this case. We cannot use quantization to smooth the straggler. Thus use dynamic batching if possible.
                self.runtime_method_name = 'db'
            else: # ILP not failed, which means the current speed may be systemetic. 
                self.mem_saving_first = False
                self.adjust_heap()
                self.runtime_method_name = 'qsync'     
    
    def runtime_method(self, pioneer_idx, straggler_idx):
        # We don't perform runtime ILP becuase it will greately degrade the performance, especially in later stage of training
        # We base on the scarcity to choose optimization scheme to reduce the cost. Use a priority queue (OLogN) to minimize the difference
        if pioneer_idx is not None and self.np_comm_list[pioneer_idx, 2] < self.np_comm_list[straggler_idx, 2]:
            if self.runtime_method_name != 'db':
                if self.np_comm_list[straggler_idx, 6]: 
                    self.enable_runtime_db = True
                    self.runtime_method_name = 'db' # straggler indicate all nodes to run db
            if self.runtime_method_name == 'db':
                # perform runtime DB 
                if not self.enable_db: return  # don't perform any further operations for the node (full qsync)
                else: # qsync + db
                      # the drop-in DB method
                    if self.is_straggler:
                        if self.bs < self.step:
                            print("Remove this straggler!")
                            pass
                            # don't change the bs
                        else:
                            # the pinoneer must have the same logic, thus reduce here
                            self.set_bs(self.bs - self.step)
                            self.adjust_g_scalar()
                            print(f"Straggler Rank {self.rank} update bs {self.bs}")      
                    if self.is_pioneer:
                        if self.comm_list[straggler_idx][4] > self.step: # check whether the straggler is removable
                            self.set_bs(self.bs + self.step)
                            self.adjust_g_scalar()
                            print(f"Pioneer Rank {self.rank} update bs {self.bs}")      
                        else:
                            print("Pioneer not available")
                    pass     
            elif self.runtime_method_name == 'qsync':
                
                # perform runtime qsync
                # runtime qsync drop in method
                # Do nothing for the 
                if self.is_quantized_node:
                    if self.is_straggler:
                        # the pinoneer must have the same logic, thus reduce here
                        if len(self.temp_decre_h) > 0:
                            print("Exceed, go finetuning")
                            self.step = 1  
                            self.observation_window_size = 20 
                        self.reduce_with_h()
                    if self.is_pioneer:
                        if len(self.temp_incre_h) > 0:
                            print("Exceed, go finetuning")
                            self.step = 1  
                            self.observation_window_size = 20 
                        self.increase_with_h()
        else:
            # for all node, go into finetuning stage.
            print("Exceed, go finetuning")
            self.step = 1  
            self.observation_window_size = 20 

        # reset
        self.is_pioneer, self.is_straggler = False, False

    # according to the memory change, modify the model
    def runtime_qsync(self):
        self.cur_window += 1
        if self.cur_window <= self.warmup_iters: # do nothing in warmup
            return 
        
        # communicate the get target information
        availability = self.check_mem_availability()
        # for qsync, we don't move batch but 
        self.comm_vector = [self.rank, self.l_bs_process_time, self.cur_bs_process_time, self.threshold, self.bs, availability, \
         self.enable_runtime_db, self.is_quantized_node]
        # communicate batchsize
        if self._world_size > 1:
            dist.all_gather_object(self.comm_list, self.comm_vector)
        pioneer_idx, straggler_idx = self.find_pioneer()
        # switch strategy: once
        self.strategy_switcher()
        # do ILP if necessary
        if self.cur_window == self.re_ILP:
            if self.is_quantized_node:
                print("Re Calculate ILP")
                self.find_optimal_bit()
                map_bit_setup_to_heap(self.bit_df, self.gap_column, self.reduce_h, self.incre_h, self.available_bits)  # reset the heap for adjustment
                # debug
                # torch.save(self.gap_column, 'gap_column.pt')
                # torch.save(self.incre_h, 'incre.pt')
                # torch.save(self.reduce_h, 'reduce.pt')
        

        # run qsync with respect to observation window-size
        if self.cur_window % self.observation_window_size == 0: 
            print(f"windows {self.cur_window}, {self.rank} Run QSync")
            if self.rank == 0:
                print(self.np_comm_list)
                print(pioneer_idx, straggler_idx)
                print("Strategy:", self.runtime_method_name, "mem first?", self.mem_saving_first)
                print(f"rank{self.rank} pioneer idx {pioneer_idx}, straggler idx {straggler_idx}")
        else: 
            return 
        self.runtime_method(pioneer_idx, straggler_idx)
       
        

    # Model Related.
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        return super().train(mode)

    def eval(self):
        return super().eval()

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        # remove the prefix "model." added by this wrapper
        new_state_dict = OrderedDict([("model." + k,  v) for k, v in state_dict.items()])
        return super().load_state_dict(new_state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super().state_dict(destination, prefix, keep_vars)

        # remove the prefix "model." added by this wrapper
        ret = OrderedDict([(k[6:], v) for k, v in ret.items()])
        return ret
    
    # forward collection
    def collect_with_forward_hook(self, fn):
        for mod in self.ava_list:
            mod.register_forward_hook(fn)
    
    def collect_with_backward_hook(self, list_grad):
        for mod in self.ava_list:
            if hasattr(mod, 'weight'):
                list_grad.append(mod.weight.grad)


class QSyncTrainer:
    def __init__(self, T = 15, max_update_times = 10):
        self.T = T # period for updating, iterations
        self.itc = T
        self.cnt = 0 # record iteration


        self.update_time = 0
        self.max_update_times = max_update_times


    def one_pass(self, model, train_pass):
        if hasattr(model, "module"):
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        
        if model_without_ddp.cnt == self.itc and self.update_time < self.max_update_times:
            print("-------- Update w_star --------")
            self.itc += self.T
            # QNode
            if model_without_ddp.is_quantized_node:
                # update variance information according to the new graidnet
                model_without_ddp.compute_omega_matrix()
                model_without_ddp.find_optimal_bit()
            model_without_ddp.get_sum_sigma_term()
            model_without_ddp.compute_new_w_star()
            # profile new activation result
            config.profile_act = True
            result = train_pass()
            # profile new gradient result
            config.profile_act = False
            model_without_ddp.adjust_gradient_with_w_star()
            nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm=2.0, norm_type=2)
            
            self.update_time += 1 # restrict the updating in later stage
        else:
            result = train_pass()
            model_without_ddp.adjust_gradient_with_w_star()
            nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm=2.0, norm_type=2)


        return result