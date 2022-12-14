import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
import torch.autograd.profiler as profiler
from torch.nn.modules.pooling import _size_2_t, _single, _pair, _triple, _MaxPoolNd, _AvgPoolNd

from .ops import *
# from test_ops import *
import time
from collections import defaultdict

from .conf import config, calculate_running_avg    
from .functions import padding_input_weight_nchw
from .quant import get_amax, get_scale, collect_scale, quant_with_buffer
from .env import *
from .cppfrontend import ext_quantization

# import indicator profiler here
from qsync.Predictor.indicator_profiler import profile_stas, profile_stas_sole_input, profile_stas_emb

# profile blocks for the layers who only have inputs and don't have weight
# some ops, like dropout, stores 1.2x mem instead of 1x activation mem
def input_only_layer_profiler_block(unique_id=None, input=None, scalar=1):
    if unique_id is None:
        return # trace 
    with torch.no_grad():
        # in context. input, weight, bias saved
        if config.profile_mem:
            config.store_mem(32, compute_tensor_bytes([input]) * scalar / config.unit, unique_id)
        
        if config.profile_act: # this op don't introduce extra variance. and the data collect here will never be used
            profile_stas_sole_input(input, None, unique_id)


def gemm_layer_profiler_block(unique_id=None, input=None, input2=None, res=None, scalar=1):
    if unique_id is None:
        return # trace 
    with torch.no_grad():
        # in context. input, weight, bias saved
        if config.profile_mem:
            config.store_mem(32, compute_tensor_bytes([input, input2]) * scalar / config.unit, unique_id)
        
        if config.profile_act: # this op don't introduce extra variance. and the data collect here will never be used
            profile_stas(input, input2, res, None, None, unique_id)

# for layers who only support fp due to implementation
def bit_reset_float_only(model, bit):
    model.bit = bit
    if model.bit == 32:
        model.float()
    elif config.enable_half_conversion:
        model.half()
    
def input_only_fp32_cast(input):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::cast_cost_fp16"): # for half conversion, always use this
            input = input.float()
    else:
        input = input.float()
    return input 

# half float conversion
def tensor_to_half(p_tensor):
    if p_tensor is None or p_tensor.dtype == torch.float16:
        return p_tensor
    else:
        return p_tensor.half()
    
def tensor_to_float(p_tensor):
    if p_tensor is None or p_tensor.dtype == torch.float32:
        return p_tensor
    else:
        return p_tensor.float()
    

# the current version only support nhwc
class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 ):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.reset_bits(32)
        self.profile = False
        
        # get quant param in advance
        # self.act_para = nn.Parameter(torch.zeros(1, device=self.weight.device))
        # self.weight_para = nn.Parameter(torch.zeros(1, device=self.weight.device))
        self.cn_2nhwc = NCHW2NHWCFunction.apply
        self.cn_2nchw = NHWC2NCHWFunction.apply
        self.torch_conv = TorchConvInterface.apply

        self.norm = None
        self.activation = None # fusion did in the Detectron
        self.used = False # in implementation of detectron, just find that some convolution is not used in the forward pass.

        self.check_weight_size()
        self.name = 'conv'
        self.unique_id = None


        self.scale_weight = None
        self.scale_inp = None
        self.scale_output = None 


        self.qweight = None # quantize weight in advance


    def collect_scales(self, input, weight):
        if not config.enable_period_collect: return 
        # print("collect")
        with torch.no_grad():
            scale_inp = collect_scale(input, config.act_quant)
            scale_weight = collect_scale(weight, config.kernel_quant)
            self.scale_inp = scale_inp
            self.scale_weight = scale_weight
            self.scale_output = scale_inp * scale_weight
    
    def quantize_weight(self):
        if self.qweight is None: 
            self.qweight = torch.empty_like(self.weight, dtype=torch.int8, requires_grad=False)

        if self.bit <= 8:
            scale_w = quant_with_buffer(self.weight, self.qweight)
            self.scale_weight = scale_w
        
        
    
    def check_weight_size(self):
        if self.weight.size(1) == 3:
            self.first_conv = True
        else:
            self.first_conv = False

    # determine layer will run with which layout
    def convert_to_nhwc(self):
        self.weight = nn.Parameter(convert_from_nchw_to_nhwc(self.weight))
    
    def convert_to_nchw(self):
        print("If nchw by default, pls don't call this method")
        # TODO: we haven't implemented the NCHW kernel right now
        self.weight = nn.Parameter(convert_from_nhwc_to_nchw(self.weight))
    
    def reset_bits(self, bit):
        self.bit = bit
        # self.check_weight_size()
        # if self.first_conv:
        #     self.fn = Conv2dFP32.apply
        #     return 
        if self.bit == 4:
            self.fn = Conv2dInt4.apply
        elif self.bit == 8:
            self.fn = Conv2dInt8.apply
        elif self.bit == 16:
            self.fn = Conv2dFP16.apply
        elif self.bit == 19:
            self.fn = Conv2dTF32.apply
        else:
            self.fn = Conv2dFP32.apply
    
    def cast_tensor(self, input):
        if self.bit == 16:
            input = tensor_to_half(input)
            weight = tensor_to_half(self.weight)
            bias = tensor_to_half(self.bias)
        else:
            weight = tensor_to_float(self.weight)
            bias = tensor_to_float(self.bias) 
            input = tensor_to_float(input)
        return weight, bias, input

    def forward(self, input):
        if config.simu:
            input_padded, weight_padded = padding_input_weight_nchw(input, self.weight)
            if self.training:
                x = self.torch_conv(input_padded, weight_padded, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups,
                                        self.bit, self.unique_id)
            else:
                x = self.torch_conv(input_padded, weight_padded, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups,
                                        32, self.unique_id)
        else:
            
            # Pytorch Backbone run in NCHW. Since we only support NHWC training. If the channels_last is not configured, do conversion.
            if not config.channels_last: 
                input = self.cn_2nhwc(input)

            if config.profile_time:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            
            # tensore core cannot be used
            # if sum([input.shape[0] % 4 + input.shape[1] % 4]):
            #     res = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups,)
            # else:
            self.collect_scales(input, self.weight)
            
            if self.training:
                res = self.fn(input, self.weight, self.qweight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups, self.unique_id, self.scale_weight, self.scale_inp, self.scale_output)
                                        
            else:
                res = Conv2dFP32.apply(input, self.weight, self.qweight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups, self.unique_id, self.scale_weight, self.scale_inp, self.scale_output)

            if config.profile_time:
                end.record()
                torch.cuda.synchronize()
                Duration = start.elapsed_time(end)
                config.store_sta(self.bit, Duration, self.unique_id)
                if not self.used: self.used = True
            
            

            if not config.channels_last:
                res = self.cn_2nchw(res) # convert back to nchw

            x = res
        # object detection
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        # print(x.is_contiguous(memory_format=torch.channels_last))
        return x

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=None):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.reset_bits(32)
        self.used = False # in implementation of detectron, just find that some convolution is not used in the forward pass.
        self.name = 'linear'
        self.unique_id = None


        # self maintained quantization params
        self.scale_weight = None
        self.scale_inp = None
        self.scale_output = None

        # register_collection_hooks(self)
        self.qweight = None # quantize weight in advance

    def collect_scales(self, input, weight):
        if not config.enable_period_collect: return 
        with torch.no_grad():
            scale_inp = collect_scale(input, config.act_quant)
            scale_weight = collect_scale(weight, config.kernel_quant)
            self.scale_inp = scale_inp
            self.scale_weight = scale_weight
            self.scale_output = scale_inp * scale_weight
    
    def quantize_weight(self):
        if self.qweight is None: 
            self.qweight = torch.empty_like(self.weight, dtype=torch.int8, requires_grad=False)

        if self.bit <= 8:
            scale_w = quant_with_buffer(self.weight, self.qweight)
            self.scale_weight = scale_w

    def reset_bits(self, bit):
        self.bit = bit
        if self.bit == 8:
            self.fn = LinearInt8.apply
        elif self.bit == 16:
            self.fn = LinearFP16.apply
        elif self.bit == 19:
            self.fn = LinearTF32.apply
        else:
            self.fn = LinearFP32.apply

        if self.bit == 32:
            self.float()
        elif config.enable_half_conversion:
            self.half()
    
    def cast_tensor(self, input):
        if self.bit == 16:
            input = tensor_to_half(input)
            weight = tensor_to_half(self.weight)
            bias = tensor_to_half(self.bias)
        else:
            weight = tensor_to_float(self.weight)
            bias = tensor_to_float(self.bias) 
            input = tensor_to_float(input)
        return weight, bias, input

    def forward(self, input):
        # weight = torch.t(weight).contiguous()

        # Torch implementation is much more efficient
        # if config.profile_time:
        #     start = torch.cuda.Event(enable_timing=True)
        #     end = torch.cuda.Event(enable_timing=True)
        #     start.record()
        # res = F.linear(input, weight, bias)
        # if config.profile_time:
        #     end.record()
        #     torch.cuda.synchronize()
        #     Duration = start.elapsed_time(end)
        #     config.store_sta(self.bit, Duration, self.id_linear)
        #     if not self.used: self.used = True
        #     config.store_current_config(res.shape, id_op=self.id_linear)
        # return res

        # # print(input.shape, self.weight.shape)
        # # import pdb; pdb.set_trace()

        self.collect_scales(input, self.weight)

        if config.profile_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.training:
            # if config.pytorch_profiler:
            #     with profiler.record_function("qsync::cast_cost_fp16"):
            #             weight, bias, input = self.cast_tensor(input)
            # else:
            #     weight, bias, input = self.cast_tensor(input)
            res = self.fn(input, self.weight, self.qweight, self.unique_id, self.scale_weight, self.scale_inp, self.scale_output) 
        else:
            res = LinearFP32.apply(input, self.weight, self.qweight, self.unique_id, self.scale_weight, self.scale_inp, self.scale_output)
        # print(self)
        # print(res)
        if config.profile_time:
            end.record()
            torch.cuda.synchronize()
            Duration = start.elapsed_time(end)
            config.store_sta(self.bit, Duration, self.unique_id)
            if not self.used: self.used = True
            config.store_current_config(res.shape, id_op=self.unique_id)
            
        if self.bias is not None:
            res += self.bias

        # print(res)
        return res



class QReLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.name = 'relu'
        self.inplace=inplace
        self.unique_id = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # if config.simu:
        #     return F.relu(input, inplace=self.inplace)
        input_only_layer_profiler_block(self.unique_id, input, scalar=0)

        if config.pytorch_profiler:
            with profiler.record_function("act::relu"):
                res = ext_quantization.act_quantized_relu(input)
        else:
            res = ext_quantization.act_quantized_relu(input)
        return res
        # res = F.relu(input, inplace=self.inplace)
        # print(self, res.is_contiguous(memory_format=torch.channels_last))
        # return res

class QDropout(nn.Dropout):
    def __init__(self, p=0.5):
        super().__init__(p=p)
        self.name = 'dropout'
        self.unique_id = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Dropout store int8 for bwd
        input_only_layer_profiler_block(self.unique_id, input, scalar=0.25)
        if self.training:
            return F.dropout(input, p=self.p)
        else:
            return super(QDropout, self).forward(input)

class QMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.name = 'maxpool2d'
        self.unique_id = None
        self.bit = 32 
    
    def reset_bits(self, bit):
        bit_reset_float_only(self, bit)
    
    def cast_tensor(self, input):
        if self.bit <= 16:
            h_input = tensor_to_half(input) 
        else:
            h_input = tensor_to_float(input)
        return h_input
    
    def forward(self, input):
        # input = self.cast_tensor(input)
        input_only_layer_profiler_block(self.unique_id, input)
        input = self.cast_tensor(input)
        res = F.max_pool2d(input, self.kernel_size, self.stride,
            self.padding, self.dilation, self.ceil_mode,
            self.return_indices)
        # print(self, res.is_contiguous(memory_format=torch.channels_last))
        return res
# https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#layer_norm
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/layer_norm.cpp

#  LayerNorm and Embedding only use half. Since greatly affected the result we got. 
class QLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(QLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.bit = 32
        self.id_ln = id(self)
        self.used = False
        self.name = 'layernorm'
        self.unique_id = None
    
    def reset_bits(self, bit):
        self.bit = bit
        if self.bit == 32:
            self.float()
        elif config.enable_half_conversion:
            self.half()
        
    
    def cast_tensor(self, input):
        if self.bit <= 16: # layer norm don't have implementation <= 16
            weight = tensor_to_half(self.weight) 
            bias = tensor_to_half(self.bias) 
            input  = tensor_to_half(input)
        else:
            weight = tensor_to_float(self.weight)
            bias = tensor_to_float(self.bias) 
            input = tensor_to_float(input)
        return weight, bias, input 
    
    def forward(self, input):
        if config.profile_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        # only support two type. 32 or half
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                weight, bias, input = self.cast_tensor(input)
        else:
            weight, bias, input = self.cast_tensor(input)
        
        # in context. input, weight, bias saved
        if config.profile_mem:
            if bias is None: 
                config.store_mem(self.bit, compute_tensor_bytes([input, weight]) / config.unit, self.unique_id)
            else:
                config.store_mem(self.bit, compute_tensor_bytes([input, weight, bias]) / config.unit, self.unique_id)
        
        
        # try:
        res = F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        # res = layer_norm.apply(input, self.normalized_shape, self.weight, self.bias, self.eps, self.bit, self.id_ln)
        # except:
        #     import pdb; pdb.set_trace()


        if config.profile_act: # collect the activation information
            profile_stas(input, weight, res, None, None, self.unique_id)

        if config.profile_time:
            end.record()
            torch.cuda.synchronize()
            Duration = start.elapsed_time(end)
            config.store_sta(self.bit, Duration, self.id_ln)
            if not self.used: self.used = True
        

        return res


class QBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(QBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.bit = 32
        self.id_bn = id(self)
        self.used = False
        self.name = 'bn'
        self.unique_id = None
    
    def reset_bits(self, bit):
        self.bit = bit
        if self.bit == 32:
            self.float()
        else:
            # bn support at most fp16
            self.bit = 16
            if config.enable_half_conversion:
                self.half()
    
    def cast_tensor(self, input):
        if self.bit <= 16: # layer norm don't have implementation <= 16
            weight = tensor_to_half(self.weight) 
            bias = tensor_to_half(self.bias) 
            input  = tensor_to_half(input)
        else:
            weight = tensor_to_float(self.weight)
            bias = tensor_to_float(self.bias) 
            input = tensor_to_float(input)
        return weight, bias, input 

    def forward(self, input):
        # only support two type. 32 or half
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                weight, bias, input = self.cast_tensor(input)
        else:
            weight, bias, input = self.cast_tensor(input)
        
        self._check_input_dim(input)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)


        if config.simu:
            return F.batch_norm(input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight, bias, bn_training, self.momentum, self.eps)
            
        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if config.profile_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        if config.profile_mem:
            if bias is None:
                config.store_mem(self.bit, compute_tensor_bytes([input]) / config.unit, self.unique_id)
            else:
                config.store_mem(self.bit, compute_tensor_bytes([input, bias]) / config.unit, self.unique_id)

        res = F.batch_norm(input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight, bias, bn_training, self.momentum, self.eps)
        
        if config.profile_act: # collect the activation information
            profile_stas(input, weight, res, None, None, self.unique_id)
        
        # print(self, res.is_contiguous(memory_format=torch.channels_last))
        # res = batch_norm.apply(
        #     input,
        #     # If buffers are not to be tracked, ensure that they won't be updated
        #     self.running_mean if not self.training or self.track_running_stats else None,
        #     self.running_var if not self.training or self.track_running_stats else None,
        #     weight, bias, bn_training, exponential_average_factor, self.eps, self.bit, self.id_bn)
        if config.profile_time:
            end.record()
            torch.cuda.synchronize()
            Duration = start.elapsed_time(end)
            config.store_sta(self.bit, Duration, self.id_bn)
            if not self.used: self.used = True
        return res

# def bwd_hook_emb(module, grad_output, grad_input):
#     grad_input = grad_input.float()
#     module.weight.grad = module.weight.grad.float()
#     return grad_input

    

class QEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None):
        super(QEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)
        self.bit = 32
        self.id_emb = id(self)
        self.used = False

        self.bwd_hook = None
        self.name = 'embedding'
        self.unique_id = None

        # self.register_full_backward_hook(backward_fp32_hook)
        # register_collection_hooks(self)

    def reset_bits(self, bit):
        self.bit = bit
        if self.bit == 32:
            self.float()
        elif config.enable_half_conversion:
            self.half()
        # if self.bit != 32:
        #     self.add_bwd_hook()
        # else:
        #     self.bwd_hook.remove()
        #     self.bwd_hook = None

    # def add_bwd_hook(self):
    #     hook = self.register_full_backward_hook(bwd_hook_emb)
    #     self.bwd_hook = hook
    def cast_tensor(self):
        if self.bit <= 16:
            weight = tensor_to_half(self.weight) 
        else:
            weight = tensor_to_float(self.weight)
        return weight

    def forward(self, input):

        if config.profile_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                weight = self.cast_tensor()
        else:
            weight = self.cast_tensor()
        
        # in context. input, weight saved
        if config.profile_mem:
            config.store_mem(self.bit, compute_tensor_bytes([input, weight]) / config.unit, self.unique_id)
        
        res = F.embedding(input, weight, padding_idx=self.padding_idx, max_norm=self.max_norm, \
                             norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)
        
        if config.profile_act: # collect the activation information
            profile_stas_emb(weight, None, res, self.unique_id)
            if not self.used: self.used = True
        

        if config.profile_time:
            end.record()
            torch.cuda.synchronize()
            Duration = start.elapsed_time(end)
            config.store_sta(self.bit, Duration, self.unique_id)
            if not self.used: self.used = True
       
        return res 

# it has bit, too
class QMatmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'matmul'
        self.unique_id = None
        self.bit = 32
        self.used = False
    
    def reset_bits(self, bit):
        if bit <= 16:
            self.bit = 16
        
    def forward(self, inp1, inp2):
        # we design like that to introduce no extra quantization error in this op
        # FP16 -> FP32 no loss.
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                if self.bit == 16: # softmax is very sensitive, thus don't change here
                    inp1 = inp1.half()
                    inp2 = inp2.half()
                else:
                    inp1 = inp1.float()
                    inp2 = inp2.float()
        else:
            if self.bit == 16: # softmax is very sensitive, thus don't change here
                inp1 = inp1.half()
                inp2 = inp2.half()
            else:
                inp1 = inp1.float()
                inp2 = inp2.float()
        res = torch.matmul(inp1, inp2)
        gemm_layer_profiler_block(self.unique_id, inp1, inp2, res)
        if not self.used: self.used = True
        # print(res)
        return res

class QAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'add'
        self.unique_id = None

    def forward(self, inp1, inp2):
        # force fp32
        if inp1.dtype != inp2.dtype:
            inp1 = tensor_to_float(inp1)
            inp2 = tensor_to_float(inp2)
        res = inp1 + inp2
        return res

# thiss version relies on the transformer, this QGELU based on it


class QGELU(TransGELU):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        self.name = 'gelu'
        self.unique_id = None
        if version.parse(version.parse(torch.__version__).base_version) < version.parse("1.4") or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        input = input_only_fp32_cast(input)
        input_only_layer_profiler_block(self.unique_id, input)
        return self.act(input)

# class QGELU(nn.GELU):
#     def __init__(self):
#         super(QGELU, self).__init__()
    
#     def forward(self, input):
#         input_only_layer_profiler_block(self.unique_id, input)
#         return F.gelu(input)

class QSoftmax(nn.Softmax):
    def __init__(self, dim=None):
        super(QSoftmax, self).__init__(dim)
        self.name = 'softmax'
        self.unique_id = None
    
    def forward(self, input):
        input = input_only_fp32_cast(input)
        input_only_layer_profiler_block(self.unique_id, input)
        res = F.softmax(input, dim=self.dim)
        return res


class ModWrapper(nn.Module):
    def __init__(self, mod:nn.Module, name):
        super().__init__()
        assert mod, "Provides mod when creating fp32 wrapper"
        self.mod = mod
        self.name = name 
        self.unique_id = None
        self.bit = 32 if not hasattr(mod, 'weight') or (hasattr(mod, 'weight') and mod.weight.dtype == torch.float32)else 16
    
    def reset_bits(self, bit):
        self.bit = bit # not change
        if self.bit == 32:
            self.mod.float()
        elif self.bit <= 16:
            self.mod.half()
            
    def forward(self, input):
        if not hasattr(self, 'bit'):
            input = input.float()
        else:
            if self.bit == 32:
                input = input.float()
            if self.bit <= 16:
                input = input.half()
        input_only_layer_profiler_block(self.unique_id, input)
        return self.mod(input)

class FP16Wrapper(nn.Module):
    def __init__(self, mod:nn.Module):
        super().__init__()
        assert mod, "Provides mod when creating fp32 wrapper"
        self.mod = mod.half()
    
    def forward(self, input):
        input = input.half()
        return self.mod(input)


import copy
def construct_qconv_by_conv(child: nn.Conv2d):
    device = child.weight.device
    new_conv = QConv2d(child.in_channels, child.out_channels,
        child.kernel_size, child.stride, child.padding, child.dilation,
        child.groups, child.bias is not None, child.padding_mode)
    
    # keys for not change
    # keys_neglect = ['_buffers', '_non_persistent_buffers_set', '_backward_hooks', \
    # '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_modules']
    # keys_neglect = []
    # new_conv.__dict__.update({k:v for k, v in child.__dict__.items() if k not in keys_neglect}) 
    # assign value
    new_conv.weight.data.copy_(child.weight.data)
    new_conv.weight.requires_grad_(child.weight.requires_grad)
    if child.bias is not None:
        # print(child.bias)
        new_conv.bias = torch.nn.Parameter(child.bias.detach().clone()) 
        new_conv.bias.requires_grad_(child.bias.requires_grad)
    # for detectron
    if hasattr(child, 'activation') and child.activation is not None:
        new_conv.activation = copy.deepcopy(child.activation) 
    if hasattr(child, 'norm') and child.norm is not None:
        new_conv.norm = copy.deepcopy(child.norm) 
    if not config.simu:
        if config.channels_last:
            new_conv = new_conv.to(memory_format=torch.channels_last)
        else:
            new_conv.convert_to_nhwc() # i havent' implemented the nchw now
    return new_conv.to(device)

# Linear tensor op only support RCR.
def construct_qlinear_by_linear(child: nn.Linear):
    device = child.weight.device

    # store C
    new_li = QLinear(child.in_features, child.out_features, child.bias is not None)
    new_li.weight.data.copy_(child.weight.data)

    # store R
    # new_li = QLinear(child.out_features, child.in_features, child.bias is not None)
    # new_li.weight.data.copy_(torch.t(child.weight).data)

    new_li.weight.requires_grad_(child.weight.requires_grad)
    if child.bias is not None:
        # print(child.bias)
        # new_li.bias = nn.Parameter(torch.empty(child.bias.shape, device=child.bias.device)) # RRR
        new_li.bias.data.copy_(child.bias.data) 
        new_li.bias.requires_grad_(child.bias.requires_grad)
    # # for detectron
    # if hasattr(child, 'activation') and child.activation is not None:
    #     new_conv.activation = copy.deepcopy(child.activation) 
    # if hasattr(child, 'norm') and child.norm is not None:
    #     new_conv.norm = copy.deepcopy(child.norm) 
    del child
    return new_li.to(device)

def construct_qln_by_ln(child: nn.LayerNorm):
    new_QLN = QLayerNorm(child.normalized_shape, child.eps, child.elementwise_affine)
    if child.elementwise_affine: 
        if child.weight is not None: new_QLN.weight, new_QLN.bias = torch.nn.Parameter(child.weight.detach().clone()), torch.nn.Parameter(child.bias.detach().clone())
    new_QLN = new_QLN.to(child.weight.device)
    return new_QLN

def construct_qemb_by_emb(child: nn.Embedding):
    new_QEmb = QEmbedding(child.num_embeddings, child.embedding_dim, child.padding_idx, child.max_norm, child.norm_type, child.scale_grad_by_freq,\
    child.sparse)
    if child.weight is not None: new_QEmb.weight = torch.nn.Parameter(child.weight.detach().clone())
    new_QEmb = new_QEmb.to(child.weight.device)
    return new_QEmb

def construct_qbn_by_bn(child: nn.BatchNorm2d):
    new_QBN = QBatchNorm2d(child.num_features, child.eps, child.momentum,
                    child.affine, child.track_running_stats)
    if child.track_running_stats: 
        if child.running_mean is not None: 
            new_QBN.running_mean.data.copy_(child.running_mean.data)
            new_QBN.running_var.data.copy_(child.running_var.data) 
        new_QBN.weight.data.copy_(child.weight.data) 
        new_QBN.bias.data.copy_(child.bias.data) 

    if config.channels_last:
        new_QBN = new_QBN.to(memory_format=torch.channels_last)
    new_QBN = new_QBN.to(child.running_mean.device)
    return new_QBN

def construct_qmaxpool_by_maxpool(child: nn.MaxPool2d):
    new_QMaxPool2d = QMaxPool2d(child.kernel_size, child.stride,
        child.padding, child.dilation, child.return_indices, child.ceil_mode)
    if config.channels_last:
        new_QMaxPool2d = new_QMaxPool2d.to(memory_format=torch.channels_last)
    return new_QMaxPool2d

def reset_bits(qmod_list, bit):
    for qmod in qmod_list:
        qmod.reset_bits(bit)
    

def assign_unique_id(module, unique_id_dict, name_to_module_mapper):
    cur_cnt = unique_id_dict[module.name]
    unique_id = f'{module.name}_{cur_cnt}'
    unique_id_dict[module.name] += 1
    module.unique_id = unique_id
    # store in mapper
    name_to_module_mapper[module.unique_id] = module
    

def convert_layers(module):
    qconv_list = []
    qli_list = []
    qbn_list = []
    qln_list = []
    qemb_list = []
    # others
    qgelu_list = []
    qmatmul_list = []
    # mappers
    unique_id_dict = defaultdict(int) # assign unique names to converted layers
    name_to_module_mapper = {}
    #  convert layers
    def convert_layers_inner(module):
        for name, child in module.named_children():
            new_mod = None 
            # print(name)
            # Do not convert layers that are already quantized
            if name in config.reject_name_list:
                new_mod = mod_wrapper = ModWrapper(child, name)
                if isinstance(child, (nn.Linear)):
                    new_mod.name = 'linear'
                setattr(module, name, mod_wrapper)
                assign_unique_id(new_mod, unique_id_dict, name_to_module_mapper)
                continue
                
            if name in ['proposal_generator']: # pass some layers
                continue

            if isinstance(child, QMatmul):
                qmatmul_list.append(child)

            if isinstance(child, (QLinear, QConv2d, QReLU, QGELU, QDropout, QSoftmax, QBatchNorm2d, QMaxPool2d, QEmbedding, QLayerNorm, QAdd, QMatmul)):
                assign_unique_id(child, unique_id_dict, name_to_module_mapper)
                continue
            
            if isinstance(child, nn.Conv2d):
                if not child.weight.requires_grad: # the freezed layers. Don't change
                    continue
                if nn.Conv2d not in config.QTarget_Module:
                    continue
                # for the modified situation, don't change the convoluition
                # if hasattr(child, 'norm') and child.norm is not None:
                #     continue
                # if hasattr(child, 'activation') and child.activation is not None:
                #     continue
                new_mod = new_conv = construct_qconv_by_conv(child) 
                # attrs = vars(child)
                # print(', '.join("{}: {}".format(item[0], item[1]) for item in attrs.items()))
                # print([k for k in attrs.keys()])
                # print(dir(child))
                # import pdb;pdb.set_trace() 
                # print(new_conv.__dict__)            
                setattr(module, name, new_conv)
                qconv_list.append(new_conv)
            elif isinstance(child, nn.Linear):
                if not child.weight.requires_grad: # the freezed layers. Don't change
                    continue
                if nn.Linear not in config.QTarget_Module:
                    continue
                new_mod = new_li = construct_qlinear_by_linear(child) 
                setattr(module, name, new_li)
                qli_list.append(new_li)

            elif isinstance(child, nn.ReLU):
                if config.simu:
                    continue # temporarily not change the layer for od
                if nn.ReLU not in config.QTarget_Module:
                    continue
                new_mod = new_QRelu = QReLU(child.inplace)
                setattr(module, name, new_QRelu)
            
            # add input conversion for softmax.
            elif isinstance(child, nn.Softmax):
                if config.simu:
                    continue 
                if nn.Softmax not in config.QTarget_Module:
                    continue
                new_mod = qsoftmax = QSoftmax(child.dim)
                setattr(module, name, qsoftmax)

            
            elif isinstance(child, nn.Dropout):
                if config.simu:
                    continue # temporarily not change the layer for od
                if nn.Dropout not in config.QTarget_Module:
                    continue
                # import pdb; pdb.set_trace()
                new_mod = qdropout = QDropout(p=child.p)
                setattr(module, name, qdropout)
            
            # elif isinstance(child, nn.GELU):
            #     if config.simu:
            #         continue # temporarily not change the layer for od
            #     if nn.GELU not in config.QTarget_Module:
            #         continue
                
            #     new_mod = qgelu = QGELU()
            #     setattr(module, name, qgelu)

            elif isinstance(child, TransGELU):
                if config.simu:
                    continue # temporarily not change the layer for od
                if TransGELU not in config.QTarget_Module:
                    continue
                new_mod = qgelu = QGELU()
                qgelu_list.append(qgelu)
                setattr(module, name, qgelu)
            
            # elif isinstance(child, nn.GELU):
            #     if config.simu:
            #         continue # temporarily not change the layer for od
            #     if nn.GELU not in config.QTarget_Module:
            #         continue
            #     new_mod = qgelu = QGELU()
            #     setattr(module, name, qgelu)

            elif isinstance(child, nn.MaxPool2d):
                if nn.MaxPool2d not in config.QTarget_Module:
                    continue
                new_mod = new_QMaxPool2d = construct_qmaxpool_by_maxpool(child)
                setattr(module, name, new_QMaxPool2d)
            
            elif isinstance(child, nn.LayerNorm):
                if not child.weight.requires_grad: # the freezed layers. Don't change
                    continue
                if nn.LayerNorm not in config.QTarget_Module:
                    continue
                new_mod = new_QLN = construct_qln_by_ln(child)
                setattr(module, name, new_QLN)
                qln_list.append(new_QLN)
            
            elif isinstance(child, nn.Embedding):
                if not child.weight.requires_grad: # the freezed layers. Don't change
                    continue
                if nn.LayerNorm not in config.QTarget_Module:
                    continue
                new_mod = new_QEmb = construct_qemb_by_emb(child)
                setattr(module, name, new_QEmb)
                qemb_list.append(new_QEmb)

            elif isinstance(child, nn.BatchNorm2d):
                if not child.weight.requires_grad: # the freezed layers. Don't change
                    continue
                if nn.BatchNorm2d not in config.QTarget_Module:
                    continue
                new_mod = new_QBN = construct_qbn_by_bn(child)
                setattr(module, name, new_QBN)
                qbn_list.append(new_QBN)
            else:
                convert_layers_inner(child)
            if new_mod is not None:
                assign_unique_id(new_mod, unique_id_dict, name_to_module_mapper)
    convert_layers_inner(module)
    return [qconv_list, qli_list, qbn_list, qln_list, qemb_list, qgelu_list, qmatmul_list], name_to_module_mapper