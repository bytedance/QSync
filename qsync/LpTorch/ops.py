import torch
from torch.autograd import Function
import torch.nn.functional as F
from .conf import config
from .functions import *
from .utils import convert_from_nchw_to_nhwc, convert_from_nhwc_to_nchw, get_memory_usage, compute_tensor_bytes, transpose_dim
from .quant import quantize_int8, dequantize_int8, dequantize_int4, qdq_bit, get_amax, get_scale, convert_to_fp16, convert_to_fp32, convert_to_target_tensor, quant_with_scale_zp

from .cppfrontend import ext_backward_func
import torch.autograd.profiler as profiler

from qsync.Predictor.indicator_profiler import profile_stas

# TODO: leave for linear, removed later
def dq_switcher(q_x, scale_x, bit):
    if bit == 32:
        return q_x
    elif bit == 19: # tf32
        return q_x 
    elif bit == 16:
        return q_x.float()
    elif bit == 8:
        return dequantize_int8(q_x, scale_x)



def q_switcher_bn(x, bit):
    if bit == 32:
        return x, 1
    elif bit == 16:
        return x.half(), 1
    elif bit == 8:
        return quantize_int8(x)


def empty_cvt(x, scale_x, bit, x_shape=None):
    return x

    
def bw_dq_switcher(indicator_ts, q_input, q_weight, scale_inp, scale_w, bit, inp_shape=None, w_shape=None, linear=False):  
    bp_funct = None
    cvt_function = empty_cvt
    conv_bp_functions = [conv2d_fp16bp_optimized, conv2d_fp16bp, conv2d_fp32bp]
    linear_bp_functions = [bp_linear_w32_a16, bp_linear_w16_a16, bp_linear_w32_a32]
    bp_funtions = conv_bp_functions if not linear else linear_bp_functions

    # use low-precision backward for all bit <= 16
    if bit > 19:
        bp_funct = bp_funtions[2] # fp32 one
        cvt_function = convert_to_fp32
    else:
        cvt_function = convert_to_fp16
        if not config.optimized_fp16_bp:
            bp_funct = bp_funtions[1] # not optimzied backward
        else:
            bp_funct = bp_funtions[0]
        
    if config.pytorch_profiler:
        with profiler.record_function("qsync::bwd_cast"):
            input = cvt_function(q_input, scale_inp, bit, inp_shape)
            if config.save_q_weight:
                weight = cvt_function(q_weight, scale_w, bit, w_shape)
            else:
                weight = q_weight
    else:
        input = cvt_function(q_input, scale_inp, bit, inp_shape)
        if config.save_q_weight:
            weight = cvt_function(q_weight, scale_w, bit, w_shape)
        else:
            weight = q_weight
    return bp_funct, input, weight

class TorchConv2d(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def run_forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, bit=32, id_conv=None):
        # for experiments
        if config.swapping:
            config.store_in_cpu(input, weight, bias, id_conv)
            ctx.stride = stride
            ctx.padding = padding 
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.bit = bit 
            ctx.id = id_conv
            output = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output 
            
        qdq_weight = qdq_bit(weight, bit, config.kernel_quant)
        qdq_inp = qdq_bit(input, bit, config.act_quant)

        if bit <= 16:
            ctx.save_for_backward(qdq_inp.half(), qdq_weight.half(), bias)
        else:
            ctx.save_for_backward(input, weight, bias)
        
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.bit = bit 
        output = F.conv2d(qdq_inp, qdq_weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        if config.profile_act: # collect the activation information
            profile_stas(input, weight, output, None, None, id_conv)

        return output
       
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def run_backward(ctx, grad_output):
         # for experiments.
        if config.swapping:
            id_conv = ctx.id
            input, weight, bias = config.fetch_from_cpu(id_conv)
            qdq_input = input.cuda()
            qdq_weight = weight.cuda()
            bias = bias.cuda()
        else:
            qdq_input, qdq_weight, bias = ctx.saved_tensors

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        qdq_input, qdq_weight = qdq_input.float(), qdq_weight.float()
        if ctx.bit < 32:
            qdq_gradoutput = qdq_bit(grad_output, 16)
        else:
            qdq_gradoutput = grad_output
        # if config.cudnn:
        #     if ctx.needs_input_grad[0]:
        #         grad_input = F.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        #     if ctx.needs_input_grad[1]:
        #         grad_weight = F.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) 
        #     # use cutlass
        grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                qdq_input, qdq_gradoutput, qdq_weight, padding, stride, dilation, groups,
                False, False, False, # benchmark
                [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)


        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class TorchConvInterface(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, bit=32, id_conv=None):
        return TorchConv2d.run_forward(ctx, input, weight, bias, stride, padding, dilation, groups, bit, id_conv)
       
    @staticmethod
    def backward(ctx, grad_output):
        return TorchConv2d.run_backward(ctx, grad_output)


class Conv2dN(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def run_forward(fwd_op, ctx, q_input, q_weight, weight=None, bias=None, stride=1, padding=0, dilation=1, groups=1, id_conv=None, o_bit=32):

        if type(stride) is tuple:
            stride_h, stride_w = stride[0], stride[1]
        if type(padding) is tuple:
            padding_h, padding_w = padding[0], padding[1]
        
        output_scale = ctx.output_scale
        if config.fuse_int_output and not type(output_scale) is float:
            output_scale_f = output_scale.item()
        else:
            output_scale_f = 1.0
            

        output = fwd_op(q_input, q_weight, stride_h, stride_w, padding_h, padding_w, output_scale_f)

        # Channel wise requires finner change
        if o_bit == 8 and output_scale_f == 1.0:
            output = dequantize_int8(output,  output_scale)
        
        inp_shape, w_shape = None, None
        if o_bit == 4:
            # torch.save(q_input, 'q_input.pt')
            pack, ctx.scale_inp, ctx.scale_w, ctx.inp_shape, ctx.w_shape = output
            output, packed_input, packed_weight = pack

        if bias is not None:
            if config.channels_last:
                bias = bias.view(1, output.shape[1], 1, 1)
            else:
                bias = bias.view(1, 1, output.shape[-1])
            output += bias

        if config.profile_mem:
            if bias is None:
                config.store_mem(o_bit, compute_tensor_bytes([q_input, q_weight]) / config.unit, id_conv)
            else:
                config.store_mem(o_bit, compute_tensor_bytes([q_input, q_weight, bias]) / config.unit, id_conv)
        
        if config.profile_time:
            # add the dequantize part into consideration. bp case
            bw_dq_switcher(weight, q_input, q_weight, ctx.scale_inp, ctx.scale_w, o_bit, inp_shape, w_shape)

        if config.profile_act: # collect the activation information
            profile_stas(q_input, q_weight, output, None, None, id_conv)
        # get_memory_usage(True)
        # print("Act Mem: %.2f MB" % (compute_tensor_bytes([input]) / 1024 ** 2))


        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.bit = o_bit


        
        if config.profile_mem or config.profile_time:
            pass
        else:
            if config.fixed_save_fp16_context and o_bit <= 8:
                pass # save fp16 context
            else:
                if o_bit < 8:
                    if config.save_q_weight:
                        ctx.save_for_backward(packed_input, packed_weight, bias)
                    else:
                        ctx.save_for_backward(packed_input, weight, bias)
                else:
                    if config.save_q_weight:
                        ctx.save_for_backward(q_input, q_weight, bias)
                    else:
                        ctx.save_for_backward(q_input, weight, bias)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def run_backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        # import pdb; pdb.set_trace()
        q_input, q_weight, bias = ctx.saved_tensors
        scale_w = ctx.scale_w
        scale_inp = ctx.scale_inp

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        bit = ctx.bit
        # import pdb; pdb.set_trace()
        # print(bit)
        if bit < 8:
            inp_shape = ctx.inp_shape
            w_shape = ctx.w_shape
        else:
            inp_shape, w_shape = None, None
        
        bp_funct, input, weight = bw_dq_switcher(grad_output, q_input, q_weight, scale_inp, scale_w, bit, inp_shape, w_shape)

        # cpu time for this 
        if config.pytorch_profiler:
            with profiler.record_function("qsync::bwd_cast_output_cpu"):
                if bit <= 19: 
                    cvted_grad_output = grad_output.half()
                else:
                    cvted_grad_output = grad_output.float()
        else:
            if bit <= 19: 
                cvted_grad_output = grad_output.half()
            else:
                cvted_grad_output = grad_output.float()
        
        if config.enable_half_conversion and config.channels_last:
            grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, cvted_grad_output, weight, padding, stride, dilation, groups,
                False, False, False, # benchmark
                [ctx.needs_input_grad[0], ctx.needs_input_grad[1]]) 
        else:
            # use cutlass
            grad_input, grad_weight = bp_funct(input, cvted_grad_output, weight, stride[0], stride[1], padding[0], padding[1], 1.0)
        
        # try:
        #     grad_input.shape
        #     grad_weight.shape
        # except:

        #     import pdb; pdb.set_trace()
        grad_weight = grad_weight.float()
            

        if bias is not None and ctx.needs_input_grad[2]:
            if config.channels_last:
                grad_bias = grad_output.sum((0,2,3)).squeeze(0)
            else:
                grad_bias = grad_output.sum((0,1,2)).squeeze(0)
        
        # grad_weight = convert_to_target_tensor(weight, grad_weight)
        # grad_bias = convert_to_target_tensor(bias, grad_bias)


        return grad_input, grad_weight, None, grad_bias, None, None, None, None, None, None, None, None



class Conv2dFP32(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, q_weight, bias=None, stride=1, padding=0, dilation=1, groups=1, id_conv=None, scale_w=None, scale_inp=None, output_scale=None):
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                input_float = input.float()
        else:
            input_float = input.float()

        ctx.scale_w, ctx.scale_inp = 1, 1
        ctx.output_scale = 1.0
        return Conv2dN.run_forward(conv2d_fp32, ctx, input_float, weight, weight, bias, stride, padding, dilation, groups, id_conv, o_bit=32)
       
    @staticmethod
    def backward(ctx, grad_output):
        return Conv2dN.run_backward(ctx, grad_output)


class Conv2dTF32(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, q_weight, bias=None, stride=1, padding=0, dilation=1, groups=1, id_conv=None, scale_w=None, scale_inp=None, output_scale=None):
        ctx.scale_w, ctx.scale_inp = 1, 1
        ctx.output_scale = 1.0
        return Conv2dN.run_forward(conv2d_tf32, ctx, input, weight,  weight, bias, stride, padding, dilation, groups, id_conv, o_bit=19)

    @staticmethod
    def backward(ctx, grad_output):
        return Conv2dN.run_backward(ctx, grad_output)
    
        

class Conv2dFP16(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, q_weight, bias=None, stride=1, padding=0, dilation=1, groups=1, id_conv=None, scale_w=None, scale_inp=None, output_scale=None):
        # import pdb; pdb.set_trace()
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                input_half = input.half()
                weight_half = weight.half()
        else:
            input_half = input.half()
            weight_half = weight.half()

        ctx.scale_w, ctx.scale_inp = 1, 1
        ctx.output_scale = 1.0
        return Conv2dN.run_forward(conv2d_fp16, ctx, input_half, weight_half, weight, bias, stride, padding, dilation, groups, id_conv, o_bit=16)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return Conv2dN.run_backward(ctx, grad_output)
        

class Conv2dInt8(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, q_weight, bias=None, stride=1, padding=0, dilation=1, groups=1, id_conv=None, scale_w=None, scale_inp=None, output_scale=None):
        if scale_inp is not None:
            input_int8 = quant_with_scale_zp(input, scale_inp, torch.tensor(0))
            scale_weight = scale_w
            if q_weight is not None: # calculate in advance
                weight_int8 = q_weight
            else:
                weight_int8 = quant_with_scale_zp(weight, scale_weight, torch.tensor(0))
            # print("here")
        else:
            if config.pytorch_profiler:
                with profiler.record_function("aten::quantize_int8"):
                    input_int8, scale_inp = quantize_int8(input, config.act_quant)
                    if q_weight is not None:
                        scale_weight = scale_w
                        weight_int8 = q_weight
                    else:
                        weight_int8, scale_weight = quantize_int8(weight, config.kernel_quant)
                    # print(id_linear, q_input.shape, q_weight.shape)
                    output_scale = scale_inp * scale_weight
            else:
                input_int8, scale_inp = quantize_int8(input, config.act_quant)
                if q_weight is not None:
                    scale_weight = scale_w
                    weight_int8 = q_weight
                else:
                    weight_int8, scale_weight = quantize_int8(weight, config.kernel_quant)
                # print(id_linear, q_input.shape, q_weight.shape)
                output_scale = scale_inp * scale_weight

        if config.fixed_save_fp16_context:
            ctx.save_for_backward(input.half(), weight.half(), bias)

        ctx.output_scale = output_scale
        ctx.scale_w, ctx.scale_inp = scale_weight, scale_inp
        return Conv2dN.run_forward(conv2d_int8, ctx, input_int8, weight_int8, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, id_conv=id_conv, o_bit=8)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return Conv2dN.run_backward(ctx, grad_output)

# TODO: Implement INT4
class Conv2dInt4(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, qweight, bias=None, stride=1, padding=0, dilation=1, groups=1, id_conv=None, scale_w=None, scale_inp=None, output_scale=None):
        # quantization of int4 not happened here
        ctx.scale_w, ctx.scale_inp = 1, 1
        ctx.output_scale = 1.0
        return Conv2dN.run_forward(conv2d_int4, ctx, input, weight, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, id_conv=id_conv, o_bit=4)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return Conv2dN.run_backward(ctx, grad_output, 4)


class LinearFP32(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, qweight, id_linear=None, scale_w=None, scale_inp=None, output_scale=None):
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                input_float = input.float()
        else:
            input_float = input.float()

        ctx.scale_w, ctx.scale_inp = 1, 1
        ctx.output_scale = 1.0
        return LinearN.run_forward(linear_fp32, ctx, input_float, weight, weight, id_linear=id_linear, o_bit=32)

    @staticmethod
    def backward(ctx, grad_output):
        return LinearN.run_backward(ctx, grad_output, 32)

class LinearFP16(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, qweight, id_linear=None, scale_w=None, scale_inp=None, output_scale=None):
        # import pdb; pdb.set_trace()
        if config.pytorch_profiler:
            with profiler.record_function("qsync::cast_cost_fp16"):
                input_half = input.half()
                weight_half = weight.half()
        else:
            input_half = input.half()
            weight_half = weight.half()

        ctx.scale_w, ctx.scale_inp = 0, 0
        ctx.output_scale = 1.0
        return LinearN.run_forward(linear_fp16, ctx, input_half, weight_half, weight, id_linear=id_linear, o_bit=16)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return LinearN.run_backward(ctx, grad_output, 16)

class LinearInt8(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, q_weight, id_linear=None, scale_w=None, scale_inp=None, output_scale=None):
        # period collection.
        if scale_inp is not None:
            input_int8 = quant_with_scale_zp(input, scale_inp, torch.tensor(0))
            scale_weight = scale_w
            if q_weight is not None: # calculate in advance
                weight_int8 = q_weight
            else:
                weight_int8 = quant_with_scale_zp(weight, scale_weight, torch.tensor(0))
            # print("here")
        else:
            if config.pytorch_profiler:
                with profiler.record_function("aten::quantize_int8"):
                    input_int8, scale_inp = quantize_int8(input, config.act_quant)
                    if q_weight is not None:
                        scale_weight = scale_w
                        weight_int8 = q_weight
                    else:
                        weight_int8, scale_weight = quantize_int8(weight, config.kernel_quant)
                    # print(id_linear, q_input.shape, q_weight.shape)
                    output_scale = scale_inp * scale_weight
            else:
                input_int8, scale_inp = quantize_int8(input, config.act_quant)
                if q_weight is not None:
                    scale_weight = scale_w
                    weight_int8 = q_weight
                else:
                    weight_int8, scale_weight = quantize_int8(weight, config.kernel_quant)
                # print(id_linear, q_input.shape, q_weight.shape)
                output_scale = scale_inp * scale_weight

        if config.fixed_save_fp16_context:
            ctx.save_for_backward(input.half(), weight.half())

        ctx.scale_w, ctx.scale_inp = scale_weight, scale_inp
        ctx.output_scale = output_scale
        return LinearN.run_forward(linear_int8, ctx, input_int8, weight_int8, weight, id_linear=id_linear, o_bit=8)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return LinearN.run_backward(ctx, grad_output, 8)

class LinearTF32(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, qweight, id_linear=None, scale_w=None, scale_inp=None, output_scale=None):
        ctx.scale_w, ctx.scale_inp = 1, 1
        ctx.output_scale = 1.0
        return LinearN.run_forward(linear_tf32, ctx, input, weight, weight, id_linear=id_linear, o_bit=19)

    @staticmethod
    def backward(ctx, grad_output):
        return LinearN.run_backward(ctx, grad_output,  19)

class LinearN(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def run_forward(fwd_op, ctx, q_input, q_weight, weight=None, id_linear=None, o_bit=32):

        ctx.inp_shape, ctx.w_shape = None, None
        output_scale = ctx.output_scale

        if config.fuse_int_output and not type(output_scale) is float:
            output_scale_f = output_scale.item()
        else:
            output_scale_f = 1.0
        

        if o_bit == 16 and config.half_to_half:
            if config.pytorch_profiler:
                with profiler.record_function("qsync::linear_fp16"):
                    output = F.linear(q_input, q_weight)
            else:
                output = F.linear(q_input, q_weight)
            # output = torch.matmul(q_input, transpose_dim(q_weight))
            # print(output.dtype)
        else:
            output = fwd_op(q_input, q_weight, torch.tensor(0), output_scale_f)
            

        if o_bit <= 8 and output_scale_f == 1.0: # not fused int output quantization 
            output = dequantize_int8(output, output_scale)

        if config.profile_mem:
            config.store_mem(o_bit, compute_tensor_bytes([q_input, q_weight]) / config.unit, id_linear)
        
        if config.profile_time:
            # add the dequantize part into consideration. bp case
            bw_dq_switcher(weight, q_input, q_weight, ctx.scale_inp, ctx.scale_w, o_bit, ctx.inp_shape, ctx.w_shape)

        if config.profile_act: # collect the activation information
            profile_stas(q_input, q_weight, output, None, None, id_linear)
        
        if config.profile_mem or config.profile_time:
            pass
        else:
            if config.save_q_weight:
                ctx.save_for_backward(q_input, q_weight)
            else:
                ctx.save_for_backward(q_input, weight)
        ctx.id_linear = id_linear
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def run_backward(ctx, grad_output, bit):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        q_input, q_weight = ctx.saved_tensors
        scale_w = ctx.scale_w
        scale_inp = ctx.scale_inp
        if bit < 8:
            inp_shape = ctx.inp_shape
            w_shape = ctx.w_shape
        else:
            inp_shape, w_shape = None, None

        bp_funct, input, weight = bw_dq_switcher(grad_output, q_input, q_weight, scale_inp, scale_w, bit, inp_shape, w_shape, linear=True)
        
        # cpu time for this 
        if config.pytorch_profiler:
            with profiler.record_function("qsync::bwd_cast_output_cpu"):
                if bit <= 16: 
                    cvted_grad_output = grad_output.half()
                else:
                    cvted_grad_output = grad_output.float()
        else:
            if bit <= 16: 
                cvted_grad_output = grad_output.half()
            else:
                cvted_grad_output = grad_output.float()

        
        if config.profile_bwd_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        grad_input, grad_weight = bp_funct(cvted_grad_output, weight, input)
        grad_weight = grad_weight.float()

        if config.profile_bwd_time:
            end.record()
            torch.cuda.synchronize()
            Duration = start.elapsed_time(end)
            config.store_sta(bit, Duration, ctx.id_linear)
            config.store_current_config(cvted_grad_output.shape, flag='bwd', id_op=ctx.id_linear)

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None


class layer_norm(Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, bit, id_ln):
        quantized, scale = q_switcher_bn(input, bit)
        output, mean, rstd = ext_backward_func.layer_norm_cuda(input, normalized_shape, weight, bias, eps)

        if config.profile_mem:
            if bias is None:
                config.store_mem(bit, compute_tensor_bytes([quantized]) / config.unit, id_ln)
            else:
                config.store_mem(bit, compute_tensor_bytes([quantized, bias]) / config.unit, id_ln)
        
        if config.profile_time:
            # add the dequantize part into consideration. bp case
            dq_switcher(weight, scale, bit)

        if config.profile_act: # collect the activation information
            with torch.no_grad():
                # in run time, this two element needn't be calculated
                D = quantized.nelement()
                if scale != 0:
                    # fixed point
                    ra_2 = scale ** 2
                else:
                    # floating point
                    ra_2 = get_scale(get_amax(quantized)).item() ** 2
                config.store_sta_act(ra_2, 1, 1, D, 1, id_ln)

        ctx.saved = (quantized, normalized_shape, mean, rstd, weight, bias)
        ctx.inp_scale = scale
        ctx.bit = bit 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        quantized, normalized_shape, mean, rstd, weight, bias = ctx.saved
        scale = ctx.inp_scale
        input = convert_to_fp32(quantized, scale_x = scale, bit=ctx.bit)
        input = input.contiguous()

        grad_input, grad_weight, grad_bias = ext_backward_func.layer_norm_backward_cuda(grad_output, input, normalized_shape, mean, rstd, weight, bias, \
        [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]])

        # if ctx.scheme:
        #     ctx.scheme.if_allocate_perlayer()
        return grad_input, None, grad_weight, grad_bias, None, None, None

class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps, bit, id_bn):
        # in experiment, bit = 8 detroy result. 
        # Considier: Using 16bit only
        # OR: abandoned the int8 BN gradient.
        quantized, scale = q_switcher_bn(input, bit)
        
        if training and weight is not None:
            output, save_mean, save_var, reserve = ext_backward_func.cudnn_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        if config.profile_mem:
            if bias is None:
                config.store_mem(bit, compute_tensor_bytes([quantized]) / config.unit, id_bn)
            else:
                config.store_mem(bit, compute_tensor_bytes([quantized, bias]) / config.unit, id_bn)
        
        if config.profile_time:
            # add the dequantize part into consideration. bp case
            dq_switcher(weight, scale, bit)

        if config.profile_act: # collect the activation information
            with torch.no_grad():
                # in run time, this two element needn't be calculated
                D = quantized.nelement()
                if scale != 0:
                    # fixed point
                    ra_2 = scale ** 2
                else:
                    # floating point
                    ra_2 = get_scale(get_amax(quantized)).item() ** 2
                config.store_sta_act(ra_2, 1, 1, D, 1, id_bn)

        ctx.saved = (quantized, weight, bias, running_mean, running_var, save_mean, save_var, training, eps, reserve)
        ctx.inp_scale = scale
        ctx.bit = bit 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None
        quantized, weight, bias, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved
        scale = ctx.inp_scale
        input = convert_to_fp32(quantized, scale_x = scale, bit=ctx.bit)
        input = input.contiguous()
        if training and weight is not None:
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
            )

        # if ctx.scheme:
        #     ctx.scheme.if_allocate_perlayer()
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None, None


# DO NCHW and NHWC conversion. To save time in implementation of NHWC training
class NCHW2NHWCFunction(Function):
    def forward(self, input):
        # print("Convert from NCHW - > NHWC")
        return convert_from_nchw_to_nhwc(input)

    def backward(self, grad_output):
        return convert_from_nhwc_to_nchw(grad_output) 
    

class NHWC2NCHWFunction(Function):
    def forward(self, input):
        # print("Convert from NHWC - > NCHW")
        return convert_from_nhwc_to_nchw(input)

    def backward(self, grad_output):
        return convert_from_nchw_to_nhwc(grad_output) 