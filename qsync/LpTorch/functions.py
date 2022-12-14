
import torch 
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from .conf import config
from .quant import get_amax, get_scale
from .utils import transpose_dim
from .cppfrontend import cutlassconv_cuda, cutlasslinear_cuda, cudnn_conv

def get_padding_num(input):
    # int8 mode 16, fp16 mode 8, else 4 for tensor
    mod_num = 1
    if input.dtype == torch.int8:
        mod_num = 16
    elif input.dtype == torch.float16:
        mod_num = 8
    else:
        mod_num = 4 # tf32 and fp32
    if config.disable_padding:
        mod_num = 1
    # else:
        # mod_num = 16
    return mod_num

# meta nchw padding
def padding_input_weight_nchw(input, weight):
    mod_num = get_padding_num(input)
    if input.size(1) % mod_num != 0:
        padding_channels = mod_num - input.size(1) % mod_num
        input_padded = F.pad(input, (0,0,0,0,0,padding_channels),"constant", 0)
        weight_padded = F.pad(weight, (0,0,0,0,0,padding_channels),"constant", 0)
    else:
        input_padded = input
        weight_padded = weight
    return input_padded, weight_padded
# meta nhwc padding
def padding_input_weight_nhwc(input, weight):
    mod_num = get_padding_num(input)
    if input.size(3) % mod_num != 0:
        padding_channels = mod_num - input.size(3) % mod_num
        input_padded = F.pad(input, (0, padding_channels),"constant", 0)
        weight_padded = F.pad(weight, (0, padding_channels),"constant", 0)
    else:
        input_padded = input
        weight_padded = weight
    return input_padded, weight_padded

def padding_input_weight(input, weight):
    if input.is_contiguous(memory_format=torch.channels_last):
        # meta is nchw.
        input_padded, weight_padded = padding_input_weight_nchw(input, weight)
    else:
        # meta is nhwc
        input_padded, weight_padded = padding_input_weight_nhwc(input, weight)
    # print(input_padded.shape, weight_padded.shape)
    return input_padded, weight_padded
    

# CUDNN
def conv2d_cudnn(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale):
    dtype = input.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        return F.conv2d(input, weight, stride=(stride_h, stride_w), padding=(padding_h, padding_w))
    else:
        # print("run CUDNN")
        # input_padded, weight_padded = padding_input_weight(input, weight)
        input_padded, weight_padded = padding_input_weight(input, weight) 
        if config.channels_last:
            return cudnn_conv.int8_conv_cl(input_padded, weight_padded, stride_h, stride_w, padding_h, padding_w, 1, 1, dq_scale)
        else:
            return cudnn_conv.int8_conv(input_padded, weight_padded, stride_h, stride_w, padding_h, padding_w, 1, 1, dq_scale)
        
# CUTLASS
def conv_funct(optimize_funct, ana_funct,\
        input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale):
    #  cutlass has special requirement on padding
    # input_padded, weight_padded = padding_input_weight(input, weight) 
    # in our experiment, the padded cost is not so linear, to ease computation, we combine it inside the cost of conv2d\
    input_padded, weight_padded = padding_input_weight(input, weight)
    if weight.size(1) <= 32:
        # print(input_padded.shape, weight_padded.shape)
        temp = optimize_funct(input_padded, weight_padded, stride_h, stride_w, padding_h, padding_w, dq_scale)
    else:
        # print(input_padded.shape, weight_padded.shape, stride_h, stride_w, padding_h, padding_w,)
        temp = ana_funct(input_padded, weight_padded, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return temp

# <=8 bit
def conv_funct_lower(optimize_funct, ana_funct,\
        input, weight, stride_h, stride_w, padding_h, padding_w):
    # collect information
    scale_inp, scale_w = get_scale(get_amax(input), 4), get_scale(get_amax(weight), 4)
    dq_scale = scale_inp * scale_w
    input_padded, weight_padded = padding_input_weight(input, weight)
    if weight.size(1) <= 32:
        temp = optimize_funct(input_padded, weight_padded, stride_h, stride_w, padding_h, padding_w, dq_scale, scale_inp, scale_w)
    else:
        temp = ana_funct(input_padded, weight_padded, stride_h, stride_w, padding_h, padding_w, dq_scale, scale_inp, scale_w)
    return temp, scale_inp, scale_w, input_padded.shape, weight_padded.shape

def conv2d_int4(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    """ only input channel(tensor.size(3)) is a multiple of 16"""
    # TODO: INT4 stil has some bugs
    assert None, "havent's support int4 yet"
    optimize_funct = cutlassconv_cuda.sp_conv_optimized_int4
    ana_funct = cutlassconv_cuda.sp_conv_int4
    return conv_funct_lower(optimize_funct, ana_funct, input, weight, stride_h, stride_w, padding_h, padding_w)

def conv2d_int8_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    """ only input channel(tensor.size(3)) is a multiple of 16"""
    if config.cudnn:
        return conv2d_cudnn(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    # cutlass
    optimize_funct = cutlassconv_cuda.sp_conv_optimized
    ana_funct = cutlassconv_cuda.sp_conv
    return conv_funct(optimize_funct, ana_funct, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    

def conv2d_fp16_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    """ only input channel(tensor.size(3)) is a multiple of 16"""
    if config.cudnn:
        return conv2d_cudnn(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    # cutlass
    if config.half_to_half and config.channels_last:
        return F.conv2d(input, weight, stride=(stride_h, stride_w), padding=(padding_h, padding_w))
    optimize_funct = cutlassconv_cuda.sp_conv_optimized_fp16
    ana_funct = cutlassconv_cuda.sp_conv_fp16
    return conv_funct(optimize_funct, ana_funct, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)


def conv2d_tf32(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    """ only input channel(tensor.size(3)) is a multiple of 16"""
    optimize_funct = cutlassconv_cuda.sp_conv_optimized_tf32
    ana_funct = cutlassconv_cuda.sp_conv_tf32
    return conv_funct(optimize_funct, ana_funct, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

def conv2d_fp32_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    """ 
        Here compare torch F.conv, cudnn fp32 and cutlass
        - 93, 93, 97. The latency different is very small and neglectable 4 / 93 ~= 4%
        - Precision are same.
    """
    if config.cudnn:
        return conv2d_cudnn(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    # use cutlass 
    optimize_funct = cutlassconv_cuda.sp_conv_optimized_fp32
    ana_funct = cutlassconv_cuda.sp_conv_fp32
    return conv_funct(optimize_funct, ana_funct, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

def conv2d_int8(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::conv_int8"):
            return conv2d_int8_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    else:
        return conv2d_int8_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)


def conv2d_fp16(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::conv_fp16"):
            return conv2d_fp16_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    else:
        return conv2d_fp16_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)



def conv2d_fp32(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::conv_fp32"):
            return conv2d_fp32_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    else:
        return conv2d_fp32_(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)


def conv2d_fp32bp(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    if weight.size(1) == 3: # the first conv
        funct = conv2d_fp32bp_first_conv
    else:
        # funct = conv2d_fp32bp_split
        funct = conv2d_fp32bp_normal
    return funct(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

def conv2d_fp32bp_normal(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    grad_input, grad_weight =cutlassconv_cuda.sp_core_convbp_fp32(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

# for the first convolution. 
def conv2d_fp32bp_first_conv(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    grad_input = None
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp32_wgrad_c3(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

def conv2d_fp32bp_split(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    # import pdb; pdb.set_trace()
    grad_input = cutlassconv_cuda.sp_core_convbp_fp32_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dq_scale)
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp32_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

def conv2d_fp16bp(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    if weight.size(1) == 3: # the first conv
        funct = conv2d_fp16bp_first_conv
    else:
        # funct = conv2d_fp32bp_split
        funct = conv2d_fp16bp_normal
    return funct(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

def conv2d_fp16bp_normal(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    return cutlassconv_cuda.sp_core_convbp_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

# for the first convolution. 
def conv2d_fp16bp_first_conv(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    input = input.float(); grad_output = grad_output.float()
    grad_input = None
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp32_wgrad_c3(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

def conv2d_fp16bp_split(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    # import pdb; pdb.set_trace()
    grad_input = cutlassconv_cuda.sp_core_convbp_fp16_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dq_scale)
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp16_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight


# Compute activation gradient using fp16, fp16 -> fp16, and compute gradient with fp16, fp16 -> fp16
# face challenge that aggregation among nodes have to be fp32
def conv2d_fp16bp_to_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    if weight.size(1) == 3: # the first conv
        funct = conv2d_fp16bp_first_conv_to_fp16
    else:
        # funct = conv2d_fp32bp_split
        funct = conv2d_fp16bp_normal_to_fp16
    return funct(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

def conv2d_fp16bp_normal_to_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    return cutlassconv_cuda.sp_core_convbp_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

# for the first convolution. 
def conv2d_fp16bp_first_conv_to_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    grad_input = None
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp16_wgrad_c3(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

def conv2d_fp16bp_split_to_fp16_to_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    # import pdb; pdb.set_trace()
    grad_input = cutlassconv_cuda.sp_core_convbp_fp16_to_fp16_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dq_scale)
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp16_to_fp16_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

#  the one actually in use. Compute activation gradient using fp16, fp16 -> fp16, and compute gradient with fp16, fp16 -> fp32
def conv2d_fp16bp_fisrt_optimized_normal(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    input = input.float(); grad_output = grad_output.float()
    grad_input = None
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp32_wgrad_c3(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

def conv2d_fp16bp_optimized_normal(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    grad_input = cutlassconv_cuda.sp_core_convbp_fp16_to_fp16_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dq_scale)
    grad_weight = cutlassconv_cuda.sp_core_convbp_fp16_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)
    return grad_input, grad_weight

def conv2d_fp16bp_optimized(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale=1.0):
    # import pdb; pdb.set_trace()
    if weight.size(1) == 3: # the first conv
        funct = conv2d_fp16bp_fisrt_optimized_normal
    else:
        # funct = conv2d_fp32bp_split
        funct = conv2d_fp16bp_optimized_normal
    return funct(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dq_scale)

# cudnn implementation
def conv2d_data_fp16(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w):
    input_padded, weight_padded = padding_input_weight(input, weight)
    return cudnn_conv.fp16_conv_bp_data(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w)


def linear_padding(p_input, weight):
    padding_num = get_padding_num(p_input)
    if p_input.shape[-2] % padding_num != 0:
        padding_channels = padding_num - p_input.shape[-1] % padding_num
        input_padded = F.pad(p_input, (0, padding_channels),"constant", 0)
        weight_padded = F.pad(weight, (0, padding_channels),"constant", 0)
    else:
        input_padded = p_input
        weight_padded = weight
    return input_padded, weight_padded

# change linear implementation here
# input.dim == 2 and bias.defined(). return torch.addmm(bias, input, weight.t())
# else: output = input.matmul(weight.t()); output += bias;

def linear_int8(input, weight, bias, dq):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::linear_int8"):
            input_padded, weight_padded = linear_padding(input, weight)
            return cutlasslinear_cuda.gemm_int8(input_padded, weight_padded, bias, dq)
    else:
        return cutlasslinear_cuda.gemm_int8(input_padded, weight_padded, bias, dq)

def linear_fp16(input, weight, bias, dq):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::linear_fp16"):
            input_padded, weight_padded = linear_padding(input, weight)
            return cutlasslinear_cuda.gemm_half(input_padded, weight_padded, bias, dq)
    else:
        input_padded, weight_padded = linear_padding(input, weight)
        return cutlasslinear_cuda.gemm_half(input_padded, weight_padded, bias, dq)

def linear_fp32(input, weight, bias, dq):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::linear_fp32"):
            input_padded, weight_padded = linear_padding(input, weight)
            return cutlasslinear_cuda.gemm_float(input_padded, weight_padded, bias, dq)
    else:
        return cutlasslinear_cuda.gemm_float(input_padded, weight_padded, bias, dq)

def linear_tf32(input, weight, bias, dq):
    if config.pytorch_profiler:
        with profiler.record_function("qsync::linear_tf32"):
            input_padded, weight_padded = linear_padding(input, weight)
            return cutlasslinear_cuda.gemm_tf32(input_padded, weight_padded, bias, dq)
    else:
        return cutlasslinear_cuda.gemm_tf32(input_padded, weight_padded, bias, dq)



# TODO: this part follows ACTNN implementation thus may not be optimal
tmp = torch.tensor(0)
def linear_bp(grad_output, weight, input, grad_weight_funct, grad_input_funct):
    grad_input = grad_weight = None
    with torch.no_grad():
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        dq = 1 # TODO: remove in implementation

        grad_output_flatten = grad_output.reshape(-1, C_out)
        input_flatten = input.reshape(-1, C_in)
        grad_input = grad_input_funct(grad_output, weight)
        grad_weight = grad_weight_funct(grad_output_flatten.t().contiguous(), input_flatten.t().contiguous(), tmp, dq)
        # if grad_weight_funct == linear_fp16:
        #     grad_weight = grad_weight_funct(grad_output_flatten.t().contiguous(), input_flatten.t().contiguous(), tmp, dq)
        # else:
        #     grad_weight = torch.matmul(grad_output_flatten.t().contiguous(), input_flatten.contiguous())

    return grad_input, grad_weight

def bp_linear_w32_a32(grad_output, weight, input):
    grad_weight_funct = linear_fp32
    grad_input_funct = torch.matmul
    return linear_bp(grad_output, weight, input, grad_weight_funct, grad_input_funct)
    
def bp_linear_w32_a16(grad_output, weight, input):
    grad_weight_funct = linear_fp16
    grad_input_funct = torch.matmul
    return linear_bp(grad_output, weight, input, grad_weight_funct, grad_input_funct)

def bp_linear_w16_a16(grad_output, weight, input):
    grad_weight_funct = torch.matmul
    grad_input_funct = torch.matmul
    return linear_bp(grad_output, weight, input, grad_weight_funct, grad_input_funct)