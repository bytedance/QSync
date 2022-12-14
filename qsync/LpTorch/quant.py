import torch 
from .conf import config
from .cppfrontend import minimax
import cuda_quantization.quant as cuda_q 

def get_scale(max_, bit=8):
    scale = max_ / (2 ** (bit-1) - 1)
    return scale

def get_amax(val):
    max_ = torch.max(torch.abs(val)) 
    return max_

# channel-wise quantization
def get_amax_channel_wise(val):
    val_shape = val.shape
    if len(val_shape) == 3:
        max_ = torch.amax(torch.abs(val), dim=(1,2))
        return max_.view(val_shape[0], 1, 1)
    else:
        max_ = torch.amax(torch.abs(val), dim=(1,2,3))
        return max_.view(val_shape[0], 1, 1, 1)

# # kernel-wise quantization
# def get_amax_kernel_wise(val):
#     max_ = torch.amax(torch.abs(val), dim=(2,3))
#     return max_.view(max_.shape[0], max_.shape[1], 1, 1)

def quant_with_scale_zp(x, scale, zp):
    res = torch.div(x - zp, scale, rounding_mode='floor').clamp(min=-127, max=127)
    return res.char()

def quant_with_scale_zp_int16(x, scale, zp):
    res = torch.div(x - zp, scale, rounding_mode='floor').clamp(min=-127, max=127)
    return res.short()

def quant_with_scale(x, scale, qtype='scale'):
    if qtype == 'scale':
        return cuda_q.quantize_int8(x, scale)
    else:
        return quant_with_scale_zp(x, scale, 0)

# @torch.jit.script
def quantize_int32(fp32_val: torch.Tensor, default_method: str ='scale'):
    amax = get_amax(fp32_val)
    scale = get_scale(amax, bit=32)
    int32_val = (fp32_val / scale).int()
    return int32_val, scale

# @torch.jit.script
def quantize_int16(fp32_val: torch.Tensor, default_method: str ='scale'):
    amax = get_amax(fp32_val)
    scale = get_scale(amax, bit=16)
    new_int16_val = quant_with_scale_zp_int16(fp32_val, scale, torch.tensor(0))
    return new_int16_val, scale

def collect_scale(fp32_val: torch.Tensor, default_method: str ='scale'):
    if default_method == 'scale':
        amax = get_amax(fp32_val)
    elif default_method == 'channel':
        amax = get_amax_channel_wise(fp32_val)
    # amax = get_amax(fp32_val)
    scale = get_scale(amax)
    return scale

# @torch.jit.script
def quantize_int8(fp32_val: torch.Tensor, default_method: str ='scale'):
    if default_method == 'scale':
        if config.optimized_int:
            return quantize_int8_optimized(fp32_val)
        else:
            amax = get_amax(fp32_val)
    elif default_method == 'channel':
        amax = get_amax_channel_wise(fp32_val)
    # amax = get_amax(fp32_val)
    scale = get_scale(amax)
    new_int8_val = quant_with_scale(fp32_val, scale, default_method)
    return new_int8_val, scale

# for cutlass, we have to create tensor in cutlass's shape, which means the quantization should be embeded to kernel
# different from the bit > 8
def quantize_int4(fp_val: torch.Tensor, default_method: str ='scale'):
    # only support scale for the moment
    if default_method == 'scale':
        if config.optimized_int:
            return quantize_int4_optimized(fp_val)
        else:
            amax = get_amax(fp_val)
    elif default_method == 'channel':
        amax = get_amax_channel_wise(fp_val)
    # amax = get_amax(fp32_val)
    scale = get_scale(amax, 4)
    new_int4_val = cutlass_quantize_int4(fp_val, scale) # but saved in int8
    return new_int4_val, scale

# can be either fp16 or 32
def quantize_int8_optimized(fp_val: torch.Tensor, default_method: str ='scale'):
    # only support scale for the moment
    amax = minimax(fp_val)
    # parts_0, parts_1 = torch.tensor_split(fp_val, 2)
    # amax = max(get_amax(parts_0), get_amax(parts_1))
    scale = get_scale(amax)
    new_int8_val = quant_with_scale(fp_val, scale)
    return new_int8_val, scale

def quantize_int4_optimized(fp_val: torch.Tensor, default_method: str ='scale'):
    # only support scale for the moment
    amax = minimax(fp_val)
    scale = get_scale(amax, 4)
    new_int4_val = cutlass_quantize_int4(fp_val, scale) # but saved in int8
    return new_int4_val, scale

def dequantize_int4(int4_tensor: torch.Tensor, shape, scale, toF32=True):
    # current version, dq int4 support only tensor-quantization
    # only support scale for the moment
    N = 1
    for i in shape:
        N *= i
    if toF32:
        new_int4_val = cutlass_dequantize_int4(int4_tensor, N, scale.item()) # but saved in int8
    else:
        new_int4_val = cutlass_dequantize_int4_fp16(int4_tensor, N, scale.item()) # but saved in int8
    print(new_int4_val)
    return new_int4_val.reshape(shape)

def dequantize_int8(fix_val, scale, toF32=True):
    fp_val = fix_val * scale
    if not toF32: fp_val = fp_val.half()
    return fp_val

def qdq_int4(fp32_val: torch.Tensor):
    PACTED, scale = quantize_int4(fp32_val)
    A_shape = fp32_val.shape
    DE_A = dequantize_int4(PACTED, A_shape, scale)
    return DE_A

def qdq_int8(fp32_val: torch.Tensor, default_method: str ='scale'):
    new_int8_val, scale = quantize_int8(fp32_val, default_method)
    return dequantize_int8(new_int8_val, scale)

def qdq_half(fp32_val: torch.Tensor):
    new_half_val = fp32_val.half()
    return new_half_val.float()

def qdq_tf32(fp32_val: torch.Tensor):
    return cutlass_tf32_qdq(fp32_val)

def qdq_bit(fp32_val: torch.Tensor, qdq_bit: int, default_method='scale'):
    if qdq_bit == 4:
        return qdq_int4(fp32_val)
    elif qdq_bit == 8:
        return qdq_int8(fp32_val, default_method)
    elif qdq_bit == 16:
        return qdq_half(fp32_val)
    elif qdq_bit == 19:
        return qdq_tf32(fp32_val)
    else:
        return fp32_val

def q_bit(fp32_val: torch.Tensor, qdq_bit: int, default_method='scale'):
    if qdq_bit == 4:
        return quantize_int4(fp32_val)
    elif qdq_bit == 8:
        return quantize_int8(fp32_val, default_method)
    elif qdq_bit == 16:
        return fp32.half(), 1
    else:
        return fp32_val, 1

def dq_bit(q_val: torch.Tensor, qdq_bit, scale):
    if qdq_bit == 8:
        return dequantize_int8(q_val, scale)
    elif qdq_bit == 16:
        return q_val.float()
    else:
        return q_val


def convert_to_fp16(x, scale_x=None, bit=None, x_shape=None):
    if bit == 32:
        return x.half()
    elif bit == 19:
        return x.half() # tf32, negelect
    elif bit == 16:
        return x
    elif bit == 8:
        return dequantize_int8(x, scale_x, toF32=False)
    elif bit == 4:
        assert x_shape, "no x_shape passed"
        return dequantize_int4(x, x_shape, scale_x, toF32=False)

def convert_to_fp32(x, scale_x=None, bit=None, x_shape=None):
    if bit == 32:
        return x
    elif bit == 19:
        return x # tf32, negelect
    elif bit == 16:
        return x.float()
    elif bit == 8:
        return dequantize_int8(x, scale_x)
    elif bit == 4:
        assert x_shape, "no x_shape passed"
        return dequantize_int4(x, x_shape, scale_x)

def convert_to_target_tensor(target_tensor, cur_tensor):
    if target_tensor is None or target_tensor.dtype == cur_tensor.dtype:
        return cur_tensor
    if target_tensor.dtype == torch.float32:
        return convert_to_fp32(cur_tensor)
    elif target_tensor.dtype == torch.float16:
        return convert_to_fp16(cur_tensor)
    return cur_tensor

def quant_with_buffer(input, input_buffer):
    scale = collect_scale(input)
    cuda_q.quantize_int8_buffer(input, input_buffer, scale)
    return scale