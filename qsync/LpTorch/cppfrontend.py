import os 
from .conf import config
import cpp_backward.quantization as ext_quantization
import cpp_backward.cudnn as ext_backward_func
import cutlasslinear_cuda
if not config.simu:
    import cutlassconv_cuda, cudnn_conv
    from cuda_quantization.quant import minimax
else:
    cutlassconv_cuda, cudnn_conv, minimax = [None] * 3