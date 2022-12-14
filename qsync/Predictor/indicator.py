# indicator that indicates the selection
# for current version, use variance based
import torch 
from qsync.LpTorch.conf import config 
from qsync.LpTorch.layers import QConv2d, QLinear, QBatchNorm2d, QLayerNorm, QEmbedding, QDropout, QSoftmax, QGELU, QMatmul, QAdd
from qsync.LpTorch.quant import qdq_bit
import numpy as np 
from numpy import linalg as LA
import math

def gradient_variance_based_indicator(available_bits, layer, RD_value, gamma, model_depth=None):
    Sv, Sw, Dv, Dw, Dvv, q_inp_norm2, q_weight_norm2 = RD_value
    var_list = []

    # gradient of weight
    # if isinstance(layer, (QBatchNorm2d, QLayerNorm)):
    #     var = DA * (idx + 1) * Sa_2
    layer_depth = layer.depth if hasattr(layer, 'depth') else None 
    if model_depth is not None and layer_depth is not None:
        scale_fwd = model_depth - layer_depth
        scale_bwd = layer_depth
    else:
        # classifier layer or layers that is not bitwidth changable
        scale_fwd = 0
        scale_bwd = 0
    if isinstance(layer, (QConv2d, QLinear, QEmbedding)):
        grad_v = layer.weight.grad if hasattr(layer.weight, 'grad') else 1
        if torch.is_tensor(grad_v) and grad_v is not None:
            grad_v_norm2 = torch.norm(grad_v).detach().cpu().item()
            mantissa_gv, exps_gv = torch.frexp(torch.max(torch.abs(grad_v.half())))
            exps_gv = abs(exps_gv.item()) # biggest one
        else: 
            grad_v_norm2 = 1
            exps_gv = 1
        
        mantissa_v, exps_v = np.frexp(np.float16(math.sqrt(q_inp_norm2 / Dv)))
        mantissa_w, exps_w = np.frexp(np.float16(math.sqrt(q_weight_norm2 / Dw)))
        # print(np.float16(math.sqrt(q_inp_norm2 / Dv)), np.float16(math.sqrt(q_weight_norm2 / Dw)))
        exps_v = abs(exps_v)
        exps_w = abs(exps_w)
        exps_v = 5 if exps_v > 5 else exps_v
        exps_w = 5 if exps_w > 5 else exps_w
        exps_gv = 5 if exps_gv > 5 else exps_gv
        # print(exps_gv, exps_v, exps_w)
        for idx, bit in enumerate(available_bits):
            if bit == 8:
                # print(Sv, Sw, Dvv)
                e = 5; k=9
                fwd_var = 1 / 6 * (q_weight_norm2 * Sv ** 2 * Dv + q_inp_norm2 * Sw ** 2 * Dw )
                bwd_var =  1 / 6 * (grad_v_norm2 * Sv ** 2 * Dv + q_inp_norm2 *  2 ** (2 * exps_v ) * (2 ** (-k)) ** 2 * Dvv)
            elif bit == 16:
                # For FP16, e=5, k=9
                # 2^4e * epsilon^ 4
                e = 5; k=9
                # print(e, -k, q_weight_norm2,  Dv, q_inp_norm2, Dw)
                fwd_var = 1 / 6 * 2 ** (-2*k) * (q_weight_norm2 * Dv * exps_v + q_inp_norm2 * Dw * exps_w)
                bwd_var =  1 / 6 * 2 ** (-2*k) * (grad_v_norm2 * Dv * exps_v + q_inp_norm2 * Dvv * exps_gv)
            elif bit == 32:
                fwd_var = 0
                bwd_var = 0
            var_list.append(gamma * scale_fwd * fwd_var + scale_bwd * bwd_var)
    else: # oneinput
        for idx, bit in enumerate(available_bits):
            if bit == 8:
                fwd_var = 1 / 6 * Sv * Dv
            elif bit == 16:
                # For FP16, e=5, k=9
                # 2^4e * epsilon^ 4
                e = 5; k=9
                fwd_var = ( 2 ** (4*exps_v ) * (2 ** (-k)) ** 4) * Dv 
            elif bit == 32:
                fwd_var = 0
            bwd_var = 0
            var_list.append(gamma * scale_fwd * fwd_var + scale_bwd * bwd_var)
    # import pdb; pdb.set_trace()
    return var_list


# Variance method proposed in https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Fixed-Point_Back-Propagation_Training_CVPR_2020_paper.pdf
def weight_variance_based_indicator(x, layer, available_bits):
    var_list = []
    for bit in available_bits:
        bit = config.map_available_bit(layer, bit)
        qdq_x = qdq_bit(x, bit)
        var = torch.log2(torch.abs((torch.sum(torch.abs(qdq_x)) - torch.sum(torch.abs(x))) / torch.sum(torch.abs(x))) + 1)
        var_list.append(var.detach().cpu().numpy().item())
    return var_list

# Hessian method. We follow the code https://github.com/amirgholami/PyHessian/blob/master/example_pyhessian_analysis.py.
# First compute hessian, get trace then calculate mean. 
# Hessian only focus on the weight 
def hessian_based_indicator(x, layer, available_bits):
    def hessian(x):
        # print(x, x.shape)
        x = np.squeeze(x)
        x_grad = np.gradient(x) 
        hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
        for k, grad_k in enumerate(x_grad):
            # iterate over dimensions
            # apply gradient again to every component of the first derivative.
            tmp_grad = np.gradient(grad_k) 
            for l, grad_kl in enumerate(tmp_grad):
                hessian[k, l, :, :] = grad_kl
        return hessian
    # var list
    var_list = []
    for bit in available_bits:
        bit = config.map_available_bit(layer, bit)
        qdq_x = qdq_bit(x, bit)
        if len(x.shape) < 2 or x.shape[0] == 1:
            var = 0
        else:
            var = np.mean(np.matrix.trace((hessian(qdq_x.detach().cpu().numpy()))))
        var_list.append(var)
    # calculate the distance to the FP32 vector
    var_list = [abs(i - var_list[-1]) for i in var_list[:-1]]
    var_list.append(0)
    return var_list


# compute omega matrix (Indicator Matrix)
def compute_with_variance(name_to_module_mapper, available_bits, gamma, RD, model_depth=None):
    # compute the norm2 for all the gradient
    omega_matrix = []
    omega_dict = {}
    idx = 0
    for layer_name, RD_value in RD.items():
        layer = name_to_module_mapper[layer_name]
        layer_grad = layer.weight.grad if hasattr(layer, 'weight') else None 
        if layer_grad is None: grad_norm = 1 # For the case that profile cannot collect gradient. Update during training.
        else: grad_norm = torch.norm(layer_grad).cpu().numpy() # Frobenius norm
        # the conv list record should be ascending, which means the idx is the accumulation times.

        if layer_grad is None or not isinstance(layer, config.DynamicModule):
            var_list = []
        else:
            if config.indicator_type == 'GVAR':
                var_list = gradient_variance_based_indicator(available_bits, layer, RD_value, gamma, model_depth)
            elif config.indicator_type == 'WVAR':
                var_list = weight_variance_based_indicator(layer.weight, layer, available_bits)
            elif config.indicator_type == 'HESS':
                var_list = hessian_based_indicator(layer.weight, layer, available_bits)
        omega_matrix.append(var_list)
        omega_dict[layer_name] = var_list
        idx += 1
    return omega_matrix, omega_dict



import heapq as hq
import numpy as np 
def get_ind_with_value(val):
    gap_column = np.zeros(val.shape)
    row, column = val.shape
    for i in range(column):
        gap_column[:,i] = val[:, i]

    return gap_column

def reduce_with_step(reduce_h, gap_column, step, increase_h, incre_temp_h, decre_temp_h):
    update_idx = []
    i = 0
    ava_bit_length = gap_column.shape[1]
    while i < step:
        if len(increase_h) <= 0:
            break

        if len(decre_temp_h) != 0:
            plan = decre_temp_h.pop()
            gap, row, bit = plan
            hq.heappush(increase_h, plan)
            update_idx.append([row, bit])
            i += 1 
            continue

        biggest = increase_h[0] 
        cur_gap, row, bit_idx = biggest
        new_bit = bit_idx - 1
        if new_bit == 0:
            # 8 or 4. no avaiable option for left
            item = hq.heappop(increase_h)
            update_idx.append([row, new_bit])
            incre_temp_h.append(biggest)
        else:
            cur_val = gap_column[row, bit_idx]
            new_val = gap_column[row, new_bit]
            gap_val = cur_val - new_val # gap: left - right
            # decrement
            if new_val < 0:
                update_idx.append([row, new_bit])
                item = hq.heappushpop(reduce_h, (gap_val, row, new_bit))
                incre_temp_h.append(biggest)
            # decrement, (we don't care this siutation), remove the idx
            else:
                hq.heappop(reduce_h)
        i += 1 # mv to next point
    return update_idx

def increase_with_step(increase_h, gap_column, step, reduce_h, incre_temp_h, decre_temp_h):
    update_idx = []
    i = 0
    ava_bit_length = gap_column.shape[1]
    while i < step:
        if len(increase_h) <= 0:
            break
        
        if len(incre_temp_h) != 0:
            plan = incre_temp_h.pop()
            gap, row, bit = plan
            hq.heappush(decre_temp_h, plan)
            update_idx.append([row, bit])
            i += 1 
            continue

        biggest = increase_h[0] 
        cur_gap, row, bit_idx = biggest # bit_idx is the current bitwidth
        new_bit = bit_idx + 1
        if new_bit == ava_bit_length - 1:
            # 32. no avaiable option for right of 32.
            item = hq.heappop(increase_h)
            update_idx.append([row, new_bit])
            decre_temp_h.append(biggest)
        else:
            cur_val = gap_column[row, bit_idx]
            new_val = gap_column[row, new_bit]
            gap_val = cur_val - new_val # gap: left - right
            # increment
            if new_val < 0:
                update_idx.append([row, new_bit])
                item = hq.heappushpop(increase_h, (gap_val, row, new_bit))
                temp_h.append(biggest)
            # decrement, (we don't care this siutation), remove the idx
            else:
                hq.heappop(increase_h)
        i += 1 # mv to next point
    return update_idx


def map_bit_setup_to_heap(df, gap_column, reduce_h, increase_h, available_bits):
    # bit setup is at least smaller than 32
    # which means we find the bit in gap column, save its left to reduce_h
    # save its right to increase h
    reduce_h.clear();increase_h.clear()
    hq.heapify(reduce_h);hq.heapify(increase_h)

    ava_length = len(available_bits)
    res = df.index[df['solution_value'] == 1].tolist()
    for i in res:
        layer = df['column_i'][i]
        bit_idx = df['column_j'][i]
        bit = available_bits[bit_idx]
        # print("Layer {} choose bit {}".format(layer, bit))
        # increment bit saves the current bit's right moving result
        # 
        cur_val = gap_column[layer][bit_idx]
        if bit_idx > 0:
            left_val = gap_column[layer][bit_idx - 1]
            hq.heappushpop(reduce_h, (left_val - cur_val, layer, bit_idx))
        if bit_idx < ava_length - 1:
            right_val = gap_column[layer][bit_idx + 1]
            hq.heappush(increase_h, (cur_val - right_val, layer, bit_idx))
    


