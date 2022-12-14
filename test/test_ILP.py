from qsync.LpTorch.ILP_solver import select_bit_with_ILP_mem_only
from qsync.LpTorch import config as qconfig
import numpy as np 

data = np.load("profile_result/profile_data_resnet50_75_GVAR.npz", allow_pickle=True)
M, E, RD, omega_dict, min_M, min_E, smax_M, max_E, ava_idx  = \
            data['M'], data['E'], data['RD'], data['omega'], data['minM'], data['minE'], data['maxM'], data['maxE'], data['ava_idx']

M = M.item()
omega_dict = omega_dict.item()
# ILP solver of qsync takes matrix 
M_matrix = []
omega_matrix = []
layer_name = []
M_max = 0
for layer, M_item in M.items():
    omega = omega_dict[layer]
    omega_length = len(omega)
    if omega_length == 0:
        continue
    layer_name.append(layer)
    M_max += M_item
    M_matrix.append([qconfig.available_bits[i] / 32 * M_item for i in range(omega_length)])
    omega_matrix.append(omega)
M_np_matx = np.array(M_matrix)
O_np_matx = np.array(omega_matrix)

# o_mean = np.mean(omega_matrix)
# o_std = np.std(omega_matrix)

# O_np_matx = (omega_matrix - o_mean) / o_std
# print(O_np_matx[:,0] > O_np_matx[:,1])
# print(M)
# print(omega_matrix)
T_B = 99999

# import pdb; pdb.set_trace()
opt_df = select_bit_with_ILP_mem_only(M_B=M_max * 0.4, M_mem=M_np_matx, M_sens=O_np_matx, available_bits = [8,16], layer_name=layer_name)
print(opt_df)