# we set the memory compression rate to be 60% compression rate
# to show the indicator difference of different models
from qsync.LpTorch.ILP_solver import select_bit_with_ILP_mem_only
import numpy as np 
import os 
HOME = os.environ['HOME']
MEM_compression_rate = 0.6

def generate_layer_bitwidth_setup(model_name, indicator_type):
    data = np.load(f"{HOME}/profile_result/profile_data_{model_name}_75_{indicator_type}.npz", allow_pickle=True)
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
        M_matrix.append([ava_bits[i] / 32 * M_item for i in range(omega_length)])
        omega_matrix.append(omega)
    M_np_matx = np.array(M_matrix)
    O_np_matx = np.array(omega_matrix)

    # o_mean = np.mean(omega_matrix)
    # o_std = np.std(omega_matrix)

    # O_np_matx = (omega_matrix - o_mean) / o_std
    # print(O_np_matx[:,0] > O_np_matx[:,1])
    # print(M)
    # print(omega_matrix)

    # import pdb; pdb.set_trace()
    opt_df = select_bit_with_ILP_mem_only(M_B=M_max * (1-MEM_compression_rate), M_mem=M_np_matx, M_sens=O_np_matx, available_bits = [8,16], layer_name=layer_name)
    # print(opt_df)
    bit_select_compress_path = f"{HOME}/profile_result/{model_name}_{indicator_type}_bitplan_{MEM_compression_rate}.npy"
    np.save(bit_select_compress_path, opt_df)
    print(f"save to ->> {bit_select_compress_path}")

    # import pdb; pdb.set_trace()

def generate_mem_bitwidth_plans(model_name, indicator_types):
    for indicator_type in indicator_types:
        generate_layer_bitwidth_setup(model_name, indicator_type)

if __name__ == '__main__':
    ava_bits = [8, 16, 32] # T4 for experiment
    # four models mentioned in our paper
    model_name = 'resnet50'
    indicator_types = ['GVAR', 'HESS', 'WVAR']
    generate_mem_bitwidth_plans(model_name, indicator_types)
    
    model_name = 'vgg16bn'
    generate_mem_bitwidth_plans(model_name, indicator_types)

    model_name = 'bert'
    generate_mem_bitwidth_plans(model_name, indicator_types)

    model_name = 'roberta'
    generate_mem_bitwidth_plans(model_name, indicator_types)