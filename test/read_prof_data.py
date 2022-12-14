import numpy as np 

data = np.load("profile_result/profile_data_bert_75_HESS.npz", allow_pickle=True)
M, E, RD, omega_matrix, min_M, min_E, smax_M, max_E, ava_idx  = \
            data['M'], data['E'], data['RD'], data['omega'], data['minM'], data['minE'], data['maxM'], data['maxE'], data['ava_idx']

# print(len(ava_idx), len(E))
# print(E)
# print(sum(E[:,1]), sum(E[:,2]))

print(omega_matrix)
M = M.item()
overall_MB = sum([M[i] for i in M])
print(M)

import pdb; pdb.set_trace()

