import torch 
from QSync.indicator import increase_with_step, reduce_with_step
gap_column_path = "/qsync_niti_based/gap_column.pt"
incre_h_path = "/qsync_niti_based/incre.pt"
reduce_h_path = "/qsync_niti_based/reduce.pt"

gap_column = torch.load(gap_column_path)
incre_h = torch.load(incre_h_path)
reduce_h = torch.load(reduce_h_path)
temp_incre_h, temp_decre_h = [], []

# print(gap_column)
# print(incre_h)
# print(reduce_h)
step = 5
for i in range(2):
    a = increase_with_step(incre_h, gap_column, step, reduce_h, temp_incre_h, temp_decre_h )
print(a)
print(temp_incre_h, temp_decre_h )
# print(incre_h)
# print(reduce_h)

for i in range(2):
    a = reduce_with_step(reduce_h, gap_column, step, incre_h, temp_incre_h, temp_decre_h )
print(temp_incre_h, temp_decre_h)
print(a)
# print(incre_h)
# print(reduce_h)