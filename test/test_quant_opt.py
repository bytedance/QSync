# show the quantization optimization performance
import torch
from qsync.LpTorch.quant import quantize_int8
from qsync.LpTorch.conf import config 
from utils import export_to_csv

BS_BASE = 64
BS_TIMES = 6

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


def calculate_duration():
    duration_data = []
    for times in range(1, BS_TIMES):
        A = torch.rand(BS_BASE * times, 64, 56, 56).cuda()
        torch.cuda.synchronize()
        start.record()
        for i in range(100):
            q_A_opt, scale_opt = quantize_int8(A)
        end.record()
        torch.cuda.synchronize()
        Duration = start.elapsed_time(end)
        duration_data.append(Duration)
    return duration_data
config.optimized_int = True
duration_opt = calculate_duration()
config.optimized_int = False
duration_normal = calculate_duration()
print("optimzied: ", duration_opt)
print("not-opt: ",duration_normal)

import pandas as pd
csv_file_name = 'opt_q'
df = pd.DataFrame(data={"not_opt":duration_normal,"opt": duration_opt})
export_to_csv(csv_file_name, df)








