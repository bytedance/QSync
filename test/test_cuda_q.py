import torch 
import cuda_quantization.quant as cuda_q 
from qsync.LpTorch.quant import quantize_int8, collect_scale, quant_with_scale_zp, quant_with_buffer

test_tensor = torch.randn(10, 30, 30, 8).cuda()
scales = collect_scale(test_tensor)

int8_1 = quant_with_scale_zp(test_tensor, scales, 0)
int8_2 = cuda_q.quantize_int8(test_tensor, scales)

print(torch.max(torch.abs(int8_1 - int8_2)))

int8_buffered = torch.empty_like(test_tensor, dtype=torch.int8)
quant_with_buffer(test_tensor, int8_buffered)

print(torch.max(torch.abs(int8_buffered - int8_2)))

# N = 1000

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# start.record()
# for _ in range(N):
#     int8_1 = quant_with_scale_zp(test_tensor, scales, 0)
# end.record()
# torch.cuda.synchronize()
# Duration = start.elapsed_time(end)
# print("Duration 1", Duration)

# start.record()
# for _ in range(N):
#     int8_2 = cuda_q.quantize_int8(test_tensor, scales)
# end.record()
# torch.cuda.synchronize()
# Duration = start.elapsed_time(end)
# print("Duration 2", Duration)




