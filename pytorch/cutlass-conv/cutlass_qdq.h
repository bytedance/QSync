#ifndef cutlass_qdq_h
#define cutlass_qdq_h

// update to grid 2
__global__ void dqd_tf32_kernel(
    float* output_float,      
    float* input_float,      
    int64_t N){
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output_float[idx] = static_cast<float>(static_cast<cutlass::tfloat32_t>(input_float[idx]));
  }
}

// converter
torch::Tensor tf32_qdq(
    torch::Tensor& fp32_tensor
)
{   auto output = torch::empty_like(fp32_tensor);
    int64_t BLOCKSIZE = 256;
    int N = fp32_tensor.numel();
    int64_t nblock_inp = (N + BLOCKSIZE - 1)/BLOCKSIZE;
    dqd_tf32_kernel<<< nblock_inp, BLOCKSIZE>>>(output.data_ptr<float>(), fp32_tensor.data_ptr<float>(), N);
    return output;
}


template <typename scalar_t>
__global__ void dequantize_int4_kernel(
    scalar_t* __restrict__ output_int4b,      
    int8_t* __restrict__ input_float,      
    int64_t N, float scale){
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  int mask = ((1 << 4) - 1);
  int local_pack;
  int8_t packed_value = input_float[idx];
  if (idx * 2 < N) { // each thread for two
    local_pack = (packed_value >> 4) & mask;
    output_int4b[idx * 2] = static_cast<scalar_t>(local_pack) * scale;
    // output_int4b[idx] = static_cast<at::Half>(input_float[idx]) * scale;
  }
  if (idx * 2 + 1 < N) { // each thread for two
    local_pack = (packed_value) & mask;
    output_int4b[idx * 2 + 1] = static_cast<scalar_t>(local_pack) * scale;
    // output_int4b[idx] = static_cast<at::Half>(input_float[idx]) * scale;
  }
}

// // converter
torch::Tensor dequantize_int4(
    torch::Tensor& tensor,
    int64_t N,
    float scale
)
{   
    // tensor here is int4 
    int64_t BLOCKSIZE = 256;
    int64_t nblock_inp = (N + BLOCKSIZE - 1)/BLOCKSIZE;

    auto fp_tensor = torch::empty({N}, torch::dtype(torch::kF32).device(tensor.device()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(fp_tensor.scalar_type(), "dequantize_int4_kernel", ([&] {
    dequantize_int4_kernel<scalar_t><<<nblock_inp, BLOCKSIZE>>>(
        fp_tensor.data_ptr<scalar_t>(), tensor.data_ptr<int8_t>(), N, scale);
    }));
    return fp_tensor;
}


torch::Tensor dequantize_int4_fp16(
    torch::Tensor& tensor,
    int64_t N,
    float scale
)
{   
    // tensor here is int4 
    int64_t BLOCKSIZE = 256;
    int64_t nblock_inp = (N + BLOCKSIZE - 1)/BLOCKSIZE;

    auto fp_tensor = torch::empty({N}, torch::dtype(torch::kF16).device(tensor.device()));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(fp_tensor.scalar_type(), "dequantize_int4_kernel", ([&] {
    dequantize_int4_kernel<scalar_t><<<nblock_inp, BLOCKSIZE>>>(
        fp_tensor.data_ptr<scalar_t>(), tensor.data_ptr<int8_t>(), N, scale);
    }));
    return fp_tensor;
}

#endif