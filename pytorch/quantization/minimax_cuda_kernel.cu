#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <cuda.h>
#include <cuda_runtime.h>

using torch::Tensor;


__device__ __inline__ c10::Half __shfl_down_sync(const unsigned mask, const c10::Half var,
                                                 const unsigned int delta, const int width) {
  __half var_ = var;
  return __shfl_down_sync(mask, var_, delta, width);
}


__device__ __inline__ c10::Half __shfl_sync(const unsigned mask, const c10::Half var,
                                            const int delta, const int width) {
  __half var_ = var;
  return __shfl_sync(mask, var_, delta, width);
}


template <typename scalar_t>
__global__ void minimax_cuda_kernel(const scalar_t* __restrict__ data,
                                    scalar_t* __restrict__ min,
                                    scalar_t* __restrict__ max,
                                    int64_t N,
                                    int64_t D) {
  scalar_t max_val, min_val;
  max_val = -1e30;
  min_val = 1e30;

  for (int64_t k1_outer = 0; k1_outer < D / 32; ++k1_outer) {
    max_val = std::max(max_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
    min_val = std::min(min_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
  }

  unsigned int mask;
  scalar_t max_val_t, min_val_t;
  mask = __activemask();

  max_val_t = __shfl_down_sync(mask, max_val, 16, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 8, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 4, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 2, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 1, 32);
  max_val = std::max(max_val, max_val_t);
  max_val = __shfl_sync(mask, max_val, 0, 32);
  max[blockIdx.x] = max_val;

  min_val_t = __shfl_down_sync(mask, min_val, 16, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 8, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 4, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 2, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 1, 32);
  min_val = std::min(min_val, min_val_t);
  min_val = __shfl_sync(mask, min_val, 0, 32);
  min[blockIdx.x] = min_val;
}


template <typename T>
__device__ inline __attribute__((always_inline)) T
quantize_ops_shfl_xor(const T val, int laneMask, int width) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}



template <typename input_t>
__global__ inline void _get_8bit_qparam_cuda_kernel(
    const input_t* __restrict__ input,
    int nrows,
    int ncols,
    int8_t* __restrict__ output,
    input_t* __restrict__ range_list) {
  const int row = (int)blockIdx.x * blockDim.y + threadIdx.y;

  // const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;

  // starting values for future reductions
#ifdef __HIP_PLATFORM_HCC__
#define HIPRT_INF_F __int_as_float(0x7f800000)
  float minimum_element = HIPRT_INF_F;
  float maximum_element = -HIPRT_INF_F;
#undef HIPRT_INF_F
#else
  float minimum_element = CUDART_INF_F;
  float maximum_element = -CUDART_INF_F;
#endif

  // always a power of 2 up to size 32. Multiple rows can share the same warp
  // when smaller than 32.
  const int lane_width = blockDim.x;

  // March warp-wise through the row, doing thread local min and max reductions.
  // This loop will only execute once when ncol <= 32
  if (row < nrows) {
    const input_t* const input_row = input + row * ncols;

    for (int col = threadIdx.x; col < ncols; col += lane_width) {
      // Get thread-local minmax. These are the smallest min and max ever seen
      // by this thread.
      minimum_element = fminf(minimum_element, input_row[col]);
      maximum_element = fmaxf(maximum_element, input_row[col]);
      // output[col] = static_cast<int8_t>((input_row[col] / scale));
    }
  }

  // Perform warp-wide min and max reductions. All threads in the warp
  // participate, even if they aren't assigned to a row, since we can't assume
  // the existence of the `*_sync` warp primitives with support for masking.
  for (int offset = lane_width >> 1; offset > 0; offset >>= 1) {
    minimum_element = fminf(
        minimum_element,
        quantize_ops_shfl_xor(minimum_element, offset, lane_width));
    maximum_element = fmaxf(
        maximum_element,
        quantize_ops_shfl_xor(maximum_element, offset, lane_width));
  }

  // only the leading thread in the warp is needed to return the final result in
  // output. Additionally, threads mapped to non-existent rows do not write to
  // the output array.
  if (threadIdx.x != 0 || row >= nrows) {
    return;
  }

  const input_t range = fmaxf(fabs(maximum_element), fabs(minimum_element)) ;
  range_list[row] = range;
}


uint64_t cuda_calc_xblock_count(int num_items, int threads_per_block){
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}
Tensor minimax_cuda(torch::Tensor& data) {

  const auto input_sizes = data.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;

  // printf("%d", nrows);
  // printf("%s", input_sizes);

  // std::cout <<  input_sizes << std::endl;
  // std::cout <<  last_dim << std::endl;
  // std::cout <<  nrows << std::endl;
  // std::cout <<  ncols << std::endl;
  // std::cout <<  ncols_aligned << std::endl;

  // create output matrix
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = ncols_aligned;
  auto output = at::empty(
      output_dims, // 4 = sizeof(float)
      data.options().dtype(torch::kInt8));


  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);

  // std::cout <<  num_blocks << std::endl;
  // we don't consider nrow < 20's case
  int blockDim_x = 1;
  blockDim_x = 32; // lanewidth

  const int rows_per_block = threads_per_block / blockDim_x;
  const auto num_blocks_warp =
          cuda_calc_xblock_count(nrows, rows_per_block);
  
  // std::cout <<  "dims" << std::endl;
  // std::cout <<  num_blocks_warp << std::endl;
  // std::cout <<  blockDim_x << std::endl;
  // std::cout <<  rows_per_block << std::endl;

  dim3 blockdim(blockDim_x, rows_per_block);

  auto range_tensor = at::empty({nrows}, data.options());

  // torch::max(data);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    data.scalar_type(), "_get_8bit_qparam_cuda_kernel", [&] {
      _get_8bit_qparam_cuda_kernel<scalar_t>
          <<<num_blocks_warp,
              dim3(blockDim_x, rows_per_block),
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              data.data_ptr<scalar_t>(),
              nrows,
              ncols,
              output.data_ptr<int8_t>(),
              range_tensor.data_ptr<scalar_t>());
  });
  auto scale_cmp = range_tensor.max();
  return scale_cmp;
  // auto res = at::div(data, scale_cmp).to(torch::kInt8);
  // // TORCH_CHECK(D % 32 == 0 && D > 32);

  // // AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "minimax_cuda", ([&] {
  // //   minimax_cuda_kernel<scalar_t><<<blocks, threads>>>(
  // //     data.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(), max.data_ptr<scalar_t>(),
  // //     N, D);
  // // }));

  // return std::make_pair(res, scale_cmp);
}




// internal function do quantization int4
template <typename scalar_t>
__global__ void quantize_int8_kernel(
    int8_t* __restrict__ int8_out,      
    scalar_t* __restrict__ input_tensor,      
    int64_t N, scalar_t* scale){
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    int8_out[idx] = static_cast<int8_t>(input_tensor[idx] / scale[0]);
  }
}



torch::Tensor quantize_int8_internal(
    torch::Tensor& tensor,
    torch::Tensor& scale
)
{   
    int64_t N = tensor.numel();
    int64_t BLOCKSIZE = 256;
    int64_t nblock_inp = (N + BLOCKSIZE - 1)/BLOCKSIZE;
    
    auto int8_out = torch::empty_like(tensor, torch::dtype(torch::kInt8).device(tensor.device()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), "quantize_int8_kernel", ([&] {
    quantize_int8_kernel<scalar_t><<<nblock_inp, BLOCKSIZE>>>(
        int8_out.data_ptr<int8_t>(), tensor.data_ptr<scalar_t>(), N, scale.data_ptr<scalar_t>());
    }));

    return int8_out;
    // packed_int4_to_int8<<<nblock_inp, BLOCKSIZE>>>(
    //    out_tensor.data_ptr<int8_t>(), cutlass_int4b, N);
}

void quantize_int8_buffered_internal(torch::Tensor& data, torch::Tensor& int8_data_buffer, torch::Tensor& scale){
    int64_t N = data.numel();
    int64_t BLOCKSIZE = 256;
    int64_t nblock_inp = (N + BLOCKSIZE - 1)/BLOCKSIZE;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "quantize_int8_kernel", ([&] {
    quantize_int8_kernel<scalar_t><<<nblock_inp, BLOCKSIZE>>>(
        int8_data_buffer.data_ptr<int8_t>(), data.data_ptr<scalar_t>(), N, scale.data_ptr<scalar_t>());
    }));
}
