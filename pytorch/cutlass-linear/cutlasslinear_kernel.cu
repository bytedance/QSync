
#include <torch/extension.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
// #include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cuda_runtime.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


#define BLOCKSIZE 1024


// half
using Half = cutlass::half_t;
// since int8 only allows RCR, we implement half and fp32 in RCR
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using ElementComputeEpilogue = float;
using MMAOp = cutlass::arch::OpClassTensorOp;
using SimtOP = cutlass::arch::OpClassSimt;

// Number of pipelines you want to use
constexpr int split_k_slices = 1;


#include "device_config.h"

void get_input_layout(torch::Tensor& input, torch::Tensor& weight, 
                    int& B, int& M, int& K, int& N, 
                    cutlass::MatrixCoord& input_size, cutlass::MatrixCoord& weight_size, cutlass::MatrixCoord& output_size)
{
    if(input.dim() == 3){
      B = input.size(0);
      M = input.size(1);
      K = input.size(2);
    }else{
      B = 1;
      M = input.size(0);
      K = input.size(1);
    }
    // weight is NK
    N = weight.size(0);
    input_size = {B*M, K};
    weight_size = {N, K};
    output_size = {B*M, N};
}

template<typename Gemm>
torch::Tensor cutlass_gemm(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, float dq_scale){

    using ElementInputA = typename Gemm::ElementA;
    using ElementInputB = typename Gemm::ElementB;
    using ElementOutput = typename Gemm::ElementC;
    
    int B, M, K;
    int N;
    cutlass::MatrixCoord input_size, weight_size, output_size;
    get_input_layout(input, weight, B, M, K, N, input_size, weight_size, output_size);

    // tensor refs
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(reinterpret_cast<ElementInputA *>(input.data_ptr()),LayoutInputA(K));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(reinterpret_cast<ElementInputB *>(weight.data_ptr()), LayoutInputB(K));

    torch::Tensor y;
    if(B == 1 && input.dim() != 3){
      y = torch::empty({M, N}, torch::dtype(torch::kF32).device(input.device()));
    }else{
      y = torch::empty({B, M, N}, torch::dtype(torch::kF32).device(input.device()));
    };

    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(y.data_ptr<ElementOutput>(), LayoutOutput(N));

    cutlass::gemm::GemmCoord problem_size(B * M, N, K);
    ElementComputeEpilogue alpha = ElementComputeEpilogue(dq_scale);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    typename Gemm::Arguments arguments{problem_size,           // <- problem size of matrix multiplication
                                      input_ref,  // <- reference to matrix A on device
                                      weight_ref,  // <- reference to matrix B on device
                                      output_ref,  // <- reference to matrix C on device
                                      output_ref,  // <- reference to matrix D on device
                                      {alpha, beta},          // <- tuple of alpha and beta
                                      split_k_slices};        // <- k-dimension split factor
    // Allocate workspace memory
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::dtype(torch::kUInt8).device(input.device()));

    cutlass::Status status;
    Gemm gemm_op;
    status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
    CUTLASS_CHECK(status);

    status = gemm_op();
    CUTLASS_CHECK(status);
    return y;
}


// ref: https://github.com/NVIDIA/cutlass/blob/master/test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sm80.cu

torch::Tensor cutlass_gemm_int8_interface(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, float dq_scale){
    using Gemm = int8_gemm::Gemm;
    return cutlass_gemm<Gemm>(input, weight, bias, dq_scale);
}


torch::Tensor cutlass_gemm_half_interface(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, float dq_scale){
    using Gemm = half_gemm::Gemm;
    return cutlass_gemm<Gemm>(input, weight, bias, dq_scale);
}

torch::Tensor cutlass_gemm_float_interface(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, float dq_scale){
    using Gemm = float_gemm::Gemm;
    return cutlass_gemm<Gemm>(input, weight, bias, dq_scale);
}

torch::Tensor cutlass_gemm_tf32_interface(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, float dq_scale){
    using Gemm = tf32_gemm::Gemm;
    return cutlass_gemm<Gemm>(input, weight, bias, dq_scale);
}



