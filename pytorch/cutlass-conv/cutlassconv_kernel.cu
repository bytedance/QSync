#include <torch/extension.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>

#include <cudnn.h>
#include <ATen/cudnn/Descriptors.h>
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

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

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp_TensorOP = cutlass::arch::OpClassTensorOp;
using MMAOp_SIMT = cutlass::arch::OpClassSimt; // fp32 can only use simt
// This code section describes how threadblocks are scheduled on GPU

constexpr int split_k_slices = 1; // no reduction
cutlass::conv::SplitKMode const &split_k_mode = cutlass::conv::SplitKMode::kSerial;
// mode (kCrossCorrelation or kConvolution)
cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;



#include "device_config.h"


void assign_4dcrd_with_tensor(cutlass::Tensor4DCoord& tensor_coord, torch::Tensor& tensor, bool contigous){
    if(contigous){
      // store with NHWC and meta with NHWC
      tensor_coord.n() = tensor.size(0);
      tensor_coord.h() = tensor.size(1);
      tensor_coord.w() = tensor.size(2);
      tensor_coord.c() = tensor.size(3);
    }
    else{
      // NCHW in format, but store tensor with NHWC
      // tensor_coord = {tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)};
      tensor_coord.n() = tensor.size(0);
      tensor_coord.c() = tensor.size(1);
      tensor_coord.h() = tensor.size(2);
      tensor_coord.w() = tensor.size(3);
    };
}

void get_relevant_tensorcoord(torch::Tensor& input, torch::Tensor& weight,
            cutlass::Tensor4DCoord& input_size, cutlass::Tensor4DCoord& filter_size, cutlass::Tensor4DCoord& padding,
            cutlass::MatrixCoord& conv_stride, cutlass::MatrixCoord& dilation, cutlass::Tensor4DCoord& output_size,
            int32_t stride_h, int32_t stride_w,
            int32_t padding_h, int32_t padding_w)
{

    int32_t n_in, h_in, w_in, c_in;
    int32_t n_weight, h_weight, w_weight, c_weight;
    if(input.is_contiguous() && weight.is_contiguous()){
      // theres is a case that H = W = 1, they are contigous for both NCHW or NHWC format.
      // there we need to ensure input and weight are both contigous thus the conv format is contiguous
      // input coord
      assign_4dcrd_with_tensor(input_size, input, true);
      // filter coord
      assign_4dcrd_with_tensor(filter_size, weight, true);

      n_in = input.size(0);
      h_in = input.size(1);
      w_in = input.size(2);
      c_in = input.size(3);

      n_weight = weight.size(0);
      h_weight = weight.size(1);
      w_weight = weight.size(2);
      c_weight = weight.size(3);

    }
    else
    {
      // input coord
      assign_4dcrd_with_tensor(input_size, input, false);
      // filter coord
      assign_4dcrd_with_tensor(filter_size, weight, false);

      n_in = input.size(0);
      c_in = input.size(1);
      h_in = input.size(2);
      w_in = input.size(3);

      n_weight = weight.size(0);
      c_weight = weight.size(1);
      h_weight = weight.size(2);
      w_weight = weight.size(3);
    };
    
    // padding, conv, dilation
    padding = {padding_h, padding_h, padding_w, padding_w};
    conv_stride = {stride_h, stride_w};
    dilation = {1, 1};


    // if(w_weight == 1){
    //   // output 
    //   output_size.n() = input.size(0);
    //   output_size.h() = (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1;
    //   output_size.w() = (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1;
    //   output_size.c() = weight.size(0);
    // }
    // else{
    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_INT8, 
                n_in, c_in, h_in, w_in));


    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_INT8, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));


    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, 1, 1,
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_INT32));

    int32_t n_out;
    int32_t c_out;
    int32_t h_out;
    int32_t w_out;

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    output_size.n() = n_out;
    output_size.c() = c_out;
    output_size.h() = h_out;
    output_size.w() = w_out;
    // };  

}

torch::Tensor create_4d_tensor(int64_t N, int64_t H, int64_t W, int64_t C, torch::Dtype OType, torch::Tensor& ref_tensor){
    // auto output = torch::empty({output_size.n(), output_size.h(), output_size.w(), output_size.c()}, torch::dtype(torch::kF32).device(input.device()));
    torch::Tensor output;
    if(ref_tensor.is_contiguous(torch::MemoryFormat::ChannelsLast)){
        output = torch::empty({N, C, H, W}, torch::dtype(OType).device(ref_tensor.device()).
        memory_format(torch::MemoryFormat::ChannelsLast));
    }else{
        //  create a tensor with nchw meta but NHWC storage
        output = torch::empty({N, H, W, C}, torch::dtype(OType).device(ref_tensor.device()));
    };
    return output;
}

#include "conv_bp.h"
#include "conv_fp.h"
#include "cutlass_qdq.h"


