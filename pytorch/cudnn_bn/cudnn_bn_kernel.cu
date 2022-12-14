#include <torch/extension.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <ATen/cudnn/Handle.h>

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

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}


torch::Tensor create_4d_tensor(int64_t N, int64_t H, int64_t W, int64_t C, torch::Dtype OType, torch::Tensor& ref_tensor){
    // auto output = torch::empty({output_size.n(), output_size.h(), output_size.w(), output_size.c()}, torch::dtype(torch::kF32).device(input.device()));
    torch::Tensor output;
    if(ref_tensor.is_contiguous(torch::MemoryFormat::ChannelsLast)){
        output = torch::empty({N, C, H, W}, torch::TensorOptions().dtype(OType).device(ref_tensor.device()).
        memory_format(torch::MemoryFormat::ChannelsLast));
    }else{
        //  create a tensor with nchw meta but NHWC storage
        output = torch::empty({N, H, W, C}, 
        torch::TensorOptions().dtype(OType).device(ref_tensor.device()));
    };
    return output;
}


// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardTraining
torch::Tensor fp32_bn_cudnn_training(torch::Tensor input, 
        torch::Tensor running_mean, torch::Tensor running_var, 
        torch::Tensor weight, torch::Tensor bias,
        bool training, float exponential_average_factor, float eps)
{
    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();
    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    int32_t n_in = input.size(0);
    int32_t c_in = input.size(1);
    int32_t h_in = input.size(2);
    int32_t w_in = input.size(3);

    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NCHW, 
                CUDNN_DATA_FLOAT, 
                n_in, c_in, h_in, w_in));
    
    float alpha = 1.0;
    float beta = 0.0;

    cudnnBatchNormMode_t

    y = torch::empty({n_in, c_in, h_in, w_in}, torch::TensorOptions().dtype(OType).device(ref_tensor.device()));



    checkCUDNN(cudnnBatchNormalizationForwardTraining(
      cudnnHandle_t                    handle,
      cudnnBatchNormMode_t             mode,
      const void                      *alpha,
      const void                      *beta,
      const cudnnTensorDescriptor_t    xDesc,
      const void                      *x,
      const cudnnTensorDescriptor_t    yDesc,
      void                            *y,
      const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
      const void                      *bnScale,
      const void                      *bnBias,
      double                           exponentialAverageFactor,
      void                            *resultRunningMean,
      void                            *resultRunningVariance,
      double                           epsilon,
      void                            *resultSaveMean,
      void                            *resultSaveInvVariance)
    );

    return y;
}


torch::Tensor tensor_core_fp32_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w){

    /* only support n_in and c_in multiply of 4 */
    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    int32_t n_in = input.size(0);
    int32_t c_in = input.size(1);
    int32_t h_in = input.size(2);
    int32_t w_in = input.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_FLOAT, 
                n_in, c_in, h_in, w_in));

    int32_t n_weight= weight.size(0);
    int32_t c_weight = weight.size(1);
    int32_t h_weight = weight.size(2);
    int32_t w_weight = weight.size(3);

    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_FLOAT, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));

    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_FLOAT));

    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_FLOAT, 
                n_out, c_out, h_out, w_out));

    //std::cout<<"create y tensor"<<std::endl;
    auto y = create_4d_tensor(n_out, h_out, w_out, c_out, torch::kF32, input);

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
   
    float alpha = 1.0;
    //float alpha = 1;
    float beta = 0.0;

    //size_t ws_size = 355968;
    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,yDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(input.device()));

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,xDesc,input.data_ptr<float>(),
                wDesc,weight.data_ptr<float>(),
                convDesc,
                algo,
                workspace.data_ptr<int32_t>(),
                ws_size,
                &beta,yDesc,
                y.data_ptr<float>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));
     return y;
}


torch::Tensor tensor_core_fp32_conv_nhwc(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w){

    /* only support n_in and c_in multiply of 4 */
    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    int32_t n_in = input.size(0);
    int32_t h_in = input.size(1);
    int32_t w_in = input.size(2);
    int32_t c_in = input.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_FLOAT, 
                n_in, c_in, h_in, w_in));

    int32_t n_weight=  weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);

    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_FLOAT, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));

    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_FLOAT));

    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_FLOAT, 
                n_out, c_out, h_out, w_out));

    //std::cout<<"create y tensor"<<std::endl;
    auto y = create_4d_tensor(n_out, h_out, w_out, c_out, torch::kF32, input);

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
   
    float alpha = 1.0;
    //float alpha = 1;
    float beta = 0.0;

    //size_t ws_size = 355968;
    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,yDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(input.device()));

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,xDesc,input.data_ptr<float>(),
                wDesc,weight.data_ptr<float>(),
                convDesc,
                algo,
                workspace.data_ptr<int32_t>(),
                ws_size,
                &beta,yDesc,
                y.data_ptr<float>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));
     return y;
}
