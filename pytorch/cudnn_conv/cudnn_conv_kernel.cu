#include <torch/extension.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Descriptors.h>

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

void tensor_core_find_algo(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t float_flag){

    /* only support n_in and c_in multiply of 4 */

    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnDataType_t input_type;
    cudnnDataType_t output_type;
    cudnnDataType_t conv_type;
    cudnnTensorFormat_t input_format;
    cudnnTensorFormat_t output_format;

    if (float_flag == 0){
        conv_type = CUDNN_DATA_INT32;

        //input_type = CUDNN_DATA_INT8;
        //input_format = CUDNN_TENSOR_NHWC;
        input_type = CUDNN_DATA_INT8x4;
        input_format = CUDNN_TENSOR_NCHW_VECT_C;

        //output_type = CUDNN_DATA_INT8x4;
        //output_format = CUDNN_TENSOR_NCHW_VECT_C;

        output_type = CUDNN_DATA_FLOAT;
        //output_type = CUDNN_DATA_INT8;
        output_format = CUDNN_TENSOR_NHWC;
    }
    else{
        conv_type = CUDNN_DATA_FLOAT;

        input_type = CUDNN_DATA_FLOAT;
        input_format = CUDNN_TENSOR_NHWC;

        output_type = CUDNN_DATA_FLOAT;
        output_format = CUDNN_TENSOR_NHWC;
    }

    int32_t n_in = input.size(0);
    int32_t h_in = input.size(1);
    int32_t w_in = input.size(2);
    int32_t c_in = input.size(3);
    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                input_format, 
                input_type, 
                n_in, c_in, h_in, w_in));
    //std::cout<<n_in<<' '<<h_in<<' '<<w_in<<' '<<c_in<<' '<<std::endl;

    int32_t n_weight= weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);
    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                input_type, 
                input_format, 
                n_weight, c_weight, h_weight, w_weight));

    //std::cout<<n_weight<<' '<<h_weight<<' '<<w_weight<<' '<<c_weight<<' '<<std::endl;

    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, dilation, dilation, 
                CUDNN_CROSS_CORRELATION,
                conv_type));

    //std::cout<<"create conv descriptor"<<std::endl;

    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));
    //std::cout<<n_out<<' '<<h_out<<' '<<w_out<<' '<<c_out<<' '<<std::endl;

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                output_format, 
                output_type, 
                n_out, c_out, h_out, w_out));

    //std::cout<<"create y tensor"<<std::endl;
    //auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    cudnnConvolutionFwdAlgoPerf_t perfResults[3];
    int32_t algo_cnt;
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
                cudnnHandle, 
                xDesc, 
                wDesc, 
                convDesc, 
                yDesc, 
                3, 
                &algo_cnt, 
                //&perfResults));
                perfResults));

    std::cout<<"float flag: "<<float_flag<<std::endl;
    std::cout<<"conv algorithm count: "<<algo_cnt<<std::endl;
    for (int i=0; i<algo_cnt;i++){
        std::cout<<"algo: "<<perfResults[i].algo<<std::endl;
        std::cout<<"time: "<<perfResults[i].time<<std::endl;
        std::cout<<"memory: "<<perfResults[i].memory<<std::endl;
        std::cout<<"mathType: "<<perfResults[i].mathType<<std::endl;
        std::cout<<"==============="<<std::endl;
    }

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // this is algo 0
    //std::cout<<"algo: "<<CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM<<std::endl;
    // this is algo 1
    //std::cout<<"algo: "<<CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM<<std::endl;
    // according to find algo function, should use algo 1
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


torch::Tensor tensor_core_int8_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        float dq_scale){

    /* only support n_in and c_in multiply of 4 */

    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    // int32_t n_in = input.size(0);
    // int32_t c_in = input.size(1);
    // int32_t h_in = input.size(2);
    // int32_t w_in = input.size(3);
    int32_t n_in = input.size(0);
    int32_t h_in = input.size(1);
    int32_t w_in = input.size(2);
    int32_t c_in = input.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_INT8, 
                n_in, c_in, h_in, w_in));

    // int32_t n_weight = weight.size(0);
    // int32_t c_weight = weight.size(1);
    // int32_t h_weight = weight.size(2);
    // int32_t w_weight = weight.size(3);

    int32_t n_weight = weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);

    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_INT8, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));


    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_INT32));

    int32_t n_out;
    int32_t c_out;
    int32_t h_out;
    int32_t w_out;

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_FLOAT, 
                n_out, c_out, h_out, w_out));

    // std::cout<<"create y tensor"<<std::endl;
    // printf( "NCHW %d, %d, %d, %d\n", n_out, c_out, h_out, w_out );
    auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kFloat32).device(input.device()));

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    

    // cudnnConvolutionFwdAlgo_t algo;
    // cudnnGetConvolutionForwardAlgorithm(cudnn,
    //     xDesc,wDesc,convDesc,yDesc,
    //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //     /*memoryLimitInBytes=*/0,
    //     &algo);
    // cudnnConvolutionFwdAlgo_t algo;
    // int RetCnt; cudnnConvolutionFwdAlgoPerf_t fwd_algo_pref_[4];
    // checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,xDesc,wDesc,convDesc,yDesc,
    //     4,
    //     &RetCnt,
    //     fwd_algo_pref_));
    
    // bool found_conv_algorithm = false;
    // for(int n=0;n<RetCnt;n++){
    //     if (fwd_algo_pref_[n].status == CUDNN_STATUS_SUCCESS &&
    //         fwd_algo_pref_[n].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED){
    //       found_conv_algorithm = true;
    //       //fwd_algo_[i]                   = fwd_algo_pref_[n].algo;
    //       //workspace_fwd_sizes_[i]        = fwd_algo_pref_[n].memory;
    //       algo = fwd_algo_pref_[n].algo;
    //       break;
    //     }
    // }
    // if(!found_conv_algorithm) LOG(ERROR) << "[Forward_gpu()]cuDNN did not return a suitable algorithm for convolution.";
   
    float alpha = dq_scale;
    //float alpha = 1;
    float beta = 0.0;

    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,yDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(input.device()));

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,xDesc,input.data_ptr<int8_t>(),
                wDesc,weight.data_ptr<int8_t>(),
                convDesc,
                algo,
                workspace.data_ptr<int32_t>(),
                ws_size,
                &beta,yDesc,
                y.data<float>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

     return y;
}


torch::Tensor tensor_core_int8_conv_channels_last(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        torch::Tensor& dq_scale){

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
                CUDNN_DATA_INT8, 
                n_in, c_in, h_in, w_in));

    int32_t n_weight = weight.size(0);
    int32_t c_weight = weight.size(1);
    int32_t h_weight = weight.size(2);
    int32_t w_weight = weight.size(3);


    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_INT8, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));


    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_INT32));

    int32_t n_out;
    int32_t c_out;
    int32_t h_out;
    int32_t w_out;

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    // cudnnTensorDescriptor_t yDesc;
    // checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    // checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
    //             CUDNN_TENSOR_NHWC, 
    //             CUDNN_DATA_FLOAT, 
    //             n_out, c_out, h_out, w_out));
    
    at::native::TensorDescriptor yDesc;

    // std::cout<<"create y tensor"<<std::endl;
    // printf( "NCHW %d, %d, %d, %d\n", n_out, c_out, h_out, w_out );
    // auto y = create_4d_tensor(n_out, h_out, w_out, c_out, torch::kFloat32, input);
    auto y = torch::empty({n_out, c_out, h_out, w_out}, torch::dtype(torch::kFloat32).device(input.device()).
        memory_format(torch::MemoryFormat::ChannelsLast));
    
    yDesc.set(y);
    

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    

    // cudnnConvolutionFwdAlgo_t algo;
    // cudnnGetConvolutionForwardAlgorithm(cudnn,
    //     xDesc,wDesc,convDesc,yDesc,
    //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //     /*memoryLimitInBytes=*/0,
    //     &algo);
    // cudnnConvolutionFwdAlgo_t algo;
    // int RetCnt; cudnnConvolutionFwdAlgoPerf_t fwd_algo_pref_[4];
    // checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,xDesc,wDesc,convDesc,yDesc,
    //     4,
    //     &RetCnt,
    //     fwd_algo_pref_));
    
    // bool found_conv_algorithm = false;
    // for(int n=0;n<RetCnt;n++){
    //     if (fwd_algo_pref_[n].status == CUDNN_STATUS_SUCCESS &&
    //         fwd_algo_pref_[n].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED){
    //       found_conv_algorithm = true;
    //       //fwd_algo_[i]                   = fwd_algo_pref_[n].algo;
    //       //workspace_fwd_sizes_[i]        = fwd_algo_pref_[n].memory;
    //       algo = fwd_algo_pref_[n].algo;
    //       break;
    //     }
    // }
    // if(!found_conv_algorithm) LOG(ERROR) << "[Forward_gpu()]cuDNN did not return a suitable algorithm for convolution.";
   
    
    
    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,yDesc.desc(),algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(input.device()));

    float alpha = dq_scale.item<float>();
    float beta = 0.0f;
    // float alpha = 1.0f;


    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,xDesc,input.data_ptr<int8_t>(),
                wDesc,weight.data_ptr<int8_t>(),
                convDesc,
                algo,
                workspace.data_ptr<int32_t>(),
                ws_size,
                &beta,yDesc.desc(),
                y.data<float>()));
    // y.mul_(dq_scale);

    //  checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

    return y;
}



torch::Tensor tensor_core_int8_conv_nchw(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        float dq_scale){

    /* only support n_in and c_in multiply of 4 */

    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    int32_t n_in = input.size(0);
    int32_t c_in = input.size(1);
    int32_t h_in = input.size(2);
    int32_t w_in = input.size(3);

    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NCHW, 
                CUDNN_DATA_INT8, 
                n_in, c_in, h_in, w_in));

    int32_t n_weight = weight.size(0);
    int32_t c_weight = weight.size(1);
    int32_t h_weight = weight.size(2);
    int32_t w_weight = weight.size(3);

    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_INT8, 
                CUDNN_TENSOR_NCHW, 
                n_weight, c_weight, h_weight, w_weight));


    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_INT32));
    

    // convert x, w to nchw_vect_c
    cudnnTensorDescriptor_t int8_4_xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&int8_4_xDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(int8_4_xDesc, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, n_in, c_in, h_in, w_in));

    cudnnFilterDescriptor_t int8_4_wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&int8_4_wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(int8_4_wDesc, 
                CUDNN_DATA_INT8x4, 
                CUDNN_TENSOR_NCHW_VECT_C, 
                n_weight, c_weight, h_weight, w_weight));

    auto int8_4_x = torch::empty({n_in, c_in, h_in, w_in}, input.options());
    auto int8_4_w = torch::empty({n_weight, c_weight, h_weight, w_weight}, weight.options());
    // y is still ok to be float NCHW

    float one = 1;float zero = 0;
    cudnnStatus_t transform_status;
    transform_status =
        cudnnTransformTensor(
            cudnnHandle,
            &one,
            xDesc,
            input.data_ptr<int8_t>(),
            &zero,
            int8_4_xDesc,
            int8_4_x.data_ptr<int8_t>());
    checkCUDNN(transform_status);
    cudnnTensorTransformDescriptor_t filter_trans_descriptor;
    checkCUDNN(cudnnCreateTensorTransformDescriptor(&filter_trans_descriptor));
    checkCUDNN(cudnnSetTensorTransformDescriptor(filter_trans_descriptor,
                                  weight.dim(),
                                  CUDNN_TENSOR_NCHW_VECT_C,
                                  NULL,
                                  NULL,
                                  NULL,
                                  CUDNN_TRANSFORM_FOLD));

    transform_status =
        cudnnTransformFilter(
            cudnnHandle,
            filter_trans_descriptor,
            &one,
            wDesc,
            weight.data_ptr<int8_t>(),
            &zero,
            int8_4_wDesc,
            int8_4_w.data_ptr<int8_t>());
    checkCUDNN(transform_status);
    checkCUDNN(cudnnDestroyTensorTransformDescriptor(filter_trans_descriptor));


    int32_t n_out;
    int32_t c_out;
    int32_t h_out;
    int32_t w_out;

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NCHW, 
                CUDNN_DATA_FLOAT, 
                n_out, c_out, h_out, w_out));

    // std::cout<<"create y tensor"<<std::endl;
    // printf( "NCHW %d, %d, %d, %d\n", n_out, c_out, h_out, w_out );
    auto y = torch::empty({n_out, c_out, h_out, w_out}, torch::dtype(torch::kFloat32).device(input.device()));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; // called tensorcore, only support this
   
    float alpha = dq_scale;
    //float alpha = 1;
    float beta = 0.0;

    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,int8_4_xDesc,int8_4_wDesc,convDesc,yDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(input.device()));

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,int8_4_xDesc,int8_4_x.data_ptr<int8_t>(),
                int8_4_wDesc,int8_4_w.data_ptr<int8_t>(),
                convDesc,
                algo,
                workspace.data_ptr<int32_t>(),
                ws_size,
                &beta,yDesc,
                y.data<float>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(int8_4_wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(int8_4_xDesc));

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


// cudnn backward fp16, fp16 situation
torch::Tensor tensor_core_backward_data(
        torch::Tensor& grad_output, 
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w){

    /* only support n_in and c_in multiply of 4 */
    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    // save the result in
    cudnnTensorDescriptor_t dxDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&dxDesc));
    int32_t n_in = input.size(0);
    int32_t h_in = input.size(1);
    int32_t w_in = input.size(2);
    int32_t c_in = input.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(dxDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_in, c_in, h_in, w_in));

    int32_t n_weight=  weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);

    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_HALF, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));
    
    int32_t n_grad_output=  grad_output.size(0);
    int32_t h_grad_output = grad_output.size(1);
    int32_t w_grad_output = grad_output.size(2);
    int32_t c_grad_output = grad_output.size(3);

    cudnnTensorDescriptor_t goDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&goDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(goDesc,
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_grad_output, c_grad_output, h_grad_output, w_grad_output));


    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_FLOAT));

    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

    auto dx = create_4d_tensor(n_in, h_in, w_in, c_in, torch::kF16, input);

    float alpha = 1.0;
    //float alpha = 1;
    float beta = 0.0;

    //size_t ws_size = 355968;
    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,wDesc,goDesc,convDesc,dxDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(input.device()));

    checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
                &alpha,wDesc,weight.data_ptr<at::Half>(),
                goDesc,grad_output.data_ptr<at::Half>(),
                convDesc,
                algo,
                workspace.data_ptr<int32_t>(),
                ws_size,
                &beta,dxDesc,
                dx.data_ptr<at::Half>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(goDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(dxDesc));
     return dx;
}


torch::Tensor tensor_core_group_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t groups){

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
                CUDNN_DATA_INT8, 
                n_in, c_in, h_in, w_in));

    int32_t n_weight= weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);
    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_INT8, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));

    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, dilation, dilation, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_INT32));

    checkCUDNN(cudnnSetConvolutionGroupCount(convDesc,groups));

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
    auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
   
    float alpha = 1.0;
    //float alpha = 1;
    float beta = 0.0;

    //size_t ws_size = 355968;
    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,yDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,xDesc,input.data<int8_t>(),
                wDesc,weight.data<int8_t>(),
                convDesc,
                algo,
                workspace.data<int32_t>(),
                ws_size,
                &beta,yDesc,
                y.data<float>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

     return y;
}


at::Tensor cudnnNhwcToNchw(const at::Tensor& input) {

  int N = input.size(0);
  int C = input.size(3);
  int H = input.size(1);
  int W = input.size(2);
  auto output = at::empty({N,C,H,W}, input.options());
  auto handle = at::native::getCudnnHandle();
  at::native::TensorDescriptor in_desc;
  at::native::TensorDescriptor out_desc;
  in_desc.set(input, 4);
  out_desc.set(output, 4);
  float alpha=1.0f;
  float beta=0.0f;
  cudnnTransformTensor(handle, &alpha, in_desc.desc(), input.data_ptr(), &beta, out_desc.desc(), output.data_ptr());
  return output;
}

at::Tensor cudnnNchwToNhwc(const at::Tensor& input) {

  int N = input.size(0);
  int C = input.size(1);
  int H = input.size(2);
  int W = input.size(3);
  auto output = at::empty({N,H,W,C}, input.options());
  auto handle = at::native::getCudnnHandle();
  at::native::TensorDescriptor in_desc;
  at::native::TensorDescriptor out_desc;
  in_desc.set(input, 4);
  out_desc.set(output, 4);
  float alpha=1.0f;
  float beta=0.0f;
  cudnnTransformTensor(handle, &alpha, in_desc.desc(), input.data_ptr(), &beta, out_desc.desc(), output.data_ptr());
  return output;
}


// torch::Tensor backward_filter(torch::Tensor& grad_output, torch::Tensor& input, 
//         torch::Tensor& weight,
//         int32_t stride_h, int32_t stride_w,
//         int32_t padding_h, int32_t padding_w,
//         int32_t dilation_h, int32_t dilation_w)
// {
//     cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();
//     CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
//                                                                      inputTensorDescriptor->descriptor(),
//                                                                      outputTensorDescriptor->descriptor(),
//                                                                      convolutionDescriptor_,
//                                                                      filterDescriptor->descriptor(),
//                                                                      algo,
//                                                                      &workspace_size));
//     return workspace_size;
// }