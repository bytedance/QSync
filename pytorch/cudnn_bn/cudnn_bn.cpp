#include <torch/extension.h>

/* Actual Tensor Core Function */
torch::Tensor fp32_bn_cudnn_training(torch::Tensor input, 
        torch::Tensor running_mean, torch::Tensor running_var, 
        torch::Tensor weight, torch::Tensor bias,
        bool training, float exponential_average_factor, float eps)

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_COMPLEX(x) AT_ASSERTM(x.is_contiguous(torch::MemoryFormat::ChannelsLast) || x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS_COMPLEX(x);

/* Extension Interface */

torch::Tensor fp32_bn_train(torch::Tensor input, 
        torch::Tensor running_mean, torch::Tensor running_var, 
        torch::Tensor weight, torch::Tensor bias,
        bool training, float exponential_average_factor, float eps){
    CHECK_INPUT(input);
    return fp32_bn_cudnn_training(input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp32_bn_train", &fp32_bn_train, "fp32 bn train");
  m.def("find_algo", &find_algo, "find the convolution forward algorithm");
}