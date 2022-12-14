#include <torch/extension.h>

/* Actual Tensor Core Function */
torch::Tensor tensor_core_int8_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        float dq_scale);

torch::Tensor tensor_core_int8_conv_channels_last(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        torch::Tensor& dq_scale);

torch::Tensor tensor_core_int8_conv_nchw(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        float dq_scale);

torch::Tensor tensor_core_fp32_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w);

torch::Tensor tensor_core_fp32_conv_nhwc(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w);

torch::Tensor tensor_core_group_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t groups);


torch::Tensor tensor_core_backward_data(
        torch::Tensor& grad_output, 
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w);

at::Tensor cudnnNhwcToNchw(const at::Tensor& input);
at::Tensor cudnnNchwToNhwc(const at::Tensor& input);

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_COMPLEX(x) AT_ASSERTM(x.is_contiguous(torch::MemoryFormat::ChannelsLast) || x.is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_CHANNEL(x) AT_ASSERTM(x.is_contiguous(torch::MemoryFormat::ChannelsLast), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS_COMPLEX(x);
#define CHECK_INPUT_CHANNEL(x) CHECK_CUDA(x); CHECK_CONTIGUOUS_CHANNEL(x);

/* Extension Interface */

torch::Tensor int8_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int8_conv(input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, dq_scale);
}

torch::Tensor int8_conv_nchw(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int8_conv_nchw(input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, dq_scale);
}

torch::Tensor int8_conv_cl(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w,
        torch::Tensor& dq_scale){

    CHECK_INPUT_CHANNEL(input);
    CHECK_INPUT_CHANNEL(weight);
    return tensor_core_int8_conv_channels_last(input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, dq_scale);

}

torch::Tensor fp32_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_fp32_conv(input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
}

torch::Tensor fp32_conv_nhwc(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_fp32_conv_nhwc(input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
}

torch::Tensor tensor_core_backward_data_interface(
        torch::Tensor& grad_output, 
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        int32_t dilation_h, int32_t dilation_w
){
    CHECK_INPUT(input); CHECK_INPUT(weight); CHECK_INPUT(grad_output);
    return tensor_core_backward_data(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
}


torch::Tensor group_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t groups){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_group_conv(input, weight, stride, padding, dilation, groups);
}

torch::Tensor cudnnNhwcToNchw_interface(const at::Tensor& input){
        CHECK_INPUT(input);
        return cudnnNhwcToNchw(input);
}

torch::Tensor cudnnNchwToNhwc_interface(const at::Tensor& input){
        CHECK_INPUT(input);
        return cudnnNchwToNhwc(input);
}

void tensor_core_find_algo(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t float_flag);

void find_algo(
        torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t float_flag){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_find_algo(input, weight, stride, padding, dilation, float_flag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_conv", &int8_conv, "int8 convolution forward Nvidia GPU tensor core");
  m.def("int8_conv_nchw", &int8_conv_nchw, "int8 convolution forward Nvidia GPU tensor core");
  m.def("int8_conv_cl", &int8_conv_cl, "int8 convolution cl forward Nvidia GPU tensor core");
  m.def("fp32_conv", &fp32_conv, "fp32 convolution forward Nvidia GPU tensor core");
  m.def("fp32_conv_nhwc", &fp32_conv_nhwc, "fp32 nhwc convolution forward Nvidia GPU tensor core");
  m.def("group_conv", &group_conv, "int8 convolution forward Nvidia GPU tensor core");
  m.def("find_algo", &find_algo, "find the convolution forward algorithm");

  m.def("cudnnNchwToNhwc_interface", &cudnnNchwToNhwc_interface, "NCHW -> NHWC");
  m.def("cudnnNhwcToNchw_interface", &cudnnNhwcToNchw_interface, "NHWC -> NCHW");
  m.def("fp16_conv_bp_data", &tensor_core_backward_data_interface, "fp16 data bp");
}