#include <torch/extension.h>

/* Actual Tensor Core Function */
//torch::Tensor tensor_core_dgrad(
        //torch::Tensor& err_in, 
        //torch::Tensor& weight);
//int4 ops
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensor_core_sp_conv_int4(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale,
        float scale_inp, float scale_w);

// accept stride and padding
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensor_core_sp_conv_optimized_int4(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale,
        float scale_inp, float scale_w);

// int8 ops
torch::Tensor tensor_core_sp_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

// accept stride and padding
torch::Tensor tensor_core_sp_conv_optimized(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

// fp 16 ops
torch::Tensor tensor_core_sp_conv_fp16(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

torch::Tensor tensor_core_sp_conv_optimized_fp16(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

// tf 32 ops
torch::Tensor tensor_core_sp_conv_tf32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

torch::Tensor tensor_core_sp_conv_optimized_tf32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

// fp 32 ops
torch::Tensor tensor_core_sp_conv_fp32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

torch::Tensor tensor_core_sp_conv_optimized_fp32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale);

// gradient out
std::tuple<torch::Tensor, torch::Tensor> tensor_core_convbp_fp32(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale);

torch::Tensor tensor_core_convbp_fp32_dgrad(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale);

torch::Tensor tensor_core_convbp_fp32_wgrad(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
);

torch::Tensor tensor_core_convbp_fp32_wgrad_c3(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
);


// gradient out
std::tuple<torch::Tensor, torch::Tensor> tensor_core_convbp_fp16(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale);

torch::Tensor tensor_core_convbp_fp16_dgrad(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale);

torch::Tensor tensor_core_convbp_fp16_wgrad(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
);

torch::Tensor tensor_core_convbp_fp16_wgrad_c3(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
);



std::tuple<torch::Tensor, torch::Tensor> tensor_core_convbp_fp16_to_fp16(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale);

torch::Tensor tensor_core_convbp_fp16_to_fp16_dgrad(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale);

torch::Tensor tensor_core_convbp_fp16_to_fp16_wgrad(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
);

torch::Tensor tensor_core_convbp_fp16_to_fp16_wgrad_c3(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
);

// conversion
torch::Tensor tf32_qdq(
    torch::Tensor& fp32_tensor
);


torch::Tensor quantize_int4(
    torch::Tensor& fp_tensor,
    float scale
);

torch::Tensor dequantize_int4(
    torch::Tensor& tensor,
    int64_t N,
    float scale
);

torch::Tensor dequantize_int4_fp16(
    torch::Tensor& tensor,
    int64_t N,
    float scale
);


#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_COMPLEX(x) AT_ASSERTM(x.is_contiguous(torch::MemoryFormat::ChannelsLast) || x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS_COMPLEX(x);

/* Extension Interface */
// stride 1, padding 1, dilation 1, kernel 3x3

// stride, padding, group
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sp_conv_int4(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale, float scale_inp, float scale_w){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_int4(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale, scale_inp, scale_w);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sp_conv_optimized_int4(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale, float scale_inp, float scale_w){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_optimized_int4(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale, scale_inp, scale_w);
}

// stride, padding, group
torch::Tensor sp_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}

torch::Tensor sp_conv_optimized(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_optimized(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}


// // newly added part, fp16
torch::Tensor sp_conv_fp16(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_fp16(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}

torch::Tensor sp_conv_optimized_fp16(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_optimized_fp16(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}

// tf32
torch::Tensor sp_conv_tf32(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_tf32(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}

torch::Tensor sp_conv_optimized_tf32(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_optimized_tf32(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}


// fp32
torch::Tensor sp_conv_fp32(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_fp32(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}

torch::Tensor sp_conv_optimized_fp32(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dq_scale){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_optimized_fp32(input, weight, stride_h, stride_w, padding_h, padding_w, dq_scale);
}

// bp related

std::tuple<torch::Tensor, torch::Tensor> sp_core_convbp_fp32(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp32(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor sp_core_convbp_fp32_dgrad(
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        torch::Tensor& input,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp32_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor sp_core_convbp_fp32_wgrad(
        torch::Tensor& grad_output,
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp32_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}

// when channel = 3
torch::Tensor sp_core_convbp_fp32_wgrad_c3(
        torch::Tensor& grad_output,
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp32_wgrad_c3(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


std::tuple<torch::Tensor, torch::Tensor> sp_core_convbp_fp16(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor sp_core_convbp_fp16_dgrad(
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        torch::Tensor& input,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor sp_core_convbp_fp16_wgrad(
        torch::Tensor& grad_output,
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}

// when channel = 3
torch::Tensor sp_core_convbp_fp16_wgrad_c3(
        torch::Tensor& grad_output,
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16_wgrad_c3(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}



std::tuple<torch::Tensor, torch::Tensor> sp_core_convbp_fp16_to_fp16(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16_to_fp16(input, grad_output, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor sp_core_convbp_fp16_to_fp16_dgrad(
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        torch::Tensor& input,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16_to_fp16_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor sp_core_convbp_fp16_to_fp16_wgrad(
        torch::Tensor& grad_output,
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16_to_fp16_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}

// when channel = 3
torch::Tensor sp_core_convbp_fp16_to_fp16_wgrad_c3(
        torch::Tensor& grad_output,
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale)
{
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);
        CHECK_INPUT(weight);
        return tensor_core_convbp_fp16_to_fp16_wgrad_c3(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}



// conversion
torch::Tensor cutlass_tf32_qdq(
    torch::Tensor& fp32_tensor
){
      CHECK_INPUT(fp32_tensor); 
      return tf32_qdq(fp32_tensor);
}

torch::Tensor cutlass_quantize_int4(
    torch::Tensor& fp_tensor,
    float scale
){
      CHECK_INPUT(fp_tensor); 
      return quantize_int4(fp_tensor, scale);
}

torch::Tensor cutlass_dequantize_int4(
    torch::Tensor& tensor,
    int64_t N,
    float scale
){
    CHECK_INPUT(tensor); 
    return dequantize_int4(tensor, N, scale);
}


torch::Tensor cutlass_dequantize_int4_fp16(
    torch::Tensor& tensor,
    int64_t N,
    float scale
){
    CHECK_INPUT(tensor); 
    return dequantize_int4_fp16(tensor, N, scale);
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sp_conv_int4", &sp_conv_int4, "int4 convolution forward Nvidia GPU tensor core");
  m.def("sp_conv_optimized_int4", &sp_conv_optimized_int4, "int4 convolution forward Nvidia GPU tensor core");

  m.def("sp_conv", &sp_conv, "int8 convolution forward Nvidia GPU tensor core");
  m.def("sp_conv_optimized", &sp_conv_optimized, "int8 convolution forward Nvidia GPU tensor core");
  
  m.def("sp_conv_fp16", &sp_conv_fp16, "fp16 convolution forward Nvidia GPU tensor core");
  m.def("sp_conv_optimized_fp16", &sp_conv_optimized_fp16, "fp16 convolution forward Nvidia GPU tensor core");

  m.def("sp_conv_tf32", &sp_conv_tf32, "fp32 convolution forward Nvidia GPU tensor core");
  m.def("sp_conv_optimized_tf32", &sp_conv_optimized_tf32, "fp32 convolution forward Nvidia GPU tensor core");

  m.def("sp_conv_fp32", &sp_conv_fp32, "fp32 convolution forward Nvidia GPU tensor core");
  m.def("sp_conv_optimized_fp32", &sp_conv_optimized_fp32, "fp32 convolution forward Nvidia GPU tensor core");


  // bp
  m.def("sp_core_convbp_fp32", &sp_core_convbp_fp32, "fp32 convolution backward Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp32_dgrad", &sp_core_convbp_fp32_dgrad, "fp32 convolution backward_dgrad Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp32_wgrad", &sp_core_convbp_fp32_wgrad, "fp32 convolution backward_wgrad Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp32_wgrad_c3", &sp_core_convbp_fp32_wgrad_c3, "fp32 convolution backward_wgrad for first conv");

  // bp fp16
  m.def("sp_core_convbp_fp16", &sp_core_convbp_fp16, "fp16 convolution backward Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp16_dgrad", &sp_core_convbp_fp16_dgrad, "fp16 convolution backward_dgrad Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp16_wgrad", &sp_core_convbp_fp16_wgrad, "fp16 convolution backward_wgrad Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp16_wgrad_c3", &sp_core_convbp_fp16_wgrad_c3, "fp16 convolution backward_wgrad for first conv");
  
  // bp fp16 2 fp16
  m.def("sp_core_convbp_fp16_to_fp16", &sp_core_convbp_fp16_to_fp16, "fp16_to_fp16 convolution backward Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp16_to_fp16_dgrad", &sp_core_convbp_fp16_to_fp16_dgrad, "fp16_to_fp16 convolution backward_dgrad Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp16_to_fp16_wgrad", &sp_core_convbp_fp16_to_fp16_wgrad, "fp16_to_fp16 convolution backward_wgrad Nvidia GPU tensor core");
  m.def("sp_core_convbp_fp16_to_fp16_wgrad_c3", &sp_core_convbp_fp16_to_fp16_wgrad_c3, "fp16_to_fp16 convolution backward_wgrad for first conv");

  // tool
  m.def("cutlass_tf32_qdq", &cutlass_tf32_qdq, "cutlass qdq conversion for tf32");
  m.def("cutlass_quantize_int4", &cutlass_quantize_int4, "quantize number to cutlass int4b");
  m.def("cutlass_dequantize_int4", &cutlass_dequantize_int4, "dequantize number to cutlass int4b");
  m.def("cutlass_dequantize_int4_fp16", &cutlass_dequantize_int4_fp16, "dequantize number to cutlass int4b");
}
