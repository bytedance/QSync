// Functions called in pytorch code.
#include <torch/extension.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

/******************************
 * CUDA Interfaces *
 ******************************/

torch::Tensor cutlass_gemm_tf32_interface(
        torch::Tensor& input, 
        torch::Tensor& weight,
        torch::Tensor& bias,
        float dq_scale);

torch::Tensor cutlass_gemm_float_interface(
        torch::Tensor& input, 
        torch::Tensor& weight,
        torch::Tensor& bias,
        float dq_scale);

torch::Tensor cutlass_gemm_half_interface(
        torch::Tensor& input, 
        torch::Tensor& weight,
        torch::Tensor& bias,
        float dq_scale);

torch::Tensor cutlass_gemm_int8_interface(
        torch::Tensor& input, 
        torch::Tensor& weight,
        torch::Tensor& bias,
        float dq_scale);

/***********************
 * Pytorch Binded Functions  *
 ***********************/

// float gemm kernel
torch::Tensor gemm_tf32(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float dq_scale)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return cutlass_gemm_tf32_interface(input, weight, bias, dq_scale);
}

torch::Tensor gemm_float(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float dq_scale)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return cutlass_gemm_float_interface(input, weight, bias, dq_scale);
}

// half gemm kernel
torch::Tensor gemm_half(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float dq_scale)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return cutlass_gemm_half_interface(input, weight, bias, dq_scale);
}

// half gemm kernel
torch::Tensor gemm_int8(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float dq_scale)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return cutlass_gemm_int8_interface(input, weight, bias, dq_scale);
}

// // int8 batch gemm kernel
// void batch_gemm_int8(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor ret,
//                     torch::Tensor input_int, torch::Tensor weight_int, torch::Tensor ret_int,
//                     int64_t B, int64_t M, int64_t K, int64_t N,
//                     torch::Tensor scale_input, torch::Tensor scale_weight, torch::Tensor scale_out)
// {
//     cutlass_batchgemm_int_interface(
//             input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), ret.data_ptr<float>(), 
//             input_int.data_ptr<int8_t>(), weight_int.data_ptr<int8_t>(), ret_int.data_ptr<int32_t>(),
//             B, M, K, N,
//             scale_input.item<float>(), scale_weight.item<float>(), scale_out.item<float>());
// }





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_int8", &gemm_int8, "cutlass implemented int8 gemm kernel");
  m.def("gemm_half", &gemm_half, "cutlass implemented half gemm kernel");
  m.def("gemm_float", &gemm_float, "cutlass implemented float gemm kernel");
  m.def("gemm_tf32", &gemm_tf32, "cutlass implemented  tf32 gemm kernel");
 
//   m.def("batchgemm_int8", &batch_gemm_int8, "cutlass implemented int8 batchgemm kernel");
}