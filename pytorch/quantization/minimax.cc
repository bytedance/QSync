#include <torch/extension.h>

#include "ext_common.h"

torch::Tensor minimax_cuda(torch::Tensor& data);

torch::Tensor minimax(torch::Tensor& data) {
  CHECK_CUDA_TENSOR_FLOAT(data);
  return minimax_cuda(data);
}


torch::Tensor quantize_int8_internal(torch::Tensor& data, torch::Tensor& scale);

torch::Tensor quantize_int8(torch::Tensor& data, torch::Tensor& scale) {
  CHECK_CUDA_TENSOR_FLOAT(data);
  return quantize_int8_internal(data, scale);
}


void quantize_int8_buffered_internal(torch::Tensor& data, torch::Tensor& int8_data_buffer, torch::Tensor& scale);

void quantize_int8_buffer(torch::Tensor& data, torch::Tensor& int8_data_buffer, torch::Tensor& scale) {
  CHECK_CUDA_TENSOR_FLOAT(data);
  quantize_int8_buffered_internal(data, int8_data_buffer, scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("minimax", &minimax);
  m.def("quantize_int8", &quantize_int8);
  m.def("quantize_int8_buffer", &quantize_int8_buffer);
}