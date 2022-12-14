
std::pair<torch::Tensor, torch::Tensor> minimax_cuda(torch::Tensor& input) {

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  printf("%d", nrows);
  printf("%s", input_sizes);

  // std::cout <<  nrows << std::endl;
  // std::cout <<  input_sizes << std::endl;

  uint64_t u_num_items = input.numel();
  
  // The number of threads can be as high as 2048 on some newer architectures,
  // but this is not portable.
  // const uint64_t blocks =
  //     u_num_items / u_threads + (u_num_items % u_threads != 0);

  const uint64_t blockDim_x = 32; // wrap <= 32
  uint64_t threads_per_block = 256;
  TORCH_CHECK(threads_per_block <= 1024, "Number of threads must be <=1024!");


  const uint64_t rows_per_block = threads_per_block  / blockDim_x;



  int64_t num_blocks = (u_num_items + rows_per_block - 1) / rows_per_block;
  // int64_t N = data.size(0);
  // int64_t D = data.size(1);

  

  auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  torch::Tensor min = torch::empty({1}, options);
  torch::Tensor max = torch::empty({1}, options);

  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "minimax_cuda", ([&] {
  //   minimax_cuda_kernel<scalar_t><<<num_blocks, rows_per_block>>>(
  //     data.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(), max.data_ptr<scalar_t>(),
  //     N, D);
  // }));

  return std::make_pair(min, max);
}
