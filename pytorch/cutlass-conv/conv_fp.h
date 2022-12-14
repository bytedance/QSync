#ifndef conv_fp_h
#define conv_fp_h

template<typename Conv2d>
torch::Tensor cutlass_conv(torch::Tensor& input, torch::Tensor& weight, 
                            int32_t stride_h, int32_t stride_w, int32_t padding_h, int32_t padding_w, 
                            float dequant_scale)
{

    using ElementInputA = typename Conv2d::ElementA;
    using ElementInputB = typename Conv2d::ElementB;
    using ElementOutput = typename Conv2d::ElementC;
    using ElementComputeEpilogue = typename Conv2d::ElementCompute;
    using ElementAccumulator = typename Conv2d::ElementAccumulator;

    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    cutlass::Tensor4DCoord output_size;

    get_relevant_tensorcoord(input, weight, 
                            input_size, filter_size, padding, conv_stride, dilation, output_size, 
                            stride_h, stride_w, padding_h, padding_w);

    torch::Tensor output;
    output = create_4d_tensor(output_size.n(), output_size.h(), output_size.w(), output_size.c(), torch::kF32, input);
    // problem
    cutlass::conv::Conv2dProblemSize problem_size(      
            input_size,
            filter_size,
            padding,
            conv_stride,
            dilation,
            output_size,
            mode,
            split_k_slices);

    
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(reinterpret_cast<ElementInputA *>(input.data_ptr()),LayoutInputA::packed(input_size));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(reinterpret_cast<ElementInputB *>(weight.data_ptr()), LayoutInputB::packed(filter_size));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(output.data_ptr<ElementOutput>(), LayoutOutput::packed(output_size));

    ElementComputeEpilogue alpha = ElementComputeEpilogue(dequant_scale);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0.0);

    // C is the accumulator tensor, D is the destination tensor. 
    typename Conv2d::Arguments arguments{
        problem_size,
        input_ref,
        weight_ref,
        output_ref,
        output_ref,
        {alpha, beta},
        split_k_mode,
    };

    Conv2d conv2d_op;
    cutlass::Status status;

    status = conv2d_op.initialize(arguments);
    CUTLASS_CHECK(status);

    status = conv2d_op();
    CUTLASS_CHECK(status);

    return output;
}


// internal function do quantization int4
template <typename scalar_t>
__global__ void quantize_int4_kernel(
    cutlass::int4b_t* __restrict__ output_int4b,      
    scalar_t* __restrict__ input_float,      
    int64_t N, float scale){
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output_int4b[idx] = static_cast<cutlass::int4b_t>(input_float[idx] / scale);
  }
}

__global__ void packed_int4_to_int8(
    int8_t* __restrict__ output_int8,      
    cutlass::int4b_t * __restrict__ input_int4,
    int64_t N){
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t target_output_idx = idx / 2;
  if (idx < N && idx % 2 == 0) {
    output_int8[target_output_idx] = static_cast<int8_t>(input_int4[idx]) << 4;
  }
  if (idx < N && idx % 2 != 0) {
    output_int8[target_output_idx] += static_cast<int8_t>(input_int4[idx]);
  }
}


void quantize_int4_internal(
    torch::Tensor& tensor,
    float scale, int64_t N,
    cutlass::int4b_t* cutlass_int4b,
    torch::Tensor& out_tensor
)
{   
    int64_t BLOCKSIZE = 256;
    int64_t nblock_inp = (N + BLOCKSIZE - 1)/BLOCKSIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), "quantize_int4_kernel", ([&] {
    quantize_int4_kernel<scalar_t><<<nblock_inp, BLOCKSIZE>>>(
        cutlass_int4b, tensor.data_ptr<scalar_t>(), N, scale);
    }));

    // packed_int4_to_int8<<<nblock_inp, BLOCKSIZE>>>(
    //    out_tensor.data_ptr<int8_t>(), cutlass_int4b, N);
}

// converter
torch::Tensor quantize_int4(
    torch::Tensor& tensor,
    float scale
)
{   
    int64_t BLOCKSIZE = 256;
    int N = tensor.numel();
    int64_t nblock_inp = (N + BLOCKSIZE - 1)/BLOCKSIZE;

    // cutlass::Tensor4DCoord tensor_size;
    // assign_4dcrd_with_tensor(tensor_size, tensor);
    int num_bytes = N * cutlass::sizeof_bits<cutlass::int4b_t>::value / 8;
    cutlass::int4b_t* int4_tensor;
    cudaMalloc((void**)&int4_tensor, num_bytes);
    // auto int4_tensor = torch::empty({tensor_size.n(), tensor_size.h(), tensor_size.w(), tensor_size.c()}, torch::dtype(torch::kInt8).device(tensor.device()));
    // cutlass::TensorRef<cutlass::int4b_t, LayoutOutput> output_ref(reinterpret_cast<cutlass::int4b_t>(int4_tensor.data_ptr()), LayoutOutput::packed(tensor_size));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), "quantize_int4_kernel", ([&] {
    quantize_int4_kernel<scalar_t><<<nblock_inp, BLOCKSIZE>>>(
        int4_tensor, tensor.data_ptr<scalar_t>(), N, scale);
    }));

    // cutlass::int4b_t* int4_h_tensor;
    // int4_h_tensor = (cutlass::int4b_t *)malloc(num_bytes);
    // cudaMemcpy(int4_h_tensor, int4_tensor, num_bytes, cudaMemcpyDeviceToHost);
    // for (int i = N - 1; i >= 0; i--) 
    //     std::cout <<  int4_h_tensor[i];
    auto output_int8 = torch::empty({N / 2 + 1},torch::dtype(torch::kInt8).device(tensor.device()));
    packed_int4_to_int8<<<nblock_inp, BLOCKSIZE>>>(
        output_int8.data_ptr<int8_t>(), int4_tensor, N);
        
    cudaFree(int4_tensor);
    return output_int8;
}

// for the kernel smaller than 8
template<typename Conv2d>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cutlass_conv_lower8(torch::Tensor& input, torch::Tensor& weight, 
                            int32_t stride_h, int32_t stride_w, int32_t padding_h, int32_t padding_w, 
                            float dequant_scale, float scale_inp, float scale_w)
{

    using ElementInputA = typename Conv2d::ElementA;
    using ElementInputB = typename Conv2d::ElementB;
    using ElementOutput = typename Conv2d::ElementC;
    using ElementComputeEpilogue = typename Conv2d::ElementCompute;
    using ElementAccumulator = typename Conv2d::ElementAccumulator;

    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    cutlass::Tensor4DCoord output_size;

    get_relevant_tensorcoord(input, weight, 
                            input_size, filter_size, padding, conv_stride, dilation, output_size, 
                            stride_h, stride_w, padding_h, padding_w);
    auto output = create_4d_tensor(output_size.n(), output_size.h(), output_size.w(), output_size.c(), torch::kF32, input);
    // problem
    cutlass::conv::Conv2dProblemSize problem_size(      
            input_size,
            filter_size,
            padding,
            conv_stride,
            dilation,
            output_size,
            mode,
            split_k_slices);


    cutlass::int4b_t* cutlass_int4b_inp;
    cutlass::int4b_t* cutlass_int4b_weight;
    int64_t input_N, weight_N;
    int64_t num_byte_input, num_byte_weight;
    input_N = input.numel(); weight_N = weight.numel();
    num_byte_input = input_N * cutlass::sizeof_bits<cutlass::int4b_t>::value / 8; 
    num_byte_weight = weight_N * cutlass::sizeof_bits<cutlass::int4b_t>::value / 8;
    
    cudaMalloc(&cutlass_int4b_inp, num_byte_input);
    cudaMalloc(&cutlass_int4b_weight, num_byte_weight);
    auto output_int8_inp = torch::empty({input_N / 2 + 1},torch::dtype(torch::kInt8).device(input.device()));
    auto output_int8_weight = torch::empty({weight_N / 2 + 1},torch::dtype(torch::kInt8).device(input.device()));


    quantize_int4_internal(input, scale_inp, input_N, cutlass_int4b_inp, output_int8_inp);
    quantize_int4_internal(weight, scale_w, weight_N, cutlass_int4b_weight, output_int8_weight);


    // cutlass::int4b_t* int4_h_tensor;
    // int4_h_tensor = (cutlass::int4b_t *)malloc(num_byte_input);
    // cudaMemcpy(int4_h_tensor, cutlass_int4b_inp, num_byte_input, cudaMemcpyDeviceToHost);
    // for (int i = input_N - 1; i >= input_N - 5; i--) 
    //     std::cout <<  int4_h_tensor[i];

    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(cutlass_int4b_inp, LayoutInputA::packed(input_size));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(cutlass_int4b_weight, LayoutInputB::packed(filter_size));
    
    // // remaining are the same
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(output.data_ptr<ElementOutput>(), LayoutOutput::packed(output_size));

    ElementComputeEpilogue alpha = ElementComputeEpilogue(dequant_scale);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0.0);

    // // C is the accumulator tensor, D is the destination tensor. 
    typename Conv2d::Arguments arguments{
        problem_size,
        input_ref,
        weight_ref,
        output_ref,
        output_ref,
        {alpha, beta},
        split_k_mode,
    };

    Conv2d conv2d_op;
    cutlass::Status status;

    status = conv2d_op.initialize(arguments);
    CUTLASS_CHECK(status);

    status = conv2d_op();
    CUTLASS_CHECK(status);

    cudaFree(cutlass_int4b_inp);
    cudaFree(cutlass_int4b_weight);


    return std::make_tuple(output, output_int8_inp, output_int8_weight);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensor_core_sp_conv_optimized_int4(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale,
        float scale_inp, float scale_w){

        // return input; // debug
        using Conv2dFpropKernel = int4_conv::Conv2dFpropKernel_OPT;
        using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
        return cutlass_conv_lower8<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale, scale_inp, scale_w);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensor_core_sp_conv_int4(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale,
        float scale_inp, float scale_w){

        // return input; // debug
        using Conv2dFpropKernel = int4_conv::Conv2dFpropKernel_ANA;
        using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
        return cutlass_conv_lower8<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale, scale_inp, scale_w);
}

torch::Tensor tensor_core_sp_conv_optimized(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){

        // return input; // debug
        using Conv2dFpropKernel = int8_conv::Conv2dFpropKernel_OPT;
        using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
        return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor tensor_core_sp_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){

        // return input; // debug
        using Conv2dFpropKernel = int8_conv::Conv2dFpropKernel_ANA;
        using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
        return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}

// //fp 16 and 32 part
torch::Tensor tensor_core_sp_conv_optimized_fp16(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){
        
        using Conv2dFpropKernel = half_conv::Conv2dFpropKernel_OPT;
        using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

        return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
        
}


torch::Tensor tensor_core_sp_conv_fp16(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){

        using Conv2dFpropKernel = half_conv::Conv2dFpropKernel_ANA;
        using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

        return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}

// tf32
torch::Tensor tensor_core_sp_conv_optimized_tf32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){

    using Conv2dFpropKernel = tf32_conv::Conv2dFpropKernel_OPT;
    using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor tensor_core_sp_conv_tf32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){
    using Conv2dFpropKernel = tf32_conv::Conv2dFpropKernel_ANA;
    using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
    return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


// fp32
torch::Tensor tensor_core_sp_conv_optimized_fp32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){

    using Conv2dFpropKernel = float_conv::Conv2dFpropKernel_OPT;
    using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
    return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor tensor_core_sp_conv_fp32(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){
    using Conv2dFpropKernel = float_conv::Conv2dFpropKernel_ANA;
    using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
    return cutlass_conv<Conv2d>(input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}
#endif