// name_A100.h
#ifndef conv_bp_h
#define conv_bp_h

template<typename Conv2d>
torch::Tensor cutlass_conv_wgrad(torch::Tensor& grad_output, torch::Tensor& input, torch::Tensor& weight, 
                            int32_t stride_h, int32_t stride_w, int32_t padding_h, int32_t padding_w, 
                            float dequant_scale, torch::Dtype OType)
{

    using ElementInputA = typename Conv2d::ElementA;
    using ElementInputB = typename Conv2d::ElementB;
    using ElementOutput = typename Conv2d::ElementC;
    using ElementComputeEpilogue = typename Conv2d::ElementCompute;
    using ElementAccumulator = typename Conv2d::ElementAccumulator;

    cutlass::Tensor4DCoord grad_output_size;
    cutlass::Tensor4DCoord input_size;
    
    cutlass::Tensor4DCoord padding = {padding_h, padding_h, padding_w, padding_w};
    cutlass::MatrixCoord conv_stride = {stride_h, stride_w};
    cutlass::MatrixCoord dilation = {1, 1};
    cutlass::Tensor4DCoord filter_size;

    if(input.is_contiguous()){
        assign_4dcrd_with_tensor(grad_output_size, grad_output, true);
        assign_4dcrd_with_tensor(input_size, input, true);
        assign_4dcrd_with_tensor(filter_size, weight, true);
    }else
    {
        assign_4dcrd_with_tensor(grad_output_size, grad_output, false);
        assign_4dcrd_with_tensor(input_size, input, false);
        assign_4dcrd_with_tensor(filter_size, weight, false);
    };

    auto output = create_4d_tensor(filter_size.n(), filter_size.h(), filter_size.w(), filter_size.c(), OType, input);
    // problem
    cutlass::conv::Conv2dProblemSize problem_size(      
            input_size,
            filter_size,
            padding,
            conv_stride,
            dilation,
            grad_output_size,
            mode,
            split_k_slices);
    
    cutlass::TensorRef<ElementInputA, LayoutInputA> grad_output_ref(reinterpret_cast<ElementInputA *>(grad_output.data_ptr()),LayoutInputA::packed(grad_output_size));
    cutlass::TensorRef<ElementInputB, LayoutInputB> input_ref(reinterpret_cast<ElementInputB *>(input.data_ptr()), LayoutInputB::packed(input_size));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(reinterpret_cast<ElementOutput *>(output.data_ptr()), LayoutOutput::packed(filter_size));

    ElementComputeEpilogue alpha = ElementComputeEpilogue(dequant_scale);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0.0);

    // C is the accumulator tensor, D is the destination tensor. 
    typename Conv2d::Arguments arguments{
        problem_size,
        grad_output_ref,
        input_ref,
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



template<typename Conv2d>
torch::Tensor cutlass_conv_dgrad(torch::Tensor& grad_output, torch::Tensor& weight, torch::Tensor& input, 
                            int32_t stride_h, int32_t stride_w, int32_t padding_h, int32_t padding_w, 
                            float dequant_scale, torch::Dtype OType)
{

    using ElementInputA = typename Conv2d::ElementA;
    using ElementInputB = typename Conv2d::ElementB;
    using ElementOutput = typename Conv2d::ElementC;
    using ElementComputeEpilogue = typename Conv2d::ElementCompute;
    using ElementAccumulator = typename Conv2d::ElementAccumulator;

    cutlass::Tensor4DCoord grad_output_size;
    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;

    if(input.is_contiguous()){
        assign_4dcrd_with_tensor(grad_output_size, grad_output, true);
        assign_4dcrd_with_tensor(input_size, input, true);
        assign_4dcrd_with_tensor(filter_size, weight, true);
    }else
    {
        assign_4dcrd_with_tensor(grad_output_size, grad_output, false);
        assign_4dcrd_with_tensor(input_size, input, false);
        assign_4dcrd_with_tensor(filter_size, weight, false);
    };
    

    cutlass::Tensor4DCoord padding = {padding_h, padding_h, padding_w, padding_w};
    cutlass::MatrixCoord conv_stride = {stride_h, stride_w};
    cutlass::MatrixCoord dilation = {1, 1};
    
    auto output = create_4d_tensor(input_size.n(), input_size.h(), input_size.w(), input_size.c(), OType, input);
    // problem
    cutlass::conv::Conv2dProblemSize problem_size(      
            input_size,
            filter_size,
            padding,
            conv_stride,
            dilation,
            grad_output_size,
            mode,
            split_k_slices);
    
    cutlass::TensorRef<ElementInputA, LayoutInputA> grad_output_ref(reinterpret_cast<ElementInputA *>(grad_output.data_ptr()),LayoutInputA::packed(grad_output_size));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(reinterpret_cast<ElementInputB *>(weight.data_ptr()), LayoutInputB::packed(filter_size));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(reinterpret_cast<ElementOutput *>(output.data_ptr()), LayoutOutput::packed(input_size));

    ElementComputeEpilogue alpha = ElementComputeEpilogue(dequant_scale);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0.0);

    // C is the accumulator tensor, D is the destination tensor. 
    typename Conv2d::Arguments arguments{
        problem_size,
        grad_output_ref,
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


// abstract dgrad and abstract wgrad
template<typename DKernel>
torch::Tensor tensor_core_convbp_dgrad(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using DConv2d = cutlass::conv::device::ImplicitGemmConvolution<DKernel>;
    torch::Dtype OType = torch::kF32;
    torch::Tensor act_gradient = cutlass_conv_dgrad<DConv2d>(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale, OType);
    return act_gradient;
}

template<typename WKernel>
torch::Tensor tensor_core_convbp_wgrad(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WConv2d = cutlass::conv::device::ImplicitGemmConvolution<WKernel>;
    torch::Dtype OType = torch::kF32;
    torch::Tensor w_gradient = cutlass_conv_wgrad<WConv2d>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale, OType);
    return w_gradient;
}





// implementation of dgrad and wgrad
torch::Tensor tensor_core_convbp_fp32_dgrad(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using DKernel = fp_bp_conv::Conv_Dgrad_Kernel;
    return tensor_core_convbp_dgrad<DKernel>(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}

// specialized wgrad
torch::Tensor tensor_core_convbp_fp32_wgrad(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WKernel = fp_bp_conv::Conv_Wgrad_Kernel;
    return tensor_core_convbp_wgrad<WKernel>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}





torch::Tensor tensor_core_convbp_fp32_wgrad_c3(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WKernel = fp_bp_conv::Conv_Wgrad_Kernel_C3;
    return tensor_core_convbp_wgrad<WKernel>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}



// tensor core
std::tuple<torch::Tensor, torch::Tensor> tensor_core_convbp_fp32(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){
    
    torch::Tensor act_gradient;
    torch::Tensor w_gradient;
    act_gradient = tensor_core_convbp_fp32_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
    w_gradient = tensor_core_convbp_fp32_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
    return std::make_pair(act_gradient, w_gradient);
}

torch::Tensor tensor_core_convbp_fp16_dgrad(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using DKernel = fp_bp_conv_fp16::Conv_Dgrad_Kernel;
    return tensor_core_convbp_dgrad<DKernel>(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor tensor_core_convbp_fp16_wgrad(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WKernel = fp_bp_conv_fp16::Conv_Wgrad_Kernel;
    return tensor_core_convbp_wgrad<WKernel>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor tensor_core_convbp_fp16_wgrad_c3(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WKernel = fp_bp_conv_fp16::Conv_Wgrad_Kernel_C3;
    return tensor_core_convbp_wgrad<WKernel>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}



std::tuple<torch::Tensor, torch::Tensor> tensor_core_convbp_fp16(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){
    
    torch::Tensor act_gradient;
    torch::Tensor w_gradient;
    act_gradient = tensor_core_convbp_fp16_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
    w_gradient = tensor_core_convbp_fp16_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
    return std::make_pair(act_gradient, w_gradient);
}

// fp16 to fp16
// output is fp16
template<typename DKernel>
torch::Tensor tensor_core_convbp_dgrad_fp16(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using DConv2d = cutlass::conv::device::ImplicitGemmConvolution<DKernel>;
    torch::Dtype OType = torch::kF16;
    torch::Tensor act_gradient = cutlass_conv_dgrad<DConv2d>(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale, OType);
    return act_gradient;
}

template<typename WKernel>
torch::Tensor tensor_core_convbp_wgrad_fp16(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WConv2d = cutlass::conv::device::ImplicitGemmConvolution<WKernel>;
    torch::Dtype OType = torch::kF16;
    torch::Tensor w_gradient = cutlass_conv_wgrad<WConv2d>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale, OType);
    return w_gradient;
}

torch::Tensor tensor_core_convbp_fp16_to_fp16_dgrad(
    torch::Tensor& grad_output,
    torch::Tensor& weight,
    torch::Tensor& input,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using DKernel = fp_bp_conv_fp16_to_fp16::Conv_Dgrad_Kernel;
    return tensor_core_convbp_dgrad_fp16<DKernel>(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


torch::Tensor tensor_core_convbp_fp16_to_fp16_wgrad(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WKernel = fp_bp_conv_fp16_to_fp16::Conv_Wgrad_Kernel;
    return tensor_core_convbp_wgrad_fp16<WKernel>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}



torch::Tensor tensor_core_convbp_fp16_to_fp16_wgrad_c3(
    torch::Tensor& grad_output,
    torch::Tensor& input,
    torch::Tensor& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    float dequant_scale
)
{
    using WKernel = fp_bp_conv_fp16_to_fp16::Conv_Wgrad_Kernel_C3;
    return tensor_core_convbp_wgrad_fp16<WKernel>(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
}


std::tuple<torch::Tensor, torch::Tensor> tensor_core_convbp_fp16_to_fp16(
        torch::Tensor& input, 
        torch::Tensor& grad_output,
        torch::Tensor& weight,
        int32_t stride_h, int32_t stride_w,
        int32_t padding_h, int32_t padding_w,
        float dequant_scale){
    
    torch::Tensor act_gradient;
    torch::Tensor w_gradient;
    act_gradient = tensor_core_convbp_fp16_to_fp16_dgrad(grad_output, weight, input, stride_h, stride_w, padding_h, padding_w, dequant_scale);
    w_gradient = tensor_core_convbp_fp16_to_fp16_wgrad(grad_output, input, weight, stride_h, stride_w, padding_h, padding_w, dequant_scale);
    return std::make_pair(act_gradient, w_gradient);
}


#endif