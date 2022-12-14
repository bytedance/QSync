// name_A100.h
#ifndef name_T4_h
#define name_T4_h

namespace sm75_space{
    using SmArch = cutlass::arch::Sm75;
    using SmArch80 = cutlass::arch::Sm75;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
    constexpr int NumStages = 2;
    namespace int4_conv
    {
        using ElementInputA = cutlass::int4b_t;
        using ElementInputB = cutlass::int4b_t;
        using ElementOutput = float;

        using ElementAccumulator = int32_t;
        using ElementComputeEpilogue = float;

        // Different for different SM and datatype. Be careful here.
        using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 128>;  // Threadblock tile shape
        // This code section describes tile size a warp will compute
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;         // Warp tile shape
        // This code section describes the size of MMA op
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;    // TensorCore instruction shape

        // This code section describes the epilogue part of the kernel, we use default value
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            ElementOutput,                                     // Data type of output matrix.
            128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                            // memory access. This becomes the vector width of
                                                            // math instructions in the epilogue too.
            ElementAccumulator,                                // Data type of accumulator
            ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;                           // Data type for alpha/beta in linear combination
            
        
        using Conv2dFpropKernel_OPT = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            NumStages, // number of stages
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            32,
            32>::Kernel;

        using Conv2dFpropKernel_ANA = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::IteratorAlgorithm::kAnalytic,
            cutlass::conv::StrideSupport::kStrided,
            32,
            32>::Kernel;
    };

    namespace int8_conv
    {
        using ElementInputA = int8_t;
        using ElementInputB = int8_t;
        using ElementOutput = float;

        using ElementAccumulator = int32_t;
        using ElementComputeEpilogue = float;

        // Different for different SM and datatype. Be careful here.
        using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;  // Threadblock tile shape
        // This code section describes tile size a warp will compute
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;         // Warp tile shape
        // This code section describes the size of MMA op
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;    // TensorCore instruction shape

        // This code section describes the epilogue part of the kernel, we use default value
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            ElementOutput,                                     // Data type of output matrix.
            128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                            // memory access. This becomes the vector width of
                                                            // math instructions in the epilogue too.
            ElementAccumulator,                                // Data type of accumulator
            ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;                           // Data type for alpha/beta in linear combination
            
        
        using Conv2dFpropKernel_OPT = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            NumStages, // number of stages
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            16,
            16>::Kernel;

        using Conv2dFpropKernel_ANA = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::IteratorAlgorithm::kAnalytic,
            cutlass::conv::StrideSupport::kStrided,
            16,
            16>::Kernel;
    };

    namespace half_conv
    {

        using ElementInputA = cutlass::half_t;            // Data type of elements in input tensor
        using ElementInputB = cutlass::half_t;            // Data type of elements in input tensor
        using ElementOutput = float;             // Data type of elements in output tensor

        using ElementAccumulator = float;
        using ElementComputeEpilogue = float;

        torch::Dtype AccType = torch::kF32;

        using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Threadblock tile shape
        using WarpShape = cutlass::gemm::GemmShape<64, 16, 32>;         // Warp tile shape
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;    // TensorCore instruction shape

        // This code section describes the epilogue part of the kernel, we use default value
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // Data type of output matrix.
        128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
        // 8,
                                                        // memory access. This becomes the vector width of
                                                        // math instructions in the epilogue too.
        ElementAccumulator,                                // Data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;                           // Data type for alpha/beta in linear combination
            
        
        using Conv2dFpropKernel_OPT = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            8,
            8>::Kernel;

        using Conv2dFpropKernel_ANA =  typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kAnalytic,
            cutlass::conv::StrideSupport::kStrided,
            8,
            8>::Kernel;
    };

    // cannot use in T4
    namespace tf32_conv
    {

        using ElementInputA = cutlass::tfloat32_t;            // Data type of elements in input tensor
        using ElementInputB = cutlass::tfloat32_t;            // Data type of elements in input tensor
        using ElementOutput = float;             // Data type of elements in output tensor

        using ElementAccumulator = float;
        using ElementComputeEpilogue = float;

        torch::Dtype AccType = torch::kF32;

        using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;  // Threadblock tile shape
        using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;         // Warp tile shape
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;    // TensorCore instruction shape

        // This code section describes the epilogue part of the kernel, we use default value
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // Data type of output matrix.
        128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
        // 8,
                                                        // memory access. This becomes the vector width of
                                                        // math instructions in the epilogue too.
        ElementAccumulator,                                // Data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;                           // Data type for alpha/beta in linear combination
            
        
        using Conv2dFpropKernel_OPT = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

        using Conv2dFpropKernel_ANA =  typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;
    };


    namespace float_conv
    {

        using ElementInputA = float;            // Data type of elements in input tensor
        using ElementInputB = float;            // Data type of elements in input tensor
        using ElementOutput = float;             // Data type of elements in output tensor

        using ElementAccumulator = float;
        using ElementComputeEpilogue = float;

        torch::Dtype AccType = torch::kF32;

        using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Threadblock tile shape
        using WarpShape = cutlass::gemm::GemmShape<128, 128, 32>;         // Warp tile shape
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;    // TensorCore instruction shape

        // This code section describes the epilogue part of the kernel, we use default value
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // Data type of output matrix.
        1,  // <- the number of elements per vectorized
                                                            // memory access. This becomes the vector width of
                                                            // math instructions in the epilogue too.
        ElementAccumulator,                                // Data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;                           // Data type for alpha/beta in linear combination
            
        
        using Conv2dFpropKernel_OPT = typename cutlass::conv::kernel::DefaultConv2dFprop<
            float, 
            cutlass::layout::TensorNHWC,
            float, 
            cutlass::layout::TensorNHWC,
            float, 
            cutlass::layout::TensorNHWC,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm50,
            cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<32, 64, 8 >,
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<
            float,
            1,
            float,
            float
            >,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            1,
            1
        >::Kernel;

        using Conv2dFpropKernel_ANA = typename cutlass::conv::kernel::DefaultConv2dFprop<
            float, 
            cutlass::layout::TensorNHWC,
            float, 
            cutlass::layout::TensorNHWC,
            float, 
            cutlass::layout::TensorNHWC,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm50,
            cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<32, 64, 8 >,
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<
            float,
            1,
            float,
            float
            >,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kAnalytic,
            cutlass::conv::StrideSupport::kStrided,
            1,
            1
        >::Kernel;
    };

    namespace fp_bp_conv
    {
        using ElementA           = float;
        using ElementB           = float;
        using ElementC           = float;
        using ElementAccumulator = float;
        using ElementCompute     = float;

        using epilogue_op = cutlass::epilogue::thread::LinearCombination<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator,
            ElementCompute
        >;

        using Conv_Dgrad_Kernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
                float, 
                cutlass::layout::TensorNHWC,
                float, 
                cutlass::layout::TensorNHWC,
                float, 
                cutlass::layout::TensorNHWC,
                float,
                cutlass::arch::OpClassSimt,
                cutlass::arch::Sm50,
                cutlass::gemm::GemmShape<128, 128, 8>,
                cutlass::gemm::GemmShape<32, 64, 8 >,
                cutlass::gemm::GemmShape<1, 1, 1>,
                cutlass::epilogue::thread::LinearCombination<
                float,
                1,
                float,
                float
                >,
                cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
                2,
                cutlass::arch::OpMultiplyAdd,
                cutlass::conv::IteratorAlgorithm::kOptimized,
                cutlass::conv::StrideSupport::kStrided,
                1,
                1
            >::Kernel;

        // align = 8
        using Conv_Wgrad_Kernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
            float, 
            cutlass::layout::TensorNHWC,
            float, 
            cutlass::layout::TensorNHWC,
            float, 
            cutlass::layout::TensorNHWC,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm50,
            cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<32, 64, 8 >,
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<
            float,
            1,
            float,
            float
            >,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            1,
            1
        >::Kernel;
            
        // align = 1
        using Conv_Wgrad_Kernel_C3 = typename cutlass::conv::kernel::DefaultConv2dWgrad<
                float, 
                cutlass::layout::TensorNHWC,
                float, 
                cutlass::layout::TensorNHWC,
                float, 
                cutlass::layout::TensorNHWC,
                float,
                cutlass::arch::OpClassSimt,
                cutlass::arch::Sm50,
                cutlass::gemm::GemmShape<128, 128, 8>,
                cutlass::gemm::GemmShape<32, 64, 8 >,
                cutlass::gemm::GemmShape<1, 1, 1>,
                cutlass::epilogue::thread::LinearCombination<
                float,
                1,
                float,
                float
                >,
                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
                2,
                cutlass::arch::OpMultiplyAdd,
                cutlass::conv::IteratorAlgorithm::kOptimized,
                cutlass::conv::StrideSupport::kStrided,
                1,
                1
            >::Kernel;
    };


    namespace fp_bp_conv_fp16
    {
        using ElementA           = cutlass::half_t;
        using ElementB           = cutlass::half_t;
        using ElementC           = float;
        using ElementAccumulator = float;
        using ElementCompute     = float;


        using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;  // Threadblock tile shape
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 32 >;         // Warp tile shape
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;    // TensorCore instruction shape

        using epilogue_op = cutlass::epilogue::thread::LinearCombination<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator,
            ElementCompute
        >;

        using Conv_Dgrad_Kernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
            ElementA, LayoutInputA,
            ElementB, LayoutInputB,
            ElementC, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            epilogue_op,
            cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            8,
            8
        >::Kernel;

        // align = 8
        using Conv_Wgrad_Kernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
            ElementA, LayoutInputA,
            ElementB, LayoutInputB,
            ElementC, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            epilogue_op,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            8,
            8
        >::Kernel;
            
        // align = 1
        using Conv_Wgrad_Kernel_C3 = typename cutlass::conv::kernel::DefaultConv2dWgrad<
            ElementA, LayoutInputA,
            ElementB, LayoutInputB,
            ElementC, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            epilogue_op,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            1,
            1
        >::Kernel;
    };

    namespace fp_bp_conv_fp16_to_fp16
    {
    
        using ElementA           = cutlass::half_t;
        using ElementB           = cutlass::half_t;
        using ElementC           = cutlass::half_t;
        using ElementAccumulator = float;
        using ElementCompute     = float;


        using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;  // Threadblock tile shape
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 32 >;         // Warp tile shape
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;    // TensorCore instruction shape

        using epilogue_op = cutlass::epilogue::thread::LinearCombination<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator,
            ElementCompute
        >;

        using Conv_Dgrad_Kernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
            cutlass::half_t, 
            cutlass::layout::TensorNHWC,
            cutlass::half_t, 
            cutlass::layout::TensorNHWC,
            cutlass::half_t, 
            cutlass::layout::TensorNHWC,
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<256, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32 >,
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t,
            8,
            float,
            float
            >,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kUnity,
            8,
            8
        >::Kernel;

        // align = 8
        using Conv_Wgrad_Kernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
            ElementA, LayoutInputA,
            ElementB, LayoutInputB,
            ElementC, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            epilogue_op,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            8,
            8
        >::Kernel;
            
        // align = 1
        using Conv_Wgrad_Kernel_C3 = typename cutlass::conv::kernel::DefaultConv2dWgrad<
            ElementA, LayoutInputA,
            ElementB, LayoutInputB,
            ElementC, LayoutOutput,
            ElementAccumulator,
            MMAOp_TensorOP,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            epilogue_op,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            1,
            1
        >::Kernel;
    };
};


#endif