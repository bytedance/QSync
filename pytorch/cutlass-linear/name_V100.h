// name_A100.h
#ifndef name_V100_h
#define name_V100_h
namespace sm70_space{
  using SmArch70 = cutlass::arch::Sm70;
  constexpr int NumStages = 2;
  // V100 dot' have int8, tf32 kernel
  namespace int8_gemm
  {
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = float;
    using ElementAccumulator = int32_t;

    // Different for different SM and datatype. Be careful here.
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;  // Threadblock tile shape
    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;         // Warp tile shape
    // This code section describes the size of MMA op
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;    // TensorCore instruction shape

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, 
        ElementInputB, LayoutInputB, 
        ElementOutput, LayoutOutput,
        ElementAccumulator, 
        MMAOp, 
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape, 
        InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;
  };


  namespace half_gemm
  {
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;  // Threadblock tile shape
    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;         // Warp tile shape
    // This code section describes the size of MMA op
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;    // TensorCore instruction shape
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, 
        ElementInputB, LayoutInputB, 
        ElementOutput, LayoutOutput,
        ElementAccumulator, 
        MMAOp, 
        SmArch70,
        ThreadblockShape,
        WarpShape, 
        InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;
  };


  namespace float_gemm
  {
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;

    using ElementAccumulator = float;

    using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;  // Threadblock tile shape
    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;         // Warp tile shape
    // This code section describes the size of MMA op
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;    // TensorCore instruction shape
    

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, 
        ElementInputB, LayoutInputB, 
        ElementOutput, LayoutOutput,
        ElementAccumulator>;
  };

  namespace tf32_gemm
  {
    using ElementInputA = cutlass::tfloat32_t; 
    using ElementInputB = cutlass::tfloat32_t; 
    using ElementOutput = float;

    using ElementAccumulator = float;
    
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 16>;  // Threadblock tile shape
    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;         // Warp tile shape
    // This code section describes the size of MMA op
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;    // TensorCore instruction shape

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, 
        ElementInputB, LayoutInputB, 
        ElementOutput, LayoutOutput,
        ElementAccumulator, 
        MMAOp, 
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape, 
        InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;
  };
};
#endif