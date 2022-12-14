// name_A100.h
#ifndef name_A100_h
#define name_A100_h
namespace sm80_space{
  using SmArch = cutlass::arch::Sm80;
  constexpr int NumStages = 3;
  namespace int8_gemm
  {
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = float;
    using ElementAccumulator = int32_t;

    // Different for different SM and datatype. Be careful here.
    using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;  // Threadblock tile shape
    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;         // Warp tile shape
    // This code section describes the size of MMA op
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, 
        ElementInputB, LayoutInputB, 
        ElementOutput, LayoutOutput,
        ElementAccumulator, 
        MMAOp, 
        SmArch,
        ThreadblockShape,
        WarpShape, 
        InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
        NumStages>;
  };


  namespace half_gemm
  {
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
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
        SmArch,
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
        SmArch,
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