# PyTorch (Underneath Kernel Implementations of LpTorch)
Here is the underneath kernel implementation for LpTorch. (structure below CUDAPI)
- cudnn-bn: cudnn implemented batchnorm2d
- cudnn-conv: cudnn implemented conv2d
- cutlass: the cutlass lib
- cutlass-conv: cutlass implemented conv2d
    - nameA100.h: support S80 and above (s86)
    - nameT4.h: support s75 and above (s75)
    - nameV100: support s70 
- cutlass-linear: same to above
- other_extension: Original PyTorch CPP interfaces
- quantization: optimized quantization for INT8
- v_temp: version controls, abandoned for the moment


### Available bits
- for the moment, we support INT8, FP16, TF32 and FP32. 
- We are going to support INT4 in the future.