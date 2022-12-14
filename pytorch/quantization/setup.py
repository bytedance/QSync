from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='cuda_quantization',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'cuda_quantization.quant',
              ['minimax.cc', 'minimax_cuda_kernel.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)