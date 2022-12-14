from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='cpp_backward',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'cpp_backward.cudnn',
              ['backward_func.cc']
          ),
            cpp_extension.CUDAExtension(
              'cpp_backward.quantization',
              ['quantization.cc', 'quantization_cuda_kernel.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)