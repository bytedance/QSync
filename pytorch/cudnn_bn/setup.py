from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cudnn_bn',
    ext_modules=[
        CUDAExtension('cudnn_bn',
                      ['cudnn_bn.cpp',
                       'cudnn_bn_kernel.cu',],
                      library_dirs=['/usr/lib/x86_64-linux-gnu'],
                      libraries=['cudnn'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages()
)