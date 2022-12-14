from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
setup(
    name='cutlassconv_cuda',
    ext_modules=[
        CUDAExtension(
            'cutlassconv_cuda',
            ['cutlassconv.cpp', 'cutlassconv_kernel.cu',],
            include_dirs=[
                os.path.join(project_root, 'pytorch/cutlass/include'),
                os.path.join(project_root, 'pytorch/cutlass/util/include')
                # '/qsync_niti_based/pytorch/cutlass/include', '/qsync_niti_based/pytorch/cutlass/util/include'
                ],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],
            libraries=['cudnn'])],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages()
)
