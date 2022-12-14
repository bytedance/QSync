from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))

setup(
    name='cutlasslinear_cuda',
    ext_modules=[
        CUDAExtension(
            'cutlasslinear_cuda',
            ['cutlasslinear.cpp',
            'cutlasslinear_kernel.cu',],
            include_dirs=[
                os.path.join(project_root, 'pytorch/cutlass/include'),
                os.path.join(project_root, 'pytorch/cutlass/util/include')]
    )],
    cmdclass={
        'build_ext': BuildExtension}
)
