from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

file_path = os.path.dirname(os.getcwd())
cutlass_path = "cutlass/cutlass/include"
file_path = os.path.join(file_path, cutlass_path)

setup(
    name='HLQ_backward',
    ext_modules=[
        CUDAExtension(name='HLQ_backward', sources=[
            'HLQ_backward.cpp',
            'HLQ_backward_kernel.cu',
        ], include_dirs=[file_path],
        extra_compile_args=["-std=c++17"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })