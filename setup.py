from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='fastfeedforward_cuda',
    ext_modules=[
        CUDAExtension(
            name='fff_cuda',
            sources=['fastfeedforward_cuda/fff_cuda.cpp', 'fastfeedforward_cuda/fff_cuda_kernel.cu'],  # Your C++ and CUDA sources
            extra_compile_args={
                'cxx': ['-O3'],  # C++ compilation flags
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
                    '-gencode', 'arch=compute_90,code=sm_90',  # Hopper
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension  # Use PyTorch's BuildExtension for the build
    },
    package_dir={'': '.'},
    install_requires=['torch>=2.0.0'],
)
