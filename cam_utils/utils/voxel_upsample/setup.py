from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

def make_cuda_ext(name, sources):
    cuda_ext = CUDAExtension(
        name=name,
        sources=[src for src in sources]
    )
    return cuda_ext

setup(
    name='voxel_upsample', 
    cmdclass={
            'build_ext': BuildExtension,
    },
    ext_modules=[
        make_cuda_ext(
            name='voxel_upsample_cuda',
            sources=[
                'src/voxel_upsample_api.cpp',
                'src/voxel_upsample.cpp',
                'src/voxel_upsample_gpu.cu',
            ]
        )
    ]
)