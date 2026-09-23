from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
cxx_compiler_flags = []

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ['-gencode=arch=compute_86,code=sm_86',
                                         '-gencode=arch=compute_80,code=sm_80',
                                         '-gencode=arch=compute_90,code=sm_90'], 
                                "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)