"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
import shutil

import setuptools

root = Path(__file__).parent.resolve()
gen_dir = root / "csrc" / "generated"

head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "128,256").split(",")
head_dims = list(map(int, head_dims))
SM90_ALLOWED_HEAD_DIMS = {(64, 64), (128, 128), (256, 256), (192, 128)}
head_dims_sm90 = [(d, d) for d in head_dims if (d, d) in SM90_ALLOWED_HEAD_DIMS]
head_dims_sm90.extend(
    [(k, v) for k, v in SM90_ALLOWED_HEAD_DIMS if k != v]
)  # Always enable (192,128)

mask_modes = [0, 1, 2]

enable_aot = os.environ.get("FLASHINFER_ENABLE_AOT", "0") == "1"
enable_f16 = os.environ.get("FLASHINFER_ENABLE_F16", "1") == "1"
enable_bf16 = os.environ.get("FLASHINFER_ENABLE_BF16", "1") == "1"
enable_fp8 = os.environ.get("FLASHINFER_ENABLE_FP8", "1") == "1"
enable_fp8_e4m3 = (
    os.environ.get("FLASHINFER_ENABLE_FP8_E4M3", "1" if enable_fp8 else "0") == "1"
)
enable_fp8_e5m2 = (
    os.environ.get("FLASHINFER_ENABLE_FP8_E5M2", "1" if enable_fp8 else "0") == "1"
)
enable_sm90 = os.environ.get("FLASHINFER_ENABLE_SM90", "1") == "1"


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_version():
    package_version = (root / "version.txt").read_text().strip()
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version is None:
        return package_version
    return f"{package_version}+{local_version}"


def generate_build_meta(aot_build_meta: dict) -> None:
    build_meta_str = f"__version__ = {get_version()!r}\n"
    if len(aot_build_meta) != 0:
        build_meta_str += f"build_meta = {aot_build_meta!r}\n"
    write_if_different(root / "flashinfer" / "_build_meta.py", build_meta_str)


def generate_cuda() -> None:
    print("Calling generate_cuda()")
    try:  # no aot_build_utils in sdist
        sys.path.append(str(root))
        from aot_build_utils import generate_dispatch_inc
        from aot_build_utils.generate import get_instantiation_cu
        from aot_build_utils.generate_aot_default_additional_params_header import (
            get_aot_default_additional_params_header_str,
        )
        from aot_build_utils.generate_sm90 import get_sm90_instantiation_cu
    except ImportError:
        print("Import failed in the function generate_cuda(): ", e)
        return

    # dispatch.inc
    write_if_different(
        gen_dir / "dispatch.inc",
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                head_dims=head_dims,
                head_dims_sm90=head_dims_sm90,
                pos_encoding_modes=[0],
                use_fp16_qk_reductions=[0],
                mask_modes=mask_modes,
            )
        ),
    )

    # _kernels
    aot_kernel_uris = get_instantiation_cu(
        argparse.Namespace(
            path=gen_dir,
            head_dims=head_dims,
            pos_encoding_modes=[0],
            use_fp16_qk_reductions=[0],
            mask_modes=mask_modes,
            enable_f16=enable_f16,
            enable_bf16=enable_bf16,
            enable_fp8_e4m3=enable_fp8_e4m3,
            enable_fp8_e5m2=enable_fp8_e5m2,
        )
    )

    # _kernels_sm90
    if enable_sm90:
        aot_kernel_uris += get_sm90_instantiation_cu(
            argparse.Namespace(
                path=gen_dir,
                head_dims=head_dims_sm90,
                pos_encoding_modes=[0],
                use_fp16_qk_reductions=[0],
                mask_modes=mask_modes,
                enable_f16=enable_f16,
                enable_bf16=enable_bf16,
            )
        )
    aot_config_str = f"""prebuilt_ops_uri = set({aot_kernel_uris})"""
    write_if_different(root / "flashinfer" / "jit" / "aot_config.py", aot_config_str)
    write_if_different(
        root / "csrc" / "aot_default_additional_params.h",
        get_aot_default_additional_params_header_str(),
    )


ext_modules = []
cmdclass = {}
install_requires = ["numpy", "torch", "ninja"]
generate_build_meta({})

if enable_aot:
    import torch
    import torch.utils.cpp_extension as torch_cpp_ext
    from packaging.version import Version
    from flashinfer.utils import check_hip_availability, check_cuda_availability

    generate_cuda()

    def get_cuda_version() -> Version:
        if torch_cpp_ext.CUDA_HOME is None:
            nvcc = "nvcc"
        else:
            nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        return Version(re.findall(r"release (\d+\.\d+),", txt)[0])

    def get_hip_version() -> Version:
        if torch_cpp_ext.ROCM_VERSION is None:
            hipcc_loc = shutil.which('hipcc')
            if hipcc_loc is None:
                # FIXME
                hipcc_loc = torch_cpp_ext.ROCM_HOME + "/bin/hipcc"
                # hipcc_loc = torch_cpp_ext.ROCM_HOME + "/hip/bin/hipcc"
            # HIP version: 5.2.0
            # Clang version: 13.0.0 (ROCm Clang 13.0.0)
            # Build configuration: Release
            txt = subprocess.check_output([hipcc_loc, "--version"], text=True)
            return Version(re.findall(r"HIP version: (\d+\.\d+),", txt)[0])
        else:
            return Version(str(torch_cpp_ext.ROCM_VERSION[0]) + str(torch_cpp_ext.ROCM_VERSION[1]))

    class NinjaBuildExtension(torch_cpp_ext.BuildExtension):
        def __init__(self, *args, **kwargs) -> None:
            # do not override env MAX_JOBS if already exists
            if not os.environ.get("MAX_JOBS"):
                max_num_jobs_cores = max(1, os.cpu_count())
                os.environ["MAX_JOBS"] = str(max_num_jobs_cores)

            super().__init__(*args, **kwargs)

    if check_hip_availability():
        # TODO
        # allowed_archs = ["native", "gfx90a", "gfx940", "gfx941", "gfx942"]
        for rocm_arch_flags in torch_cpp_ext._get_rocm_arch_flags():
            # arch = str(re.search(r"\-\-offload\-arch=(\w+)", rocm_arch_flags).group(1))
            # if arch not in allowed_archs:
                # raise RuntimeError("AMD ROCm archs mismatch")
            pass
        cuda_version = get_hip_version()
    else:
        # cuda arch check for fp8 at the moment.
        for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
            arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
            if arch < 75:
                raise RuntimeError("FlashInfer requires sm75+")
        cuda_version = get_cuda_version()
    torch_full_version = Version(torch.__version__)
    torch_version = f"{torch_full_version.major}.{torch_full_version.minor}"
    cmdclass["build_ext"] = NinjaBuildExtension
    # install_requires = [f"torch == {torch_version}+rocm{str(torch_cpp_ext.ROCM_VERSION[0])}.{str(torch_cpp_ext.ROCM_VERSION[1])}"] if check_hip_availability() else [f"torch == {torch_version}"]
    # Workaround for missed PyTorch for ROCm in FlashInfer's Wheel hub https://flashinfer.ai/whl/,
    # where there are only PyTorch versions for CUDA platform.
    install_requires = [] if check_hip_availability() else [f"torch == {torch_version}"]

    aot_build_meta = {}
    aot_build_meta["cuda_major"] = cuda_version.major
    aot_build_meta["cuda_minor"] = cuda_version.minor
    aot_build_meta["torch"] = torch_version
    aot_build_meta["python"] = platform.python_version()
    if check_hip_availability():
        aot_build_meta["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH")
    else:
        aot_build_meta["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
    generate_build_meta(aot_build_meta)

    if enable_f16:
        torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_F16")
    if enable_bf16:
        torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
        if check_cuda_availability():
            torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
        elif check_hip_availability():
            # FIXME
            torch_cpp_ext.COMMON_HIP_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
            torch_cpp_ext.COMMON_HIPCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
    if enable_fp8_e4m3:
        if check_cuda_availability():
            torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8_E4M3")
        elif check_hip_availability():
            # FIXME
            torch_cpp_ext.COMMON_HIP_FLAGS.append("-DFLASHINFER_ENABLE_FP8_E4M3")
            torch_cpp_ext.COMMON_HIPCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8_E4M3")
    if enable_fp8_e5m2:
        if check_cuda_availability():
            torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8_E5M2")
        elif check_hip_availability():
            # FIXME
            torch_cpp_ext.COMMON_HIP_FLAGS.append("-DFLASHINFER_ENABLE_FP8_E5M2")
            torch_cpp_ext.COMMON_HIPCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8_E5M2")

    for flag in [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

    cutlass = root / "3rdparty" / "cutlass"
    include_dirs = [
        root.resolve() / "include",
        cutlass.resolve() / "include",  # for group gemm
        cutlass.resolve() / "tools" / "util" / "include",
    ]
    cxx_flags = [
        "-O3",
        "-DPy_LIMITED_API=0x03080000",
    ]
    if check_hip_availability():
        # FIXME
        cxx_flags += ["-I/opt/rocm/include", "-D__HIP_PLATFORM_AMD__"]
    else:
        cxx_flags += ["-Wno-switch-bool"]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--threads=1",
        "-Xfatbin",
        "-compress-all",
        "-use_fast_math",
        "-DPy_LIMITED_API=0x03080000",
    ]
    libraries = [
        "cublas",
        "cublasLt",
    ] if not check_hip_availability() else ["hiprtc", "amdhip64", "c10", "torch", "torch_python", "hipblas", "hipblaslt"]
    sm90a_flags = "-gencode arch=compute_90a,code=sm_90a".split()
    # FIXME: ROCm/HIP compiler flags
    hipcc_flags = [
        "-O3",
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "--amdgpu-target=gfx942",
        "--offload-arch=gfx942",
        "-ffast-math",
        "-I/opt/rocm/include",
        "-L/opt/rocm/lib",
        "-lamdhip64",
        "-D__HIP_PLATFORM_AMD__=1",
    ]
    kernel_sources = [
        "csrc/bmm_fp8.cu",
        "csrc/cascade.cu",
        "csrc/group_gemm.cu",
        "csrc/norm.cu",
        "csrc/page.cu",
        "csrc/quantization.cu",
        "csrc/rope.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
        "csrc/batch_decode.cu",
        "csrc/batch_prefill.cu",
        "csrc/single_decode.cu",
        "csrc/single_prefill.cu",
        "csrc/flashinfer_ops.cu",
    ]
    kernel_sm90_sources = [
        "csrc/group_gemm_sm90.cu",
        "csrc/single_prefill_sm90.cu",
        "csrc/batch_prefill_sm90.cu",
        "csrc/flashinfer_ops_sm90.cu",
    ]
    # decode_sources = list(gen_dir.glob("*decode_head*.cu"))
    decode_sources = list(gen_dir.glob("batch_*decode_head*.cu"))
    prefill_sources = [
        f for f in gen_dir.glob("*prefill_head*.cu") if "_sm90" not in f.name
    ]
    prefill_sm90_sources = list(gen_dir.glob("*prefill_head*_sm90.cu"))
    ext_modules = [
        torch_cpp_ext.CUDAExtension(
            name="flashinfer.flashinfer_kernels",
            # sources=kernel_sources + decode_sources + prefill_sources,
            sources=["csrc/batch_decode.cu"] + decode_sources,
            include_dirs=include_dirs,
            libraries=libraries,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": hipcc_flags if check_hip_availability() else nvcc_flags,
            },
            py_limited_api=True,
        )
    ]
    if enable_sm90:
        ext_modules += [
            torch_cpp_ext.CUDAExtension(
                name="flashinfer.flashinfer_kernels_sm90",
                sources=kernel_sm90_sources + prefill_sm90_sources,
                include_dirs=include_dirs,
                libraries=libraries,
                extra_compile_args={
                    "cxx": cxx_flags,
                    "nvcc": nvcc_flags + sm90a_flags,
                },
                py_limited_api=True,
            ),
        ]

setuptools.setup(
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    options={"bdist_wheel": {"py_limited_api": "cp38"}},
)
