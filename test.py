import torch
import ctypes
import matplotlib.pyplot as plt
import triton
from triton._C.libtriton import amd
import csv
from dataclasses import dataclass
import inspect

torch.cuda.set_device(3)


def get_cublas_tflops(dtype):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[dtype]
    cublas_workspace = torch.empty(32 * 4024 * 2024, device="cuda", dtype=torch.uint8)
    cublas = amd.hipblas.HipBlasLt(cublas_workspace)
    device = "cuda"
    M, N, K = 8192, 8192, 8192
    a = torch.randn(M, K, device=device, dtype=torch.float32).to(dtype)
    b = torch.randn(K, N, device=device, dtype=torch.float32).to(dtype).T
    c = torch.empty((M, N), device=device, dtype=dtype)
    time_ms = triton.testing.do_bench(lambda: cublas.matmul(a, b, c), rep=1000)
    return 2 * M * N * K / time_ms * 1e-9


get_cublas_tflops("bf16")
