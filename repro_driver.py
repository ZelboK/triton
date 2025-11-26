import pathlib

import torch

import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    with pl.scope("load_x"):
        x = tl.load(x_ptr + offsets, mask=mask)
    with pl.scope("load_y"):
        y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def run(deactivate_first: bool):
    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    temp_file = pathlib.Path("/tmp/repro_driver.hatchet")
    if temp_file.exists():
        temp_file.unlink()
    session_id = proton.start(str(temp_file.with_suffix("")))
    if deactivate_first:
        proton.deactivate(session_id)
        proton.activate()
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
    proton.finalize()
    print("session:", session_id)
    print("file exists:", temp_file.exists())
    if temp_file.exists():
        print(temp_file.read_text())


if __name__ == "__main__":
    import sys

    deactivate = "--deactivate" in sys.argv
    run(deactivate)
