import torch
import triton
import triton.language as tl
import triton.profiler as proton
import json
import tempfile
import os


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        x = torch.rand(256, device='cuda')
        y = torch.rand(256, device='cuda')
        output = torch.empty_like(x)
        temp_file_driver = os.path.join(tmp, 'test_driver.hatchet')

        print(f"Starting profiling session...")
        session_id = proton.start(temp_file_driver.replace('.hatchet', ''))
        print(f"Session ID: {session_id}")

        print("Running kernel...")
        add_kernel[(1, )](x, y, output, 256, BLOCK_SIZE=1024)
        torch.cuda.synchronize()
        print("Kernel finished")

        print("Finalizing...")
        proton.finalize()

        print(f'Output file exists: {os.path.exists(temp_file_driver)}')
        with open(temp_file_driver, 'r') as f:
            data = json.load(f)
            print('Data structure:')
            print(json.dumps(data, indent=2)[:2000])
