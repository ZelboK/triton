import pytest
import subprocess
import json
import os
import pathlib
import sys


def test_help():
    # Only check if the viewer can be invoked
    subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)


def test_rocprofiler_multi_client_shutdown(tmp_path: pathlib.Path):
    script = """
import pathlib
import sys

import torch

if torch.version.hip is None:
    raise SystemExit(77)

import triton.profiler as proton

session = proton.start(str(pathlib.Path(sys.argv[1]).with_suffix("")))
proton.finalize(session)
"""
    env = os.environ.copy()
    env.pop("ROCPROFILER_REGISTER_FORCE_LOAD", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "multi_client.hatchet")],
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    if result.returncode == 77:
        pytest.skip("Requires a HIP PyTorch build")
    assert result.returncode == 0, result.stderr


def test_rocprofiler_late_dispatch_after_session_destroy(tmp_path: pathlib.Path):
    script = """
import json
import pathlib
import sys

import torch

if torch.version.hip is None:
    raise SystemExit(77)

import triton.profiler as proton

profile0 = pathlib.Path(sys.argv[1])
profile1 = pathlib.Path(sys.argv[2])

session0 = proton.start(str(profile0.with_suffix("")), backend="rocprofiler")
session1 = proton.start(str(profile1.with_suffix("")), backend="rocprofiler")
with proton.scope("first_session"):
    torch.ones((16,), device="cuda")
proton.finalize(session0)

with proton.scope("second_session"):
    torch.ones((16,), device="cuda")
proton.finalize(session1)

with profile1.open() as f:
    data = json.load(f)
assert [child["frame"]["name"] for child in data[0]["children"]] == [
    "first_session",
    "second_session",
]
"""
    env = os.environ.copy()
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("ROCPROFILER_REGISTER_FORCE_LOAD", None)
    env.pop("ROCPROFILER_REGISTER_LIBRARY", None)
    env["PROTON_TEST_DEFER_FIRST_ROCPROFILER_DISPATCH"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "late_dispatch_0.hatchet"),
            str(tmp_path / "late_dispatch_1.hatchet"),
        ],
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    if result.returncode == 77:
        pytest.skip("Requires a HIP PyTorch build")
    assert result.returncode == 0, (f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


@pytest.mark.parametrize("mode", ["script", "python", "pytest"])
def test_exec(mode, tmp_path: pathlib.Path):
    file_path = __file__
    helper_file = file_path.replace("test_cmd.py", "helper.py")
    temp_file = tmp_path / "test_exec.hatchet"
    name = str(temp_file.with_suffix(""))
    if mode == "script":
        subprocess.check_call(["proton", "-n", name, helper_file, "test"], stdout=subprocess.DEVNULL)
    elif mode == "python":
        subprocess.check_call([sys.executable, "-m", "triton.profiler.proton", "-n", name, helper_file, "test"],
                              stdout=subprocess.DEVNULL)
    elif mode == "pytest":
        subprocess.check_call(["proton", "-n", name, "pytest", "-k", "test_main", helper_file],
                              stdout=subprocess.DEVNULL)
    with temp_file.open() as f:
        data = json.load(f, )
    kernels = data[0]["children"]
    assert len(kernels) == 2
    assert kernels[0]["frame"]["name"] == "test" or kernels[1]["frame"]["name"] == "test"
