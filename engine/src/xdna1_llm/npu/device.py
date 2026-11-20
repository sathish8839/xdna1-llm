"""Thin pyxrt wrapper for mlir-aie npu1 designs.

Facts about the Windows XDNA1 driver that shape this module (measured, see docs/benchmarks.md):
  * at most 5 hw_contexts can be open at once (the 6th fails with 0xc01e0009);
  * alternating between two contexts costs ~1.45 ms per call, while swapping buffers inside one context is free;
  * destroying pyxrt objects at interpreter exit can access-violate, so callers keep them alive and exit
    with os._exit() (see `hard_exit`).
mlir-aie kernels take (opcode=3, instr_bo, instr_len, *buffers) with buffers on group ids 3, 4, 5, ...
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from . import NpuError

MAX_CONTEXTS = 5
TESTED_DRIVERS = ("32.0.203.297",)  # Windows NPU driver versions the kernels are known to run on
DRIVER_HELP = "https://github.com/sathish8839/xdna1-llm/blob/main/docs/setup.md#npu-driver"

try:
    import pyxrt as xrt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pyxrt not found. Copy pyxrt.pyd from the XRT Windows SDK (xrt_sdk/xrt/python/pyxrt.pyd, "
        "XRT release 2.21.75) into your Python 3.13 environment's site-packages. See docs/setup.md."
    ) from e

TO_DEV = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEV = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
COMPLETED = xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


class Npu:
    """One XRT device plus a registry of loaded kernels (one hw_context each)."""

    def __init__(self, index: int = 0):
        try:
            self.dev = xrt.device(index)
        except Exception as e:
            raise NpuError(
                "No AMD NPU found. xdna1-llm needs a Ryzen 7040, 8040 or 200 series processor with AMD's NPU driver "
                f"installed (Device Manager > Neural processors). See {DRIVER_HELP} ({e})"
            ) from e
        self.name = self.dev.get_info(xrt.xrt_info_device.name)
        self.driver = driver_version(self.dev)
        if self.driver not in TESTED_DRIVERS:
            print(f"note: NPU driver {self.driver} is untested; xdna1-llm is tested on {', '.join(TESTED_DRIVERS)}")
        self.kernels: dict[str, Kernel] = {}

    def load(self, tag: str, xclbin_path: str, insts_path: str) -> Kernel:
        if tag in self.kernels:
            return self.kernels[tag]
        if len(self.kernels) >= MAX_CONTEXTS:
            raise RuntimeError(
                f"XDNA1 driver allows {MAX_CONTEXTS} concurrent contexts; already loaded {list(self.kernels)}"
            )
        k = Kernel(self, tag, xclbin_path, insts_path)
        self.kernels[tag] = k
        return k

    def bo(self, nbytes: int, group_id: int, cacheable: bool = False):
        return xrt.bo(self.dev, nbytes, xrt.bo.cacheable if cacheable else xrt.bo.host_only, group_id)


class Kernel:
    def __init__(self, npu: Npu, tag: str, xclbin_path: str, insts_path: str):
        self.npu, self.tag = npu, tag
        try:
            xb = xrt.xclbin(xclbin_path)
            npu.dev.register_xclbin(xb)
            self.ctx = xrt.hw_context(npu.dev, xb.get_uuid())
            self.kernel = xrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        except Exception as e:
            raise NpuError(
                f"The NPU driver ({npu.driver}) did not accept the kernel {os.path.basename(xclbin_path)}. "
                f"xdna1-llm is tested on NPU driver {', '.join(TESTED_DRIVERS)}; newer drivers may not run "
                f"first-generation (npu1) kernels. See {DRIVER_HELP} ({e})"
            ) from e
        self.programs: dict[str, tuple] = {}
        self.add_program("default", insts_path)

    def add_program(self, name: str, insts_path: str):
        """Register another instruction stream (host-side DMA program) for this context. Designs whose core
        programs do not depend on a size (e.g. the decode GEMV's N) run every size this way, in one context."""
        insts = np.fromfile(insts_path, dtype=np.uint32)
        bo = self.npu.bo(insts.nbytes, self.kernel.group_id(1), cacheable=True)
        bo.write(insts.tobytes(), 0)
        bo.sync(TO_DEV)
        self.programs[name] = (bo, len(insts))

    def group(self, arg_index: int) -> int:
        """Group id for the i-th data buffer (0-based) of an mlir-aie runtime sequence."""
        return self.kernel.group_id(3 + arg_index)

    def start(self, *bos, program: str = "default"):
        bo_instr, n = self.programs[program]
        return self.kernel(3, bo_instr, n, *bos)

    def run(self, *bos, program: str = "default"):
        finish(self.start(*bos, program=program), self.tag)


_PENDING = {
    getattr(xrt.ert_cmd_state, f"ERT_CMD_STATE_{s}")
    for s in ("NEW", "QUEUED", "RUNNING", "SUBMITTED")
    if hasattr(xrt.ert_cmd_state, f"ERT_CMD_STATE_{s}")
}


def finish(run, what: str = "kernel", spin_s: float = 0.05):
    """Wait for a started run. Polls first: for sub-millisecond decode calls that is ~0.05 ms faster than
    run.wait(), which sleeps until the driver wakes it. Falls back to wait() after spin_s."""
    t_end = time.perf_counter() + spin_s
    st = run.state()
    while st in _PENDING and time.perf_counter() < t_end:
        st = run.state()
    if st in _PENDING:
        st = run.wait()
    if st != COMPLETED:
        raise RuntimeError(f"{what}: {st}")


def mapped(bo, dtype, shape=None) -> np.ndarray:
    """Zero-copy numpy view of a host_only buffer object."""
    a = np.frombuffer(bo.map(), dtype=dtype)
    return a.reshape(shape) if shape is not None else a


def driver_version(dev) -> str:
    """The Windows NPU driver's version, as XRT reports it ("unknown" if it does not)."""
    try:
        host = json.loads(dev.get_info(xrt.xrt_info_device.host))
        return next(d["version"] for d in host.get("drivers", []) if "NPU" in d.get("name", ""))
    except (ValueError, KeyError, StopIteration, RuntimeError):
        return "unknown"


def hard_exit(code: int = 0):
    """Exit without running pyxrt destructors (they can access-violate on Windows at shutdown)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
