"""Thin pyxrt wrapper for mlir-aie npu1 designs.

Facts about the Windows XDNA1 driver that shape this module (measured, see docs/benchmarks.md):
  * at most 5 hw_contexts can be open at once (the 6th fails with 0xc01e0009);
  * alternating between two contexts costs ~1.45 ms per call, while swapping buffers inside one context is free;
  * destroying pyxrt objects at interpreter exit can access-violate, so callers keep them alive and exit
    with os._exit() (see `hard_exit`).
mlir-aie kernels take (opcode=3, instr_bo, instr_len, *buffers) with buffers on group ids 3, 4, 5, ...
"""

from __future__ import annotations

import os
import sys

import numpy as np

MAX_CONTEXTS = 5

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
        self.dev = xrt.device(index)
        self.name = self.dev.get_info(xrt.xrt_info_device.name)
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
        xb = xrt.xclbin(xclbin_path)
        npu.dev.register_xclbin(xb)
        self.ctx = xrt.hw_context(npu.dev, xb.get_uuid())
        self.kernel = xrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        self.insts = np.fromfile(insts_path, dtype=np.uint32)
        self.bo_instr = npu.bo(self.insts.nbytes, self.kernel.group_id(1), cacheable=True)
        self.bo_instr.write(self.insts.tobytes(), 0)
        self.bo_instr.sync(TO_DEV)

    def group(self, arg_index: int) -> int:
        """Group id for the i-th data buffer (0-based) of an mlir-aie runtime sequence."""
        return self.kernel.group_id(3 + arg_index)

    def start(self, *bos):
        return self.kernel(3, self.bo_instr, len(self.insts), *bos)

    def run(self, *bos):
        st = self.start(*bos).wait()
        if st != COMPLETED:
            raise RuntimeError(f"{self.tag}: {st}")


def mapped(bo, dtype, shape=None) -> np.ndarray:
    """Zero-copy numpy view of a host_only buffer object."""
    a = np.frombuffer(bo.map(), dtype=dtype)
    return a.reshape(shape) if shape is not None else a


def hard_exit(code: int = 0):
    """Exit without running pyxrt destructors (they can access-violate on Windows at shutdown)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
