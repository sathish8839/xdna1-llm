"""NPU runtime: pyxrt device/context handling and the chunked linear-layer engine."""


class NpuError(RuntimeError):
    """The NPU cannot run the engine (no device, or the driver refuses the kernels); the message says why.
    Defined here, not in device.py, so it can be caught without importing pyxrt."""
