"""Clear errors when the NPU cannot run the engine (needs pyxrt and an NPU; skipped otherwise)."""

import pytest

pytestmark = pytest.mark.npu


def test_a_missing_device_says_what_to_install():
    from xdna1_llm.npu import NpuError
    from xdna1_llm.npu.device import Npu

    with pytest.raises(NpuError, match="No AMD NPU found"):
        Npu(index=99)


def test_a_rejected_kernel_names_the_driver(tmp_path, npu):
    from xdna1_llm.npu import NpuError
    from xdna1_llm.npu.device import Npu

    bad = tmp_path / "bad.xclbin"
    bad.write_bytes(b"not an xclbin")
    with pytest.raises(NpuError, match=rf"NPU driver \({npu.driver}\) did not accept the kernel bad.xclbin"):
        Npu().load("bad", str(bad), str(tmp_path / "bad.bin"))  # its own registry: other tests' contexts stay open
