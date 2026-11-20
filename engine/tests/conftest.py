"""Tests marked `npu` need an XDNA1 NPU, pyxrt and built kernels; they are skipped when any of those is missing."""

import os

import pytest

_npu_used = False


def _npu_skip_reason():
    try:
        import pyxrt  # noqa: F401
    except ImportError:
        return "pyxrt not importable (copy pyxrt.pyd from the XRT Windows SDK into the venv)"
    from xdna1_llm.npu.linear import default_kernel_dir

    if not os.path.isdir(default_kernel_dir()):
        return f"no kernels in {default_kernel_dir()} (run kernels/build_kernels.sh)"
    return None


def pytest_collection_modifyitems(config, items):
    if not any("npu" in item.keywords for item in items):
        return
    reason = _npu_skip_reason()
    if reason:
        for item in items:
            if "npu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def npu():
    global _npu_used
    from xdna1_llm.npu.device import Npu

    _npu_used = True
    return Npu()


def pytest_unconfigure(config):
    # pyxrt objects can access-violate when destroyed at interpreter exit; skip teardown once the NPU was opened.
    if _npu_used:
        from xdna1_llm.npu.device import hard_exit

        hard_exit(getattr(config, "_xdna1_exitstatus", 0))


def pytest_sessionfinish(session, exitstatus):
    session.config._xdna1_exitstatus = int(exitstatus)
