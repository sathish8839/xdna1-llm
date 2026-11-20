import numpy as np

from xdna1_llm.weight_cache import WeightCache


def test_get_makes_once_then_maps_the_saved_array(tmp_path):
    cache, calls = WeightCache(tmp_path / "m"), []

    def make():
        calls.append(1)
        return np.arange(12, dtype=np.uint16).reshape(3, 4)

    first = cache.get("gemm/blk.0.qkv", make)
    again = cache.get("gemm/blk.0.qkv", make)
    np.testing.assert_array_equal(first, again)
    assert len(calls) == 1 and isinstance(again, np.memmap) and not again.flags.writeable
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_changed_model_gets_a_new_cache_and_the_old_one_goes(tmp_path):
    gguf = tmp_path / "Model-Q4_K_M.gguf"
    gguf.write_bytes(b"one")
    other = tmp_path / "Model-Q4_K_M-imatrix.gguf"
    other.write_bytes(b"x")
    a = WeightCache.for_model(str(gguf), tmp_path / "cache")
    keep = WeightCache.for_model(str(other), tmp_path / "cache")
    a.get("w", lambda: np.zeros(2))
    keep.get("w", lambda: np.zeros(2))
    assert WeightCache.for_model(str(gguf), tmp_path / "cache").root == a.root  # same file: same cache
    (tmp_path / "copy").mkdir()
    copy = tmp_path / "copy" / gguf.name
    copy.write_bytes(gguf.read_bytes())
    assert WeightCache.for_model(str(copy), tmp_path / "cache").root == a.root  # a copy elsewhere too

    gguf.write_bytes(b"two")
    b = WeightCache.for_model(str(gguf), tmp_path / "cache")
    assert b.root != a.root and not a.root.exists() and keep.root.exists()


def test_a_cache_that_cannot_be_written_still_returns_the_array(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    out = WeightCache(blocker / "m").get("w", lambda: np.ones(3))
    np.testing.assert_array_equal(out, np.ones(3))
