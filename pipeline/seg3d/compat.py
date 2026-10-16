"""Environment shims so the vendored Sonata/PTv3 runs on this machine.

* torch_scatter (does not build for torch 2.11) -> pure-torch `segment_csr` via torch.segment_reduce.
* flash-attn is never imported unless enable_flash=True (sm_75 has none).
* puts `_vendor` on sys.path so `import sonata` resolves to the vendored copy.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def setup() -> None:
    try:
        import torch_scatter  # noqa: F401
    except Exception:                                   # noqa: BLE001
        import torch

        def segment_csr(src, indptr, out=None, reduce="sum"):
            return torch.segment_reduce(src, reduce, offsets=indptr, axis=0)

        m = types.ModuleType("torch_scatter")
        m.segment_csr = segment_csr
        sys.modules["torch_scatter"] = m
    v = str(Path(__file__).parent / "_vendor")
    if v not in sys.path:
        sys.path.insert(0, v)
