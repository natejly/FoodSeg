import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.lift_errors import classify, item_kind, size_bin, snap_iou  # noqa: E402


def test_classify():
    assert classify(0.9, 1.0, {1: 0.95}, {})[0] == "ok"
    assert classify(0.4, 0.0, {}, {})[0] == "missed"
    assert classify(0.4, 1.0, {1: 0.5, 2: 0.45}, {})[0] == "fragmented"
    assert classify(0.4, 1.0, {1: 0.9}, {1: 0.5})[0] == "merged"
    assert classify(0.6, 1.0, {1: 0.8}, {})[0] == "boundary"
    assert classify(0.3, 0.5, {1: 0.3}, {})[0] == "under-covered"


def test_snap_iou():
    # g=100, pred=100, overlap 80; 20 FN all near boundary, 20 FP all near -> perfect
    assert snap_iou(80, 100, 100, np.full(20, 0.002), np.full(20, 0.002), 0.003) == 1.0
    # nothing near: unchanged 80/120
    assert abs(snap_iou(80, 100, 100, np.full(20, 0.05), np.full(20, 0.05), 0.003) - 80 / 120) < 1e-9


def test_kind_size():
    assert item_kind({"source": "metafood3d", "shape": {"kind": "pile"}}) == "pile"
    assert item_kind({"source": "procedural_topping", "shape": {"kind": "topping"}}) == "topping"
    assert item_kind({"source": "metafood3d", "shape": {"kind": "single"}}) == "scan"
    assert size_bin(10) == "<15 ml" and size_bin(500) == ">=150 ml"


def test_batch_of():
    from eval.lift_errors import batch_of
    assert batch_of("s000001") == "v12" and batch_of("v13/s000001") == "v13"
