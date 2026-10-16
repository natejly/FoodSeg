"""Toy tests for cross-view id repair (labels.repair_ids)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.labels import _FrameVotes, repair_ids  # noqa: E402


def _fv(pos, spans):
    """spans: {id: (start, stop)} point ranges seen in this frame."""
    idx = np.concatenate([np.arange(a, b) for a, b in spans.values()]).astype(np.int32)
    lab = np.concatenate([np.full(b - a, k) for k, (a, b) in spans.items()]).astype(np.int32)
    return _FrameVotes(pos, idx, lab, np.ones(len(idx), np.float32))


def test_merge_on_tracker_break():
    # object on points 0..100: id 1 in frames 0-4, id 5 (tracker break) in frames 5-9; id 2 elsewhere throughout
    fvs = [_fv(i, {1: (0, 100), 2: (200, 260)}) for i in range(5)]
    fvs += [_fv(i, {5: (10, 100), 2: (200, 260)}) for i in range(5, 10)]
    remap, st = repair_ids(fvs, 300, min_points=10)
    assert remap == {5: 1} and st["merges"] == 1


def test_no_merge_for_cooccurring_nested_ids():
    # garnish (id 2) always sits inside rice (id 1): overlap is 100% but they co-occur -> keep separate
    fvs = [_fv(i, {1: (0, 100), 2: (10, 40)}) for i in range(8)]
    remap, st = repair_ids(fvs, 300, min_points=10)
    assert remap == {} and st["merges"] == 0


def test_no_merge_across_concepts_or_special_labels():
    fvs = [_fv(i, {1: (0, 100)}) for i in range(4)] + [_fv(i, {5: (0, 100)}) for i in range(4, 8)]
    assert repair_ids(fvs, 300, min_points=10, groups={1: 0, 5: 1})[0] == {}
    fvs = [_fv(i, {254: (0, 100)}) for i in range(4)] + [_fv(i, {253: (0, 100)}) for i in range(4, 8)]
    assert repair_ids(fvs, 300, min_points=10)[0] == {}


def test_disjoint_ids_untouched():
    fvs = [_fv(i, {1: (0, 100)}) for i in range(4)] + [_fv(i, {5: (150, 250)}) for i in range(4, 8)]
    assert repair_ids(fvs, 300, min_points=10)[0] == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
