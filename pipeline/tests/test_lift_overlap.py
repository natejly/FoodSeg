"""segment.py without SAM 3: overlap resolution by score, container/clutter classes, re-prompt windows."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.segment import LabelAccumulator, associate, load_masks, load_scores, segment_scan  # noqa: E402


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_overlap_resolved_by_score_not_order():
    a = LabelAccumulator(max_side=0)
    low, high = _rect(20, 20, 0, 10, 0, 20), _rect(20, 20, 5, 15, 0, 20)
    a.add(0, 1, 0.6, low)          # added first but lower score
    a.add(0, 2, 0.9, high)
    lab = a.frame(0)
    assert (lab[5:15] == 2).all() and (lab[:5] == 1).all()
    a2 = LabelAccumulator(max_side=0)   # reversed order gives the same answer
    a2.add(0, 2, 0.9, high)
    a2.add(0, 1, 0.6, low)
    assert (a2.frame(0) == lab).all()
    assert a.present(0) == {1: 0.6, 2: 0.9}


def test_container_prior_lets_food_win():
    a = LabelAccumulator(max_side=0)
    plate = _rect(10, 10, 0, 10, 0, 10)
    food = _rect(10, 10, 3, 6, 3, 6)
    a.add(0, 254, 0.95, plate, prior=-0.15)
    a.add(0, 1, 0.85, food)
    assert (a.frame(0)[3:6, 3:6] == 1).all() and a.frame(0)[0, 0] == 254


def test_associate_by_iou():
    prev = {5: np.zeros((10, 10), np.uint8)}
    prev[5][:, :5] = 3
    new = {5: {0: _rect(10, 10, 0, 10, 0, 5), 1: _rect(10, 10, 0, 10, 7, 10)}}
    assert associate(prev, new) == {0: 3}


class StubPredictor:
    """Mimics the sam3 request API. A 'banana' only appears from frame 40 on; ids restart per prompt."""
    def __init__(self, n, h=48, w=64):
        self.n, self.h, self.w, self.text, self.start = n, h, w, None, 0

    def handle_request(self, r):
        if r["type"] == "start_session":
            return {"session_id": "s"}
        if r["type"] == "add_prompt":
            self.text, self.start = r["text"], r["frame_index"]
        return {}

    def handle_stream_request(self, r):
        fwd = r["propagation_direction"] == "forward"
        lo = r["start_frame_index"]
        cnt = r.get("max_frame_num_to_track") or self.n
        frames = range(lo, min(self.n, lo + cnt)) if fwd else range(lo, max(-1, lo - cnt), -1)
        boxes = {"rice": (0, 20, 0, 30), "banana": (25, 45, 35, 60), "plate": (0, 48, 0, 64), "fork": (40, 48, 0, 8)}
        for fi in frames:
            if self.text == "banana" and fi < 40:
                continue                      # not visible yet, and tracker cannot see it from an earlier prompt
            if self.text not in boxes:
                continue
            y0, y1, x0, x1 = boxes[self.text]
            m = np.zeros((self.h, self.w), bool)
            m[y0:y1, x0:x1] = True
            yield {"frame_index": fi, "outputs": {"out_obj_ids": np.array([0]),
                   "out_probs": np.array([0.9 if self.text != "plate" else 0.95]), "out_binary_masks": m[None]}}


def test_segment_scan_with_stub():
    n = 70
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "rgb").mkdir()
        for i in range(n):
            Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(d / "rgb" / f"{i:06d}.jpg")
        meta = segment_scan(d, ["rice", "banana"], predictor=StubPredictor(n), reprompt_every=30,
                            container_prompts=("plate",), clutter_prompts=("fork",), max_side=0)
        masks, meta2 = load_masks(d)
        names = {v["name"] for v in meta.values()}
        assert names == {"rice", "banana"}, meta
        rice = [k for k, v in meta.items() if v["name"] == "rice"]
        assert len(rice) == 1, f"re-prompt windows must keep one rice id, got {rice}"   # association worked
        banana = [k for k, v in meta.items() if v["name"] == "banana"][0]
        assert (masks[10] == 254).any() and (masks[10] == 253).any() and (masks[10] == rice[0]).any()
        assert not (masks[10] == banana).any() and (masks[50] == banana).any()   # found mid-orbit
        assert masks[10][5, 5] == rice[0] and masks[10][30, 30] == 254            # food beats the plate prior
        sc = load_scores(d)
        assert sc[50][banana] == 0.9 and sc[50][254] == 0.95
        table = json.loads((d / "masks/concepts.json").read_text())
        assert table["container_label"] == 254 and table["clutter_label"] == 253
        assert {c["name"]: c["labels"] for c in table["concepts"]}["plate"] == [254]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
