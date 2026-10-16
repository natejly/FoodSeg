"""segment.py v2 prompt logic with stub predictors: food primary, simplified names, split rule, image mode + linking."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import foodscan.labels as L  # noqa: E402
from foodscan import segment as S  # noqa: E402
from foodscan.labels import _FrameVotes  # noqa: E402

H, W = 40, 60


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_prompt_names_simplify_dedupe_ban():
    out = S.prompt_names(["Tuna_salad", "lettuce", "dish", "Meal", "food", "chicken breast", "chicken thighs",
                          "red onion slices"], cap=10)
    assert out == ["salad", "chicken", "onion slices"]          # table, dedupe, last two words, no dish/meal/food
    assert S.prompt_names([f"item{i}" for i in range(30)], cap=12) == [f"item{i}" for i in range(12)]
    th = S.name_thresholds(["salad", "egg"], 0.2, {"egg": 0.15})
    assert th == {"salad": 0.3, "egg": 0.15}


def test_refine_split_when_name_covers_subregion():
    food = {1: (0.6, rect(0, 40, 0, 40))}                       # one blob holding two items
    dets = [(2, "rice", 0.5, rect(0, 40, 0, 20))]               # name mask = left half: IoU 0.5 < 0.7
    out = S.refine_frame(food, dets)
    assert set(out) == {1, 2}
    assert out[2][2] == "rice" and out[2][1].sum() == 40 * 20
    assert out[1][2] is None and out[1][1].sum() == 40 * 20     # right half stays unnamed "food"
    assert not (out[1][1] & out[2][1]).any()


def test_refine_names_without_split_when_iou_high():
    food = {1: (0.6, rect(0, 40, 0, 20))}
    dets = [(2, "rice", 0.5, rect(0, 40, 0, 18))]               # IoU 0.9 -> just name it
    out = S.refine_frame(food, dets)
    assert set(out) == {1} and out[1][2] == "rice"


def test_refine_keeps_food_only_and_names_by_majority():
    food = {1: (0.5, rect(0, 10, 0, 10)), 2: (0.5, rect(20, 30, 0, 10)), 3: (0.5, rect(0, 10, 30, 40))}
    dets = [(10, "egg", 0.4, rect(0, 10, 0, 6) | rect(10, 19, 0, 60)),   # big mask: covers 60 % of 1, no split (contain < 0.5)
            (11, "bread", 0.3, rect(0, 10, 0, 30))]        # covers 100 % of 1 -> wins by coverage
    out = S.refine_frame(food, dets)
    assert set(out) == {1, 2, 3} and out[1][2] == "bread"
    assert out[2][2] is None and out[3][2] is None and out[3][1].sum() == 100   # food-only instances survive


def test_refine_unmatched_name_added_only_when_strong():
    food = {1: (0.6, rect(0, 40, 0, 40))}
    dets = [(2, "egg", 0.5, rect(0, 40, 0, 39)),                # n inside f, IoU 0.975 -> naming only
            (3, "cup", 0.9, rect(0, 10, 50, 60)),               # elsewhere, strong -> new named instance
            (4, "fig", 0.2, rect(30, 40, 50, 60))]              # elsewhere but weak -> ignored
    out = S.refine_frame(food, dets)
    assert set(out) == {1, 3} and out[1][2] == "egg" and out[3][2] == "cup"
    # a name mask covering all of f (and much more) names it without replacing its id
    out = S.refine_frame({1: (0.6, rect(10, 20, 10, 20))}, [(2, "salad", 0.45, rect(0, 40, 0, 60))])
    assert set(out) == {1} and out[1][1].sum() == 100 and out[1][2] == "salad"


class VideoStub:
    """Video API. 'food' returns two objects (a left blob = rice+egg, a right one); 'egg' the right half of the blob."""
    def __init__(self, n):
        self.n, self.text, self.prompts, self.oom_left = n, None, [], 0

    def handle_request(self, r):
        if r["type"] == "start_session":
            return {"session_id": "s"}
        if r["type"] == "add_prompt":
            self.text = r["text"]
            self.prompts.append((r["text"], r["output_prob_thresh"]))
        return {}

    def handle_stream_request(self, r):
        if self.oom_left:
            self.oom_left -= 1
            raise RuntimeError("CUDA out of memory")
        boxes = {"food": [(0, 40, 0, 30), (0, 20, 40, 60)], "egg": [(0, 40, 15, 30)], "plate": [(0, 40, 0, 60)]}
        sc = {"food": 0.6, "egg": 0.5, "plate": 0.6}
        for fi in range(r["start_frame_index"], self.n):
            bx = boxes.get(self.text, [])
            if not bx:
                continue
            yield {"frame_index": fi, "outputs": {"out_obj_ids": np.arange(len(bx)),
                   "out_probs": np.full(len(bx), sc[self.text]),
                   "out_binary_masks": np.stack([rect(*b) for b in bx])}}


def _frames(d, n):
    (d / "rgb").mkdir()
    for i in range(n):
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(d / "rgb" / f"{i:06d}.jpg")


def test_video_mode_food_primary_split_and_name():
    n = 3
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _frames(d, n)
        stub = VideoStub(n)
        meta = S.segment_scan(d, ["boiled egg", "dish"], predictor=stub, container_prompts=("plate",),
                              clutter_prompts=(), max_side=0, food_thresh=0.3, name_thresh=0.15)
        prompts = dict(stub.prompts)
        assert prompts["food"] == 0.3 and prompts["egg"] == 0.15 and "dish" not in prompts
        assert sorted(v["name"] for v in meta.values()) == ["egg", "food", "food"]
        masks, _ = S.load_masks(d)
        egg = [k for k, v in meta.items() if v["name"] == "egg"][0]
        assert len(meta) == 3 and (masks[0] == egg).sum() == 40 * 15          # blob split, egg half carved out
        assert masks[0][5, 5] != egg and masks[0][5, 5] in meta and masks[0][5, 50] in meta   # rest is "food"
        assert masks[1][5, 5] == masks[0][5, 5]                              # consistent across frames
        assert masks[0][30, 50] == 254                                       # plate fills what food left
        table = json.loads((d / "masks/concepts.json").read_text())
        assert table["mode"] == "video"
        c = {x["name"]: x for x in table["concepts"]}
        assert c["food"]["labels"] and c["egg"]["labels"] == [egg]


def test_video_mode_oom_rebuilds_predictor_and_retries():
    n = 2
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _frames(d, n)
        first, second = VideoStub(n), VideoStub(n)
        first.oom_left = 1
        built = []
        meta = S.segment_scan(d, [], predictor=first, rebuild=lambda: built.append(1) or second,
                              container_prompts=(), clutter_prompts=(), max_side=0)
        assert built == [1] and len(meta) == 2
        assert json.loads((d / "masks/concepts.json").read_text())["oom_rebuilds"] == 1


class ImageStub:
    """Image API: the same two-blob frame for every image; the second 'frame' shifts nothing."""
    def __init__(self):
        self.calls = []

    def set_image(self, img):
        return img

    def detect(self, st, prompt):
        self.calls.append(prompt)
        boxes = {"food": [(0, 40, 0, 30), (0, 20, 40, 60)], "egg": [(0, 40, 15, 30)], "plate": [(30, 40, 30, 60)]}
        sc = {"food": [0.6, 0.5], "egg": [0.5], "plate": [0.8]}
        bx = boxes.get(prompt, [])
        if not bx:
            return np.zeros((0, H, W), bool), np.zeros(0, np.float32)
        return np.stack([rect(*b) for b in bx]), np.array(sc[prompt], np.float32)


def _fake_votes(monkeypatch, per_frame_points):
    """collect_votes returns, per frame, the 3D points (ranges) that each raw id covers."""
    def collect(scan, masks, points, normals=None, **kw):
        out = []
        for pos, fi in enumerate(sorted(masks)):
            m = masks[fi]
            idx, lab = [], []
            for g in np.unique(m):
                if g == 0 or g not in per_frame_points[fi]:
                    continue
                a, b = per_frame_points[fi][g]
                idx.append(np.arange(a, b))
                lab.append(np.full(b - a, g))
            out.append(_FrameVotes(pos, np.concatenate(idx).astype(np.int32), np.concatenate(lab).astype(np.int32),
                                   np.ones(sum(len(i) for i in idx), np.float32)))
        return out
    monkeypatch.setattr(L, "collect_votes", collect)


def test_image_mode_links_instances_across_frames(monkeypatch):
    n = 3
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _frames(d, n)
        stub = ImageStub()
        captured = {}

        def cap_linker(raw, name_px, fs):
            captured["raw"], captured["npx"] = raw, name_px
            return {fi: np.where(m >= S.FIRST_RAW_ID, 1, m).astype(np.uint8) for fi, m in raw.items()}, \
                {1: {"name": "food", "concept_index": 0, "kind": "food", "mean_score": 0.5}}, fs, {}
        S.segment_scan(d, ["egg"], mode="image", predictor=stub, container_prompts=("plate",), clutter_prompts=(),
                       max_side=0, linker=cap_linker)
        raw = captured["raw"]
        ids = [set(np.unique(raw[fi])) - {0, 253, 254} for fi in range(n)]
        assert all(ids[i].isdisjoint(ids[j]) for i in range(n) for j in range(i + 1, n))   # unique per frame
        assert all(min(s) >= S.FIRST_RAW_ID for s in ids) and raw[0].dtype == np.uint16
        assert stub.calls.count("food") == n and set(stub.calls) == {"food", "egg", "plate"}
        assert (raw[0] == 254).any()


def test_link_instances_merges_across_frames(monkeypatch):
    # object X (points 0..100) is raw id 1001/1011/1021 in frames 0/1/2; Y (200..260) is 1002/1012/1022
    n = 3
    raw, pp = {}, {}
    for fi in range(n):
        m = np.zeros((H, W), np.uint16)
        a, b = S.FIRST_RAW_ID + 10 * fi + 1, S.FIRST_RAW_ID + 10 * fi + 2
        m[0:20, 0:20], m[20:40, 30:50] = a, b
        m[0:5, 55:60] = 254
        raw[fi] = m
        pp[fi] = {a: (0, 100), b: (200, 260), 65002: (400, 420)}
    # fake collect_votes keys the aux id the way link_instances encodes it (254 -> 65002)
    _fake_votes(monkeypatch, pp)
    pts = np.zeros((500, 3))
    npx = {S.FIRST_RAW_ID + 10 * fi + 1: {"egg": 100} for fi in range(n)}
    npx.update({S.FIRST_RAW_ID + 10 * fi + 2: {"food": 60} for fi in range(n)})
    fs = {fi: {S.FIRST_RAW_ID + 10 * fi + 1: 0.7, S.FIRST_RAW_ID + 10 * fi + 2: 0.4, 254: 0.9} for fi in range(n)}
    maps, meta, fsc, st = S.link_instances(None, raw, npx, fs, pts, min_points=10)
    assert st["raw_ids"] == 6 and st["clusters"] == 2
    ids = {int(maps[fi][5, 5]) for fi in range(n)}
    assert len(ids) == 1 and ids != {0}
    x = ids.pop()
    assert len({int(maps[fi][30, 40]) for fi in range(n)}) == 1 and int(maps[0][30, 40]) != x
    assert meta[x]["name"] == "egg" and meta[int(maps[0][30, 40])]["name"] == "food"
    assert maps[0].dtype == np.uint8 and maps[0][1, 57] == 254 and fsc[1][x] == 0.7
    assert L.CLUTTER == 253                                                     # patch restored


def _pixel_votes(monkeypatch):
    """collect_votes where point p = pixel (y*W+x) of every frame: same region in two frames = same 3D points."""
    def collect(scan, masks, points, normals=None, *a, features=False, **kw):
        out = []
        for pos, fi in enumerate(sorted(masks)):
            m = masks[fi]
            fv = _FrameVotes(pos, np.arange(m.size, dtype=np.int32), m.ravel().astype(np.int32),
                             np.ones(m.size, np.float32))
            if features:
                fv.edge, fv.score, fv.qual = (np.full(m.size, 1e3, np.float32),) * 3
            out.append(fv)
        return out
    monkeypatch.setattr(L, "collect_votes", collect)


def test_image_mode_covote_linker_consistent_ids_and_lift(monkeypatch):
    _pixel_votes(monkeypatch)
    n = 3
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _frames(d, n)
        pts = np.zeros((H * W, 3))
        link = lambda raw, npx, fs: S.link_instances(None, raw, npx, fs, pts, min_points=10)   # covote default
        meta = S.segment_scan(d, ["egg"], mode="image", predictor=ImageStub(), container_prompts=("plate",),
                              clutter_prompts=(), max_side=0, linker=link)
        masks, meta2 = S.load_masks(d)
        assert set(meta) == set(meta2) and len(meta) == 3
        assert json.loads((d / "masks/concepts.json").read_text())["link"]["linker"] == "covote"
        for fi in range(1, n):                                       # same id on the same object in every frame
            assert (masks[fi] == masks[0]).all()
        assert set(np.unique(masks[0])) - {0, 254} == set(meta)
        egg = [k for k, v in meta.items() if v["name"] == "egg"]
        assert len(egg) == 1 and (masks[0][20, 20] == egg[0]) and meta[egg[0]]["n_frames"] == n
        scores = S.load_scores(d)
        assert all(set(scores[fi]) == set(np.unique(masks[fi])) - {0} for fi in range(n))
        lift = L.lift_for_meal(None, masks, pts, None, meta2, weighting="cos+qual", save_to=d / "lift.npz")
        z = L.load_lift(d / "lift.npz")
        assert {"lift_label", "lift_share", "lift_nvotes", "lift_agree"} <= set(z)
        lab = z["lift_label"].reshape(H, W)
        assert lab[20, 20] == egg[0] and lab[5, 5] in meta and lab[5, 5] != egg[0] and lab[30, 50] == 254
        assert lift.stats["n_frames"] == n and z["meta"]["instances"].keys() == {str(k) for k in meta}


def test_link_instances_repair_method_still_available(monkeypatch):
    _pixel_votes(monkeypatch)
    raw = {}
    for fi in range(3):
        m = np.zeros((H, W), np.uint16)
        m[0:20, 0:20] = S.FIRST_RAW_ID + fi
        raw[fi] = m
    for method in ("repair", "covote"):
        maps, meta, _, st = S.link_instances(None, raw, {}, {fi: {} for fi in raw}, np.zeros((H * W, 3)),
                                             min_points=10, method=method)
        assert st["linker"] == method and len(meta) == 1 and len({int(maps[fi][5, 5]) for fi in raw}) == 1


REPLAY_SID = "s000000"
REPLAY_RAW = Path(__file__).resolve().parents[2] / "data/synth/hard_v12/scans" / REPLAY_SID / "masks_sam3_v2i" / "raw"


import pytest  # noqa: E402


@pytest.mark.skipif(not REPLAY_RAW.exists(), reason="hard_v12 stored raw maps not present")
def test_replay_stored_raw_maps_matches_lift_sweep(tmp_path):
    """segment.py --from-masks path on one real-SAM scan: lift inst mIoU equals the lift_sweep combined row (0.690)."""
    pytest.importorskip("open3d")
    from foodscan.replay_lift import replay_scan
    r = replay_scan(REPLAY_SID, tmp_path)
    assert abs(float(np.mean(r["iou"])) - 0.6895) < 0.01
    z = L.load_lift(tmp_path / REPLAY_SID / "lift.npz")
    assert z["lift_label"].shape == z["xyz"].shape[:1]
    inst = json.loads((tmp_path / REPLAY_SID / "masks/instances.json").read_text())
    maps, _ = S.load_masks(tmp_path / REPLAY_SID, tmp_path / REPLAY_SID / "masks")
    ids = set().union(*[set(np.unique(m)) for m in maps.values()]) - {0, 253, 254}
    assert ids <= {int(k) for k in inst}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
