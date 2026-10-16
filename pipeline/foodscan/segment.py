"""SAM 3 / SAM 3.1 text-prompted video segmentation of a FoodScan capture.

    python -m foodscan.segment SCAN_DIR --names "white rice,grilled chicken,broccoli" [--sam31]
        [--mode video|image] [--food-thresh 0.3] [--name-thresh 0.2]
        [--reprompt-every 30] [--no-aux] [--max-side 960]
        [--linker covote|repair] [--cluster-thresh 0.8]          # image mode id linking (default covote)
        [--from-masks DIR/(raw) --out-dir OUT --points-npz CLOUD.npz --lift-out lift.npz]   # CPU replay

Writes into SCAN_DIR/masks/:
  NNNNNN.png        uint8 label map (0 = background, 1..K = food instance ids consistent
                    across frames, 253 = clutter, 254 = container) at <= --max-side px
                    (the lift maps depth pixels to mask pixels proportionally, so any size works)
  NNNNNN.json       {"<label>": best_score} for every label present in that frame
  instances.json    {id: {"name", "concept_index", "mean_score", "kind": "food"}}  (food only,
                    same schema as before; load_masks/run_meal read it unchanged)
  concepts.json     concept -> label table: {"clutter_label", "container_label",
                    "concepts": [{"name", "kind", "concept_index", "labels": [...]}]}

Prompt strategy (v2, from results/sam3_gap): the primary concept is the generic "food" at
--food-thresh (0.3), which gives one instance per item/blob. A secondary pass prompts the
item names, simplified to 1-2 words (SIMPLE_NAMES) at per-concept thresholds (--name-thresh,
0.15-0.3), and refine_frame() uses them to (a) split a "food" instance when a name mask covers a
sub-region (IoU to the instance < 0.7, > 5 % of its area), (b) name instances by majority
overlap. "food"-only instances stay (named "food"); "dish"/"meal" are never used.
--mode video propagates every concept through the SAM 3 video tracker (ids consistent via
tracking + re-association); --mode image runs the image model on every frame and links the
per-frame instances in 3D (link_instances: lift to the fused cloud, then labels.cluster_ids co-vote
clustering (default, cosine cut 0.8; results/lift_sweep) or labels.repair_ids via --linker repair).
--from-masks replays stored raw maps (DIR/raw) through the same linker with no predictor.

Protocol (from facebookresearch/sam3 source, Sept 2026): a text prompt resets the
session state, so each concept is one propagation.
  * Overlaps between concepts/instances are resolved per pixel by SAM 3 object score
    (out_probs; the API exposes no per-pixel logit), not by prompt order. Containers get a
    score prior of -0.15 so a plate mask that swallows the food loses to the food.
  * Containers ("plate","bowl","cup") -> 254 and clutter ("fork","knife","spoon","napkin",
    "phone") -> 253 are prompted as classes so the lift sees them instead of background.
  * Every --reprompt-every frames (default 30) each concept is re-prompted at that frame and
    propagated forward (and `backward_frames` back), so items first visible mid-orbit are
    found. Instance ids restart per prompt; they are re-associated to the previous window by
    mask IoU on the overlapping frames (>= 0.3, one-to-one) -- remaining splits are repaired
    in 3D by labels.repair_ids. reprompt_every=0 gives the old single full-video pass.
Frames are read from SCAN_DIR/rgb (the predictor sorts by int(stem)). Needs a GPU box with the
sam3 repo installed (conda env `foodseg`, fp16 patches on the 6 GB card, see memory) and gated
weights; `segment_scan(..., predictor=stub)` lets tests inject a fake predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLUTTER, CONTAINER = 253, 254
CONTAINER_PROMPTS = ("plate", "bowl", "cup")
CLUTTER_PROMPTS = ("fork", "knife", "spoon", "napkin", "phone")
KIND_PRIOR = {"food": 0.0, "clutter": 0.0, "container": -0.15}

FOOD = "food"                 # primary generic concept; "dish"/"meal" are useless on renders (sam3_gap)
FOOD_THRESH = 0.3             # generic food hits 86 % of synthetic items at 0.3 (0.59 at 0.5)
NAME_THRESH = 0.2             # secondary name pass; 0.15 gave the best recall, 0.5 misses half
VAGUE_NAME_THRESH = 0.3       # one-word category names fire on unrelated food, so they need more score
VAGUE_NAMES = frozenset({"meat", "fruit", "vegetable", "sauce", "salad", "grain", "dip", "fish"})
BANNED_PROMPTS = frozenset({"", "food", "dish", "meal"})
AUX_THRESH = {"container": 0.3, "clutter": 0.3}   # probe on 264 synthetic frames: 0.3-0.5 dets land on GT food 0.1 %, own class 57-89 %
FIRST_RAW_ID = 1000           # image mode: per-frame unique instance ids (uint16 maps) before linking

# item name (lower, spaces) -> simplified 1-2 word prompt (same table as eval/sam3_gap.py)
SIMPLE_NAMES = {
    "black beans": "beans", "blueberries": "berries", "boiled egg": "egg", "bread roll": "bread",
    "brown rice": "rice", "cake slice": "cake", "cherry tomato": "tomato", "chicken breast": "chicken",
    "chicken thighs": "chicken", "cookies": "cookie", "couscous": "grain", "falafel": "fried ball",
    "gravy": "sauce", "green beans": "beans", "hummus": "dip", "kiwi": "fruit", "lettuce": "salad",
    "mashed potato": "potato", "melons": "melon", "melted cheese": "cheese", "mixed greens": "salad",
    "omelet": "egg", "papaya": "fruit", "pasta mixed dishes": "pasta", "pesto": "sauce",
    "pork chop": "meat", "radicchio": "lettuce", "salmon grill": "fish", "sausages": "sausage",
    "sour cream": "sauce", "squash": "vegetable", "tuna salad": "salad", "yeast bread": "bread",
}


def simplify(name: str) -> str:
    """1-2 word prompt for an item name: table lookup, else the last two words."""
    n = " ".join(name.replace("_", " ").lower().split())
    if n in SIMPLE_NAMES:
        return SIMPLE_NAMES[n]
    w = n.split()
    return " ".join(w[-2:]) if len(w) > 2 else n


def prompt_names(names, cap: int = 12, simplify_names: bool = True) -> list[str]:
    """Simplified, deduplicated secondary prompts; never food/dish/meal; at most `cap`."""
    out: list[str] = []
    for nm in names:
        s = simplify(nm) if simplify_names else " ".join(nm.replace("_", " ").lower().split())
        if s not in BANNED_PROMPTS and s not in out:
            out.append(s)
    return out[:cap] if cap else out


def name_thresholds(prompts, base: float = NAME_THRESH, overrides: dict | None = None) -> dict[str, float]:
    th = {p: (max(base, VAGUE_NAME_THRESH) if p in VAGUE_NAMES else base) for p in prompts}
    th.update(overrides or {})
    return th


class LabelAccumulator:
    """Per-frame label maps where every pixel goes to the highest-scoring claimant.

    `add(fi, label, score, mask, prior)` competes on score + prior; ties keep the earlier
    claimant. `scores[fi][label]` keeps the best raw score seen for that label.
    """

    def __init__(self, max_side: int = 960, dtype=np.uint8):
        self.max_side = max_side
        self.dtype = dtype
        self.lab: dict[int, np.ndarray] = {}
        self.eff: dict[int, np.ndarray] = {}
        self.scores: dict[int, dict[int, float]] = {}
        self.size: tuple[int, int] | None = None   # (h, w) of stored maps

    def _fit(self, m: np.ndarray) -> np.ndarray:
        h, w = m.shape
        if self.size is None:
            sc = min(1.0, self.max_side / max(h, w)) if self.max_side else 1.0
            self.size = (max(1, round(h * sc)), max(1, round(w * sc)))
        if (h, w) != self.size:
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(self.size[::-1], Image.NEAREST)) > 0
        return m.astype(bool)

    def add(self, fi: int, label: int, score: float, mask: np.ndarray, prior: float = 0.0) -> None:
        m = self._fit(np.asarray(mask).astype(bool))
        if fi not in self.lab:
            self.lab[fi] = np.zeros(self.size, self.dtype)
            self.eff[fi] = np.full(self.size, -np.inf, np.float32)
            self.scores[fi] = {}
        eff = float(score) + prior
        win = m & (eff > self.eff[fi])
        self.lab[fi][win] = label
        self.eff[fi][win] = eff
        if m.any():
            self.scores[fi][label] = max(self.scores[fi].get(label, -np.inf), float(score))

    def frame(self, fi: int) -> np.ndarray:
        return self.lab.get(fi, np.zeros(self.size or (1, 1), self.dtype))

    def present(self, fi: int) -> dict[int, float]:
        """Best score per label that still owns >= 1 pixel after competition."""
        lab = self.lab.get(fi)
        if lab is None:
            return {}
        return {k: v for k, v in self.scores[fi].items() if (lab == k).any()}


def associate(prev_maps: dict[int, np.ndarray], new_masks: dict[int, dict[int, np.ndarray]],
              min_iou: float = 0.3) -> dict[int, int]:
    """Match new local ids to previous global ids by pooled IoU over shared frames.

    prev_maps[fi]: uint8 map of the concept's previous global ids; new_masks[fi][oid]: bool mask.
    Returns {oid: gid} for the matched ones (greedy by IoU, one-to-one).
    """
    inter: dict[tuple[int, int], int] = {}
    area_new: dict[int, int] = {}
    area_old: dict[int, int] = {}
    for fi, objs in new_masks.items():
        pm = prev_maps.get(fi)
        if pm is None:
            continue
        for g in np.unique(pm):
            if g:
                area_old[int(g)] = area_old.get(int(g), 0) + int((pm == g).sum())
        for oid, m in objs.items():
            area_new[oid] = area_new.get(oid, 0) + int(m.sum())
            for g in np.unique(pm[m]):
                if g:
                    inter[(oid, int(g))] = inter.get((oid, int(g)), 0) + int((m & (pm == g)).sum())
    cands = sorted(((i / (area_new[o] + area_old[g] - i), o, g) for (o, g), i in inter.items()), reverse=True)
    out, used = {}, set()
    for iou, o, g in cands:
        if iou >= min_iou and o not in out and g not in used:
            out[o] = g
            used.add(g)
    return out


def _windows(n: int, every: int, overlap: int, back: int):
    """(prompt_frame, first_frame, last_frame_exclusive) per window."""
    if not every or every >= n:
        return [(0, 0, n)]
    return [(s, max(0, s - back) if s else 0, min(n, s + every + overlap)) for s in range(0, n, every)]


def _stream(pred, sid, direction, start, count, thresh):
    req = dict(type="propagate_in_video", session_id=sid, propagation_direction=direction,
               start_frame_index=start, output_prob_thresh=thresh, evict_cached_frame_outputs=True)
    if count is not None:
        req["max_frame_num_to_track"] = count
    yield from pred.handle_stream_request(req)


# ---------------------------------------------------------------------------
# per-frame food/name reconciliation (shared by both modes)
# ---------------------------------------------------------------------------

def refine_frame(food: dict, dets: list, split_iou: float = 0.7, split_area: float = 0.05,
                 name_cover: float = 0.5, add_thresh: float = 0.5, min_px: int = 30) -> dict:
    """Combine generic-food instances with name detections for one frame.

    food: {gid: (score, mask)}; dets: [(gid, name, score, mask)] (gids unique across both).
    Returns {gid: (score, mask, name or None)}.
      * split: a name mask n and a food instance f with IoU(n, f) < `split_iou`, n mostly inside f
        (|n&f| / |n| >= `name_cover`) and |n&f| > `split_area` * |f| carves n&f out of f as its own
        instance (gid of the detection, named; skipped when n covers all but <= `split_area` of f). The rest of f stays "food" (merged into the largest
        carved piece if it is <= `split_area` of f).
      * naming: every other instance takes the name whose masks cover >= `name_cover` of it
        (largest coverage wins); the rest stay unnamed (None -> "food").
      * a name mask with score >= `add_thresh` that covers no instance becomes a named instance.
    """
    dets = sorted(dets, key=lambda d: -d[2])
    claimed = {g: np.zeros_like(m, bool) for g, (_, m) in food.items()}
    pieces: dict[int, list] = {}
    owner: dict[int, dict[int, int]] = {g: {} for g in food}
    for dg, name, ds, dm in dets:
        dn = int(dm.sum())
        if dn < min_px:
            continue
        for fg, (fs, fm) in food.items():
            fa = int(fm.sum())
            both = int((dm & fm).sum())
            if fa == 0 or both == 0:
                continue
            piece = dm & fm & ~claimed[fg]
            k = int(piece.sum())
            if (k <= split_area * fa or both / (fa + dn - both) >= split_iou or both / dn < name_cover
                    or fa - both <= split_area * fa):      # n covers (nearly) all of f: name it, no split
                continue
            claimed[fg] |= piece
            pieces.setdefault(dg, [ds, np.zeros_like(dm), name])[1] |= piece
            owner[fg][dg] = owner[fg].get(dg, 0) + k
    out: dict[int, list] = {}
    for fg, (fs, fm) in food.items():
        rem = fm & ~claimed[fg]
        if owner[fg] and int(rem.sum()) <= split_area * int(fm.sum()):
            pieces[max(owner[fg], key=owner[fg].get)][1] |= rem        # tiny leftover goes to its biggest piece
        elif int(rem.sum()) >= 1:
            out[fg] = [fs, rem, None]
    for dg, rec in pieces.items():
        out[dg] = rec
    by_name: dict[str, np.ndarray] = {}
    for dg, name, ds, dm in dets:
        by_name[name] = by_name[name] | dm if name in by_name else dm.copy()
    for g, rec in out.items():
        if rec[2] is not None:
            continue
        a = max(int(rec[1].sum()), 1)
        best, bn = 0.0, None
        for nm, um in by_name.items():
            c = int((rec[1] & um).sum()) / a
            if c > best:
                best, bn = c, nm
        if best >= name_cover:
            rec[2] = bn
    cover = np.zeros_like(dets[0][3], bool) if dets else None
    for rec in out.values():
        if cover is not None:
            cover |= rec[1]
    for dg, name, ds, dm in dets:
        dn = int(dm.sum())
        if dg in out or dn < min_px or ds < add_thresh or cover is None:
            continue
        if int((dm & cover).sum()) / dn < name_cover:
            out[dg] = [ds, dm & ~cover, name]
    return {g: (r[0], r[1], r[2]) for g, r in out.items()}


def _choose_name(px: dict[str, int], min_share: float = 0.3) -> str:
    """Majority name by pixels; a name needs >= min_share of the instance's pixels, else "food"."""
    tot = sum(px.values())
    named = {k: v for k, v in px.items() if k != FOOD}
    if tot and named:
        k = max(named, key=named.get)
        if named[k] >= min_share * tot:
            return k
    return FOOD


def _pk(m: np.ndarray):
    return np.packbits(m), m.shape


def _upk(p) -> np.ndarray:
    bits, shape = p
    return np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)


def _is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


class _Session:
    """Predictor + video session that can be rebuilt after a CUDA OOM (`rebuild() -> new predictor`)."""

    def __init__(self, pred, rebuild, frames_dir: Path | None):
        self.pred, self.rebuild_fn, self.frames_dir, self.sid = pred, rebuild, frames_dir, None
        self.rebuilds = 0
        if frames_dir is not None:
            self.open()

    def open(self) -> None:
        self.sid = self.pred.handle_request(dict(type="start_session", resource_path=str(self.frames_dir),
                                                 offload_video_to_cpu=True))["session_id"]

    def run(self, fn):
        """fn(pred, sid) or fn(pred); on OOM rebuild the predictor once and retry."""
        try:
            return fn(self.pred, self.sid)
        except Exception as e:
            if not (_is_oom(e) and self.rebuild_fn):
                raise
            print(f"CUDA OOM ({type(e).__name__}); rebuilding predictor and retrying", flush=True)
            self.pred = None
            _free_gpu()
            self.pred = self.rebuild_fn()
            self.rebuilds += 1
            if self.frames_dir is not None:
                self.open()
            _free_gpu()
            return fn(self.pred, self.sid)


def _emit(acc: "LabelAccumulator", fi: int, res: dict, name_px: dict, scores: dict) -> None:
    for g, (s, m, nm) in res.items():
        acc.add(fi, g, s, m)
        scores.setdefault(g, []).append(float(s))
        px = name_px.setdefault(g, {})
        px[nm or FOOD] = px.get(nm or FOOD, 0) + int(m.sum())


def _video_concept(sess: _Session, name: str, thresh: float, wins, n: int, reprompt_every: int,
                   backward_frames: int, acc: "LabelAccumulator", alloc, fixed: int | None, prior: float):
    """One concept through the video tracker. Returns {fi: {gid: (score, packed mask)}} (food tracks)
    or None (container/clutter: written straight into `acc` under the fixed label)."""
    mark = alloc.n

    def go(pred, sid):
        alloc.n = mark
        tracks: dict[int, dict[int, tuple]] = {}
        prev_maps: dict[int, np.ndarray] = {}
        for (s, lo, hi) in wins:
            pred.handle_request(dict(type="reset_session", session_id=sid))
            pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=s, text=name,
                                     output_prob_thresh=thresh))
            streams = [("forward", s, hi - s)]
            if s > lo:
                streams.append(("backward", s, s - lo))
            win: dict[int, dict[int, tuple[float, np.ndarray]]] = {}
            for direction, start, count in streams:
                cnt = None if (len(wins) == 1) else count
                for r in _stream(pred, sid, direction, start, cnt, thresh):
                    o, fi = r["outputs"], r["frame_index"]
                    for oid, p, m in zip(o["out_obj_ids"], o["out_probs"], o["out_binary_masks"]):
                        win.setdefault(fi, {})[int(oid)] = (float(p), np.asarray(m).astype(bool))
            if fixed is not None:
                for fi, objs in win.items():
                    for p, m in objs.values():
                        acc.add(fi, fixed, p, m, prior)
                continue
            shared = {fi: {oid: acc._fit(m) for oid, (_, m) in objs.items()}
                      for fi, objs in win.items() if fi in prev_maps}
            match = associate(prev_maps, shared)
            local = {}
            for oid in sorted({oid for objs in win.values() for oid in objs}):
                local[oid] = match[oid] if oid in match else alloc.take()
            for fi, objs in win.items():
                cmap = None
                for oid, (p, m) in objs.items():
                    g = local[oid]
                    mf = acc._fit(m)
                    tracks.setdefault(fi, {})[g] = (p, _pk(mf))
                    if cmap is None:
                        cmap = np.zeros(acc.size, np.uint8)
                    cmap[mf & (cmap == 0)] = g
                if cmap is not None:
                    prev_maps[fi] = cmap
            nxt = s + reprompt_every if reprompt_every else n
            prev_maps = {fi: mp for fi, mp in prev_maps.items() if fi >= nxt - backward_frames}
        return tracks

    tr = sess.run(go)
    return None if fixed is not None else tr


class _Alloc:
    def __init__(self, n: int, limit: int):
        self.n, self.limit = n, limit

    def take(self) -> int:
        if self.n >= self.limit:
            raise RuntimeError(f"more than {self.limit} food instances")
        self.n += 1
        return self.n - 1


def segment_scan(scan_dir: Path, names: list[str], mode: str = "video", food_thresh: float = FOOD_THRESH,
                 name_thresh: float = NAME_THRESH, name_thresholds_: dict | None = None,
                 aux_thresh: dict | None = None, sam31: bool = False, out_dir: Path | None = None,
                 container_prompts=CONTAINER_PROMPTS, clutter_prompts=CLUTTER_PROMPTS,
                 reprompt_every: int = 30, overlap: int = 4, backward_frames: int = 4, max_side: int = 960,
                 predictor=None, rebuild=None, max_names: int = 12, max_concepts_per_pass: int = 6,
                 max_instances: int = 40, image_side: int = 1008, simplify_names: bool = True,
                 split_iou: float = 0.7, split_area: float = 0.05, linker=None, probe=None,
                 from_raw: Path | None = None) -> dict:
    """Segment a capture: generic "food" instances + simplified-name pass (see module docstring).

    mode "video": SAM 3 video predictor (`predictor` has handle_request / handle_stream_request).
    mode "image": per-frame image model (`predictor` has set_image(PIL) -> state and
    detect(state, prompt) -> (masks (N,H,W) bool, scores (N,)) at the backend's score floor);
    instances are unique per frame and `linker(raw_maps, name_px, frame_scores)` (see make_linker)
    links them across frames in 3D. `rebuild()` returns a fresh predictor and is used to retry
    once after a CUDA OOM. `probe(fi, prompt, kind, masks, scores)` sees every raw image-mode detection.
    """
    assert mode in ("video", "image"), mode
    if from_raw is not None:
        mode = "image"
    scan_dir = Path(scan_dir)
    frames_dir = scan_dir / "rgb"
    out_dir = Path(out_dir or scan_dir / "masks")
    if from_raw is not None:                     # replay: stored raw per-frame maps instead of the predictor
        if linker is None:
            raise ValueError("from_raw needs a linker (make_linker(scan_dir))")
        raw, name_px, frame_scores = load_raw(Path(from_raw).parent if Path(from_raw).name == "raw" else from_raw)
        stems = {fi: f"{fi:06d}" for fi in raw}
        maps, meta, frame_scores, link_stats = linker(raw, name_px, frame_scores)
        prompts = prompt_names(names, max_names, simplify_names)
        nth = name_thresholds(prompts, name_thresh, name_thresholds_)
        ath = {**AUX_THRESH, **(aux_thresh or {})}
        aux = ([(nm, "container") for nm in container_prompts] + [(nm, "clutter") for nm in clutter_prompts])
        return _write_outputs(out_dir, stems, maps, meta, frame_scores, prompts, nth, aux, ath, "image",
                              food_thresh, link_stats, 0)
    frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    n = len(frame_files)
    stems = [p.stem for p in frame_files]
    prompts = prompt_names(names, max_names, simplify_names)
    nth = name_thresholds(prompts, name_thresh, name_thresholds_)
    ath = {**AUX_THRESH, **(aux_thresh or {})}
    aux = ([(nm, "container") for nm in container_prompts] + [(nm, "clutter") for nm in clutter_prompts])
    fixed_of = {"container": CONTAINER, "clutter": CLUTTER}

    pred = predictor
    if pred is None and mode == "video":
        if sam31:
            from sam3.model_builder import build_sam3_multiplex_video_predictor
            pred = build_sam3_multiplex_video_predictor()
        else:
            from sam3.model_builder import build_sam3_video_predictor
            pred = build_sam3_video_predictor(gpus_to_use=[0], async_loading_frames=True)
    elif pred is None:
        pred = Sam3ImageBackend(min_thresh=min([food_thresh, *nth.values(), *ath.values()]))
    sess = _Session(pred, rebuild, frames_dir if mode == "video" else None)

    acc = LabelAccumulator(max_side, np.uint16 if mode == "image" else np.uint8)
    name_px: dict[int, dict[str, int]] = {}
    gscores: dict[int, list[float]] = {}
    if mode == "video":
        alloc = _Alloc(1, CLUTTER)
        wins = _windows(n, reprompt_every, overlap, backward_frames)
        # image size is only known after the first mask; fix it from the first frame file
        w0, h0 = Image.open(frame_files[0]).size
        acc._fit(np.zeros((h0, w0), bool))
        tracks = {FOOD: _video_concept(sess, FOOD, food_thresh, wins, n, reprompt_every, backward_frames,
                                       acc, alloc, None, 0.0)}
        print(f"concept '{FOOD}' (food, thresh {food_thresh}): "
              f"{len({g for t in tracks[FOOD].values() for g in t})} tracks")
        for i0 in range(0, len(prompts), max(1, max_concepts_per_pass)):
            for nm in prompts[i0:i0 + max(1, max_concepts_per_pass)]:
                tracks[nm] = _video_concept(sess, nm, nth[nm], wins, n, reprompt_every, backward_frames,
                                            acc, alloc, None, 0.0)
                print(f"concept '{nm}' (name, thresh {nth[nm]}): "
                      f"{len({g for t in tracks[nm].values() for g in t})} tracks")
            _free_gpu()
        for nm, kind in aux:
            _video_concept(sess, nm, ath[kind], wins, n, reprompt_every, backward_frames, acc, alloc,
                           fixed_of[kind], KIND_PRIOR[kind])
            print(f"concept '{nm}' ({kind}, thresh {ath[kind]}): class")
        for fi in range(n):
            food = {g: (p, _upk(m)) for g, (p, m) in tracks[FOOD].get(fi, {}).items()}
            dets = [(g, nm, p, _upk(m)) for nm in prompts for g, (p, m) in tracks[nm].get(fi, {}).items()]
            if food or dets:
                _emit(acc, fi, refine_frame(food, dets, split_iou, split_area), name_px, gscores)
        sess.pred.handle_request(dict(type="close_session", session_id=sess.sid))
    else:
        raw = {"next": FIRST_RAW_ID}

        def take() -> int:
            raw["next"] += 1
            return raw["next"] - 1

        for fi, fpath in enumerate(frame_files):
            def go(backend, _sid=None, fi=fi, fpath=fpath):
                img = Image.open(fpath).convert("RGB")
                sc = image_side / max(img.size)
                if sc < 1:
                    img = img.resize((round(img.width * sc), round(img.height * sc)), Image.BILINEAR)
                st = backend.set_image(img)

                def det(prompt, kind, thr):
                    ms, ss = backend.detect(st, prompt)
                    ms = np.asarray(ms).astype(bool)
                    ss = np.asarray(ss, np.float32)
                    if probe is not None and len(ss):
                        probe(fi, prompt, kind, ms, ss)
                    keep = np.argsort(-ss)[:max_instances]
                    return [(float(ss[i]), acc._fit(ms[i])) for i in keep if ss[i] >= thr]
                return ([(s, m) for s, m in det(FOOD, "food", food_thresh)],
                        {nm: det(nm, "name", nth[nm]) for nm in prompts},
                        [(k, s, m) for nm, k in aux for s, m in det(nm, k, ath[k])])

            food_l, name_d, aux_d = sess.run(go)
            for k, s, m in aux_d:
                acc.add(fi, fixed_of[k], s, m, KIND_PRIOR[k])
            food = {take(): (s, m) for s, m in food_l}
            dets = [(take(), nm, s, m) for nm, lst in name_d.items() for s, m in lst]
            if food or dets:
                _emit(acc, fi, refine_frame(food, dets, split_iou, split_area), name_px, gscores)
            if fi % 6 == 5:
                _free_gpu()

    if hasattr(sess.pred, "shutdown"):
        sess.pred.shutdown()

    frame_scores = {fi: acc.present(fi) for fi in range(n)}
    maps = {fi: acc.frame(fi) for fi in range(n)}
    link_stats = None
    if mode == "image":
        if linker is None:
            raise ValueError("image mode needs a linker (make_linker(scan_dir)) to give instances consistent ids")
        raw_dir = out_dir / "raw"          # unlinked per-frame ids: lets link_instances be re-tuned on CPU
        raw_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(n):
            Image.fromarray(maps[fi]).save(raw_dir / f"{stems[fi]}.png")
        (raw_dir / "raw.json").write_text(json.dumps(
            {"name_px": {str(k): v for k, v in name_px.items()},
             "frame_scores": {str(fi): {str(k): v for k, v in d.items()} for fi, d in frame_scores.items()}}))
        maps, meta, frame_scores, link_stats = linker(maps, name_px, frame_scores)
    else:
        meta = {}
        for g, px in name_px.items():
            meta[g] = {"name": _choose_name(px), "concept_index": 0, "mean_score": float(np.mean(gscores[g])),
                       "kind": "food"}
    return _write_outputs(out_dir, stems, maps, meta, frame_scores, prompts, nth, aux, ath, mode,
                          food_thresh, link_stats, sess.rebuilds)


def _write_outputs(out_dir, stems, maps, meta, frame_scores, prompts, nth, aux, ath, mode, food_thresh,
                   link_stats, rebuilds) -> dict:
    """masks/NNNNNN.png|json, instances.json, concepts.json for (already linked) maps; returns meta."""
    fixed_of = {"container": CONTAINER, "clutter": CLUTTER}
    table = [{"name": FOOD, "kind": "food", "concept_index": 0, "thresh": food_thresh,
              "labels": sorted(g for g, v in meta.items() if v["name"] == FOOD)}]
    for i, nm in enumerate(prompts, 1):
        table.append({"name": nm, "kind": "name", "concept_index": i, "thresh": nth[nm],
                      "labels": sorted(g for g, v in meta.items() if v["name"] == nm)})
    for j, (nm, kind) in enumerate(aux):
        table.append({"name": nm, "kind": kind, "concept_index": len(prompts) + 1 + j, "thresh": ath[kind],
                      "labels": [fixed_of[kind]]})
    out_dir.mkdir(parents=True, exist_ok=True)
    for fi in sorted(maps):
        Image.fromarray(maps[fi]).save(out_dir / f"{stems[fi]}.png")
        (out_dir / f"{stems[fi]}.json").write_text(json.dumps({str(k): v for k, v in frame_scores[fi].items()}))
    (out_dir / "instances.json").write_text(json.dumps({str(k): v for k, v in meta.items()}, indent=1))
    (out_dir / "concepts.json").write_text(json.dumps(
        {"clutter_label": CLUTTER, "container_label": CONTAINER, "mode": mode, "food_thresh": food_thresh,
         "link": link_stats, "oom_rebuilds": rebuilds, "concepts": table}, indent=1, default=str))
    named = sum(v["name"] != FOOD for v in meta.values())
    print(f"{mode}: {len(meta)} food instances ({named} named), prompts {prompts}")
    return meta


# ---------------------------------------------------------------------------
# image mode: SAM 3 image model + 3D linking
# ---------------------------------------------------------------------------

class Sam3ImageBackend:
    """SAM 3 image model (fp16 on the 6 GB Turing card, as eval/sam3_gap.py). detect() returns every
    mask above `min_thresh`, highest score first, capped at `cap`."""

    def __init__(self, min_thresh: float = 0.15, cap: int = 60, device: str = "cuda"):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        model = build_sam3_image_model(device="cpu")
        model.backbone.half()
        self.proc = Sam3Processor(model.to(device), device=device, confidence_threshold=min_thresh)
        self.cap = cap

    def set_image(self, img):
        import torch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return self.proc.set_image(img)

    def detect(self, state, prompt: str):
        import torch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o = self.proc.set_text_prompt(state=state, prompt=prompt)
        if not len(o["scores"]):
            return np.zeros((0, *state["masks"].shape[-2:]), bool), np.zeros(0, np.float32)
        m = o["masks"].reshape(-1, *o["masks"].shape[-2:]).bool().cpu().numpy()
        s = o["scores"].float().cpu().numpy()
        top = np.argsort(-s)[:self.cap]
        return m[top], s[top]

    def shutdown(self):
        self.proc = None
        _free_gpu()


def link_instances(scan, raw_maps: dict, name_px: dict, frame_scores: dict, points: np.ndarray,
                   normals: np.ndarray | None = None, overlap_thresh: float = 0.5, min_support: int = 1,
                   min_points: int = 8, max_points: int = 150000, max_ids: int = 252, min_frames: int = 2,
                   method: str = "covote", cluster_thresh: float = 0.8, **kw):
    """Give per-frame-unique instance ids (>= FIRST_RAW_ID, 253/254 = clutter/container) consistent ids.

    Lifts every frame's raw map onto `points` (labels.collect_votes) and merges ids with
    `method`: "covote" (default; labels.cluster_ids: average-linkage clustering of the id x id co-vote
    cosine matrix cut at distance `cluster_thresh` = 0.8, the results/lift_sweep winner) or "repair"
    (labels.repair_ids, view consensus: two masks from different frames sharing > `overlap_thresh` of
    the smaller one's 3D points link; same-frame ids never merge). min_support=1 because per-frame
    ids give each id pair a single node pair. Returns (uint8 maps, meta, frame_scores, stats);
    clusters seen in fewer than `min_frames` frames (one-frame noise) and clusters beyond `max_ids`
    (fewest frames first) are dropped to background.
    """
    from . import labels as L
    big = 65001
    work = {}
    for fi, m in raw_maps.items():
        w = m.astype(np.int32)
        w[m == CLUTTER], w[m == CONTAINER] = big, big + 1
        work[fi] = w
    old = L.CLUTTER
    L.CLUTTER = big                       # repair_ids treats ids >= CLUTTER as aux; raw food ids exceed 253
    try:
        fvs = L.collect_votes(scan, work, points, normals)
        if method == "covote":
            remap, st = L.cluster_ids(fvs, len(points), thresh=cluster_thresh, sim="cosine", min_points=min_points,
                                      max_points=max_points, aux_ids={big, big + 1}, **kw)
        elif method == "repair":
            remap, st = L.repair_ids(fvs, len(points), None, overlap_thresh=overlap_thresh, min_points=min_points,
                                     min_support=min_support, max_points=max_points, **kw)
        else:
            raise ValueError(f"unknown linker {method!r}; use 'covote' or 'repair'")
    finally:
        L.CLUTTER = old
    root = lambda g: remap.get(g, g)
    px: dict[int, int] = {}
    frs: dict[int, set] = {}
    for fi, m in raw_maps.items():
        ids, c = np.unique(m[m > 0], return_counts=True)
        for g, k in zip(ids, c):
            if g in (CLUTTER, CONTAINER):
                continue
            px[int(g)] = px.get(int(g), 0) + int(k)
            frs.setdefault(root(int(g)), set()).add(fi)
    csize: dict[int, int] = {}
    for g, k in px.items():
        csize[root(g)] = csize.get(root(g), 0) + k
    order = sorted((r for r in csize if len(frs[r]) >= min_frames), key=lambda r: (-len(frs[r]), -csize[r]))[:max_ids]
    final = {r: i + 1 for i, r in enumerate(order)}
    lut = np.zeros(65536, np.uint8)
    lut[CLUTTER], lut[CONTAINER] = CLUTTER, CONTAINER
    for g in px:
        lut[g] = final.get(root(g), 0)
    maps = {fi: lut[m] for fi, m in raw_maps.items()}
    meta, merged_px = {}, {}
    for g, k in px.items():
        r = root(g)
        if r in final:
            d = merged_px.setdefault(final[r], {})
            for nm, v in name_px.get(g, {}).items():
                d[nm] = d.get(nm, 0) + v
    for r, f in final.items():
        meta[f] = {"name": _choose_name(merged_px.get(f, {})), "concept_index": 0, "kind": "food",
                   "n_frames": len(frs[r]), "n_raw": sum(1 for g in px if root(g) == r), "mean_score": 0.0}
    fsc = {}
    sc_acc: dict[int, list[float]] = {}
    for fi, d in frame_scores.items():
        o = {}
        for g, s in d.items():
            f = int(lut[g]) if g < 65536 else 0
            if f:
                o[f] = max(o.get(f, -1.0), s)
                sc_acc.setdefault(f, []).append(s)
        fsc[fi] = o
    for f in meta:
        meta[f]["mean_score"] = float(np.mean(sc_acc.get(f, [0.0])))
    stats = {"linker": method, "raw_ids": len(px), "clusters": len(final), "dropped": len(csize) - len(final),
             "repair": st}
    return maps, meta, fsc, stats


def load_raw(out_dir: Path):
    """Read the unlinked image-mode maps written next to the linked ones: (raw_maps, name_px, frame_scores)."""
    raw_dir = Path(out_dir) / "raw"
    j = json.loads((raw_dir / "raw.json").read_text())
    maps = {int(p.stem): np.asarray(Image.open(p)) for p in raw_dir.glob("[0-9]*.png")}
    return (maps, {int(k): v for k, v in j["name_px"].items()},
            {int(fi): {int(k): v for k, v in d.items()} for fi, d in j["frame_scores"].items()})


def make_linker(scan_dir: Path, points: np.ndarray | None = None, normals: np.ndarray | None = None,
                voxel: float = 0.003, **kw):
    """linker for segment_scan(mode="image"): fuses the scan unless points/normals are given."""
    from .loader import Scan
    scan = Scan(Path(scan_dir))
    if points is None:
        from .fuse import fuse
        _, pcd = fuse(scan, voxel=voxel, stride=2)
        pcd.estimate_normals()
        points, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    return lambda raw, npx, fs: link_instances(scan, raw, npx, fs, points, normals, **kw)


def load_masks(scan_dir: Path, mdir: Path | None = None) -> tuple[dict[int, np.ndarray], dict]:
    """Read masks written by segment_scan into the {frame_index: label_map} form labels.py wants.

    Old-format directories (PNG + instances.json only) read fine; extras are optional.
    """
    mdir = Path(mdir) if mdir is not None else Path(scan_dir) / "masks"
    masks = {int(p.stem): np.asarray(Image.open(p)) for p in mdir.glob("*.png")}
    meta = json.loads((mdir / "instances.json").read_text())
    return masks, {int(k): v for k, v in meta.items()}


def load_scores(scan_dir: Path) -> dict[int, dict[int, float]]:
    """Per-frame {label: best_score} from masks/NNNNNN.json (empty if absent)."""
    mdir = Path(scan_dir) / "masks"
    return {int(p.stem): {int(k): v for k, v in json.loads(p.read_text()).items()}
            for p in mdir.glob("[0-9]*.json")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--names", default="", help="comma-separated food names (from the VLM); optional")
    ap.add_argument("--mode", choices=["video", "image"], default="video")
    ap.add_argument("--food-thresh", type=float, default=FOOD_THRESH)
    ap.add_argument("--name-thresh", type=float, default=NAME_THRESH)
    ap.add_argument("--max-names", type=int, default=12)
    ap.add_argument("--sam31", action="store_true", help="use SAM 3.1 Object Multiplex predictor")
    ap.add_argument("--reprompt-every", type=int, default=30, help="re-prompt every N frames (0 = once)")
    ap.add_argument("--max-side", type=int, default=960, help="stored mask long side (0 = native)")
    ap.add_argument("--no-aux", action="store_true", help="skip container/clutter prompts")
    ap.add_argument("--linker", choices=["repair", "covote"], default="covote",
                    help="image mode: how per-frame instances get consistent ids (covote = co-vote clustering, "
                         "default; repair = labels.repair_ids view consensus); video mode ignores it")
    ap.add_argument("--cluster-thresh", type=float, default=0.8, help="covote: average-linkage cut (distance)")
    ap.add_argument("--from-masks", default=None,
                    help="replay: a masks dir holding raw/ (unlinked image-mode maps); no predictor/GPU is used")
    ap.add_argument("--out-dir", default=None, help="where to write masks (default SCAN_DIR/masks)")
    ap.add_argument("--points-npz", default=None,
                    help="cloud (xyz [+ normal]) to link/lift on instead of fusing the scan (e.g. gt/seg_points.npz)")
    ap.add_argument("--lift-out", default=None,
                    help="also write lift.npz here with labels.lift_for_meal (the call run_meal makes)")
    ap.add_argument("--weighting", default=None, help="lift vote weighting (default cos+qual for covote, else cos)")
    args = ap.parse_args()
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    sd = Path(args.scan_dir)
    pts = nrm = None
    if args.points_npz:
        z = np.load(args.points_npz)
        pts = z["xyz"].astype(np.float64)
        nrm = z["normal"].astype(np.float64) if "normal" in z.files else None
    image = args.mode == "image" or args.from_masks is not None
    linker = (make_linker(sd, pts, nrm, method=args.linker, cluster_thresh=args.cluster_thresh)
              if image else None)
    meta = segment_scan(sd, names, mode=args.mode, from_raw=Path(args.from_masks) if args.from_masks else None,
                        out_dir=Path(args.out_dir) if args.out_dir else None, food_thresh=args.food_thresh, name_thresh=args.name_thresh,
                        sam31=args.sam31, reprompt_every=args.reprompt_every, max_side=args.max_side,
                        max_names=args.max_names,
                        container_prompts=() if args.no_aux else CONTAINER_PROMPTS,
                        clutter_prompts=() if args.no_aux else CLUTTER_PROMPTS,
                        linker=linker)
    print(json.dumps(meta, indent=1))
    if args.lift_out:
        from . import labels as L
        from .loader import Scan
        out = Path(args.out_dir) if args.out_dir else sd / "masks"
        masks, meta = load_masks(out.parent, out)
        scan = Scan(sd)
        if pts is None:
            from .fuse import fuse
            _, pcd = fuse(scan, voxel=0.003, stride=2)
            pts = np.asarray(pcd.points)
            nrm = np.asarray(pcd.normals) if pcd.has_normals() else L.estimate_normals(pts)
        if nrm is None:
            nrm = L.estimate_normals(pts)
        w = args.weighting or (L.DEFAULT_LINKED_WEIGHTING if image and args.linker == "covote" else "cos")
        L.lift_for_meal(scan, masks, pts, nrm, meta, weighting=w, save_to=args.lift_out)


if __name__ == "__main__":
    main()
