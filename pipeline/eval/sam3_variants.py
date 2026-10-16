"""Does a better 2D mask source raise the lifted-label ceiling? SAM 3 vs SAM 3.1 image mode, 3D-guided prompts, low-threshold union.

Run from pipeline/ in the `foodseg` env. Stages (GPU ones under flock):

    python -m eval.sam3_variants prep                     # CPU: reference A lift (stored masks) + 3D-guided prompts for C
    python -m eval.sam3_variants gpu --model sam3 --c     # GPU: all text detections >= 0.15 + box/point prompts for C
    python -m eval.sam3_variants gpu --model sam31        # GPU: same text detections with the SAM 3.1 detector
    python -m eval.sam3_variants replay [--procs 3]       # CPU: variants -> lift -> metrics (results/sam3_variants/results.json)
    python -m eval.sam3_variants report                   # CPU: tables.md

Variants (12 scans of results/lift_errors/scans40 with the most fragmented/merged items, 6 per batch):
  A   stored hard_v12/13 masks_sam3_v2i raw maps, co-vote linker + cos+qual lift (= results/lift_errors/scans40 replay)
  A'  the same pipeline fed by this script's fresh SAM 3 detections (run-to-run sanity check, fp16 noise)
  B   SAM 3.1 detector (facebook/sam3.1 sam3.1_multiplex.pt, `detector.*` weights loaded into the SAM 3 image model)
  C   3D-guided second pass: every A-lifted instance is projected into every frame; the SAM 3 interactive predictor
      (box + positive points + negatives at the neighbouring instances) gives one mask per instance per frame; the
      masks carry the lifted id, so no 2D linking is needed. C-norepair skips labels.repair_ids in the lift.
  D   union of text detections down to 0.15 (food and every name), usual aux thresholds;
      D = cos+qual weights, D-score = cos+score+qual (per-mask SAM score as vote weight).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline"))

OUT = REPO / "results" / "sam3_variants"
WORK = OUT / "work"
SIDS12 = ["v12/s000008", "v12/s000010", "v12/s000003", "v12/s000019", "v12/s000001", "v12/s000013",
          "v13/s000004", "v13/s000017", "v13/s000018", "v13/s000010", "v13/s000015", "v13/s000006"]
# held-out confirmation set: the first six scans per batch NOT in SIDS12 (not chosen on errors); SAM 3 text detections only
SIDS_CONFIRM = ["v12/s000000", "v12/s000002", "v12/s000004", "v12/s000005", "v12/s000006", "v12/s000007",
                "v13/s000000", "v13/s000001", "v13/s000002", "v13/s000003", "v13/s000005", "v13/s000007"]
FLOOR = 0.15
AUX_STORE = 0.5            # the stored run used 0.5 for container / clutter prompts
SIZE = (720, 960)          # stored label-map size (segment_scan max_side=960)
IMAGE_SIDE = 1008
MIN_PTS_PROMPT = 25        # visible points of an instance in a frame before it gets a prompt
MIN_INST_POINTS = 100
CKPT31 = "sam3.1_multiplex.pt"

VARIANTS = {
    # name: (source, text thresholds, lift weighting, repair)
    "A": ("stored", None, "cos+qual", True),
    "A'": ("det_sam3", "base", "cos+qual", True),
    "B": ("det_sam31", "base", "cos+qual", True),
    "C": ("cmasks", None, "cos+qual", True),
    "C-norepair": ("cmasks", None, "cos+qual", False),
    "D": ("det_sam3", "union", "cos+qual", True),
    "D-score": ("det_sam3", "union", "cos+score+qual", True),
}


def tag(sid: str) -> str:
    return sid.replace("/", "__")


def roots(sid: str):
    from eval.lift_sweep import roots as r
    return r(sid)


def oracle_names(sid: str) -> list[str]:
    from eval.sam3_synth_votes import concepts_for
    return concepts_for(roots(sid)[1], "names")


def frame_files(sd: Path):
    return sorted((sd / "rgb").glob("*.jpg"), key=lambda p: int(p.stem))


# ---------------------------------------------------------------------------------------------- pure helpers
def pack(m: np.ndarray) -> np.ndarray:
    return np.packbits(m.reshape(len(m), -1), axis=1)


def unpack(p: np.ndarray, shape=SIZE) -> np.ndarray:
    return np.unpackbits(p, axis=1, count=shape[0] * shape[1]).reshape(len(p), *shape).astype(bool)


def to_store(masks: np.ndarray) -> np.ndarray:
    """(n, H, W) bool at image resolution -> (n, 720, 960) bool, same NEAREST resize as LabelAccumulator._fit."""
    if not len(masks):
        return np.zeros((0, *SIZE), bool)
    if masks.shape[1:] == SIZE:
        return masks.astype(bool)
    return np.stack([np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(SIZE[::-1], Image.NEAREST)) > 0
                     for m in masks])


def instance_prompt(px: np.ndarray, others: list[np.ndarray], pad: float = 0.06, n_pos: int = 3, n_neg: int = 3,
                    shape=SIZE) -> dict:
    """Box + positive points + negatives for one instance in one frame, in the pixel frame of `px`.

    px: (n, 2) visible (x, y) pixels of the instance; others: pixel arrays of the neighbouring instances.
    Box = 2-98 percentile range padded by `pad`; positives = the pixel nearest the median, then up to n_pos - 1
    farthest-point picks within 35 % of the box diagonal of it (so they stay on the item); negatives = for the
    neighbours with the most pixels inside the box, the neighbour pixel nearest its in-box median.
    """
    x0, y0 = np.percentile(px, 2, axis=0)
    x1, y1 = np.percentile(px, 98, axis=0)
    w, h = max(x1 - x0, 4.0), max(y1 - y0, 4.0)
    box = [max(0.0, x0 - pad * w), max(0.0, y0 - pad * h),
           min(shape[1] - 1.0, x1 + pad * w), min(shape[0] - 1.0, y1 + pad * h)]
    med = np.median(px, axis=0)
    p1 = px[np.argmin(((px - med) ** 2).sum(1))]
    pos = [p1]
    diag = float(np.hypot(w, h))
    cand = px[((px - p1) ** 2).sum(1) < (0.35 * diag) ** 2]
    while len(pos) < n_pos and len(cand) > 1:
        d = np.min([((cand - q) ** 2).sum(1) for q in pos], axis=0)
        if d.max() < (0.08 * diag) ** 2:
            break
        pos.append(cand[np.argmax(d)])
    ins = []
    for o in others:
        if not len(o):
            continue
        k = (o[:, 0] >= box[0]) & (o[:, 0] <= box[2]) & (o[:, 1] >= box[1]) & (o[:, 1] <= box[3])
        if k.sum():
            ins.append((int(k.sum()), o[k]))
    neg = []
    for _, o in sorted(ins, key=lambda t: -t[0])[:n_neg]:
        m = np.median(o, axis=0)
        neg.append(o[np.argmin(((o - m) ** 2).sum(1))])
    return {"box": [float(v) for v in box], "pos": [[float(a), float(b)] for a, b in pos],
            "neg": [[float(a), float(b)] for a, b in neg]}


def compose_instance_maps(items: list[tuple[int, float, np.ndarray]], base: np.ndarray | None = None) -> np.ndarray:
    """Per-frame uint8 label map from (id, score, bool mask) instance masks: each pixel goes to the highest score;
    instance masks overwrite the aux (253/254) pixels of `base`."""
    out = np.zeros(SIZE, np.uint8) if base is None else base.copy()
    eff = np.full(SIZE, -np.inf, np.float32)
    for k, s, m in items:
        win = m & (s > eff)
        out[win] = k
        eff[win] = s
    return out


def count_errors(gi: np.ndarray, pi: np.ndarray, frac: float = 0.20) -> tuple[int, int, int]:
    """(fragmented, merged, n_items) with the lift_errors definitions on a voxel scene: fragmented = >= 2 predicted ids
    each hold > frac of the item; merged = an id holding > frac of the item also holds > frac of another item."""
    items = [g for g in np.unique(gi) if g > 0]
    if not items:
        return 0, 0, 0
    ids = [i for i in np.unique(pi) if i > 0]
    share = np.zeros((len(items), max(len(ids), 1)))
    for a, g in enumerate(items):
        m = gi == g
        p = pi[m]
        for b, i in enumerate(ids):
            share[a, b] = (p == i).sum() / m.sum()
    big = share > frac
    frag = int((big.sum(1) >= 2).sum())
    n_items_per_id = big.sum(0)
    merged = int(sum(bool((big[a] & (n_items_per_id >= 2)).any()) for a in range(len(items))))
    return frag, merged, len(items)


# ---------------------------------------------------------------------------------------------- scene / metrics
_SCENE: dict = {}


class _Scene:
    pass


def scene(sid: str):
    if sid in _SCENE:
        return _SCENE[sid]
    from seg3d.data import load_scan, semantic_of, voxelize
    d = load_scan(roots(sid)[0], "manifest")
    vi, _ = voxelize(d["xyz"], 2e-3)
    lab = d["label"][vi]
    sem = semantic_of(lab)
    s = _Scene()
    s.vi = vi
    s.sc = dict(xyz=d["xyz"][vi].astype(np.float64), label=lab, sem=sem,
                inst=np.where(sem == 3, lab, 0).astype(np.int64),
                ignore=np.isin(lab, d["ignore_ids"]) if d["ignore_ids"] else np.zeros(len(vi), bool))
    _SCENE.clear()
    _SCENE[sid] = s
    return s


def evaluate(sid: str, lift_full: np.ndarray) -> dict:
    from eval.lift_sweep import evaluate as ev
    from seg3d import metrics as M
    s = scene(sid)
    r = ev(s, lift_full)
    sel = ~s.sc["ignore"]
    _, pi = M.lift_baseline_pred(np.asarray(lift_full)[s.vi][sel])
    r["frag20"], r["merged20"], _ = count_errors(s.sc["inst"][sel], pi)
    r["pq"] = float(r["tp_iou"] / max(r["tp"] + .5 * r["fp"] + .5 * r["fn"], 1e-9))
    r["inst"] = float(np.mean(r["iou"])) if r["iou"] else float("nan")
    return r


def stats2d(sid: str, raw_dir: Path) -> dict:
    """Single-frame item stats of raw per-frame id maps against gt/seg_ids: hit rates, mean best IoU, 2D fragments/merges."""
    sd = roots(sid)[0]
    best, frag, merge, n = [], 0, 0, 0
    for p in sorted(raw_dir.glob("[0-9]*.png")):
        g = sd / "gt" / "seg_ids" / f"{int(p.stem):06d}.png"
        if not g.exists():
            continue
        m = np.asarray(Image.open(p)).astype(np.int32)
        gt = np.asarray(Image.open(g))
        if gt.shape != m.shape:
            gt = np.asarray(Image.fromarray(gt).resize(m.shape[::-1], Image.NEAREST))
        food = (m > 0) & (m != 253) & (m != 254)
        gids = [i for i in np.unique(gt[(gt > 0) & (gt < 252)]) if (gt == i).sum() >= 150]
        cov = {}
        for gi in gids:
            gm = gt == gi
            cov[gi] = (int(gm.sum()), {int(i): int(((m == i) & gm).sum()) for i in np.unique(m[gm & food])})
        for gi in gids:
            sz, row = cov[gi]
            b = 0.0
            for i, inter in row.items():
                b = max(b, inter / (sz + int((m == i).sum()) - inter))
            best.append(b)
            big = [i for i, v in row.items() if v > 0.2 * sz]
            frag += len(big) >= 2
            merge += any(sum(1 for g2 in gids if g2 != gi and cov[g2][1].get(i, 0) > 0.2 * cov[g2][0]) for i in big)
            n += 1
    b = np.array(best) if best else np.zeros(1)
    return dict(n=n, hit30=float((b >= .3).mean()), hit50=float((b >= .5).mean()), miou=float(b.mean()),
                frag2d=frag / max(n, 1), merge2d=merge / max(n, 1))


# ---------------------------------------------------------------------------------------------- GPU backend
class Backend:
    """SAM 3 / SAM 3.1 image detector (fp16 backbone as Sam3ImageBackend) + optional SAM 3 interactive predictor."""

    def __init__(self, version: str, interactive: bool, device: str = "cuda", floor: float = FLOOR):
        import glob
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        kw = dict(device="cpu", enable_inst_interactivity=interactive)
        if version == "sam31":
            pat = str(Path.home() / ".cache/huggingface/hub/models--facebook--sam3.1/snapshots/*" / CKPT31)
            kw.update(checkpoint_path=glob.glob(pat)[0], load_from_HF=False)
        model = build_sam3_image_model(**kw)
        self.device = device
        if device == "cuda":
            model.backbone.half()
        else:                                    # CPU smoke tests only: addmm_act hard-codes bf16
            model.backbone.to(torch.bfloat16)
        self.proc = Sam3Processor(model.to(device), device=device, confidence_threshold=floor)
        self.interactive = interactive
        self.torch = torch

    def _ac(self):
        t = self.torch
        return t.autocast("cuda", dtype=t.float16) if self.device == "cuda" else t.autocast("cpu", dtype=t.bfloat16)

    def set_image(self, img):
        with self.torch.no_grad(), self._ac():
            return self.proc.set_image(img)

    def detect(self, state, prompt: str, cap: int = 60):
        with self.torch.no_grad(), self._ac():
            o = self.proc.set_text_prompt(state=state, prompt=prompt)
        if not len(o["scores"]):
            return np.zeros((0, *state["masks"].shape[-2:]), bool), np.zeros(0, np.float32)
        m = o["masks"].reshape(-1, *o["masks"].shape[-2:]).bool().cpu().numpy()
        s = o["scores"].float().cpu().numpy()
        top = np.argsort(-s)[:cap]
        return m[top], s[top]

    def predict(self, state, box, pos, neg):
        """One mask for box + positive + negative points (pixels of the set image); fp32 tracker heads."""
        st = dict(state)
        bo = dict(state["backbone_out"])
        sb = dict(bo["sam2_backbone_out"])
        sb["backbone_fpn"] = [f.float() for f in sb["backbone_fpn"]]
        sb["vision_pos_enc"] = [f.float() for f in sb["vision_pos_enc"]]
        bo["sam2_backbone_out"] = sb
        st["backbone_out"] = bo
        pts = np.array(list(pos) + list(neg), np.float32)
        lab = np.array([1] * len(pos) + [0] * len(neg), np.int32)
        with self.torch.no_grad():
            masks, ious, _ = self.proc.model.predict_inst(st, point_coords=pts, point_labels=lab,
                                                          box=np.array(box, np.float32), multimask_output=False)
        return np.asarray(masks)[0].astype(bool), float(np.asarray(ious).reshape(-1)[0])


# ---------------------------------------------------------------------------------------------- stage: prep (CPU)
class ReplayBackend:
    """Stands in for the image backend in segment_scan: serves the detections stored by the GPU stage."""

    def __init__(self, path: Path):
        self.z = np.load(path)
        self.keys = set(self.z.files)
        self.fi = -1

    def set_image(self, img):
        self.fi += 1
        return self.fi

    def detect(self, state, prompt):
        k = f"{state}|{prompt}"
        if k + "|s" not in self.keys:
            return np.zeros((0, *SIZE), bool), np.zeros(0, np.float32)
        return unpack(self.z[k + "|m"]), self.z[k + "|s"].astype(np.float32)


def replay_lift(sid: str, variant: str):
    """Link (A, A', B, D*) or take (C*) per-frame maps, lift. Writes work/<variant>/<tag>/. -> (lift_label, dir)."""
    from foodscan import labels as L
    from foodscan.loader import Scan
    from foodscan.segment import (FOOD_THRESH, NAME_THRESH, load_masks, make_linker, prompt_names, segment_scan)
    sd, rawsd = roots(sid)
    z = np.load(sd / "gt/seg_points.npz")
    xyz, nrm = z["xyz"].astype(np.float64), z["normal"].astype(np.float64)
    src, thr, weighting, repair = VARIANTS[variant]
    od = WORK / variant.replace("'", "p") / tag(sid)
    od.mkdir(parents=True, exist_ok=True)
    scores = None
    if src == "cmasks":
        masks, meta, scores = compose_c(sid)
        (od / "raw").mkdir(exist_ok=True)
        for fi, m in masks.items():
            Image.fromarray(m).save(od / "raw" / f"{fi:06d}.png")
    else:
        link = make_linker(sd, xyz, nrm, method="covote", cluster_thresh=0.8)
        if src == "stored":
            segment_scan(sd, [], from_raw=rawsd / "masks_sam3_v2i", out_dir=od / "masks", linker=link)
        else:
            names = oracle_names(sid)
            prompts = prompt_names(names, 12, True)
            if thr == "union":
                kw = dict(food_thresh=FLOOR, name_thresh=FLOOR, name_thresholds_={p: FLOOR for p in prompts})
            else:
                kw = dict(food_thresh=FOOD_THRESH, name_thresh=NAME_THRESH)
            segment_scan(sd, names, mode="image", predictor=ReplayBackend(WORK / src / f"{tag(sid)}.npz"), linker=link,
                         out_dir=od / "masks", max_side=960, aux_thresh={"container": AUX_STORE, "clutter": AUX_STORE}, **kw)
            if (od / "masks" / "raw").exists():
                (od / "raw").mkdir(exist_ok=True)
                for p in (od / "masks" / "raw").glob("[0-9]*.png"):
                    p.replace(od / "raw" / p.name)
        masks, meta = load_masks(od, od / "masks")
        if "score" in weighting:
            scores = {int(p.stem): {int(k): v for k, v in json.loads(p.read_text()).items()}
                      for p in (od / "masks").glob("[0-9]*.json")}
    lift = L.lift_for_meal(Scan(sd), masks, xyz, nrm, meta, weighting=weighting, repair=repair, scores=scores,
                           save_to=od / "lift.npz")
    return lift.lift_label, od


def c_prompts(sid: str, lift_label: np.ndarray) -> dict:
    """Per frame, one prompt per A-lifted instance (box + positives + negatives), pixel coords of the 960x720 maps."""
    from foodscan import labels as L
    from foodscan.loader import Scan
    sd = roots(sid)[0]
    z = np.load(sd / "gt/seg_points.npz")
    xyz, nrm = z["xyz"].astype(np.float64), z["normal"].astype(np.float64)
    ids = [int(i) for i in np.unique(lift_label) if 0 < i < 252 and (lift_label == i).sum() >= MIN_INST_POINTS]
    lab = np.where(np.isin(lift_label, ids), lift_label, 0)
    keep = np.nonzero(lab > 0)[0]
    out = {}
    for fr in Scan(sd).frames:
        idx, mu, mv, _ = L.visible_indices(fr, xyz[keep], SIZE, depth_tol=0.01, min_conf=1, normals=nrm[keep],
                                           max_grazing_deg=75.0)
        if not len(idx):
            continue
        li = lab[keep][idx]
        pix = np.stack([mu, mv], 1).astype(np.float64)
        per = {i: pix[li == i] for i in ids}
        lst = []
        for i in ids:
            if len(per[i]) < MIN_PTS_PROMPT:
                continue
            d = instance_prompt(per[i], [per[j] for j in ids if j != i])
            d["id"], d["n_vis"] = i, int(len(per[i]))
            lst.append(d)
        if lst:
            out[int(fr.index)] = lst
    return out


def compose_c(sid: str):
    """Stored C masks -> per-frame uint8 maps (id = A-lifted instance, aux from the stored SAM 3 raw maps), meta, scores."""
    z = np.load(WORK / "cmasks" / f"{tag(sid)}.npz")
    rawsd = roots(sid)[1]
    base_raw = {int(p.stem): np.asarray(Image.open(p)) for p in (rawsd / "masks_sam3_v2i" / "raw").glob("[0-9]*.png")}
    masks, scores, ids = {}, {}, set()
    for fi in sorted(base_raw):
        b = base_raw[fi]
        base = np.where((b == 253) | (b == 254), b, 0).astype(np.uint8)
        its = []
        if f"{fi}|id" in z.files:
            m = unpack(z[f"{fi}|m"])
            its = [(int(k), float(s), m[j]) for j, (k, s) in enumerate(zip(z[f"{fi}|id"], z[f"{fi}|s"]))]
        masks[fi] = compose_instance_maps(its, base)
        scores[fi] = {k: s for k, s, _ in its}
        ids |= {k for k, _, _ in its}
    meta = {k: {"name": "food", "concept_index": 0, "kind": "food"} for k in sorted(ids)}
    return masks, meta, scores


def _prep_job(sid):
    lab, _ = replay_lift(sid, "A")
    pr = c_prompts(sid, lab)
    (WORK / "prompts").mkdir(parents=True, exist_ok=True)
    (WORK / "prompts" / f"{tag(sid)}.json").write_text(json.dumps(pr))
    return sid, int(lab.max()), sum(len(v) for v in pr.values())


def cmd_prep(a) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as p:
        for sid, k, n in p.imap_unordered(_prep_job, a.only or SIDS12):
            print(f"{sid}: A lift max id {k}, {n} instance-frame prompts", flush=True)


# ---------------------------------------------------------------------------------------------- stage: gpu
def cmd_gpu(a) -> None:
    import torch
    from foodscan.segment import CLUTTER_PROMPTS, CONTAINER_PROMPTS, FOOD, prompt_names
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    interactive = a.c and a.model == "sam3"
    be = Backend(a.model, interactive, dev)
    print(f"{a.model} loaded in {time.time() - t0:.0f}s, interactive={interactive}, "
          f"gpu {torch.cuda.memory_allocated() / 2**20 if dev == 'cuda' else 0:.0f} MiB", flush=True)
    ddir = WORK / (f"det_{a.model}" + (".smoke" if a.frames else ""))
    ddir.mkdir(parents=True, exist_ok=True)
    (WORK / "cmasks").mkdir(parents=True, exist_ok=True)
    times_p = OUT / f"gpu_times_{a.model}.json"
    times = json.loads(times_p.read_text()) if times_p.exists() else {}
    aux = [(n, "aux") for n in CONTAINER_PROMPTS + CLUTTER_PROMPTS]
    for sid in (a.only or SIDS12):
        if sid in SIDS_CONFIRM:
            interactive = False
        dpath = ddir / f"{tag(sid)}.npz"
        if dpath.exists() and not a.force:
            print(f"{sid}: done, skip", flush=True)
            continue
        sd = roots(sid)[0]
        prompts = prompt_names(oracle_names(sid), 12, True)
        pr = json.loads((WORK / "prompts" / f"{tag(sid)}.json").read_text()) if interactive else {}
        files = frame_files(sd)[:a.frames or None]
        det, cm = {}, {}
        t = time.time()
        for fi, fp in enumerate(files):
            img = Image.open(fp).convert("RGB")
            sc = IMAGE_SIDE / max(img.size)
            if sc < 1:
                img = img.resize((round(img.width * sc), round(img.height * sc)), Image.BILINEAR)
            st = be.set_image(img)
            for prompt, kind in [(FOOD, "food")] + [(p, "name") for p in prompts] + aux:
                ms, ss = be.detect(st, prompt)
                keep = ss >= (FLOOR if kind in ("food", "name") else AUX_STORE)
                if keep.any():
                    det[f"{fi}|{prompt}|s"] = ss[keep].astype(np.float16)
                    det[f"{fi}|{prompt}|m"] = pack(to_store(ms[keep]))
            if interactive and str(fi) in pr:
                f = img.width / SIZE[1]                      # 960-px map coordinates -> image pixels
                ids, scs, mm = [], [], []
                for q in pr[str(fi)]:
                    sc_ = lambda v: [x * f for x in v]  # noqa: E731
                    try:
                        m, iou = be.predict(st, sc_(q["box"]), [sc_(p) for p in q["pos"]], [sc_(p) for p in q["neg"]])
                    except Exception as e:  # noqa: BLE001
                        print(f"{sid} f{fi} id{q['id']}: predict failed {type(e).__name__}: {e}", flush=True)
                        if a.strict:
                            raise
                        continue
                    ids.append(q["id"])
                    scs.append(iou)
                    mm.append(to_store(m[None])[0])
                if ids:
                    cm[f"{fi}|id"], cm[f"{fi}|s"], cm[f"{fi}|m"] = np.array(ids), np.array(scs, np.float32), pack(np.stack(mm))
            if fi % 6 == 5 and dev == "cuda":
                torch.cuda.empty_cache()
        times[sid] = round(time.time() - t, 1)
        np.savez_compressed(dpath, **det)
        if interactive and not a.frames:
            np.savez_compressed(WORK / "cmasks" / f"{tag(sid)}.npz", **cm)
        peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
        print(f"{sid}: {len(files)} frames, {len(prompts)} name prompts, {times[sid]:.0f}s, peak {peak:.0f} MiB, "
              f"C masks {sum(len(v) for k, v in cm.items() if k.endswith('|id'))}", flush=True)
        if not a.frames:
            times_p.write_text(json.dumps(times, indent=1))
    print(f"total {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------- stage: replay
def _replay_job(args):
    sid, variants = args
    res = {}
    for v in variants:
        t = time.time()
        try:
            lab, od = replay_lift(sid, v)
        except FileNotFoundError as e:
            res[v] = {"error": str(e)}
            continue
        r = evaluate(sid, lab)
        r["t_cpu"] = round(time.time() - t, 1)
        r["s2d"] = stats2d(sid, roots(sid)[1] / "masks_sam3_v2i" / "raw" if v == "A" else od / "raw")
        res[v] = r
    return sid, res


def cmd_replay(a) -> None:
    variants = a.variants or list(VARIANTS)
    path = OUT / "results.json"
    allr = json.loads(path.read_text()) if path.exists() else {}
    with Pool(a.procs) as p:
        for sid, res in p.imap_unordered(_replay_job, [(s, variants) for s in (a.only or SIDS12)]):
            allr.setdefault(sid, {}).update(res)
            print(sid, {v: round(r["inst"], 3) if "inst" in r else r for v, r in res.items()}, flush=True)
            path.write_text(json.dumps(allr, indent=1))


# ---------------------------------------------------------------------------------------------- stage: report
def agg(rows):
    from eval.lift_sweep import pooled
    p = pooled(rows)
    p["merged"] = sum(r["merged20"] for r in rows)
    p["frag20"] = sum(r["frag20"] for r in rows)
    return p


def cmd_report(a) -> None:
    from eval.lift_sweep import bootstrap
    allr = json.loads((OUT / "results.json").read_text())
    pool = SIDS_CONFIRM if a.set == "confirm" else SIDS12
    sids = [s for s in pool if s in allr and "A" in allr[s]]
    names = [v for v in VARIANTS if all(v in allr[s] and "inst" in allr[s][v] for s in sids)]
    times = {m: json.loads((OUT / f"gpu_times_{m}.json").read_text()) for m in ("sam3", "sam31")
             if (OUT / f"gpu_times_{m}.json").exists()}

    def block(sub, title):
        L = [f"### {title} ({len(sub)} scans)", "",
             "| variant | inst mIoU | PQ | exact count | fragmented | merged | items | d inst vs A (95% CI) | scans +/-/= | d PQ vs A (95% CI) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        base = [allr[s]["A"] for s in sub]
        for v in names:
            rows = [allr[s][v] for s in sub]
            p = agg(rows)
            d = [np.mean(r["iou"]) - np.mean(b["iou"]) for r, b in zip(rows, base)]
            dq = [r["pq"] - b["pq"] for r, b in zip(rows, base)]
            m, lo, hi = bootstrap(d)
            mq, loq, hiq = bootstrap(dq)
            sg = f"{sum(x > 1e-4 for x in d)}/{sum(x < -1e-4 for x in d)}/{sum(abs(x) <= 1e-4 for x in d)}"
            L.append(f"| {v} | {p['inst']:.3f} | {p['pq']:.3f} | {p['exact']:.2f} | {p['frag20']} | {p['merged']} | {p['n_items']} | "
                     f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {sg} | {mq:+.3f} [{loq:+.3f}, {hiq:+.3f}] |")
        return L + [""]

    def block2d(sub):
        L = ["### Single-frame item stats (raw per-frame id maps vs gt/seg_ids, items >= 150 px per frame)", "",
             "| variant | frame-items | hit IoU>=0.3 | hit IoU>=0.5 | mean best IoU | fragmented in frame | merged in frame |",
             "|---|---|---|---|---|---|---|"]
        for v in names:
            s = [allr[x][v]["s2d"] for x in sub]
            w = np.array([x["n"] for x in s], float)
            f = lambda k: float(np.average([x[k] for x in s], weights=w))  # noqa: E731
            L.append(f"| {v} | {int(w.sum())} | {f('hit30'):.3f} | {f('hit50'):.3f} | {f('miou'):.3f} | {f('frag2d'):.3f} | {f('merge2d'):.3f} |")
        return L + [""]

    L = ["## Lift metrics (2 mm canonical voxel scene, co-vote linker, same as results/lift_errors/scans40)", ""]
    L += block(sids, "All 12 scans")
    for b in ("v12", "v13"):
        L += block([s for s in sids if s.startswith(b)], f"Batch hard_{b}")
    L += ["## Single frames", ""] + block2d(sids)
    L += ["## Per scan inst mIoU", "", "| scan | items | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for s in sids:
        L.append(f"| {s} | {allr[s]['A']['n_items']} | " + " | ".join(f"{allr[s][v]['inst']:.3f}" for v in names) + " |")
    L += ["", "## GPU cost (RTX 2060, fp16, 24 frames per scan)", ""]
    for m, t in times.items():
        L.append(f"- {m}: mean {np.mean(list(t.values())):.0f} s/scan over {len(t)} scans (all text detections at floor 0.15"
                 f"{' + 3D-guided box/point prompts' if m == 'sam3' else ''}).")
    (OUT / ("tables_confirm.md" if a.set == "confirm" else "tables.md")).write_text("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "gpu", "replay", "report"])
    ap.add_argument("--model", choices=["sam3", "sam31"], default="sam3")
    ap.add_argument("--c", action="store_true", help="gpu: also run the 3D-guided box/point prompts (SAM 3 only)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=0, help="gpu: first N frames only (smoke test, separate output)")
    ap.add_argument("--set", choices=["main", "confirm"], default="main")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    {"prep": cmd_prep, "gpu": cmd_gpu, "replay": cmd_replay, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
