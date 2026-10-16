"""Dataset over synthetic fused-scan dirs  <root>/scans/<id>/gt/seg_points.npz  (+ <id>/gt.json).

Data contract (see README): seg_points.npz keys xyz,rgb,normal,label,dist [+ lift_label,lift_share,
lift_nvotes,boundary].  Label ids: per-scene food items 1..251, 252 table, 253 clutter, 254 container,
0 table/artefact.  Scenes are canonicalised on load: z up (gt.json table_plane_arkit normal, default
ARKit +y), z=0 on the table plane, xy centred on the container (or food) centroid, metres.

Network coordinates = metres * coord_scale (default 10): Sonata/PTv3 was pretrained on room-scale scenes
with 2 cm voxels; x10 makes a 2 mm voxel look like 2 cm to it.  All *targets* (offsets) are in the scaled
units; metrics/eval use metres.

Per-point features (30 ch): [coord_centred(3), rgb*2-1 (3), normal (3), lift one-hot over 16 random slots
+ overflow (17), lift is-container, lift is-clutter (2), lift_share (1), lift_nvotes/nv_norm (1)] - the
first 9 match Sonata's (coord,color,normal) input so the pretrained stem can be reused.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import augment as A

N_SLOTS = 16
FEAT_DIM = 3 + 3 + 3 + (N_SLOTS + 1) + 2 + 1 + 1          # 30
GROUPS = {"coord": slice(0, 3), "rgb": slice(3, 6), "normal": slice(6, 9), "lift": slice(9, 30)}
CLASSES = ["table", "container", "clutter", "food"]
DEFAULT_ROOTS = ["data/synth/hard", "data/synth/hard_v3"]


@dataclass
class DataConfig:
    roots: list = field(default_factory=lambda: list(DEFAULT_ROOTS))
    voxel_mm: float = 2.0                  # eval / base voxel size
    coord_scale: float = 10.0
    nv_norm: float = 20.0
    val_frac: float = 0.2
    split_seed: int = 0
    split_by: str = "scan"                 # "scan" | "category" (hold out whole food categories)
    max_points: int = 20000                # train voxel cap (keeps objects, thins table) - memory
    max_points_eval: int = 40000
    epoch_repeat: int = 20                 # train dataset length multiplier (few scans)
    val_lift_seed: int = 1234              # deterministic lift corruption for val on old files
    split_file: str = ""                   # json {"train": [scan dirs rel. to repo], "val": [...]}; overrides the random split
    zero_groups: list = field(default_factory=list)   # feature groups always zeroed (train + eval): coord|rgb|normal|lift
    koc: bool = False                      # build keep-or-change candidates (set automatically by decoder=koc)
    koc_k: int = 4                         # nearest distinct other lift labels per point
    koc_r_mm: float = 25.0                 # candidate / neighbourhood radius
    koc_m: int = 16                        # points averaged per candidate embedding
    item_filter: str = "manifest"          # manifest | points | none: items that are tiny/invisible are IGNORED (loss weight 0,
                                           # removed from the instance/semantic metrics) instead of dropping the scan
    item_min_points: int = 50              # "points" rule (and fallback when the manifest has no dropped_items): fused points


def list_scans(roots, base=None):
    base = Path(base or Path(__file__).resolve().parents[2])
    out = []
    for r in roots:
        out += sorted((base / r / "scans").glob("*/gt/seg_points.npz"))
    return [p.parent.parent for p in out]


def scan_categories(scan: Path):
    try:
        return sorted({it["category"] for it in json.loads((scan / "gt.json").read_text())["items"]})
    except Exception:                                       # noqa: BLE001
        return []


_MANIFESTS: dict = {}


def _manifest(root: Path) -> dict:
    if root not in _MANIFESTS:
        mf = root / "manifest.jsonl"
        _MANIFESTS[root] = {j["scan"]: j for j in map(json.loads, mf.read_text().splitlines()) if j} if mf.exists() else {}
    return _MANIFESTS[root]


def item_status(scan: Path, label, mode: str = "manifest", min_points: int = 50):
    """-> (valid_ids, ignored_ids).  Food item ids present in `label` (fused points; ids with no points are absent).
    manifest: ignore the manifest's `dropped_items` of this scan (falls back to the points rule if the manifest has no such
    key); points: ignore items with < min_points fused points; none: nothing ignored."""
    lab = np.asarray(label)
    ids, cnt = np.unique(lab[(lab > 0) & (lab < 252)], return_counts=True)
    small = {int(i) for i, c in zip(ids, cnt) if c < min_points}
    if mode == "none":
        bad = set()
    elif mode == "manifest" and "dropped_items" in (rec := _manifest(scan.parent.parent).get(scan.name, {})):
        bad = {int(i) for i in rec["dropped_items"]}
    else:
        bad = small
    valid = sorted(int(i) for i in ids if int(i) not in bad)
    return valid, sorted(bad & {int(i) for i in ids})


def split_scans(scans, cfg: DataConfig):
    if cfg.split_file:
        base = Path(__file__).resolve().parents[2]
        sp = json.loads((base / cfg.split_file).read_text())
        return [base / p for p in sp["train"]], [base / p for p in sp["val"]]
    rng = np.random.default_rng(cfg.split_seed)
    n_val = max(1, int(math.ceil(cfg.val_frac * len(scans))))
    if cfg.split_by == "category":
        cats = sorted({c for s in scans for c in scan_categories(s)})
        order = list(rng.permutation(len(cats)))
        held, val = set(), []
        for i in order:
            held.add(cats[i])
            val = [s for s in scans if held & set(scan_categories(s))]
            if len(val) >= n_val:
                break
        train = [s for s in scans if s not in val]
        if train and val:
            return train, val
    perm = rng.permutation(len(scans))
    val = [scans[i] for i in perm[:n_val]]
    train = [scans[i] for i in perm[n_val:]]
    return train, val


def _rot_to_z(n):
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(n @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def load_scan(scan: Path, item_filter: str = "none", item_min_points: int = 50) -> dict:
    z = np.load(scan / "gt" / "seg_points.npz")
    meta = json.loads((scan / "gt.json").read_text()) if (scan / "gt.json").exists() else {}
    plane = meta.get("table_plane_arkit", {})
    n = plane.get("normal", [0, 1, 0])
    R = _rot_to_z(n)
    xyz = z["xyz"].astype(np.float64) @ R.T
    nrm = z["normal"].astype(np.float64) @ R.T
    label = z["label"].astype(np.int32)
    tab = np.isin(label, (0, 252))
    z0 = float(np.array(plane["point"]) @ R.T @ np.array([0, 0, 1.0])) if "point" in plane else \
        float(np.percentile(xyz[tab, 2], 10)) if tab.any() else float(xyz[:, 2].min())
    obj = (label == 254) | A.is_food(label)
    if obj.any() and np.median(xyz[obj, 2]) < z0:             # table normal pointed the wrong way
        xyz[:, 2] *= -1
        nrm[:, 2] *= -1
        xyz[:, 1] *= -1
        nrm[:, 1] *= -1
        z0 = -z0
    xyz[:, 2] -= z0
    cm = label == 254
    ctr = xyz[cm if cm.sum() > 50 else (A.is_food(label) if A.is_food(label).sum() > 50 else slice(None)), :2].mean(0)
    xyz[:, :2] -= ctr
    rgb = z["rgb"].astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0
    d = dict(xyz=xyz.astype(np.float32), rgb=rgb, normal=nrm.astype(np.float32), label=label,
             has_lift="lift_label" in z.files, name=f"{scan.parent.parent.name}/{scan.name}")
    if d["has_lift"]:
        d["lift_label"] = z["lift_label"].astype(np.int32)
        d["lift_share"] = z["lift_share"].astype(np.float32)
        d["lift_nvotes"] = z["lift_nvotes"].astype(np.int16)
    if "boundary" in z.files:
        d["boundary"] = z["boundary"].astype(np.uint8)
    d["real_votes"] = bool(meta.get("votes", {}).get("mode") == "sam3")
    d["valid_ids"], d["ignore_ids"] = item_status(scan, label, item_filter, item_min_points)
    d["items"] = {int(it["id"]): dict(volume_ml=it.get("volume_ml"), mass_g=it.get("mass_g"),
                                      category=it.get("category")) for it in meta.get("items", [])}
    return d


def semantic_of(label):
    s = np.zeros(len(label), np.int64)
    s[label == 254] = 1
    s[label == 253] = 2
    s[A.is_food(label)] = 3
    return s


def voxelize(xyz, vox_m, rng=None):
    """One (random if rng) point per voxel. Returns indices and integer grid coords."""
    g = np.floor((xyz - xyz.min(0)) / vox_m).astype(np.int64)
    key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
    order = rng.permutation(len(key)) if rng is not None else np.arange(len(key))
    _, first = np.unique(key[order], return_index=True)
    idx = np.sort(order[first])
    return idx, g[idx]


def lift_features(lift, share, nv, rng, cfg: DataConfig):
    """one-hot over 16 randomly permuted slots (+overflow) for food ids; container/clutter flags."""
    feats = np.zeros((len(lift), N_SLOTS + 1 + 2 + 2), np.float32)
    ids = [int(i) for i in np.unique(lift[A.is_food(lift)])]
    perm = rng.permutation(N_SLOTS)
    for k, i in enumerate(rng.permutation(ids) if ids else []):
        slot = int(perm[k]) if k < N_SLOTS else N_SLOTS
        feats[lift == i, slot] = 1.0
    feats[lift == 254, N_SLOTS + 1] = 1.0
    feats[lift == 253, N_SLOTS + 2] = 1.0
    feats[:, N_SLOTS + 3] = share * (lift > 0)
    feats[:, N_SLOTS + 4] = np.minimum(nv / cfg.nv_norm, 1.0) * (lift > 0)
    return feats


class FoodSegDataset:
    def __init__(self, scans, cfg: DataConfig, aug: A.AugConfig, train: bool, seed: int = 0):
        self.scans, self.cfg, self.aug, self.train, self.seed = list(scans), cfg, aug, train, seed
        self.cache: dict = {}
        self.epoch = 0

    def __len__(self):
        return len(self.scans) * (self.cfg.epoch_repeat if self.train else 1)

    def set_epoch(self, e):
        self.epoch = e

    def scene(self, i):
        i %= len(self.scans)
        if i not in self.cache:
            self.cache[i] = load_scan(self.scans[i], self.cfg.item_filter, self.cfg.item_min_points)
        return self.cache[i]

    def __getitem__(self, idx):
        cfg, aug = self.cfg, self.aug
        base = self.scene(idx)
        rng = np.random.default_rng(
            (self.seed, self.epoch, idx) if self.train else (cfg.val_lift_seed, idx % len(self.scans)))
        for _try in range(4):
            d = {k: v for k, v in base.items()}
            vox = cfg.voxel_mm * 1e-3
            if self.train and aug.enable:
                vox = rng.uniform(*aug.voxel_mm) * 1e-3
                d = A.geometric(d, rng, aug)
                if aug.color:
                    d["rgb"] = A.color(d["rgb"], rng, aug)
            # lift prior (synthesised + corrupted when absent; real one corrupted with p)
            d = A.apply_lift(d, rng, aug, vox, base["has_lift"], force_corrupt=not self.train)
            if not (self.train and aug.enable) or A.is_food(d["label"]).sum() > 30 or not A.is_food(base["label"]).any():
                break
        vi, grid = voxelize(d["xyz"], vox, rng if self.train else None)
        cap = cfg.max_points if self.train else cfg.max_points_eval
        if len(vi) > cap:                         # keep all object voxels, thin out table
            lab = d["label"][vi]
            obj = np.nonzero((lab != 0) & (lab != 252))[0]
            tab = np.nonzero((lab == 0) | (lab == 252))[0]
            keep = np.sort(np.r_[obj[: cap], rng.choice(tab, max(cap - len(obj), 0), replace=False)
                                 if len(tab) else tab])
            vi, grid = vi[keep], grid[keep]
        xyz = d["xyz"][vi]
        label = d["label"][vi]
        lift, share, nv = d["lift_label"][vi], d["lift_share"][vi], d["lift_nvotes"][vi]
        sem = semantic_of(label)
        inst = np.where(sem == 3, label, 0).astype(np.int64)
        # offset to instance centroid (of the sampled voxels)
        off = np.zeros_like(xyz)
        cent = {}
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            cent[int(i)] = xyz[m].mean(0)
            off[m] = cent[int(i)] - xyz[m]
        bnd = d["boundary"][vi] if "boundary" in d else np.zeros(len(vi), np.uint8)
        ign = np.isin(label, base["ignore_ids"]) if base["ignore_ids"] else np.zeros(len(vi), bool)
        # features
        S = cfg.coord_scale
        coord = xyz * S
        f = np.zeros((len(vi), FEAT_DIM), np.float32)
        f[:, 0:3] = coord - coord.mean(0)
        f[:, 3:6] = d["rgb"][vi] * 2 - 1
        f[:, 6:9] = d["normal"][vi]
        f[:, 9:30] = lift_features(lift, share, nv, rng, cfg)
        if self.train and aug.enable:
            keep = A.channel_dropout(None, rng, aug)
            for g, k in (("rgb", "rgb"), ("normal", "normal"), ("lift", "lift")):
                if not keep[k]:
                    f[:, GROUPS[g]] = 0
        for g in cfg.zero_groups:
            f[:, GROUPS[g]] = 0
        sc = d.get("scale", 1.0)
        kc = {}
        if cfg.koc:
            from .koc import build_candidates
            kc = build_candidates(xyz, lift, label, cfg.koc_k, cfg.koc_r_mm * 1e-3, cfg.koc_m)
        return dict(**kc, **dict(coord=coord.astype(np.float32), grid_coord=(grid - grid.min(0)).astype(np.int32), feat=f,
                    sem=sem, inst=inst, offset=(off * S).astype(np.float32),
                    weight=((1 - bnd) * (~ign)).astype(np.float32), ignore=ign, lift=lift.astype(np.int64),
                    lift_share=share.astype(np.float32), boundary=bnd.astype(np.uint8), label=label.astype(np.int64),
                    xyz=xyz.astype(np.float32), name=base["name"], vox=vox,
                    items={k: dict(v, volume_ml=(v["volume_ml"] or 0) * sc ** 3) for k, v in base["items"].items()}))


def collate(batch):
    import torch
    out = {}
    for k in ("coord", "grid_coord", "feat", "sem", "inst", "offset", "weight", "lift", "label", "xyz"):
        out[k] = torch.from_numpy(np.concatenate([b[k] for b in batch]))
    if "cand_id" in batch[0]:
        starts = np.cumsum([0] + [len(b["coord"]) for b in batch[:-1]])
        for k in ("cand_id", "cand_valid", "cand_role", "cand_cls", "cand_dist", "cand_tgt"):
            if k in batch[0]:
                out[k] = torch.from_numpy(np.concatenate([b[k] for b in batch]))
        out["cand_nb"] = torch.from_numpy(np.concatenate([np.where(b["cand_nb"] >= 0, b["cand_nb"] + o, -1)
                                                          for b, o in zip(batch, starts)]))
    out["batch"] = torch.cat([torch.full((len(b["coord"]),), i, dtype=torch.long) for i, b in enumerate(batch)])
    out["n"] = torch.tensor([len(b["coord"]) for b in batch])
    out["offset_ptr"] = torch.cumsum(out["n"], 0)
    out["name"] = [b["name"] for b in batch]
    out["items"] = [b["items"] for b in batch]
    out["vox"] = [b["vox"] for b in batch]
    return out
