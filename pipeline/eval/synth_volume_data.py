"""Scan loading and labelling sources for the synthetic volume harness.

Frame: every cloud is moved into the *dish frame* (z up, z = 0 on the table, origin on the container
axis) with gt.json `world_to_arkit` (dish -> ARKit, rigid):  p_dish = (p_arkit - t) @ R.

Labelling sources (each -> Source(name, pts, inst, scene_ok)):
    gt            seg_points.npz `label`            (instance ids 1..251 == gt.json item ids)
    lift          seg_points.npz `lift_label`       (SAM-mask lift; ids are matched to GT items by IoU)
    pred:<run>    results/seg3d/<run>/preds/<scan>.npz  `pred_inst` where `pred_sem == 3` (food)
                  (matched by IoU).  File name is "<batch>__<scan>.npz".  Keys: xyz (canonicalised,
                  voxelised subset of the scan: z up, z = 0 on the table, xy centred on the container),
                  label (GT item ids), gt_sem, pred_sem (0 table,1 container,2 clutter,3 food),
                  pred_inst (0 = none), pred_offset, lift.  Points are mapped back to the dish frame by
                  nearest-neighbour against pipeline.seg3d.data.load_scan's canonical cloud.
    gtsub:<run>   GT labels on the same voxel subset as pred:<run>   (so pred - gtsub is pure segmentation)
    gt_top        gt/points.npz noise-free GT surface samples, keeping each item's top (visible-from-above)
                  surface only: the estimator floor with no sensor noise, sparsity or occlusion.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.eval.synth_volume_methods import top_envelope  # noqa: E402

FOOD_MAX = 251          # item ids 1..251; 252 table, 253 clutter, 254 container, 0 artefact
MATCH_IOU = 0.1


@dataclass
class Scan:
    batch: str
    sid: str
    dir: Path
    gt: dict
    ok: bool
    manifest: dict
    R: np.ndarray | None = None          # dish -> arkit linear part
    t: np.ndarray | None = None
    xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # dish frame
    label: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    lift: np.ndarray | None = None
    npz_keys: tuple = ()
    problem: str | None = None

    @property
    def key(self) -> str:
        return f"{self.batch}/{self.sid}"

    @property
    def items(self) -> list[dict]:
        return self.gt.get("items", []) or []


def to_dish(p_arkit: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (np.asarray(p_arkit, np.float64) - t) @ R


def read_manifest(batch_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(batch_dir.glob("manifest.*")):
        if f.suffix == ".jsonl":
            for line in f.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    out[d.get("scan", d.get("scene_id"))] = d
            break
    return out


def load_scan(batch_dir: Path, sid: str, manifest: dict | None = None) -> Scan:
    d = batch_dir / "scans" / sid
    man = (manifest or {}).get(sid, {})
    gt: dict = {}
    try:
        gt = json.loads((d / "gt.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    sc = Scan(batch_dir.name, sid, d, gt, bool(man.get("ok", False)), man)
    W = gt.get("world_to_arkit")
    if W is None:
        sc.problem = "no world_to_arkit"
        return sc
    W = np.array(W, np.float64)
    sc.R, sc.t = W[:3, :3], W[:3, 3]
    try:
        z = np.load(d / "gt" / "seg_points.npz")
        sc.npz_keys = tuple(z.files)
        sc.xyz = to_dish(z["xyz"], sc.R, sc.t)
        sc.label = z["label"].astype(np.int32)
        if "lift_label" in z.files and len(z["lift_label"]) == len(sc.xyz):
            sc.lift = z["lift_label"].astype(np.int32)
    except Exception as e:  # noqa: BLE001
        sc.problem = f"seg_points: {e}"
    if len(sc.xyz) == 0:
        sc.problem = sc.problem or "empty cloud"
    return sc


def container_points(sc: Scan) -> np.ndarray:
    """GT container surface samples in the dish frame (container.npz is already dish-frame; gt/points.npz
    label 254 is ARKit and is merged in for density)."""
    parts = []
    f = sc.dir / "gt" / "container.npz"
    if f.exists():
        parts.append(np.load(f)["points"].astype(np.float64))
    f = sc.dir / "gt" / "points.npz"
    if f.exists() and sc.R is not None:
        z = np.load(f)
        m = z["label"] == 254
        if m.any():
            parts.append(to_dish(z["xyz"][m], sc.R, sc.t))
    return np.vstack(parts) if parts else np.zeros((0, 3))


@dataclass
class Source:
    name: str
    pts: np.ndarray                       # (N,3) dish frame
    inst: np.ndarray                      # (N,) int instance id, <=0 or >251 = not food
    gt_label: np.ndarray | None = None    # GT item id per point (for IoU matching); None -> ids are GT ids
    match: bool = False                   # True: instance ids are arbitrary -> Hungarian-match to GT items


def match_instances(gt_label: np.ndarray, inst: np.ndarray, min_iou: float = MATCH_IOU) -> dict[int, tuple[int, float]]:
    """GT item id -> (instance id, IoU), Hungarian on IoU over food ids; pairs below min_iou dropped."""
    g_ids = [int(i) for i in np.unique(gt_label) if 0 < i <= FOOD_MAX]
    p_ids = [int(i) for i in np.unique(inst) if 0 < i <= FOOD_MAX]
    if not g_ids or not p_ids:
        return {}
    iou = np.zeros((len(g_ids), len(p_ids)))
    for a, g in enumerate(g_ids):
        gm = gt_label == g
        for b, p in enumerate(p_ids):
            pm = inst == p
            iou[a, b] = (gm & pm).sum() / max((gm | pm).sum(), 1)
    r, c = linear_sum_assignment(-iou)
    return {g_ids[a]: (p_ids[b], float(iou[a, b])) for a, b in zip(r, c) if iou[a, b] >= min_iou}


def sources_for(sc: Scan, pred_dirs: list[Path]) -> list[Source]:
    out: list[Source] = []
    if len(sc.xyz):
        out.append(Source("gt", sc.xyz, sc.label))
        if sc.lift is not None:
            out.append(Source("lift", sc.xyz, sc.lift, gt_label=sc.label, match=True))
    top = gt_top_source(sc)
    if top is not None:
        out.append(top)
    for pd in pred_dirs:
        out += pred_sources(sc, pd)
    return out


def gt_top_source(sc: Scan) -> Source | None:
    f = sc.dir / "gt" / "points.npz"
    if not f.exists() or sc.R is None:
        return None
    try:
        z = np.load(f)
        xyz, lab = to_dish(z["xyz"], sc.R, sc.t), z["label"].astype(np.int32)
    except Exception:  # noqa: BLE001
        return None
    keep_p, keep_l = [], []
    for g in np.unique(lab):
        if 0 < g <= FOOD_MAX:
            p = top_envelope(xyz[lab == g], tol=0.0015)
            keep_p.append(p)
            keep_l.append(np.full(len(p), g, np.int32))
    if not keep_p:
        return Source("gt_top", np.zeros((0, 3)), np.zeros(0, np.int32))
    return Source("gt_top", np.vstack(keep_p), np.concatenate(keep_l))


def pred_sources(sc: Scan, pred_dir: Path) -> list[Source]:
    """pred:<run> and gtsub:<run> for one scan, if `<pred_dir>/<batch>__<sid>.npz` exists."""
    f = pred_dir / f"{sc.batch}__{sc.sid}.npz"
    if not f.exists() or len(sc.xyz) == 0:
        return []
    run = pred_dir.parent.name
    z = np.load(f)
    pxyz = z["xyz"].astype(np.float64)
    canon = canonical_xyz(sc)
    if canon is None:
        return []
    d, idx = cKDTree(canon).query(pxyz)
    if np.mean(d < 1e-3) < 0.99:
        return []                                   # canonical frame mismatch: do not guess
    pts = sc.xyz[idx]
    inst = np.where(z["pred_sem"] == 3, z["pred_inst"], 0).astype(np.int32)
    gl = z["label"].astype(np.int32)
    return [Source(f"pred:{run}", pts, inst, gt_label=gl, match=True),
            Source(f"gtsub:{run}", pts, gl)]


def canonical_xyz(sc: Scan) -> np.ndarray | None:
    """The seg3d dataset's canonical cloud for this scan (read-only import of pipeline.seg3d.data.load_scan)."""
    try:
        from pipeline.seg3d.data import load_scan as seg3d_load
        return np.asarray(seg3d_load(sc.dir)["xyz"], np.float64)
    except Exception:  # noqa: BLE001
        return None
