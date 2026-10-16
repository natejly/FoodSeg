"""Manual per-item labelling of a scan's fused point cloud, so segmentation (not just mass) can be scored.

    python -m tools.label_items label  SCAN [--grow lift|geodesic] [--dish "rice, chicken"] [--annotator me]
    python -m tools.label_items label  SCAN --seeds seeds.json [--out-dir DIR]        # headless
    python -m tools.label_items export SCAN [--labels labels_manual.npz] [--scale 0.5]  # masks_manual/*.png
    python -m tools.label_items score  SCAN [--pred refined.npz] [--ref labels_manual.npz]

SCAN is a real scan dir (run_meal.py output: lift.npz with xyz [+ normals], fused*_points.ply, frames.jsonl, rgb/)
or a synthetic dir (gt/seg_points.npz stands in for lift.npz + fused cloud).

Label ids: 1..N = items (scene-local, in labelling order), 0 = background, 253 clutter, 254 container.

seeds.json:
    {"grow": "lift",                                   # or "geodesic"; optional, CLI --grow default
     "items": [{"name": "rice", "seeds": [10, 99]}, {"name": "sauce", "seeds": [4000]}],
     "marks": [{"cls": "container", "seeds": [7]}]}    # cls: container | clutter | background
Two items whose seeds sit in the same lift instance are a *split* (points of that instance go to the nearest seed
by geodesic distance). One item with seeds in two lift instances is a *merge*.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

CLUTTER, CONTAINER, BG = 253, 254, 0
CLS = {"container": CONTAINER, "clutter": CLUTTER, "background": BG, "bg": BG}


# ----------------------------------------------------------------------------------------------- scan adapter
def _srgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    c = np.clip(rgb01, 0, 1)
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _estimate_normals(xyz):
    import open3d as o3d
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    p.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    p.orient_normals_to_align_with_direction(np.array([0.0, 1.0, 0.0]))
    return np.asarray(p.normals, np.float32)


def _dbscan_prior(xyz, eps=0.01, min_points=20):
    import open3d as o3d
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    lab = np.asarray(p.cluster_dbscan(eps=eps, min_points=min_points))
    return (lab + 1).astype(np.int32)                      # noise -> 0


def load_scan(scan_dir) -> dict:
    """-> dict(xyz, rgb (0..1), normals, prior (int32 per point, lift_label or DBSCAN ids), has_lift, kind, names)."""
    scan = Path(scan_dir)
    names = []
    inst = scan / "masks" / "instances.json"
    if inst.exists():
        try:
            names = sorted({v["name"] for v in json.loads(inst.read_text()).values() if isinstance(v, dict) and "name" in v})
        except Exception:                                   # noqa: BLE001
            pass
    d = dict(scan=scan, names=names, extra={})
    syn = scan / "gt" / "seg_points.npz"
    if (scan / "lift.npz").exists():
        z = np.load(scan / "lift.npz", allow_pickle=False)
        d["kind"] = "real"
        d["prior"] = z["lift_label"].astype(np.int32)
        d["has_lift"] = True
        d["extra"] = {k: z[k] for k in ("lift_share", "lift_nvotes", "lift_agree") if k in z.files}
        xyz = z["xyz"].astype(np.float32) if "xyz" in z.files else None
        rgb = None
        plys = [scan / "fused_points.ply"] + sorted(scan.glob("*_points.ply"))
        for ply in plys:
            if ply.exists():
                import open3d as o3d
                pc = o3d.io.read_point_cloud(str(ply))
                pxyz, prgb = np.asarray(pc.points, np.float32), np.asarray(pc.colors, np.float32)
                if xyz is None:
                    xyz, rgb = pxyz, prgb
                elif len(pxyz) == len(xyz):
                    rgb = prgb
                elif len(prgb):
                    rgb = prgb[cKDTree(pxyz).query(xyz)[1]]
                break
        if xyz is None:
            raise FileNotFoundError(f"{scan}: lift.npz has no xyz and no fused*_points.ply found")
        d["xyz"] = xyz
        d["rgb"] = rgb if rgb is not None and len(rgb) == len(xyz) else np.full((len(xyz), 3), 0.6, np.float32)
        d["normals"] = z["normals"].astype(np.float32) if "normals" in z.files else _estimate_normals(xyz)
    elif syn.exists():
        z = np.load(syn, allow_pickle=False)
        d["kind"] = "synthetic"
        d["xyz"] = z["xyz"].astype(np.float32)
        d["rgb"] = (z["rgb"].astype(np.float32) / 255.0) if z["rgb"].dtype == np.uint8 else z["rgb"].astype(np.float32)
        d["normals"] = z["normal"].astype(np.float32)
        d["has_lift"] = "lift_label" in z.files
        d["prior"] = z["lift_label"].astype(np.int32) if d["has_lift"] else _dbscan_prior(d["xyz"])
        d["gt_label"] = z["label"].astype(np.int32)
        d["extra"] = {k: z[k] for k in ("lift_share", "lift_nvotes") if k in z.files}
        gj = scan / "gt.json"
        if gj.exists():
            d["gt_names"] = {int(i["id"]): i["name"] for i in json.loads(gj.read_text()).get("items", [])}
    else:
        plys = sorted(scan.glob("*_points.ply"))
        if not plys:
            raise FileNotFoundError(f"{scan}: no lift.npz, gt/seg_points.npz or *_points.ply")
        import open3d as o3d
        pc = o3d.io.read_point_cloud(str(plys[0]))
        d.update(kind="real-nolift", xyz=np.asarray(pc.points, np.float32), has_lift=False,
                 rgb=np.asarray(pc.colors, np.float32))
        d["normals"] = np.asarray(pc.normals, np.float32) if pc.has_normals() else _estimate_normals(d["xyz"])
        d["prior"] = _dbscan_prior(d["xyz"])
    return d


# ----------------------------------------------------------------------------------------------- growing
class Geometry:
    """kNN graph over the cloud with colour/normal-pruned edges (built lazily, cached)."""

    def __init__(self, xyz, rgb, normals, k=12, max_edge_mult=4.0, de_max=20.0, normal_deg=50.0):
        self.xyz, self.n = xyz, len(xyz)
        tree = cKDTree(xyz)
        self.tree = tree
        dist, nb = tree.query(xyz, k=min(k + 1, self.n))
        med = float(np.median(dist[:, 1]))
        self.spacing = med
        row = np.repeat(np.arange(self.n), dist.shape[1] - 1)
        col, w = nb[:, 1:].ravel(), dist[:, 1:].ravel()
        keep = w <= max_edge_mult * med
        row, col, w = row[keep], col[keep], w[keep]
        self.row, self.col, self.w = row, col, w
        lab = _srgb_to_lab(rgb)
        de = np.linalg.norm(lab[row] - lab[col], axis=1)
        dots = np.abs((normals[row] * normals[col]).sum(1)) if normals is not None else np.ones(len(row))
        ok = (de <= de_max) & (dots >= np.cos(np.deg2rad(normal_deg)))
        self.G_full = self._mk(row, col, w)
        self.G_pruned = self._mk(row[ok], col[ok], w[ok])

    def _mk(self, r, c, w):
        g = sparse.coo_matrix((np.maximum(w, 1e-9), (r, c)), shape=(self.n, self.n)).tocsr()
        return g.maximum(g.T)                                            # symmetric

    def multi_source(self, seeds_per_item, pruned=True, allowed=None, radius=np.inf):
        """seeds_per_item: list of int arrays. -> (owner index into list or -1, dist). `allowed`: node mask."""
        G = self.G_pruned if pruned else self.G_full
        flat = np.concatenate([np.asarray(s, int) for s in seeds_per_item])
        who = np.concatenate([np.full(len(s), i) for i, s in enumerate(seeds_per_item)])
        if allowed is not None:
            idx = np.nonzero(allowed | np.isin(np.arange(self.n), flat))[0]
            G = G[idx][:, idx]
            remap = -np.ones(self.n, int)
            remap[idx] = np.arange(len(idx))
            src = remap[flat]
        else:
            idx, src = None, flat
        d, _, s = dijkstra(G, directed=False, indices=src, min_only=True, return_predecessors=True,
                           limit=radius)
        d = np.asarray(d)
        reached = np.isfinite(d) & (s >= 0)
        pos = {int(f): i for i, f in enumerate(src)}
        owner = -np.ones(len(d), int)
        sl = np.nonzero(reached)[0]
        owner[sl] = np.array([who[pos[int(x)]] for x in s[sl]]) if len(sl) else owner[sl]
        if idx is None:
            return owner, d
        o_full, d_full = -np.ones(self.n, int), np.full(self.n, np.inf)
        o_full[idx], d_full[idx] = owner, d
        return o_full, d_full


class Session:
    """Replayable list of labelling operations -> per-point labels (undo = drop the last op)."""

    def __init__(self, scan: dict, grow="lift", radius=0.12, geo: Geometry | None = None, annotator=None):
        self.s, self.grow, self.radius = scan, grow, radius
        self.xyz, self.prior = scan["xyz"], scan["prior"]
        self.geo = geo
        self.ops: list[dict] = []
        self.annotator = annotator or getpass.getuser()
        self._cache = None

    @property
    def geometry(self) -> Geometry:
        if self.geo is None:
            self.geo = Geometry(self.xyz, self.s["rgb"], self.s["normals"])
        return self.geo

    # --- ops
    def add_items(self, items: list[dict], grow: str | None = None):
        """items: [{"name", "seeds":[idx...]}]. Several items = a group (shared lift ids are split by geodesic)."""
        self._push({"kind": "items", "grow": grow or self.grow,
                    "items": [{"name": it["name"], "seeds": [int(x) for x in it["seeds"]]} for it in items]})

    def add_mark(self, cls, seeds, grow: str | None = None):
        c = CLS[cls] if isinstance(cls, str) else int(cls)
        self._push({"kind": "mark", "cls": c, "grow": grow or self.grow, "seeds": [int(x) for x in seeds]})

    def undo(self):
        if self.ops:
            self.ops.pop()
            self._cache = None

    def _push(self, op):
        if any(len(it["seeds"]) == 0 for it in op.get("items", [{"seeds": op.get("seeds", [])}])):
            raise ValueError("every item / mark needs at least one seed")
        seeds = [x for it in op.get("items", [op]) for x in it["seeds"]]
        if seeds and not (0 <= min(seeds) and max(seeds) < len(self.xyz)):
            raise ValueError(f"seed index out of range (cloud has {len(self.xyz)} points)")
        self.ops.append(op)
        self._cache = None

    # --- region growing
    def _lift_ids(self, seeds):
        ids = np.unique(self.prior[np.asarray(seeds, int)])
        return ids[ids > 0]

    def _geodesic_regions(self, seed_sets):
        allowed = ~np.isin(self.prior, (CLUTTER, CONTAINER)) if self.s["has_lift"] else None
        owner, _ = self.geometry.multi_source(seed_sets, pruned=True, allowed=allowed, radius=self.radius)
        return owner

    def _split_regions(self, region, seed_sets):
        """Assign `region` points to the nearest seed set by geodesic distance (euclidean fallback)."""
        owner, _ = self.geometry.multi_source(seed_sets, pruned=False, allowed=region, radius=np.inf)
        idx = np.nonzero(region & (owner < 0))[0]
        if len(idx):
            flat = np.concatenate(seed_sets)
            who = np.concatenate([np.full(len(s), i) for i, s in enumerate(seed_sets)])
            owner[idx] = who[cKDTree(self.xyz[flat]).query(self.xyz[idx])[1]]
        owner[~region] = -1
        return owner

    def _regions(self, seed_sets, grow):
        """-> list of boolean masks, one per seed set (non-overlapping within the call)."""
        n = len(self.xyz)
        masks = [np.zeros(n, bool) for _ in seed_sets]
        if grow == "geodesic":
            owner = self._geodesic_regions(seed_sets)
            for i, m in enumerate(masks):
                m[owner == i] = True
            return masks
        ids = [set(self._lift_ids(s).tolist()) for s in seed_sets]
        no_lift = [i for i, s in enumerate(ids) if not s]
        # union-find over sets sharing lift ids
        parent = list(range(len(seed_sets)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] & ids[j]:
                    parent[find(j)] = find(i)
        comps: dict[int, list[int]] = {}
        for i in range(len(ids)):
            if ids[i]:
                comps.setdefault(find(i), []).append(i)
        for members in comps.values():
            region = np.isin(self.prior, sorted(set().union(*[ids[i] for i in members])))
            if len(members) == 1:
                masks[members[0]] = region
            else:
                owner = self._split_regions(region, [np.asarray(seed_sets[i], int) for i in members])
                for k, i in enumerate(members):
                    masks[i] = owner == k
        if no_lift:                                         # seeds on lift-background: grow geodesically
            owner = self._geodesic_regions([seed_sets[i] for i in no_lift])
            for k, i in enumerate(no_lift):
                masks[i] = owner == k
        return masks

    def result(self):
        """-> (label int32, names {id: name}). Replays all ops."""
        if self._cache is not None:
            return self._cache
        label = np.zeros(len(self.xyz), np.int32)
        names: dict[int, str] = {}
        nxt = 1
        for op in self.ops:
            if op["kind"] == "mark":
                label[self._regions([op["seeds"]], op["grow"])[0]] = op["cls"]
            else:
                masks = self._regions([it["seeds"] for it in op["items"]], op["grow"])
                for it, m in zip(op["items"], masks):
                    label[m] = nxt
                    names[nxt] = it["name"]
                    nxt += 1
        self._cache = (label, names)
        return self._cache

    def apply_seeds_spec(self, spec: dict):
        grow = spec.get("grow") or self.grow
        if spec.get("items"):
            self.add_items(spec["items"], grow)
        for m in spec.get("marks", []):
            self.add_mark(m["cls"], m["seeds"], grow)

    # --- output
    def save(self, out_dir=None, method=None) -> Path:
        out = Path(out_dir) if out_dir else self.s["scan"]
        out.mkdir(parents=True, exist_ok=True)
        label, names = self.result()
        method = method or f"manual:{self.grow}"
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        allseeds = sorted({x for op in self.ops for it in op.get("items", [op]) for x in it["seeds"]})
        np.savez_compressed(out / "labels_manual.npz", label=label.astype(np.int32),
                            names=np.array(json.dumps({str(k): v for k, v in names.items()})),
                            seeds=np.array(json.dumps({"ops": self.ops, "indices": allseeds})),
                            method=np.array(method), timestamp=np.array(stamp), annotator=np.array(self.annotator))
        summ = dict(scan=str(self.s["scan"]), method=method, timestamp=stamp, annotator=self.annotator,
                    n_points=int(len(label)), n_items=len(names),
                    items=[dict(id=k, name=v, points=int((label == k).sum())) for k, v in names.items()],
                    container_points=int((label == CONTAINER).sum()), clutter_points=int((label == CLUTTER).sum()),
                    background_points=int((label == 0).sum()))
        (out / "labels_manual.json").write_text(json.dumps(summ, indent=1))
        return out / "labels_manual.npz"


# ----------------------------------------------------------------------------------------------- interactive
def _palette(n):
    rng = np.random.default_rng(3)
    base = np.array([[0.90, 0.10, 0.10], [0.10, 0.60, 0.90], [0.95, 0.80, 0.10], [0.20, 0.75, 0.30],
                     [0.80, 0.30, 0.85], [1.00, 0.50, 0.10], [0.10, 0.85, 0.80], [0.60, 0.40, 0.20]])
    return np.vstack([base, rng.uniform(0.2, 1, (max(n, 1), 3))])


def colour_view(scan, label, names):
    """Items saturated, container dark blue, clutter brown; unlabelled points muted by their lift instance."""
    prior = scan["prior"]
    pal = _palette(max(int(prior.max()), 1) + 1)
    col = np.full((len(prior), 3), 0.55)
    m = (prior > 0) & (prior < 252)
    col[m] = 0.55 * 0.4 + 0.6 * pal[prior[m] % len(pal)] * 0.7        # pastel lift instances
    col[prior == CONTAINER] = [0.35, 0.40, 0.55]
    col[prior == CLUTTER] = [0.55, 0.45, 0.35]
    col[prior == 0] = 0.85 if scan["has_lift"] else 0.55
    for k in names:
        col[label == k] = pal[k % len(pal)]
    col[label == CONTAINER] = [0.1, 0.15, 0.45]
    col[label == CLUTTER] = [0.4, 0.25, 0.1]
    return col


HELP = """
 viewer : shift+left-click = pick a seed (shift+right-click = un-pick), drag = rotate, wheel = zoom, Q/close = done picking
 prompt : <name>              new item from the picked seeds (merge = seeds in 2 lift instances)
          <n>                 new item named candidate #n from the list
          split a b*2 c       one item per name; b*2 takes two picks (picks consumed in order) -> split a lift instance
          con | clu | bg      mark the picked instance(s) container / clutter / background
          u                   undo last   l  list items   g lift|geodesic  switch grow mode   w save   q save+quit
          (empty line)        re-open the viewer"""


def interactive(session: Session, candidates=(), view_pts=400_000):
    import open3d as o3d
    s = session.s
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s["xyz"].astype(np.float64)))
    cam = None
    print(HELP)
    print(f"{len(s['xyz'])} points, {'lift' if s['has_lift'] else 'DBSCAN'} prior, grow={session.grow}")
    picks: list[int] = []
    while True:
        label, names = session.result()
        pc.colors = o3d.utility.Vector3dVector(colour_view(s, label, names))
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"label_items [{s['scan'].name}] shift+click seeds, Q when done", width=1400, height=900)
        vis.add_geometry(pc)
        if cam is not None:
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            except Exception:                                  # noqa: BLE001
                pass
        vis.run()
        try:
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        except Exception:                                      # noqa: BLE001
            cam = None
        picks = [int(i) for i in vis.get_picked_points()]
        vis.destroy_window()
        print(f"picked {len(picks)} seeds: {picks}")
        if candidates:
            print("candidates: " + "  ".join(f"[{i + 1}] {c}" for i, c in enumerate(candidates)))
        while True:
            try:
                cmd = input("cmd> ").strip()
            except EOFError:
                cmd = "q"
            try:
                if not cmd:
                    break
                head, *rest = cmd.split()
                if cmd in ("q", "w"):
                    p = session.save()
                    print(f"saved {p}")
                    if cmd == "q":
                        return
                elif cmd == "u":
                    session.undo()
                    break
                elif cmd == "l":
                    for k, v in names.items():
                        print(f"  {k}: {v} ({int((label == k).sum())} pts)")
                elif head == "g" and rest and rest[0] in ("lift", "geodesic"):
                    session.grow = rest[0]
                    print("grow =", session.grow)
                elif cmd in ("con", "clu", "bg"):
                    session.add_mark({"con": "container", "clu": "clutter", "bg": "background"}[cmd], need(picks))
                    picks = []
                    break
                elif head == "split":
                    items, cur = [], 0
                    for tok in rest:
                        nm, _, k = tok.partition("*")
                        k = int(k or 1)
                        items.append({"name": nm, "seeds": need(picks)[cur:cur + k]})
                        cur += k
                    if cur > len(picks):
                        raise ValueError(f"split needs {cur} picks, have {len(picks)}")
                    session.add_items(items)
                    picks = []
                    break
                else:
                    name = candidates[int(cmd) - 1] if cmd.isdigit() and 0 < int(cmd) <= len(candidates) else cmd
                    session.add_items([{"name": name, "seeds": need(picks)}])
                    picks = []
                    break
            except Exception as e:                             # noqa: BLE001
                print("!", e)


def need(picks):
    if not picks:
        raise ValueError("no seeds picked - press empty line to reopen the viewer")
    return list(picks)


# ----------------------------------------------------------------------------------------------- frame export
def export_frames(scan_dir, label, out_dir=None, xyz=None, scale=0.5, fill_px=8, depth_tol=0.015, min_vis=1):
    """Project per-point labels into each rgb frame (nearest-depth gate against the frame's depth map) and write
    uint8 label PNGs (value = item id, 0 bg, 253/254) to <scan>/masks_manual/NNNNNN.png. -> list of paths."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    try:
        from foodscan.loader import Scan
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from foodscan.loader import Scan
    scan_dir = Path(scan_dir)
    if xyz is None:
        xyz = load_scan(scan_dir)["xyz"]
    out = Path(out_dir) if out_dir else scan_dir / "masks_manual"
    out.mkdir(parents=True, exist_ok=True)
    label = np.asarray(label)
    scan = Scan(scan_dir)
    paths = []
    for fr in scan.frames:
        W, H = fr.rgb_size
        w, h = max(int(W * scale), 1), max(int(H * scale), 1)
        img = project_labels(fr, xyz, label, (w, h), fill_px, depth_tol, scale)
        p = out / f"{fr.index:06d}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths


def project_frame_points(fr, xyz, depth_tol=0.015):
    """-> (idx, u_rgb, v_rgb, z) of points inside the rgb image and agreeing with the frame depth (gate)."""
    T = fr.T_wc
    pc = xyz @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = fr.K_rgb
        u, v = K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]
        Kd = fr.K_depth
        ud, vd = np.floor(Kd[0, 0] * pc[:, 0] / z + Kd[0, 2]).astype(int), np.floor(Kd[1, 1] * pc[:, 1] / z + Kd[1, 2]).astype(int)
    W, H = fr.rgb_size
    dw, dh = fr.depth_size
    ok = (z > 0.05) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H) \
        & (ud >= 0) & (ud < dw) & (vd >= 0) & (vd < dh)
    idx = np.nonzero(ok)[0]
    d = fr.depth()[vd[idx], ud[idx]]
    gate = np.isfinite(d) & (d > 0) & (np.abs(d - z[idx]) < depth_tol)
    idx = idx[gate]
    return idx, u[idx], v[idx], z[idx]


def project_labels(fr, xyz, label, size, fill_px, depth_tol, scale):
    from scipy.ndimage import distance_transform_edt
    w, h = size
    idx, u, v, z = project_frame_points(fr, xyz, depth_tol)
    img = np.zeros((h, w), np.uint8)
    if not len(idx):
        return img
    ui, vi = np.clip((u * scale).astype(int), 0, w - 1), np.clip((v * scale).astype(int), 0, h - 1)
    order = np.argsort(-z)                                  # far first, nearest written last
    ui, vi, idx = ui[order], vi[order], idx[order]
    lab = np.zeros((h, w), np.int32) - 1
    lab[vi, ui] = label[idx]
    occ = lab >= 0
    dist, (iy, ix) = distance_transform_edt(~occ, return_indices=True)
    fill = dist <= fill_px
    img[fill] = lab[iy[fill], ix[fill]].clip(0, 255)
    return img


# ----------------------------------------------------------------------------------------------- scoring
try:
    _here = Path(__file__).resolve().parents[1]
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from seg3d.metrics import boundary_band, match      # reuse the benchmark's metric code
except Exception:                                       # noqa: BLE001  (minimal fallbacks)
    from scipy.optimize import linear_sum_assignment

    def boundary_band(xyz, label, width=0.003):
        pairs = cKDTree(xyz).query_pairs(width, output_type="ndarray")
        band = np.zeros(len(xyz), bool)
        if len(pairs):
            band[pairs[label[pairs[:, 0]] != label[pairs[:, 1]]].ravel()] = True
        return band

    def match(gt, pred):
        g_ids = [i for i in np.unique(gt) if i > 0]
        p_ids = [i for i in np.unique(pred) if i > 0]
        iou = np.zeros((len(g_ids), max(len(p_ids), 1)))
        for a, g in enumerate(g_ids):
            for b, p in enumerate(p_ids):
                iou[a, b] = ((gt == g) & (pred == p)).sum() / max(((gt == g) | (pred == p)).sum(), 1)
        r, c = linear_sum_assignment(-iou) if g_ids and p_ids else ([], [])
        return g_ids, p_ids, {g_ids[a]: (p_ids[b], iou[a, b]) for a, b in zip(r, c)}


def _food(lab):
    lab = np.asarray(lab, np.int64)
    return np.where((lab > 0) & (lab < 252), lab, 0)


def score_labels(xyz, ref, pred, ref_names=None, band_mm=3.0) -> dict:
    """Food-instance metrics of `pred` against reference `ref` (both per-point int labels)."""
    gi, pi = _food(ref), _food(pred)
    g_ids, p_ids, m = match(gi, pi)
    ious = [m[g][1] if g in m else 0.0 for g in g_ids]
    tp = [i for i in ious if i > 0.5]
    fp = sum(1 for p in p_ids if p not in {v[0] for v in m.values() if v[1] > 0.5})
    fn = len(g_ids) - len(tp)
    denom = len(tp) + 0.5 * fp + 0.5 * fn
    band = boundary_band(xyz, np.asarray(ref), band_mm / 1000.0)
    bnd = []
    for g in g_ids:
        gm = gi == g
        pm = (pi == m[g][0]) if g in m else np.zeros_like(gm)
        if (gm & band).any():
            bnd.append(((gm & pm) & band).sum() / max(((gm | pm) & band).sum(), 1))
    names = ref_names or {}
    return dict(inst_miou=float(np.mean(ious)) if ious else float("nan"),
                pq=float(sum(tp) / denom) if denom else float("nan"),
                boundary_iou=float(np.mean(bnd)) if bnd else float("nan"),
                n_ref=len(g_ids), n_pred=len(p_ids), exact_count=bool(len(g_ids) == len(p_ids)),
                per_item=[dict(id=int(g), name=names.get(int(g), names.get(str(g), "?")),
                               matched_pred=int(m[g][0]) if g in m else None, iou=float(m[g][1]) if g in m else 0.0,
                               points=int((gi == g).sum())) for g in g_ids])


def _load_pred(path, n):
    z = np.load(path, allow_pickle=False)
    for k in ("label", "refined", "pred", "inst", "instance", "lift_label"):
        if k in z.files and z[k].shape == (n,):
            return z[k].astype(np.int64), k
    raise KeyError(f"{path}: no per-point label array of length {n} among {z.files}")


def cmd_score(args):
    scan = Path(args.scan)
    sc = load_scan(scan)
    xyz = sc["xyz"]
    ref_path = Path(args.ref) if args.ref else scan / "labels_manual.npz"
    z = np.load(ref_path, allow_pickle=False)
    ref, names = z["label"], {int(k): v for k, v in json.loads(str(z["names"])).items()}
    preds = {"lift": sc["prior"]} if sc["has_lift"] else {}
    if args.pred:
        p, key = _load_pred(args.pred, len(xyz))
        preds[f"{Path(args.pred).name}:{key}"] = p
    out = {}
    for nm, p in preds.items():
        r = score_labels(xyz, ref, p, names)
        out[nm] = r
        print(f"\n== {nm} vs manual ({ref_path.name}) ==")
        print(f"instance mIoU {r['inst_miou']:.3f}  PQ {r['pq']:.3f}  boundary IoU(3mm) {r['boundary_iou']:.3f}  "
              f"count pred/ref {r['n_pred']}/{r['n_ref']} exact={r['exact_count']}")
        for it in r["per_item"]:
            print(f"   {it['id']:>3} {it['name']:<24} IoU {it['iou']:.3f}  pred {it['matched_pred']}  ({it['points']} pts)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------------------------- CLI
def cmd_label(args):
    scan = load_scan(args.scan)
    if args.dish:
        scan["names"] = [x.strip() for x in args.dish.split(",") if x.strip()] + scan["names"]
    sess = Session(scan, grow=args.grow, radius=args.radius, annotator=args.annotator)
    if args.seeds:
        sess.apply_seeds_spec(json.loads(Path(args.seeds).read_text()))
        p = sess.save(args.out_dir, method=f"manual:{args.grow}:batch")
        print(f"saved {p}")
    else:
        interactive(sess, scan["names"])
        p = (Path(args.out_dir) if args.out_dir else scan["scan"]) / "labels_manual.npz"
    if args.export_frames:
        z = np.load(p, allow_pickle=False)
        out = (Path(args.out_dir) if args.out_dir else scan["scan"]) / "masks_manual"
        paths = export_frames(scan["scan"], z["label"], out, xyz=scan["xyz"], scale=args.scale, fill_px=args.fill_px)
        print(f"wrote {len(paths)} masks to {out}")


def cmd_export(args):
    scan = Path(args.scan)
    lab = Path(args.labels) if args.labels else scan / "labels_manual.npz"
    z = np.load(lab, allow_pickle=False)
    out = Path(args.out_dir) if args.out_dir else scan / "masks_manual"
    paths = export_frames(scan, z["label"], out, scale=args.scale, fill_px=args.fill_px)
    print(f"wrote {len(paths)} masks to {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label")
    a.add_argument("scan")
    a.add_argument("--grow", choices=["lift", "geodesic"], default="lift")
    a.add_argument("--radius", type=float, default=0.12, help="geodesic radius cap (m)")
    a.add_argument("--seeds", help="seeds.json: non-interactive")
    a.add_argument("--dish", help="comma-separated candidate names")
    a.add_argument("--annotator", default=None)
    a.add_argument("--out-dir", default=None, help="default: the scan dir")
    a.add_argument("--export-frames", action="store_true")
    a.add_argument("--scale", type=float, default=0.5)
    a.add_argument("--fill-px", type=float, default=8)
    a.set_defaults(fn=cmd_label)
    e = sub.add_parser("export")
    e.add_argument("scan")
    e.add_argument("--labels")
    e.add_argument("--out-dir")
    e.add_argument("--scale", type=float, default=0.5)
    e.add_argument("--fill-px", type=float, default=8)
    e.set_defaults(fn=cmd_export)
    s = sub.add_parser("score")
    s.add_argument("scan")
    s.add_argument("--ref", help="default: <scan>/labels_manual.npz")
    s.add_argument("--pred", help="refined prediction npz (per-point label array)")
    s.add_argument("--json")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
