"""Point-level boundary refinement of lifted SAM 3 labels (alpha-expansion graph cut on a kNN graph).

    python -m foodscan.boundary_refine cache            # soft votes of the 11 hard_v12 scans -> results/boundary_refine/cache
    python -m foodscan.boundary_refine grid             # (w, s_c, s_n, s_z, band) grid -> results/boundary_refine/grid_*.json
    python -m foodscan.boundary_refine report           # LOO / all-scan tables -> results/boundary_refine/summary.md

Energy over the labels of the points within `band` of a lift-label boundary (other points stay fixed):
  sum_i -log p_i(l_i)  +  sum_(i,j) w * exp(-dLab^2/2sc^2) exp(-ang^2/2sn^2) exp(-dz^2/2sz^2) [l_i != l_j]
p_i = normalised lift vote shares over the candidate ids found within `cand_mm` (floor for unvoted ones; label 0 =
background/table is a candidate like any id). Solved by alpha-expansion (PyMaxflow), exact per binary move.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

BIG = 1e6


@dataclass(frozen=True)
class RefineParams:
    w: float = 1.0
    s_c: float = 5.0           # Lab colour scale
    s_n: float = 10.0          # normal-angle scale, degrees
    s_z: float = 3.0           # height-step scale, mm (<=0 or inf: off)
    band_mm: float | None = 6.0  # None = every point may change
    k: int = 12
    radius_mm: float = 8.0
    cand_mm: float = 12.0
    floor: float = 0.02
    sweeps: int = 3


def _lab(rgb):
    import cv2
    c = np.asarray(rgb, np.float32)
    if c.max() > 1.5:
        c = c / 255.0
    return cv2.cvtColor(np.clip(c, 0, 1).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


class Graph:
    """kNN graph + per-edge discontinuity features (independent of w / scales, reused across a grid)."""

    def __init__(self, xyz, rgb, normal, up_z, k=12, radius_mm=8.0):
        xyz = np.asarray(xyz, np.float64)
        tree = cKDTree(xyz)
        d, nb = tree.query(xyz, k=k + 1, distance_upper_bound=radius_mm * 1e-3)
        n = len(xyz)
        i = np.repeat(np.arange(n), k)
        j = nb[:, 1:].ravel()
        ok = j < n
        i, j = i[ok], j[ok]
        a, b = np.minimum(i, j), np.maximum(i, j)
        key = np.unique(a.astype(np.int64) * n + b)
        self.i, self.j = (key // n).astype(np.int64), (key % n).astype(np.int64)
        lab = _lab(rgb)
        self.dcol2 = ((lab[self.i] - lab[self.j]) ** 2).sum(1)
        nrm = np.asarray(normal, np.float64)
        cosang = np.clip((nrm[self.i] * nrm[self.j]).sum(1), -1, 1)
        self.ang = np.degrees(np.arccos(np.abs(cosang)))      # sign-agnostic: fused normals may flip
        self.dz = np.abs(np.asarray(up_z, np.float64)[self.i] - np.asarray(up_z, np.float64)[self.j]) * 1e3
        self.xyz, self.tree, self.n = xyz, tree, n

    def weights(self, p: RefineParams):
        w = p.w * np.exp(-self.dcol2 / (2 * p.s_c ** 2)) * np.exp(-self.ang ** 2 / (2 * p.s_n ** 2))
        if p.s_z and np.isfinite(p.s_z) and p.s_z > 0:
            w = w * np.exp(-self.dz ** 2 / (2 * p.s_z ** 2))
        return w


def pseudo_votes(lift, share, graph: Graph, cand_mm=12.0):
    """No soft votes stored: own label gets `share`, the remainder spreads over the other labels within cand_mm."""
    labs, inv = np.unique(lift, return_inverse=True)
    n, L = len(lift), len(labs)
    pairs = graph.tree.query_pairs(cand_mm * 1e-3, output_type="ndarray")
    A = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, L))
    hist = (A @ onehot).toarray()
    hist[np.arange(n), inv] = 0
    tot = hist.sum(1, keepdims=True)
    own = np.clip(share, 0.5, 1.0) if share is not None else np.full(n, 0.9)
    rest = np.where(tot[:, 0] > 0, 1 - own, 0.0)
    v = hist / np.maximum(tot, 1) * rest[:, None]
    v[np.arange(n), inv] += 1 - rest
    return v.astype(np.float32), labs.astype(np.int64)


def _band(graph: Graph, lift, band_mm):
    if band_mm is None:
        return np.ones(graph.n, bool)
    pairs = graph.tree.query_pairs(band_mm * 1e-3, output_type="ndarray")
    diff = lift[pairs[:, 0]] != lift[pairs[:, 1]]
    m = np.zeros(graph.n, bool)
    m[pairs[diff].ravel()] = True
    return m


def unary_costs(lift, votes, ids, graph: Graph, p: RefineParams, free):
    """-> (labels array (L,), U (n, L) with BIG for non-candidates, current local label idx)."""
    n = len(lift)
    labs = np.unique(np.concatenate([np.unique(lift), np.zeros(1, np.int64)]))
    col = {int(v): c for c, v in enumerate(ids)}
    inv = np.searchsorted(labs, lift)
    pairs = graph.tree.query_pairs(p.cand_mm * 1e-3, output_type="ndarray")
    A = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    A = (A + A.T + sparse.identity(n)).tocsr()
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, len(labs)))
    cand = (A @ onehot).toarray() > 0
    V = np.zeros((n, len(labs)), np.float64)
    nonfood_cols = [c for c, v in enumerate(ids) if not 1 <= int(v) < 252]
    for li, lab in enumerate(labs):
        if lab == 0:                                  # background / table / container / clutter: one non-food class
            V[:, li] = votes[:, nonfood_cols].sum(1) if nonfood_cols else 0.0
        elif int(lab) in col:
            V[:, li] = votes[:, col[int(lab)]]
    tot = V.sum(1, keepdims=True)
    P = np.where(tot > 0, V / np.maximum(tot, 1e-12), 1.0 / np.maximum(cand.sum(1, keepdims=True), 1))
    P = np.where(cand, np.maximum(P, p.floor), 0.0)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    cand[np.arange(n), inv] = True
    P[np.arange(n), inv] = np.maximum(P[np.arange(n), inv], p.floor)
    U = np.where(cand, -np.log(np.maximum(P, 1e-9)), BIG)
    return labs, U, inv


def alpha_expansion(U, cur, free, gi, gj, gw, sweeps=3):
    """Min sum U[i,l_i] + sum_edges gw[e]*[l_i!=l_j] over `free` nodes; non-free nodes keep `cur`. -> local labels."""
    import maxflow
    lab = cur.copy()
    fidx = np.flatnonzero(free)
    if len(fidx) == 0:
        return lab
    pos = -np.ones(len(lab), np.int64)
    pos[fidx] = np.arange(len(fidx))
    fi, fj = free[gi], free[gj]
    both = fi & fj
    ei, ej, ew = pos[gi[both]], pos[gj[both]], gw[both]
    # free-fixed edges: the fixed side's label is a constant
    m1 = fi & ~fj
    m2 = fj & ~fi
    si = np.concatenate([pos[gi[m1]], pos[gj[m2]]])
    sf = np.concatenate([gj[m1], gi[m2]])
    sw = np.concatenate([gw[m1], gw[m2]])
    fixed_lab = lab[sf]
    Uf = U[fidx]
    nf = len(fidx)
    for _ in range(sweeps):
        changed = False
        # most frequent labels first
        order = np.argsort(-np.bincount(lab[fidx], minlength=U.shape[1]))
        for alpha in order:
            cl = lab[fidx]
            if not (Uf[:, alpha] < BIG).any():
                continue
            d0 = Uf[np.arange(nf), cl]
            d1 = Uf[:, alpha].copy()
            # pairwise between free nodes
            li, lj = cl[ei], cl[ej]
            A_ = ew * (li != lj)
            B_ = ew * (li != alpha)
            C_ = ew * (alpha != lj)
            u1 = np.zeros(nf)
            np.add.at(u1, ei, C_ - A_)
            np.add.at(u1, ej, -C_)
            d1 = d1 + u1
            # free-fixed edges
            c0 = sw * (cl[si] != fixed_lab)
            c1 = sw * (alpha != fixed_lab)
            d0 = d0 + np.bincount(si, weights=c0, minlength=nf)
            d1 = d1 + np.bincount(si, weights=c1, minlength=nf)
            cap = B_ + C_ - A_
            g = maxflow.Graph[float](nf, len(ei))
            g.add_nodes(nf)
            g.add_edges(ei, ej, cap, np.zeros_like(cap))
            m0 = np.minimum(d0, d1)
            g.add_grid_tedges(np.arange(nf), d1 - m0, d0 - m0)
            g.maxflow()
            seg = g.get_grid_segments(np.arange(nf))
            new = np.where(seg & (Uf[:, alpha] < BIG), alpha, cl)
            if (new != cl).any():
                changed = True
                lab[fidx] = new
        if not changed:
            break
    return lab


def refine_labels(lift, votes, ids, graph: Graph, p: RefineParams, share=None):
    """lift (n,) int, votes (n,C)|None, ids (C,)|None -> refined labels (n,), info."""
    lift0 = np.asarray(lift).astype(np.int64)
    food0 = (lift0 >= 1) & (lift0 < 252)
    lift = np.where(food0, lift0, 0)                 # non-food labels (0/252/253/254) collapse to one class
    if votes is None:
        votes, ids = pseudo_votes(lift, share, graph, p.cand_mm)
    free = _band(graph, lift, p.band_mm)
    labs, U, cur = unary_costs(lift, votes, ids, graph, p, free)
    new = alpha_expansion(U, cur, free, graph.i, graph.j, graph.weights(p), p.sweeps)
    out = labs[new]
    out = np.where((out == 0) & ~food0, lift0, out)  # untouched non-food points keep their original (e.g. 254) label
    return out.astype(np.int32), dict(n_free=int(free.sum()), n_changed=int((out != lift0).sum()))


def refine_cloud(xyz, rgb, normal, up, lift, votes=None, ids=None, share=None, params: RefineParams | None = None,
                 voxel_mm=2.0):
    """Full-cloud entry (run_meal): voxelise at 2 mm, refine on the voxel scene, propagate changes to all points.
    `up` = unit vector of the height axis. Returns refined labels (n,)."""
    p = params or RefineParams()
    xyz = np.asarray(xyz, np.float64)
    g = np.floor((xyz - xyz.min(0)) / (voxel_mm * 1e-3)).astype(np.int64)
    key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
    _, vi = np.unique(key, return_index=True)
    vi = np.sort(vi)
    up = np.asarray(up, np.float64)
    gr = Graph(xyz[vi], np.asarray(rgb)[vi], np.asarray(normal)[vi], xyz[vi] @ up, p.k, p.radius_mm)
    v = None if votes is None else votes[vi]
    new_v, info = refine_labels(np.asarray(lift)[vi], v, ids, gr, p, None if share is None else np.asarray(share)[vi])
    out = np.asarray(lift).copy()
    out[vi] = new_v
    rest = np.setdiff1d(np.arange(len(xyz)), vi)
    if len(rest):
        d, nn = gr.tree.query(xyz[rest])
        ch = new_v[nn] != np.asarray(lift)[vi][nn]
        out[rest[ch]] = new_v[nn[ch]]
    return out, info


# ------------------------------------------------------------------------------------------------ evaluation harness
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline"))
OUT = REPO / "results/boundary_refine"
SANDBOX = REPO / "results/lift_replay"
SCANS = OUT / "snap/hard_v12_snap/scans"   # frozen copy of the 11 hard_v12 seg_points (the live data dir was regenerated mid-study)
LIVE = REPO / "data/synth/hard_v12_sam3_v2_image/scans"
SIDS = ["s000000", "s000001", "s000002", "s000007", "s000009", "s000011", "s000012", "s000013", "s000014",
        "s000016", "s000017"]


def cache_scan(sid):
    """Re-run labels.lift_for_meal on the stored masks of the covote replay -> soft votes (N, C)."""
    from foodscan import labels as L
    from foodscan.loader import Scan
    from foodscan.segment import load_masks
    sd, od = LIVE / sid, SANDBOX / sid
    z = np.load(sd / "gt/seg_points.npz")
    xyz, nrm = z["xyz"].astype(np.float64), z["normal"].astype(np.float64)
    masks, meta = load_masks(od, od / "masks")
    lift = L.lift_for_meal(Scan(sd), masks, xyz, nrm, meta, weighting=L.DEFAULT_LINKED_WEIGHTING)
    ref = np.load(od / "lift.npz")["lift_label"]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "cache" / f"{sid}.npz", votes=lift.votes.astype(np.float32), ids=lift.ids,
                        lift_label=lift.lift_label, lift_share=lift.lift_share)
    return sid, bool((ref == lift.lift_label).all()), lift.votes.shape


class Scene:
    """One scan on the sweep's 2 mm voxel scene (same construction as replay_lift.evaluate_lift)."""

    def __init__(self, path, kind="v12", cache=None):
        from seg3d.data import load_scan, semantic_of, voxelize
        path = Path(path)
        d = load_scan(path, "manifest")
        vi, _ = voxelize(d["xyz"], 2e-3)
        self.vi, self.name = vi, f"{path.parent.parent.name}/{path.name}"
        lab = d["label"][vi]
        sem = semantic_of(lab)
        self.sc = dict(xyz=d["xyz"][vi].astype(np.float64), label=lab, sem=sem, inst=np.where(sem == 3, lab, 0).astype(np.int64),
                       ignore=np.isin(lab, d["ignore_ids"]) if d["ignore_ids"] else np.zeros(len(vi), bool))
        self.n_full = len(d["xyz"])
        if kind in ("v12", "r40"):
            c = np.load(cache or (OUT / "cache" / f"{path.name}.npz"))
            self.lift_full = c["lift_label"].astype(np.int64)
            self.votes, self.ids, self.share = c["votes"][vi], c["ids"], c["lift_share"][vi]
        else:
            self.lift_full = d["lift_label"].astype(np.int64)
            self.votes = self.ids = None
            self.share = d["lift_share"][vi]
        self.lift = self.lift_full[vi]
        t = time.time()
        self.graph = Graph(d["xyz"][vi], d["rgb"][vi], d["normal"][vi], d["xyz"][vi, 2], 12, 8.0)
        self.t_graph = time.time() - t
        self.base = self.score(self.lift)

    def score(self, lab):
        from eval.lift_sweep import evaluate
        from seg3d import metrics as M
        full = self.lift_full.copy()
        full[self.vi] = lab
        r = evaluate(SimpleNamespace(sc=self.sc, vi=self.vi), full)
        sel = ~self.sc["ignore"]
        gs, gi = self.sc["sem"][sel], self.sc["inst"][sel]
        bs, bi = M.lift_baseline_pred(self.lift[sel])
        ps, pi = M.lift_baseline_pred(np.asarray(lab)[sel])
        rb, rr = M.point_correct(bs, bi, gs, gi), M.point_correct(ps, pi, gs, gi)
        r["corr"], r["broken"], r["nsel"] = int((~rb & rr).sum()), int((rb & ~rr).sum()), int(sel.sum())
        r["n_err"] = int((~rb).sum())
        r["changed"] = int((np.asarray(lab) != self.lift).sum())
        return r

    def oracle(self, band_mm):
        """Snap lift-wrong points within band_mm of a lift-label boundary to their GT item (matched id)."""
        from seg3d import metrics as M
        sel = ~self.sc["ignore"]
        gs, gi = self.sc["sem"][sel], self.sc["inst"][sel]
        bs, bi = M.lift_baseline_pred(self.lift[sel])
        rb = M.point_correct(bs, bi, gs, gi)
        _, _, m = M.match(gi, bi)
        band = _band(self.graph, np.where((self.lift >= 1) & (self.lift < 252), self.lift, 0), band_mm)[sel]
        tgt = np.zeros(len(gi), np.int64)
        for g in np.unique(gi[gi > 0]):
            tgt[gi == g] = m[g][0] if g in m else 100000 + int(g)
        lab = self.lift.copy()
        idx = np.flatnonzero(sel)[band & ~rb]
        lab[idx] = tgt[np.flatnonzero(band & ~rb)]
        return self.score(lab)

    def run(self, p: RefineParams):
        t = time.time()
        lab, info = refine_labels(self.lift, self.votes, self.ids, self.graph, p, self.share)
        dt = time.time() - t
        return dict(self.score(lab), t=dt, n_free=info["n_free"])


# ------------------------------------------------------------------------------------------------ 40-scan set (hard_v12 + hard_v13)
REPLAY40 = REPO / "results/lift_replay40"
CACHE40 = OUT / "cache40"
LIFT_BATCHES = ("v12", "v13")


def sids40():
    """['v12/s000000', ..., 'v13/s000019'] (sid convention of eval.lift_sweep.roots, always with the batch prefix)."""
    from eval.lift_sweep import ROOTS
    return [f"{b}/{p.name}" for b in LIFT_BATCHES for p in sorted(ROOTS[b][0].iterdir()) if (p / "gt/seg_points.npz").exists()]


def tag40(sid):
    return sid.replace("/", "__")


def replay40_scan(sid, linker="covote", cluster_thresh=0.8):
    """Co-vote-linked lift of one real-vote scan from its stored raw SAM 3 maps (any batch): segment_scan(from_raw) ->
    labels.lift_for_meal. Writes results/lift_replay40/<tag>/lift.npz and the soft-vote cache cache40/<tag>.npz."""
    import tempfile
    from eval.lift_sweep import roots
    from foodscan import labels as L
    from foodscan.loader import Scan
    from foodscan.segment import load_masks, make_linker, segment_scan
    sd, rawsd = roots(sid)
    z = np.load(sd / "gt/seg_points.npz")
    xyz, nrm = z["xyz"].astype(np.float64), z["normal"].astype(np.float64)
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        od = Path(td)
        segment_scan(sd, [], from_raw=rawsd / "masks_sam3_v2i", out_dir=od / "masks",
                     linker=make_linker(sd, xyz, nrm, method=linker, cluster_thresh=cluster_thresh))
        masks, meta = load_masks(od, od / "masks")
        (REPLAY40 / tag40(sid)).mkdir(parents=True, exist_ok=True)
        lift = L.lift_for_meal(Scan(sd), masks, xyz, nrm, meta, weighting=L.DEFAULT_LINKED_WEIGHTING,
                               save_to=REPLAY40 / tag40(sid) / "lift.npz")
    CACHE40.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE40 / f"{tag40(sid)}.npz", votes=lift.votes.astype(np.float32), ids=lift.ids,
                        lift_label=lift.lift_label, lift_share=lift.lift_share, lift_nvotes=lift.lift_nvotes)
    return sid, int(lift.lift_label.max())


def scene40(sid):
    from eval.lift_sweep import roots
    return Scene(roots(sid)[0], "r40", cache=CACHE40 / f"{tag40(sid)}.npz")


def grid_union():
    """The 486 configs of the 11-scan study (grid_v12 + grid_v12b)."""
    import itertools
    out = []
    for band in (3.0, 6.0, 12.0):
        for ws, cs, ns, zs in (((0.5, 1.0, 2.0), (3.0, 5.0, 10.0), (10.0, 15.0, 30.0), (0.0, 3.0, 6.0)),
                               ((0.7, 1.0, 1.4), (4.0, 5.0, 7.0), (5.0, 7.5, 10.0), (2.0, 3.0, 4.5))):
            out += [RefineParams(w=w, s_c=c, s_n=n, s_z=z, band_mm=band) for w, c, n, z in itertools.product(ws, cs, ns, zs)]
    return list(dict.fromkeys(out))


def _grid40_job(a):
    sid, cfgs, tag = a
    sc = scene40(sid)
    res = {key(p): sc.run(p) for p in cfgs}
    (OUT / f"grid_{tag}").mkdir(parents=True, exist_ok=True)
    json.dump(dict(name=sid, base=sc.base, res=res, t_graph=sc.t_graph, n_vox=len(sc.vi)),
              open(OUT / f"grid_{tag}" / (tag40(sid) + ".json"), "w"), default=float)
    return sid


def _replay40_job(sid):
    return replay40_scan(sid)


def report40(tag="r40"):
    """40-scan tables: fixed 11-scan params (no retune) and leave-one-batch-out retune, per batch and pooled."""
    from eval.lift_sweep import bootstrap, pooled
    R = _load_grid(tag)
    names = sorted(R)
    batch = {s: s.split("/")[0] for s in names}
    keys = list(R[names[0]]["res"])
    fixed = key(RefineParams())
    assert fixed in keys, "fixed 11-scan params not in the grid"
    P = lambda ss, k: pooled([R[s]["res"][k] for s in ss])      # noqa: E731

    def pick(train):
        return max(keys, key=lambda k: P(train, k)["inst"])

    lobo, picks = {}, {}
    for b in LIFT_BATCHES:
        tr = [s for s in names if batch[s] != b]
        k = pick(tr)
        picks[b] = k
        for s in names:
            if batch[s] == b:
                lobo[s] = R[s]["res"][k]
    allk = pick(names)

    def row(label, sub, rowof):
        rows = [rowof(s) for s in sub]
        base = [R[s]["base"] for s in sub]
        pl, pb = pooled(rows), pooled(base)
        d_item = bootstrap(np.concatenate([r["iou"] for r in rows]) - np.concatenate([r["iou"] for r in base]))
        d_scan = bootstrap(np.array(pl["scan"]) - np.array(pb["scan"]))
        corr, brk = sum(r["corr"] for r in rows), sum(r["broken"] for r in rows)
        nerr, nsel = sum(r["n_err"] for r in rows), sum(r["nsel"] for r in rows)
        return dict(label=label, n=len(sub), inst=pl["inst"], base_inst=pb["inst"], pq=pl["pq"], base_pq=pb["pq"],
                    bnd=pl["bnd"], base_bnd=pb["bnd"], d_scan=d_scan, d_item=d_item, corr=corr / max(nerr, 1),
                    broken_pct=brk / max(nsel, 1) * 100, broken=brk, corrected=corr, n_err=nerr,
                    n_up=int(sum(np.mean(a["iou"]) > np.mean(b["iou"]) + 1e-9 for a, b in zip(rows, base))),
                    n_down=int(sum(np.mean(a["iou"]) < np.mean(b["iou"]) - 1e-9 for a, b in zip(rows, base))))

    arms = [("fixed 11-scan params", lambda s: R[s]["res"][fixed]),
            ("leave-one-batch-out retune", lambda s: lobo[s]),
            ("all-40 oracle-tuned (optimistic)", lambda s: R[s]["res"][allk])]
    out = {}
    L = ["| arm | set | n scans | lift inst mIoU | CRF inst mIoU | d scan-mIoU (95% CI) | d items (95% CI) | lift PQ -> CRF PQ | boundary IoU lift -> CRF | scans +/-  | corrected / errors | broken (% pts) | broken pts |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, rowof in arms:
        for sname, sub in (("v12", [s for s in names if batch[s] == "v12"]), ("v13", [s for s in names if batch[s] == "v13"]), ("pooled", names)):
            r = row(label, sub, rowof)
            out[f"{label}|{sname}"] = r
            L.append(f"| {label} | {sname} | {r['n']} | {r['base_inst']:.3f} | {r['inst']:.3f} | {r['d_scan'][0]:+.3f} [{r['d_scan'][1]:+.3f}, {r['d_scan'][2]:+.3f}] | "
                     f"{r['d_item'][0]:+.3f} [{r['d_item'][1]:+.3f}, {r['d_item'][2]:+.3f}] | {r['base_pq']:.3f} -> {r['pq']:.3f} | "
                     f"{r['base_bnd']:.3f} -> {r['bnd']:.3f} | {r['n_up']}/{r['n_down']} | {r['corr']:.3f} | {r['broken_pct']:.3f} | {r['broken']} |")
    def pj(k):
        c = json.loads(k)
        return f"band {c['band_mm']:g} w={c['w']} sc={c['s_c']} sn={c['s_n']} sz={c['s_z']}"
    L += ["", "LOBO picks: " + "; ".join(f"train on other batch -> test {b}: {pj(k)}" for b, k in picks.items()) + f"; all-40 best: {pj(allk)}"]
    L += ["", "Per-scan inst mIoU (lift -> fixed -> LOBO):", "", "| scan | lift | fixed | LOBO |", "|---|---|---|---|"]
    for s in names:
        L.append(f"| {s} | {np.mean(R[s]['base']['iou']):.3f} | {np.mean(R[s]['res'][fixed]['iou']):.3f} | {np.mean(lobo[s]['iou']):.3f} |")
    txt = "\n".join(L)
    (OUT / f"tables_{tag}.md").write_text(txt)
    json.dump(dict(rows=out, picks=picks, all40=allk), open(OUT / f"report_{tag}.json", "w"), default=float, indent=1)
    print(txt)
    return out


def grid_configs(full=True):
    out = []
    for band in (3.0, 6.0, 12.0):
        for w in (0.5, 1.0, 2.0, 4.0):
            for s_c in (5.0, 10.0, 20.0):
                for s_n in (15.0, 30.0, 60.0):
                    for s_z in (3.0, 6.0, 0.0):
                        out.append(RefineParams(w=w, s_c=s_c, s_n=s_n, s_z=s_z, band_mm=band))
    return out


def key(p: RefineParams):
    return json.dumps(asdict(p), sort_keys=True)


def _cache_job(sid):
    return cache_scan(sid)


def _grid_job(a):
    path, kind, cfgs, tag = a
    sc = Scene(path, kind)
    res = {key(p): sc.run(p) for p in cfgs}
    orc = {str(b): sc.oracle(b) for b in (3.0, 6.0, 12.0)}
    (OUT / f"grid_{tag}").mkdir(parents=True, exist_ok=True)
    json.dump(dict(name=sc.name, base=sc.base, oracle=orc, res=res, t_graph=sc.t_graph, n_vox=len(sc.vi)),
              open(OUT / f"grid_{tag}" / (sc.name.replace("/", "__") + ".json"), "w"), default=float)
    return sc.name


def _load_grid(tag):
    rows = {}
    for t in tag.split(","):                       # several grid dirs merge their configs per scan
        for f in sorted((OUT / f"grid_{t}").glob("*.json")):
            d = json.load(open(f))
            if d["name"] in rows:
                rows[d["name"]]["res"].update(d["res"])
            else:
                rows[d["name"]] = d
    return rows


def report(tag="v12", pick_from=None):
    """Tables for one grid dir. Selection (all-scan and leave-one-scan-out) maximises pooled item IoU."""
    from eval.lift_sweep import bootstrap, loso_pick, pooled
    R = _load_grid(tag)
    names = sorted(R)
    keys = list(R[names[0]]["res"])
    table = {k: {s: R[s]["res"][k] for s in names} for k in keys}
    P = {k: pooled([table[k][s] for s in names]) for k in keys}
    base = pooled([R[s]["base"] for s in names])
    best = max(keys, key=lambda k: P[k]["inst"])
    loo_rows, picks = loso_pick(table, keys, names)

    def line(label, rows, k=None):
        pl = pooled(rows)
        b_iou = np.concatenate([R[s]["base"]["iou"] for s in names])
        iou = np.concatenate([r["iou"] for r in rows])
        d_item = bootstrap(iou - b_iou)
        d_scan = bootstrap(np.array(pl["scan"]) - np.array(base["scan"]))
        corr, brk = sum(r.get("corr", 0) for r in rows), sum(r.get("broken", 0) for r in rows)
        nerr, nsel = sum(r.get("n_err", 0) for r in rows), sum(r.get("nsel", 1) for r in rows)
        t = np.mean([r.get("t", 0) for r in rows]) + np.mean([R[s]["t_graph"] for s in names])
        return (f"| {label} | {pl['inst']:.3f} | {d_item[0]:+.3f} [{d_item[1]:+.3f}, {d_item[2]:+.3f}] | "
                f"{d_scan[0]:+.3f} [{d_scan[1]:+.3f}, {d_scan[2]:+.3f}] | {pl['pq']:.3f} | {pl['bnd']:.3f} | "
                f"{corr / max(nerr, 1):.3f} | {brk / max(nsel, 1) * 100:.3f} | {t:.1f} |"), pl["inst"] - base["inst"]

    L = ["| setting | inst mIoU | d inst (items, 95% CI) | d scan-mIoU (11 scans, 95% CI) | PQ | boundary IoU 3mm | corrected / lift errors | broken (% of points) | s / scan |",
         "|---|---|---|---|---|---|---|---|---|"]
    b = [R[s]["base"] for s in names]
    L.append(line("lift (co-vote, reference)", [dict(r, t=0, corr=0, broken=0) for r in b])[0].replace("+0.000 [+0.000, +0.000]", "-"))
    gains = {}
    for label, rows in [("all-scan-tuned " + f"{best}", [table[best][s] for s in names]), ("leave-one-scan-out", loo_rows)]:
        l, g = line(label, rows)
        L.append(l)
        gains[label[:8]] = g
    for bd in (3.0, 6.0, 12.0):
        kb = max([k for k in keys if json.loads(k)["band_mm"] == bd], key=lambda k: P[k]["inst"])
        l, g = line(f"best at band {bd:g} mm", [table[kb][s] for s in names])
        L.append(l)
        gains[f"b{bd:g}"] = (g, kb)
    L.append("")
    L.append("Oracle (snap lift-wrong points within band of a lift-label boundary to GT item; our band definition):")
    L.append("")
    L.append("| band | oracle inst mIoU | gain | realised by best-at-band | realised by all-scan-tuned | realised by LOO |")
    L.append("|---|---|---|---|---|---|")
    for bd in (3.0, 6.0, 12.0):
        o = pooled([R[s]["oracle"][str(bd)] for s in names])
        og = o["inst"] - base["inst"]
        L.append(f"| {bd:g} mm | {o['inst']:.3f} | {og:+.3f} | {gains[f'b{bd:g}'][0] / og:.1%} | "
                 f"{gains['all-scan'] / og:.1%} | {gains['leave-on'] / og:.1%} |")
    L.append("")
    L.append("LOO picks: " + "; ".join(f"{s}: {json.loads(p)['band_mm']:g}mm w={json.loads(p)['w']} sc={json.loads(p)['s_c']} sn={json.loads(p)['s_n']} sz={json.loads(p)['s_z']}" for s, p in zip(names, picks)))
    L.append("")
    L.append("Per-scan inst mIoU (lift -> all-scan-tuned -> LOO):")
    L.append("")
    L.append("| scan | n vox | lift | tuned | LOO | oracle 6mm |")
    L.append("|---|---|---|---|---|---|")
    for s, lr in zip(names, loo_rows):
        L.append(f"| {s} | {R[s]['n_vox']} | {np.mean(R[s]['base']['iou']):.3f} | {np.mean(table[best][s]['iou']):.3f} | "
                 f"{np.mean(lr['iou']):.3f} | {np.mean(R[s]['oracle']['6.0']['iou']):.3f} |")
    txt = "\n".join(L)
    (OUT / f"tables_{tag.replace(',', '+')}.md").write_text(txt)
    print(txt)
    return best, picks


def report_val(tag="val", extra=""):
    """Fixed-parameter rows (no selection on these scans) for the pseudo-vote val set, all scans and the harder subset."""
    from eval.lift_sweep import bootstrap, pooled
    R = _load_grid(tag)
    names = sorted(R)
    keys = list(R[names[0]]["res"])
    hard = [s for s in names if np.mean(R[s]["base"]["iou"]) < 0.95]
    L = []
    for label, sub in (("all", names), ("lift mIoU < 0.95", hard)):
        base = pooled([R[s]["base"] for s in sub])
        L += ["", f"**{label}: {len(sub)} scans, {base['n_items']} items**", "",
              "| setting | inst mIoU | d inst (items, 95% CI) | PQ | boundary IoU | corrected / lift errors | broken % pts | s / scan |",
              "|---|---|---|---|---|---|---|---|",
              f"| lift (pseudo votes) | {base['inst']:.3f} | - | {base['pq']:.3f} | {base['bnd']:.3f} | - | - | - |"]
        b_iou = np.concatenate([R[s]["base"]["iou"] for s in sub])
        for k in keys:
            rows = [R[s]["res"][k] for s in sub]
            pl = pooled(rows)
            d = bootstrap(np.concatenate([r["iou"] for r in rows]) - b_iou)
            corr, brk = sum(r["corr"] for r in rows), sum(r["broken"] for r in rows)
            nerr, nsel = sum(r["n_err"] for r in rows), sum(r["nsel"] for r in rows)
            c = json.loads(k)
            L.append(f"| band {c['band_mm']} w={c['w']} sc={c['s_c']} sn={c['s_n']} sz={c['s_z']} | {pl['inst']:.3f} | "
                     f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {pl['pq']:.3f} | {pl['bnd']:.3f} | {corr / max(nerr, 1):.3f} | "
                     f"{brk / max(nsel, 1) * 100:.3f} | {np.mean([r['t'] for r in rows]):.1f} |")
    txt = "\n".join(L)
    (OUT / f"tables_{tag}.md").write_text(txt)
    print(txt)


def main():
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "grid", "val", "time", "report", "replay40", "grid40", "report40"])
    ap.add_argument("--procs", type=int, default=22)
    ap.add_argument("--configs", default=None, help="json list of RefineParams dicts (default: the full grid)")
    ap.add_argument("--tag", default="v12")
    ap.add_argument("--only-lowacc", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = [RefineParams(**c) for c in json.load(open(a.configs))] if a.configs else grid_configs()
    if a.cmd == "replay40":
        with Pool(a.procs) as p:
            for r in p.imap_unordered(_replay40_job, sids40()):
                print(r, flush=True)
    elif a.cmd == "grid40":
        cf = [RefineParams(**c) for c in json.load(open(a.configs))] if a.configs else grid_union()
        with Pool(a.procs) as p:
            for r in p.imap_unordered(_grid40_job, [(s, cf, a.tag) for s in sids40()]):
                print(r, flush=True)
    elif a.cmd == "report40":
        report40(a.tag)
    elif a.cmd == "cache":
        with Pool(a.procs) as p:
            for r in p.imap_unordered(_cache_job, SIDS):
                print(r, flush=True)
    elif a.cmd == "report":
        report_val(a.tag) if a.tag.startswith("val") or a.tag.endswith("c") else report(a.tag)
    elif a.cmd == "time":
        sc = Scene(SCANS / SIDS[0], "v12")
        print("graph", sc.t_graph, "n", len(sc.vi), "base", np.mean(sc.base["iou"]))
        for p in cfgs[:3]:
            r = sc.run(p)
            print(p, np.mean(r["iou"]), r["t"], r["n_free"], r["corr"], r["broken"])
    elif a.cmd == "grid":
        jobs = [(SCANS / s, "v12", cfgs, a.tag) for s in SIDS]
        with Pool(a.procs) as p:
            for r in p.imap_unordered(_grid_job, jobs):
                print(r, flush=True)
    elif a.cmd == "val":
        split = json.load(open(REPO / "results/seg3d/split_v3.json"))["val"]
        jobs = []
        for s in split:
            s = OUT / "snap" / Path(s).relative_to("data/synth")      # frozen copies, see SCANS
            if not (s / "gt/seg_points.npz").exists():
                continue
            z = np.load(s / "gt/seg_points.npz")
            if "lift_label" in z.files:
                jobs.append((s, "pseudo", cfgs, a.tag))
        with Pool(a.procs) as p:
            for r in p.imap_unordered(_grid_job, jobs):
                print(r, flush=True)


if __name__ == "__main__":
    main()
