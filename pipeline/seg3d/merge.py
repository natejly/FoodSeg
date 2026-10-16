"""Explicit fragment merging for lifted SAM 3 instances on fused food scans (CPU only, no learned point backbone).

Fragment graph: nodes = lift food instances (ids 1..251; 252/253/254 are never nodes), edges = pairs of instances that come within
d_contact of each other.  Each edge gets seam / colour / normal / height / size / vote / convexity features (FEATS).  Two deciders:
  heuristic : contact_ratio > t1 & colour dE < t2 & normal angle < t3 & height step < t4
  learned   : classifier probability (HistGradientBoosting / logistic regression) of "same GT item" > 0.5
both followed by the same safety rules (convexity of the union >= c_min, low-share side vs a large side, seam owned by the container),
and merged agglomeratively (best score first, features recomputed for the merged node).

  python -m seg3d.merge run            full experiment (writes results/seg3d/merge/*.json, *.md)
  python -m seg3d.merge apply SCAN_DIR [--out merged_lift.npy]   merge one scan with the heuristic (HEUR_FINAL)  (cwd .../FoodSeg/pipeline)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import augment as A
from . import metrics as M

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/seg3d/merge"
FEATS = ["contact_ratio", "n_seam", "seam_frac", "dE", "chi2", "ang", "hstep", "size_ratio", "log_n_small", "log_n_large",
         "share_small", "share_large", "nv_small", "nv_large", "convexity", "cont_frac", "n_small", "n_large"]
HEUR0 = dict(t1=0.15, t2=15.0, t3=40.0, t4=0.006)                 # t3 degrees, t4 metres
HEUR_FINAL = dict(t1=0.3, t2=8, t3=60, t4=0.0015)                  # grid optimum on all 11 real scans (results/seg3d/merge/results.json)
SAFE0 = dict(c_min=0.6, s_min=0.35, large_pts=400, cont_max=0.5)


# ----------------------------------------------------------------------------- scenes
def load_scene(scan: Path, item_filter="manifest", voxel_mm=2.0) -> dict:
    """Canonicalised, deterministically voxelised scan (same recipe as the evaluator).  Contains the stored lift."""
    from .data import load_scan, voxelize, semantic_of
    d = load_scan(Path(scan), item_filter)
    vi, _ = voxelize(d["xyz"], voxel_mm * 1e-3)
    sc = dict(name=d["name"], xyz=d["xyz"][vi].astype(np.float64), rgb=d["rgb"][vi], normal=d["normal"][vi].astype(np.float64),
              label=d["label"][vi], real=d["real_votes"], has_lift=d["has_lift"], valid_ids=d["valid_ids"], ignore_ids=d["ignore_ids"],
              vox=voxel_mm * 1e-3)
    if d["has_lift"]:
        sc.update(lift=d["lift_label"][vi], share=d["lift_share"][vi], nv=d["lift_nvotes"][vi])
    sc["sem"] = semantic_of(sc["label"])
    sc["inst"] = np.where(sc["sem"] == 3, sc["label"], 0).astype(np.int64)
    sc["ignore"] = np.isin(sc["label"], sc["ignore_ids"]) if sc["ignore_ids"] else np.zeros(len(vi), bool)
    return sc


def pseudo_lift(sc: dict, seed: int, n_frag=None, corrupt=True) -> dict:
    """Pseudo-vote scene: lift from GT (or the stored lift), R1 corruption, then 1-3 `fragment_lift` calls."""
    rng = np.random.default_rng((seed, 77))
    c = A.AugConfig()
    if sc.get("has_lift") and not sc.get("real"):
        lift, share, nv = sc["lift"], sc["share"], sc["nv"]
    else:
        lift, share, nv = A.synth_lift(sc["label"], rng)
    if corrupt:
        lift, share, nv = A.corrupt_lift(sc["xyz"], lift, share, nv, rng, c, sc["vox"])
    n_frag = int(rng.integers(1, 4)) if n_frag is None else n_frag
    for _ in range(n_frag):
        lift, share, nv, _n = A.fragment_lift(sc["xyz"], lift, share, nv, rng, c)
    return dict(sc, lift=lift.astype(np.int32), share=share.astype(np.float32), nv=nv.astype(np.int16))


# ----------------------------------------------------------------------------- features
def _lab(rgb):
    rgb = np.clip(np.asarray(rgb, np.float64), 0, 1)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = lin @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]]) / np.array([0.9505, 1, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _hull_ml(p, rng):
    if len(p) > 2000:
        p = p[rng.choice(len(p), 2000, replace=False)]
    return M.hull_vol_ml(p)


class SceneGeo:
    """Static geometry of one scene + a feature cache keyed by the member sets of the two nodes."""

    def __init__(self, sc, d_contact=0.004, d_near=0.006):
        self.sc, self.dn = sc, d_near
        self.xyz, self.lift = sc["xyz"], sc["lift"].astype(np.int64)
        self.lab = _lab(sc["rgb"])
        self.tree = cKDTree(self.xyz)
        pr = self.tree.query_pairs(d_contact, output_type="ndarray")
        li, lj = self.lift[pr[:, 0]], self.lift[pr[:, 1]]
        d = li != lj
        self.pd = pr[d]                                                        # all differing-label pairs (boundary counts)
        food = lambda l: (l > 0) & (l < 252)                                   # noqa: E731
        fm = food(li[d]) & food(lj[d])
        self.pf = self.pd[fm]                                                  # food-food pairs
        cm = self.lift == 254
        self.ctree = cKDTree(self.xyz[cm]) if cm.sum() > 10 else None
        self.d_contact = d_contact
        self.cache: dict = {}
        self.ids = [int(i) for i in np.unique(self.lift[food(self.lift)])]

    # -- one edge
    def _feat(self, la, lb, ia, jb, a, b):
        sc, xyz = self.sc, self.xyz
        pa, pb = np.nonzero(la)[0], np.nonzero(lb)[0]
        na, nb = len(pa), len(pb)
        sa, sb = np.unique(ia), np.unique(jb)
        small_is_a = na <= nb
        n_seam_small = len(sa) if small_is_a else len(sb)
        sp = np.unique(np.r_[sa, sb])
        if len(sp) > 300:
            sp = sp[np.random.default_rng(0).choice(len(sp), 300, replace=False)]
        nbrs = self.tree.query_ball_point(xyz[sp], self.dn)
        near = np.unique(np.concatenate([np.asarray(x, np.int64) for x in nbrs])) if len(nbrs) else np.r_[sa, sb]
        na_, nb_ = near[la[near]], near[lb[near]]
        na_ = na_ if len(na_) else sa
        nb_ = nb_ if len(nb_) else sb
        dE = float(np.linalg.norm(self.lab[na_].mean(0) - self.lab[nb_].mean(0)))
        ha = np.histogramdd(np.floor(np.clip(sc["rgb"][pa], 0, 0.999) * 4), bins=[np.arange(5)] * 3)[0].ravel() / na
        hb = np.histogramdd(np.floor(np.clip(sc["rgb"][pb], 0, 0.999) * 4), bins=[np.arange(5)] * 3)[0].ravel() / nb
        chi2 = float(0.5 * np.sum((ha - hb) ** 2 / (ha + hb + 1e-9)))
        ma, mb = sc["normal"][na_].mean(0), sc["normal"][nb_].mean(0)
        ma, mb = ma / (np.linalg.norm(ma) + 1e-9), mb / (np.linalg.norm(mb) + 1e-9)
        ang = float(np.degrees(np.arccos(np.clip(ma @ mb, -1, 1))))
        hstep = float(np.median(np.abs(xyz[ia, 2] - xyz[jb, 2])))
        rng = np.random.default_rng(0)
        va, vb = _hull_ml(xyz[pa], rng), _hull_ml(xyz[pb], rng)
        vu = _hull_ml(xyz[np.r_[pa, pb]], rng)
        conv = float(min((va + vb) / vu, 1.5)) if vu > 0 and va > 0 and vb > 0 else 1.0
        if self.ctree is not None:
            cont = float((self.ctree.query(xyz[sp], k=1)[0] < self.d_contact).mean())
        else:
            cont = 0.0
        sh = lambda p: float(sc["share"][p].mean())                           # noqa: E731
        nv = lambda p: float(sc["nv"][p].mean())                              # noqa: E731
        ps, pl = (pa, pb) if small_is_a else (pb, pa)
        return dict(n_seam=float(len(sa) + len(sb)), n_seam_small=float(n_seam_small), seam_frac=n_seam_small / min(na, nb),
                    dE=dE, chi2=chi2, ang=ang, hstep=hstep, size_ratio=min(na, nb) / max(na, nb), log_n_small=math.log(min(na, nb)),
                    log_n_large=math.log(max(na, nb)), share_small=sh(ps), share_large=sh(pl), nv_small=nv(ps), nv_large=nv(pl),
                    convexity=conv, cont_frac=cont, n_small=min(na, nb), n_large=max(na, nb))

    def edges(self, root, members):
        """root: int array id -> current node id.  Returns list of dict(a, b, f) with f including contact_ratio."""
        lab = root[self.lift]
        ra, rb = root[self.lift[self.pf[:, 0]]], root[self.lift[self.pf[:, 1]]]
        k = ra != rb
        pf, ra, rb = self.pf[k], ra[k], rb[k]
        if not len(pf):
            return []
        sw = ra > rb
        i0 = np.where(sw, pf[:, 1], pf[:, 0])
        j0 = np.where(sw, pf[:, 0], pf[:, 1])
        a_, b_ = np.minimum(ra, rb), np.maximum(ra, rb)
        key = a_.astype(np.int64) * 512 + b_
        order = np.argsort(key, kind="stable")
        key, i0, j0 = key[order], i0[order], j0[order]
        cuts = np.nonzero(np.diff(key))[0] + 1
        bnd_idx = np.unique(self.pd[root[self.lift[self.pd[:, 0]]] != root[self.lift[self.pd[:, 1]]]])
        bnd = np.bincount(lab[bnd_idx], minlength=len(root))
        out = []
        for ia, jb, kk in zip(np.split(i0, cuts), np.split(j0, cuts), np.split(key, cuts)):
            a, b = int(kk[0] // 512), int(kk[0] % 512)
            ck = (members[a], members[b])
            if ck not in self.cache:
                self.cache[ck] = self._feat(lab == a, lab == b, ia, jb, a, b)
            f = dict(self.cache[ck])
            small = a if (lab == a).sum() <= (lab == b).sum() else b
            f["contact_ratio"] = f["n_seam_small"] / max(bnd[small], 1)
            out.append(dict(a=a, b=b, f=f))
        return out


def fvec(f):
    return np.array([f[k] for k in FEATS], np.float64)


# ----------------------------------------------------------------------------- deciders
def safe(f, c_min=0.6, s_min=0.35, large_pts=400, cont_max=0.5):
    if f["convexity"] < c_min or f["cont_frac"] > cont_max:
        return False
    if (f["share_small"] < s_min and f["n_large"] >= large_pts) or (f["share_large"] < s_min and f["n_small"] >= large_pts):
        return False
    return True


def heuristic_decider(t1, t2, t3, t4, safety=SAFE0):
    def decide(f):
        if f["contact_ratio"] > t1 and f["dE"] < t2 and f["ang"] < t3 and f["hstep"] < t4 and (safety is None or safe(f, **safety)):
            return f["contact_ratio"]
        return None
    return decide


def learned_decider(model, thr=0.5, safety=SAFE0):
    def decide(f):
        if safety is not None and not safe(f, **safety):
            return None
        p = float(model.predict_proba(fvec(f)[None])[0, 1])
        return p if p > thr else None
    return decide


def gt_majority(sc, lab, ids):
    """GT food item (majority) and purity of every node id; -1 when it has no GT food points."""
    out = {}
    for i in ids:
        g = sc["label"][lab == i]
        g = g[(g > 0) & (g < 252)]
        if len(g):
            v, c = np.unique(g, return_counts=True)
            out[i] = (int(v[c.argmax()]), float(c.max() / len(g)))
        else:
            out[i] = (-1, 0.0)
    return out


def agglomerate(geo: SceneGeo, decide, collect=None, tag=None):
    """Merge best-score edge first until no edge passes `decide`.  Returns root array (id -> node id).  `collect`: list receiving
    (fvec, same_item or -1, tag) for every distinct edge state encountered (training data / edge classifier evaluation)."""
    root = np.arange(512)
    members = {i: frozenset([i]) for i in range(512)}
    seen = set()
    while True:
        es = geo.edges(root, members)
        if collect is not None and es:
            lab = root[geo.lift]
            gm = gt_majority(geo.sc, lab, {e[k] for e in es for k in "ab"})
            for e in es:
                key = (members[e["a"]], members[e["b"]])
                if key in seen:
                    continue
                seen.add(key)
                (ga, pa), (gb, pb) = gm[e["a"]], gm[e["b"]]
                same = -1 if (ga < 0 or gb < 0 or min(pa, pb) < 0.5) else int(ga == gb)
                collect.append((fvec(e["f"]), same, tag))
        best, bs = None, -1
        for e in es:
            s = decide(e["f"], ) if not callable(getattr(decide, "oracle", None)) else decide.oracle(e, root, geo)
            if s is not None and s > bs:
                best, bs = e, s
        if best is None:
            return root
        a, b = best["a"], best["b"]
        na, nb = (root[geo.lift] == a).sum(), (root[geo.lift] == b).sum()
        keep, gone = (a, b) if na >= nb else (b, a)
        members[keep] = members[keep] | members[gone]
        root[root == gone] = keep


def oracle_decider():
    def decide(f):
        return None
    def oracle(e, root, geo):
        lab = root[geo.lift]
        gm = gt_majority(geo.sc, lab, [e["a"], e["b"]])
        (ga, pa), (gb, pb) = gm[e["a"]], gm[e["b"]]
        return e["f"]["contact_ratio"] + 1.0 if ga > 0 and ga == gb and min(pa, pb) >= 0.5 else None
    decide.oracle = oracle
    return decide


def merge_scene(sc, decide, geo=None):
    """sc needs xyz rgb normal lift share nv.  Returns merged lift array (same shape as sc['lift'])."""
    geo = geo or SceneGeo(sc)
    return agglomerate(geo, decide)[sc["lift"].astype(np.int64)].astype(np.int32), geo


# ----------------------------------------------------------------------------- evaluation
def scan_eval(sc, lift_pred):
    """-> Accumulator.result()-like dict + item lists for one scan, ignoring points of ignored items."""
    sel = ~sc["ignore"]
    ps, pi = M.lift_baseline_pred(np.asarray(lift_pred)[sel])
    acc = M.Accumulator(volumes=False)
    acc.add(ps, pi, sc["xyz"][sel], sc["sem"][sel], sc["inst"][sel], sc["label"][sel], {}, lift=None)
    return acc


def item_table(sc, lift0, lift1):
    sel = ~sc["ignore"]
    gi = sc["inst"][sel]
    p0, p1 = M.lift_baseline_pred(np.asarray(lift0)[sel])[1], M.lift_baseline_pred(np.asarray(lift1)[sel])[1]
    g_ids, _, m0 = M.match(gi, p0)
    _, _, m1 = M.match(gi, p1)
    rows = []
    for g in g_ids:
        mk = gi == g
        n = mk.sum()
        _, c0 = np.unique(p0[mk & (p0 > 0)], return_counts=True)
        _, c1 = np.unique(p1[mk & (p1 > 0)], return_counts=True)
        i0, i1 = float(m0[g][1]) if g in m0 else 0.0, float(m1[g][1]) if g in m1 else 0.0
        rows.append(dict(iou0=i0, iou1=i1, frag=int((c0 >= 0.15 * n).sum()) >= 2,
                         whole=bool(i1 > 0.5 and len(c1) and c1.max() / n >= 0.85)))
    return rows


def fast_iou(sc, lift_pred):
    """mean item IoU (ignored items removed), no Accumulator; for grid search."""
    sel = ~sc["ignore"]
    pi = M.lift_baseline_pred(np.asarray(lift_pred)[sel])[1]
    g_ids, _, m = M.match(sc["inst"][sel], pi)
    return [m[g][1] if g in m else 0.0 for g in g_ids]


def summarize(scs, lifts0, lifts1, groups=None):
    """Pooled metrics of lifts1 vs lifts0 over scenes (lists aligned)."""
    res = {}
    for nm, ls in (("lift", lifts0), ("merge", lifts1)):
        accs = [scan_eval(s, l) for s, l in zip(scs, ls)]
        ious = [x for a in accs for x in a.inst_iou]
        tp, tpi = sum(a.tp for a in accs), sum(a.tp_iou for a in accs)
        fp, fn = sum(a.fp for a in accs), sum(a.fn for a in accs)
        res[nm] = dict(inst=float(np.mean(ious)), pq=float(tpi / max(tp + .5 * fp + .5 * fn, 1e-9)),
                       count=float(np.mean([a.count_ok for a in accs])), scan=[float(np.mean(a.inst_iou)) if a.inst_iou else np.nan for a in accs],
                       n_items=len(ious), item=ious)
    rows = [r for s, l0, l1 in zip(scs, lifts0, lifts1) for r in item_table(s, l0, l1)]
    res["n_frag"] = sum(r["frag"] for r in rows)
    res["frag_made_whole"] = sum(r["frag"] and r["whole"] for r in rows)
    res["n_correct_lift"] = sum(r["iou0"] > 0.5 for r in rows)
    res["broken"] = sum(r["iou0"] > 0.5 and r["iou1"] <= 0.5 for r in rows)
    res["fixed"] = sum(r["iou0"] <= 0.5 and r["iou1"] > 0.5 for r in rows)
    res["n_merges"] = int(sum(len(set(np.unique(l0[l0 != l1]).tolist())) for l0, l1 in zip(lifts0, lifts1)))
    d = np.array(res["merge"]["scan"]) - np.array(res["lift"]["scan"])
    if groups is not None:                                             # average pseudo seeds per scan first
        gs = sorted(set(groups))
        d = np.array([np.nanmean([x for x, g in zip(d, groups) if g == k]) for k in gs])
    ok = ~np.isnan(d)
    rng = np.random.default_rng(0)
    bs = [d[ok][rng.integers(0, ok.sum(), ok.sum())].mean() for _ in range(2000)]
    res["d_scan"] = [float(d[ok].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return res


# ----------------------------------------------------------------------------- experiment
GRID = dict(t1=[0.1, 0.2, 0.3, 0.45, 0.6, 0.8], t2=[3, 5, 8, 12, 18, 1e9], t3=[30, 45, 60, 90, 181], t4=[0.0015, 0.003, 0.006, 0.012, 1e9])
REAL_DIR = REPO / "results/seg3d/r4_data/real_v12/scans"
PSEUDO_ROOT = REPO / "data/synth/hard_v12"


def _grid_configs():
    import itertools
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def _w_real(sid):
    """Worker: all grid configs + oracle edges + initial edges for one real scan."""
    sc = load_scene(REAL_DIR / sid)
    geo = SceneGeo(sc)
    base = fast_iou(sc, sc["lift"])
    memo, table = {}, {}
    for cfg in _grid_configs() + [dict(t1=-1, t2=1e9, t3=181, t4=1e9, nosafe=True)]:
        for safety in ((SAFE0,) if not cfg.get("nosafe") else (None,)):
            c = {k: v for k, v in cfg.items() if k != "nosafe"}
            root = agglomerate(geo, heuristic_decider(**c, safety=safety))
            k = tuple(root[geo.ids])
            if k not in memo:
                memo[k] = fast_iou(sc, root[sc["lift"].astype(np.int64)])
            table[json.dumps(cfg, sort_keys=True)] = (memo[k], int(sum(root[i] != i for i in geo.ids)))
    col = []
    agglomerate(geo, oracle_decider(), collect=col, tag=sid)
    init = []
    agglomerate(geo, lambda f: None, collect=init, tag=sid)
    return sid, base, table, col, init


def _w_pseudo_edges(args):
    scan, seed = args
    sc = pseudo_lift(load_scene(REPO / scan), seed)
    col = []
    agglomerate(SceneGeo(sc), oracle_decider(), collect=col, tag=f"{scan}#{seed}")
    return col


def _w_pseudo_eval(args):
    scan, seed, heur, models, safety = args
    sc = pseudo_lift(load_scene(REPO / scan), seed)
    geo = SceneGeo(sc)
    out = {"lift": sc["lift"]}
    out["heur"] = agglomerate(geo, heuristic_decider(**heur, safety=safety))[sc["lift"].astype(np.int64)].astype(np.int32)
    out["heur_nosafe"] = agglomerate(geo, heuristic_decider(**heur, safety=None))[sc["lift"].astype(np.int64)].astype(np.int32)
    for k, m in models.items():
        out[k] = agglomerate(geo, learned_decider(m, safety=safety))[sc["lift"].astype(np.int64)].astype(np.int32)
    out["oracle"] = agglomerate(geo, oracle_decider())[sc["lift"].astype(np.int64)].astype(np.int32)
    return out


def make_model(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def fit(kind, rows_pseudo, rows_real, w_real=5.0):
    rows = [(x, y, 1.0) for x, y, _ in rows_pseudo if y >= 0] + [(x, y, w_real) for x, y, _ in rows_real if y >= 0]
    X, y, w = np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    m = make_model(kind)
    if kind == "hgb":
        m.fit(X, y, sample_weight=w)
    else:
        m.fit(X, y, logisticregression__sample_weight=w)
    return m


def run_experiment(n_seed_train=8, n_seed_val=3, procs=22):
    import multiprocessing as mp
    from sklearn.inspection import permutation_importance
    OUT.mkdir(parents=True, exist_ok=True)
    sids = sorted(p.name for p in REAL_DIR.iterdir())
    split = json.loads((REPO / "results/seg3d/split_v3.json").read_text())
    ctx = mp.get_context("fork")
    with ctx.Pool(procs) as pool:
        real = pool.map(_w_real, sids)
        pseudo_jobs = [(s, k) for s in split["train"] for k in range(n_seed_train)]
        pe = pool.map(_w_pseudo_edges, pseudo_jobs, chunksize=2)
    pseudo_rows = [r for c in pe for r in c]
    print("pseudo edges", len(pseudo_rows), "pos", sum(r[1] == 1 for r in pseudo_rows), "neg", sum(r[1] == 0 for r in pseudo_rows), flush=True)
    real_rows = {sid: col for sid, _, _, col, _ in real}
    real_init = {sid: init for sid, _, _, _, init in real}
    print("real edges", sum(len(c) for c in real_rows.values()), "init", sum(len(c) for c in real_init.values()), flush=True)
    cfgs = [c for c in _grid_configs()]
    keyof = lambda c: json.dumps(c, sort_keys=True)                                       # noqa: E731
    tab = {sid: t for sid, _, t, _, _ in real}
    nosafe_key = keyof(dict(t1=-1, t2=1e9, t3=181, t4=1e9, nosafe=True))

    def pick(train_sids):
        best, bs = None, -1
        for c in cfgs:
            v = [x for s in train_sids for x in tab[s][keyof(c)][0]]
            nm = sum(tab[s][keyof(c)][1] for s in train_sids)
            sc_ = np.mean(v) - 1e-6 * nm
            if sc_ > bs:
                best, bs = c, sc_
        return best

    # ---- real LOSO
    scenes = {s: load_scene(REAL_DIR / s) for s in sids}
    geos = {s: SceneGeo(scenes[s]) for s in sids}
    lifts = {k: [] for k in ("lift", "heur0", "heur", "heur_nosafe", "hgb", "hgb_pseudo", "lr", "oracle")}
    chosen, pr_edges = {}, {k: [] for k in ("heur", "hgb", "lr", "hgb_pseudo")}
    hgb_all = {}
    models_pseudo = {k: fit(k, pseudo_rows, []) for k in ("hgb",)}
    for s in sids:
        sc, geo = scenes[s], geos[s]
        tr = [t for t in sids if t != s]
        ch = pick(tr)
        chosen[s] = ch
        li = sc["lift"].astype(np.int64)
        mapit = lambda root: root[li].astype(np.int32)                                    # noqa: E731
        lifts["lift"].append(sc["lift"])
        lifts["heur0"].append(mapit(agglomerate(geo, heuristic_decider(**HEUR0))))
        lifts["heur"].append(mapit(agglomerate(geo, heuristic_decider(**ch))))
        lifts["heur_nosafe"].append(mapit(agglomerate(geo, heuristic_decider(**ch, safety=None))))
        rr = [r for t in tr for r in real_rows[t]]
        mh, ml = fit("hgb", pseudo_rows, rr), fit("lr", pseudo_rows, rr)
        lifts["hgb"].append(mapit(agglomerate(geo, learned_decider(mh))))
        lifts["lr"].append(mapit(agglomerate(geo, learned_decider(ml))))
        lifts["hgb_pseudo"].append(mapit(agglomerate(geo, learned_decider(models_pseudo["hgb"]))))
        lifts["oracle"].append(mapit(agglomerate(geo, oracle_decider())))
        for x, y, _ in real_init[s]:
            if y < 0:
                continue
            f = dict(zip(FEATS, x))
            pr_edges["heur"].append((y, heuristic_decider(**ch)(f) is not None))
            pr_edges["hgb"].append((y, mh.predict_proba(x[None])[0, 1] > 0.5 and safe(f, **SAFE0)))
            pr_edges["lr"].append((y, ml.predict_proba(x[None])[0, 1] > 0.5 and safe(f, **SAFE0)))
            pr_edges["hgb_pseudo"].append((y, models_pseudo["hgb"].predict_proba(x[None])[0, 1] > 0.5 and safe(f, **SAFE0)))
        print("loso", s, ch, flush=True)
    scs = [scenes[s] for s in sids]
    real_res = {k: summarize(scs, lifts["lift"], v) for k, v in lifts.items() if k != "lift"}
    real_res["lift"] = summarize(scs, lifts["lift"], lifts["lift"])["lift"]

    def prf(l):
        y = np.array([a for a, _ in l]); p = np.array([bool(b) for _, b in l])
        tp = int((y.astype(bool) & p).sum())
        return dict(n=len(y), pos=int(y.sum()), pred_pos=int(p.sum()), tp=tp, precision=tp / max(p.sum(), 1), recall=tp / max(y.sum(), 1))
    pr = {k: prf(v) for k, v in pr_edges.items()}

    # ---- importances (full real fit) + final thresholds on all real scans
    allrr = [r for s in sids for r in real_rows[s]]
    hfull, lfull = fit("hgb", pseudo_rows, allrr), fit("lr", pseudo_rows, allrr)
    Xr = np.array([x for s in sids for x, y, _ in real_init[s] if y >= 0]); yr = np.array([y for s in sids for x, y, _ in real_init[s] if y >= 0])
    pi = permutation_importance(hfull, Xr, yr, n_repeats=20, random_state=0, scoring="average_precision")
    imp = sorted(zip(FEATS, pi.importances_mean.tolist()), key=lambda t: -t[1])
    coefs = lfull[-1].coef_[0]
    lr_imp = sorted(zip(FEATS, coefs.tolist()), key=lambda t: -abs(t[1]))
    final_cfg = pick(sids)

    # ---- pseudo val (real scans with a pseudo copy in val excluded from threshold/model fitting)
    val_names = {Path(p).name for p in split["val"] if "hard_v12" in p}
    fit_sids = [s for s in sids if s not in val_names]
    pv_cfg = pick(fit_sids)
    rr_v = [r for s in fit_sids for r in real_rows[s]]
    pv_models = {"hgb": fit("hgb", pseudo_rows, rr_v), "hgb_pseudo": models_pseudo["hgb"], "lr": fit("lr", pseudo_rows, rr_v)}
    jobs = [(s, k, pv_cfg, pv_models, SAFE0) for s in split["val"] for k in range(n_seed_val)]
    with ctx.Pool(procs) as pool:
        pv = pool.map(_w_pseudo_eval, jobs, chunksize=2)
    pscs = [pseudo_lift(load_scene(REPO / s), k) for s in split["val"] for k in range(n_seed_val)]
    groups = [s for s in split["val"] for k in range(n_seed_val)]
    pseudo_res = {k: summarize(pscs, [o["lift"] for o in pv], [o[k] for o in pv], groups=groups) for k in pv[0] if k != "lift"}
    pseudo_res["lift"] = summarize(pscs, [o["lift"] for o in pv], [o["lift"] for o in pv])["lift"]

    out = dict(real=real_res, pseudo_val=pseudo_res, edge_pr_real=pr, chosen_loso=chosen, final_cfg=final_cfg, pseudo_val_cfg=pv_cfg,
               hgb_perm_importance=imp, lr_coef=lr_imp, n_pseudo_edges=len(pseudo_rows), n_real_edges=len(allrr), sids=sids,
               n_pseudo_pos=int(sum(r[1] == 1 for r in pseudo_rows)), n_real_pos=int(sum(r[1] == 1 for r in allrr)),
               n_real_init=int(len(yr)), n_real_init_pos=int(yr.sum()), grid=GRID, safety=SAFE0, heur0=HEUR0)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=lambda o: float(o) if np.isfinite(o) else None))
    # save merged real lifts for inspection
    np.savez_compressed(OUT / "real_merged_lifts.npz", **{f"{k}__{s}": v[i] for k, v in lifts.items() for i, s in enumerate(sids)})
    return out


def apply_scan(scan, out=None):
    sc = load_scene(Path(scan))
    if not sc.get("has_lift"):
        raise SystemExit("scan has no lift_label in gt/seg_points.npz")
    merged, _ = merge_scene(sc, heuristic_decider(**HEUR_FINAL))
    print("lift ids", len(np.unique(sc["lift"])), "-> merged", len(np.unique(merged)))
    if out:
        np.save(out, merged)
    return merged


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_experiment()
    elif len(sys.argv) > 2 and sys.argv[1] == "apply":
        apply_scan(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        print(__doc__)
