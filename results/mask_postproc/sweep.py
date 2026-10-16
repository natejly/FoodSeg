"""2D mask post-processing sweep (CPU).  python results/mask_postproc/sweep.py [--only NAME ...] [--procs 8]

Each config = segment.PostOpts fields (+ co-vote cluster_thresh) replayed through
segment_scan(from_raw) -> co-vote linker -> labels.lift_for_meal -> seg3d metrics on the 11 hard_v12 scans.
Decision precision uses GT seg_ids (diagnostics only; the judge is never seen by the algorithm).
Results -> results/mask_postproc/results.json (per-config, per-scan raw numbers + decision logs).
"""
import argparse, json, sys, functools
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
CACHE = str(HERE / "cache")
RAW = REPO / "data/synth/hard_v12/scans"

CONFIGS = {
    "baseline": {},
    "A2_depth": dict(depth_split=True),
    "A2_depth_dz5_ang30": dict(depth_split=True, dz_cut=0.005, ang_cut=30.0),
    "A2_depth_dz12_ang50": dict(depth_split=True, dz_cut=0.012, ang_cut=50.0),
    "A3_colour": dict(color_split=True),
    "B1_merge": dict(merge_same=True),
    "B2_score0.4": dict(min_score=0.4),
    "B2_score0.5": dict(min_score=0.5),
    "B2_score0.3": dict(min_score=0.3),
    "B2_area0.3pct": dict(min_area_frac=0.003),
    "B2_compact0.5": dict(min_compact=0.5),
    "B2_all_s0.4": dict(min_score=0.4, min_area_frac=0.003, min_compact=0.5),
}


# ------------------------------------------------------------------ GT judge (diagnostics only)
@functools.lru_cache(maxsize=4096)
def gt_frame(sid, fi):
    a = Image.open(RAW / sid / "gt/seg_ids" / f"{fi:06d}.png").resize((960, 720), Image.NEAREST)
    return np.asarray(a)


def hist(sid, fi, m):
    g = gt_frame(sid, fi)[m]
    ids, c = np.unique(g, return_counts=True)
    return ids, c


def dom(sid, fi, m):
    ids, c = hist(sid, fi, m)
    if not len(ids):
        return 0, 0.0
    k = c.argmax()
    return int(ids[k]), float(c[k] / c.sum())


def is_food(g):
    return 0 < g < 253


def make_judge(sid):
    def judge(kind, fi, masks):
        if kind == "drop":
            g, pur = dom(sid, fi, masks[0])
            return (not is_food(g)) or pur < 0.5
        d = [dom(sid, fi, m) for m in sorted(masks, key=lambda m: -m.sum())]
        if kind == "merge":
            return is_food(d[0][0]) and d[0][0] == d[1][0]
        if kind == "split":
            (g0, p0), (g1, p1) = d[0], d[1]
            return is_food(g0) and is_food(g1) and g0 != g1 and p0 >= 0.5 and p1 >= 0.5
    return judge


def mixed_masks(sid, raw_dir):
    """Reference: raw food masks holding >= 2 GT food ids each > 20 % (what a perfect splitter could act on)."""
    n = tot = 0
    for p in sorted(Path(raw_dir).glob("[0-9]*.png")):
        fi = int(p.stem)
        r = np.asarray(Image.open(p))
        for g in np.unique(r[r >= 1000]):
            m = r == g
            ids, c = hist(sid, fi, m)
            fo = [(i, k) for i, k in zip(ids, c) if is_food(int(i))]
            tot += 1
            n += int(sum(k > 0.2 * m.sum() for _, k in fo) >= 2)
    return n, tot


# ------------------------------------------------------------------ per-scan job
def types(sid, lift_full):
    """Primary error types per GT item (eval.lift_errors.classify) on the 2 mm scene."""
    from seg3d import metrics as M
    from seg3d.data import load_scan, semantic_of, voxelize
    from eval.lift_errors import classify, FRAC
    from foodscan.replay_lift import SCANS
    d = load_scan(SCANS / sid, "manifest")
    vi, _ = voxelize(d["xyz"], 2e-3)
    lab = d["label"][vi]
    sem = semantic_of(lab)
    gi = np.where(sem == 3, lab, 0).astype(np.int64)
    ign = np.isin(lab, d["ignore_ids"]) if d["ignore_ids"] else np.zeros(len(vi), bool)
    pi = np.asarray(lift_full)[vi]
    pi = np.where((pi > 0) & (pi < 252), pi, 0)[~ign]
    gi = gi[~ign]
    g_ids, _, m = M.match(gi, pi)
    out = {t: 0 for t in ["ok", "boundary", "fragmented", "merged", "under-covered", "missed"]}
    flags = dict(frag=0, merged=0)
    for g in g_ids:
        gm = gi == g
        n = gm.sum()
        iou = m[g][1] if g in m else 0.0
        ids, cnt = np.unique(pi[gm][pi[gm] > 0], return_counts=True)
        fr = {int(i): c / n for i, c in zip(ids, cnt)}
        oc = {}
        for h in g_ids:
            if h == g:
                continue
            hm = gi == h
            hi, hc = np.unique(pi[hm][pi[hm] > 0], return_counts=True)
            for i, c in zip(hi, hc):
                oc[int(i)] = max(oc.get(int(i), 0.0), c / hm.sum())
        t, f = classify(iou, float((pi[gm] > 0).mean()), fr, oc)
        out[t] += 1
        flags["frag"] += int(f["fragmented"])
        flags["merged"] += int(f["merged"])
    return out, flags


def job(args):
    sid, name, post, ct = args
    from foodscan.replay_lift import replay_scan
    from foodscan.segment import PostOpts
    opts = PostOpts(cache_dir=CACHE, **post)
    res, lift = replay_scan(sid, HERE / "out" / name, cluster_thresh=ct, post=opts, post_judge=make_judge(sid),
                            return_lift=True)
    t, f = types(sid, lift)
    res["types"], res["flags"] = t, f
    res["log"] = opts.log
    return sid, res


def summarize_log(logs):
    out = {}
    for r in logs:
        k = f"{r['kind']}:{r['rule']}"
        o = out.setdefault(k, [0, 0])
        o[1] += 1
        o[0] += int(bool(r.get("ok")))
    return out


def run(name, post, ct=0.8, procs=8):
    from foodscan.replay_lift import SIDS
    with Pool(min(procs, len(SIDS))) as p:
        rs = dict(p.map(job, [(s, name, post, ct) for s in SIDS]))
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--extra", default=None, help="JSON {name: {post fields..., cluster_thresh}} added to CONFIGS")
    a = ap.parse_args()
    cfgs = dict(CONFIGS)
    if a.extra:
        cfgs.update(json.loads(Path(a.extra).read_text()))
    f = HERE / "results.json"
    allr = json.loads(f.read_text()) if f.exists() else {}
    for name in (a.only or list(cfgs)):
        post = dict(cfgs[name])
        ct = post.pop("cluster_thresh", 0.8)
        rs = run(name, post, ct, a.procs)
        allr[name] = dict(post=post, cluster_thresh=ct, scans=rs)
        from eval.lift_sweep import pooled
        from foodscan.replay_lift import SIDS
        p = pooled([rs[s] for s in SIDS])
        ty = {t: sum(rs[s]["types"][t] for s in SIDS) for t in rs[SIDS[0]]["types"]}
        print(name, f"inst {p['inst']:.4f} pq {p['pq']:.4f} exact {p['exact']:.3f} frag {p['frag']} ids "
              f"{sum(rs[s]['n_ids'] for s in SIDS)} types {ty}", summarize_log([r for s in SIDS for r in rs[s]['log']]), flush=True)
        f.write_text(json.dumps(allr, default=str))


if __name__ == "__main__":
    main()
