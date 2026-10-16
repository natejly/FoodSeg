"""CPU sweep over the lift's own knobs on the 11 real-SAM 3 hard_v12 scans (no SAM, no GPU).

    python -m eval.lift_sweep run      [--procs 11] [--only s000000 ...]   # phase 1 + combined grid, writes results/lift_sweep/results.json
    python -m eval.lift_sweep report                                       # tables -> results/lift_sweep/summary.md

Pipeline replayed per scan, from the stored per-frame *raw* SAM 3 maps (per-frame-unique ids,
data/synth/hard_v12/scans/<sid>/masks_sam3_v2i/raw) and scores:
    collect_votes (one projection pass, with score/edge/quality features)
    -> instance linking (repair_ids / mutual-best / co-vote clustering; one-frame clusters dropped, as in
       segment.link_instances) -> vote weighting -> decision rule -> optional smoothing / superpoint majority.
The cloud is the scan's seg_points.npz (same points the stored lift was computed on); metrics are computed on
the 2 mm voxelised canonical scene exactly like seg3d/merge.py (manifest-dropped items ignored), so "lift" here is
comparable to the 0.632 in results/seg3d/ablations.md. Reference rows: `stored` (the V.lift labels in
seg_points.npz), `lift_scan_stored_masks` (labels.lift_scan on the linked masks_sam3 maps) and `current`
(this script's replay with today's defaults; all deltas/CIs are paired against `current`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline"))

from foodscan import labels as L  # noqa: E402
from seg3d import metrics as M  # noqa: E402

OUT = REPO / "results" / "lift_sweep"
SCANS = REPO / "data/synth/hard_v12_sam3_v2_image/scans"
RAWROOT = REPO / "data/synth/hard_v12/scans"
SIDS = ["s000000", "s000001", "s000002", "s000007", "s000009", "s000011", "s000012", "s000013", "s000014",
        "s000016", "s000017"]
ROOTS = {"v12": (SCANS, RAWROOT),
         "v13": (REPO / "data/synth/hard_v13_sam3_v2_image/scans", REPO / "data/synth/hard_v13/scans")}


def roots(sid):
    """sid 'v13/s000003' -> (voted scan dir, raw-mask scan dir); a bare sid is hard_v12."""
    v, _, n = sid.rpartition("/")
    a, b = ROOTS[v or "v12"]
    return a / n, b / n


BIG = 65001
AUX = {BIG, BIG + 1}
MAX_IDS = 252

BASE = dict(link=("repair", 0.5), weighting="cos", edge_px=6.0, min_nvotes=2, margin=0.0, bg_abstain=False,
            smooth=0.0, sp=None, repair2=True, min_frames=2)


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


def _norm(v):
    return tuple(_norm(x) for x in v) if isinstance(v, (list, tuple)) else v


def name_of(c):
    d = {k: _norm(v) for k, v in c.items() if _norm(BASE.get(k)) != _norm(v)}
    return "current" if not d else " ".join(f"{k}={v}" for k, v in d.items())


# ------------------------------------------------------------------------------------------------ config grids
def grids():
    """family -> list of configs (each a full dict)."""
    g = {}
    w = [cfg(weighting="cos+score"), cfg(weighting="cos+qual")]
    for px in (3.0, 6.0, 12.0):
        w += [cfg(weighting="cos+edge", edge_px=px), cfg(weighting="cos+score+edge", edge_px=px),
              cfg(weighting="cos+score+edge+qual", edge_px=px)]
    g["weighting"] = [cfg()] + w
    link = [cfg(link=("repair", t)) for t in (0.3, 0.7)] + [cfg(link=("mutual", t)) for t in (0.3, 0.5)]
    link += [cfg(link=("cluster-cosine", t)) for t in (0.5, 0.65, 0.8, 0.9, 0.95)]
    link += [cfg(link=("cluster-min", t)) for t in (0.4, 0.6, 0.8)]
    link += [cfg(min_frames=m) for m in (1, 3)] + [cfg(link=("cluster-cosine", 0.9), min_frames=m) for m in (1, 3)]
    link += [cfg(repair2=False), cfg(link=("cluster-cosine", 0.9), repair2=False)]
    g["linking"] = [cfg()] + link
    dec = []
    for nv in (1, 2, 3):
        for m in (0.0, 0.1, 0.2):
            for bg in (False, True):
                dec.append(cfg(min_nvotes=nv, margin=m, bg_abstain=bg))
    g["decision"] = [c for c in dec if name_of(c) != "current"] + [cfg()]
    sp = [cfg(smooth=0.004), cfg(smooth=0.002), cfg(smooth=0.008)]
    for n in (200, 400, 600):
        for feat in ("geom", "geom+colour"):
            for mode in ("hard", "soft"):
                sp.append(cfg(sp=(n, feat, mode)))
    g["geometry"] = [cfg()] + sp
    return g


def uniq(cfgs):
    seen, out = set(), []
    for c in cfgs:
        k = json.dumps(c, sort_keys=True, default=str)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


# ------------------------------------------------------------------------------------------------ scan setup
class ScanData:
    def __init__(self, sid):
        from foodscan.loader import Scan
        from foodscan.segment import load_raw
        from PIL import Image
        from seg3d.data import load_scan, semantic_of, voxelize
        self.sid = sid
        sd = roots(sid)[0]
        z = np.load(sd / "gt" / "seg_points.npz")
        self.xyz = z["xyz"].astype(np.float64)
        self.nrm = z["normal"].astype(np.float64)
        self.rgb = z["rgb"]
        self.stored = z["lift_label"].astype(np.int32)
        d = load_scan(sd, "manifest")
        vi, _ = voxelize(d["xyz"], 2e-3)
        self.vi = vi
        lab = d["label"][vi]
        sem = semantic_of(lab)
        self.sc = dict(xyz=d["xyz"][vi].astype(np.float64), label=lab, sem=sem,
                       inst=np.where(sem == 3, lab, 0).astype(np.int64),
                       ignore=np.isin(lab, d["ignore_ids"]) if d["ignore_ids"] else np.zeros(len(vi), bool))
        self.scan = Scan(sd)
        raw, _, fsc = load_raw(roots(sid)[1] / "masks_sam3_v2i")
        self.raw_maps = raw
        self.work = {}
        self.work_scores = {}
        for fi, m in raw.items():
            w = m.astype(np.int32)
            w[m == L.CLUTTER], w[m == L.CONTAINER] = BIG, BIG + 1
            self.work[fi] = w
            self.work_scores[fi] = {(BIG if k == L.CLUTTER else BIG + 1 if k == L.CONTAINER else k): v
                                    for k, v in fsc[fi].items()}
        self.stored_masks = {int(p.stem): np.asarray(Image.open(p)) for p in (sd / "masks_sam3").glob("[0-9]*.png")}
        self.n = len(self.xyz)
        self.fvs = L.collect_votes(self.scan, self.work, self.xyz, self.nrm, scores=self.work_scores, features=True)
        self._link, self._sp = {}, {}
        # frames in which each raw id has pixels (linker's frame count)
        self.raw_frames = {}
        for fi, m in raw.items():
            for g in np.unique(m):
                if g > 0 and g not in (L.CLUTTER, L.CONTAINER):
                    self.raw_frames.setdefault(int(g), set()).add(fi)
        self.raw_px = {}
        for m in raw.values():
            ids, c = np.unique(m[m > 0], return_counts=True)
            for g, k in zip(ids, c):
                if g not in (L.CLUTTER, L.CONTAINER):
                    self.raw_px[int(g)] = self.raw_px.get(int(g), 0) + int(k)

    def linked(self, link, min_frames=2):
        """-> (relabelled fvs, repair2 remap cache) for a linking rule; mimics segment.link_instances."""
        mkey = (link, min_frames)
        if mkey in self._link:
            return self._link[mkey]
        kind, t = link
        if kind == "repair":
            remap, _ = L.repair_ids(self.fvs, self.n, None, overlap_thresh=t, min_points=8, min_support=1,
                                    max_points=150000, aux_ids=AUX)
        elif kind == "mutual":
            remap, _ = L.repair_ids(self.fvs, self.n, None, overlap_thresh=t, min_points=8, min_support=1,
                                    max_points=150000, aux_ids=AUX, mutual=True)
        elif kind.startswith("cluster-"):
            remap, _ = L.cluster_ids(self.fvs, self.n, thresh=t, sim=kind.split("-")[1], min_points=8,
                                     max_points=150000, aux_ids=AUX)
        else:
            raise ValueError(kind)
        root = lambda g: remap.get(g, g)               # noqa: E731
        frs, csize = {}, {}
        for g, fr in self.raw_frames.items():
            frs.setdefault(root(g), set()).update(fr)
            csize[root(g)] = csize.get(root(g), 0) + self.raw_px[g]
        order = sorted((r for r in csize if len(frs[r]) >= min_frames), key=lambda r: (-len(frs[r]), -csize[r]))[:MAX_IDS]
        final = {r: i + 1 for i, r in enumerate(order)}
        lut = np.zeros(65536, np.int32)
        lut[BIG], lut[BIG + 1] = L.CLUTTER, L.CONTAINER
        for g in self.raw_px:
            lut[g] = final.get(root(g), 0)
        fvs = []
        for fv in self.fvs:
            fvs.append(L._FrameVotes(fv.pos, fv.idx, lut[fv.lab], fv.w, fv.score, fv.edge, fv.qual))
        out = (fvs, {"clusters": len(final), "raw_ids": len(self.raw_px)})
        self._link[mkey] = out
        return out

    def superpoints(self, n, feat):
        k = (n, feat)
        if k not in self._sp:
            self._sp[k] = L.compute_superpoints(self.xyz, self.nrm, self.rgb, n_target=n,
                                                w_color=1.0 if feat == "geom+colour" else 0.0)
        return self._sp[k]

    def lift(self, c):
        fvs, _ = self.linked(tuple(c["link"]), c["min_frames"])
        sp, spm = None, "hard"
        if c["sp"]:
            sp, spm = self.superpoints(c["sp"][0], c["sp"][1]), c["sp"][2]
        key = ("rm", tuple(c["link"]), c["min_frames"])
        if c["repair2"] and key not in self._link:
            self._link[key] = L.repair_ids(fvs, self.n)[0]
        r = L.lift_from_votes(fvs, self.n, self.xyz, min_nvotes=c["min_nvotes"], repair=False, agree=False,
                              smooth_radius=c["smooth"], weighting=c["weighting"],
                              weight_kwargs=dict(edge_px=c["edge_px"]), margin=c["margin"],
                              bg_abstain=c["bg_abstain"], superpoints=sp, sp_mode=spm,
                              remap=self._link[key] if c["repair2"] else {})
        return r.lift_label


# ------------------------------------------------------------------------------------------------ metrics
def evaluate(sd: ScanData, lift_full):
    sc = sd.sc
    lift = np.asarray(lift_full)[sd.vi]
    sel = ~sc["ignore"]
    ps, pi = M.lift_baseline_pred(lift[sel])
    gi, xyz = sc["inst"][sel], sc["xyz"][sel]
    acc = M.Accumulator(volumes=False)
    acc.add(ps, pi, xyz, sc["sem"][sel], gi, sc["label"][sel], {}, lift=None)
    g_ids, _, m = M.match(gi, pi)
    frag = noid = 0
    for g in g_ids:
        mk = gi == g
        p = pi[mk]
        _, cnt = np.unique(p[p > 0], return_counts=True)
        frag += int((cnt >= 0.15 * mk.sum()).sum() >= 2)
        noid += int((p > 0).mean() < 0.05)
    # oracle merge of whole lift ids: ids whose GT-majority item (purity >= .5) coincide become one instance
    lab = sc["label"][sel]
    new = pi.copy()
    for i in np.unique(pi[pi > 0]):
        g = lab[(pi == i) & (lab > 0) & (lab < 252)]
        if len(g):
            v, c = np.unique(g, return_counts=True)
            if c.max() / len(g) >= 0.5:
                new[pi == i] = 100000 + int(v[c.argmax()])
    go, _, mo = M.match(gi, new)
    ious_o = [mo[g][1] if g in mo else 0.0 for g in go]
    return dict(iou=[float(x) for x in acc.inst_iou], tp=acc.tp, tp_iou=float(acc.tp_iou), fp=acc.fp, fn=acc.fn,
                count_ok=int(acc.count_ok), bnd=[float(x) for x in acc.bnd_iou], frag=frag, noid=noid,
                n_items=len(g_ids), oracle_iou=[float(x) for x in ious_o], n_ids=int(len(np.unique(pi[pi > 0]))))


def sp_oracle(sd: ScanData, n, feat):
    """Diagnostic: label every superpoint with its GT-majority item -> instance mIoU = ceiling of superpoint relabelling."""
    sp = sd.superpoints(n, feat)
    lab = np.zeros(sd.n, np.int32)
    gt = np.load(roots(sd.sid)[0] / "gt" / "seg_points.npz")["label"]
    for s in range(sp.max() + 1):
        m = sp == s
        g = gt[m]
        v, c = np.unique(g, return_counts=True)
        lab[m] = v[c.argmax()]
    lab = np.where((lab > 0) & (lab < 252), lab, np.where(lab == 254, 254, 0))
    return evaluate(sd, lab)


def run_scan(args):
    sid, cfgs, extras = args
    t0 = time.time()
    sd = ScanData(sid)
    t_setup = time.time() - t0
    res = {}
    if extras:
        res["__stored"] = evaluate(sd, sd.stored)
        r = L.lift_scan(sd.scan, sd.stored_masks, sd.xyz, normals=sd.nrm, agree=False)
        res["__lift_scan_stored_masks"] = evaluate(sd, r.lift_label)
        for n in (200, 400, 600):
            for feat in ("geom", "geom+colour"):
                res[f"__sp_oracle {n} {feat}"] = sp_oracle(sd, n, feat)
        res["__meta"] = dict(n_points=sd.n, n_voxels=len(sd.vi), n_raw_ids=len(sd.raw_px), t_setup=t_setup,
                             n_sp={f"{n} {f}": int(sd.superpoints(n, f).max() + 1) for n in (200, 400, 600)
                                   for f in ("geom", "geom+colour")})
    for c in cfgs:
        t = time.time()
        res[json.dumps(c, sort_keys=True, default=str)] = dict(evaluate(sd, sd.lift(c)), t=time.time() - t)
    return sid, res


# ------------------------------------------------------------------------------------------------ aggregation
def pooled(rows):
    """rows: list of per-scan evaluate() dicts -> union metrics."""
    iou = [x for r in rows for x in r["iou"]]
    tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
    return dict(inst=float(np.mean(iou)), pq=float(sum(r["tp_iou"] for r in rows) / max(tp + .5 * fp + .5 * fn, 1e-9)),
                exact=float(np.mean([r["count_ok"] for r in rows])), frag=sum(r["frag"] for r in rows),
                noid=sum(r["noid"] for r in rows), n_items=len(iou),
                bnd=float(np.mean([x for r in rows for x in r["bnd"]])) if any(r["bnd"] for r in rows) else float("nan"),
                oracle=float(np.mean([x for r in rows for x in r["oracle_iou"]])),
                scan=[float(np.mean(r["iou"])) if r["iou"] else float("nan") for r in rows])


def bootstrap(delta, n=2000, seed=0):
    d = np.asarray(delta, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def loso_pick(table, names, sids):
    """Leave-one-scan-out selection among config `names`: the held-out scan's row comes from the config
    that maximises pooled item IoU on the other scans. -> (rows per scan, picks)."""
    rows, picks = [], []
    for i, s in enumerate(sids):
        best, bv = None, -1
        for nm in names:
            v = np.mean([x for j, t in enumerate(sids) if j != i for x in table[nm][t]["iou"]])
            if v > bv:
                best, bv = nm, v
        rows.append(table[best][s])
        picks.append(best)
    return rows, picks


def cmd_run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    sids = a.only or (SIDS + [f"v13/s{i:06d}" for i in range(20)] + [f"v12/s{i:06d}" for i in range(20) if f"s{i:06d}" not in SIDS]
                      if a.all40 else SIDS)
    g = grids()
    t0 = time.time()
    all1 = uniq([c for v in g.values() for c in v])
    print(f"phase 1: {len(all1)} configs x {len(sids)} scans", flush=True)
    with Pool(min(a.procs, len(sids))) as p:
        out1 = dict(p.map(run_scan, [(s, all1, True) for s in sids]))
    print(f"phase 1 done {time.time() - t0:.0f}s", flush=True)
    (OUT / "phase1.json").write_text(json.dumps(out1))

    # nominate the best of each family on all scans (documented as slightly optimistic), cross them
    def best_of(fam):
        cs = g[fam]
        sc = [(pooled([out1[s][json.dumps(c, sort_keys=True, default=str)] for s in sids])["inst"], i) for i, c in enumerate(cs)]
        return cs[max(sc)[1]]
    bw, bl, bd, bs = best_of("weighting"), best_of("linking"), best_of("decision"), best_of("geometry")
    combos = []
    for w in (cfg(), bw):
        for l in (cfg(), bl):
            for d in (cfg(), bd):
                for s in (cfg(), bs):
                    combos.append(cfg(weighting=w["weighting"], edge_px=w["edge_px"], link=l["link"], min_frames=l["min_frames"], repair2=l["repair2"],
                                      min_nvotes=d["min_nvotes"], margin=d["margin"], bg_abstain=d["bg_abstain"],
                                      smooth=s["smooth"], sp=s["sp"]))
    combos = uniq(combos)
    done = {json.dumps(c, sort_keys=True, default=str) for c in all1}
    new = [c for c in combos if json.dumps(c, sort_keys=True, default=str) not in done]
    print(f"phase 2: {len(new)} new combined configs; nominated {[name_of(c) for c in (bw, bl, bd, bs)]}", flush=True)
    with Pool(min(a.procs, len(sids))) as p:
        out2 = dict(p.map(run_scan, [(s, new, False) for s in sids]))
    for s in sids:
        out1[s].update(out2[s])
    (OUT / "results.json").write_text(json.dumps(dict(sids=sids, results=out1, grids={k: [json.dumps(c, sort_keys=True, default=str) for c in v] for k, v in g.items()},
                                                      combos=[json.dumps(c, sort_keys=True, default=str) for c in combos],
                                                      nominated=[json.dumps(c, sort_keys=True, default=str) for c in (bw, bl, bd, bs)],
                                                      runtime_s=time.time() - t0)))
    (OUT / "phase1.json").unlink(missing_ok=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


def key_of(c):
    return json.dumps(c, sort_keys=True, default=str)


def cmd_report():
    R = json.loads((OUT / "results.json").read_text())
    sids, res = R["sids"], R["results"]
    table = {}
    for s in sids:
        for k, v in res[s].items():
            table.setdefault(k, {})[s] = v
    base_k = key_of(cfg())
    base = pooled([table[base_k][s] for s in sids])

    def row(label, rows, note=""):
        p = pooled(rows)
        dl = np.array(p["scan"]) - np.array(base["scan"])
        d, lo, hi = bootstrap(dl)
        note = (note + " " if note else "") + f"scans +/-/=: {int((dl > .005).sum())}/{int((dl < -.005).sum())}/{int((abs(dl) <= .005).sum())}"
        return (f"| {label} | {p['inst']:.3f} | {p['pq']:.3f} | {p['exact']:.2f} | {p['frag']} / {p['n_items']} | {p['noid']} | "
                f"{p['bnd']:.3f} | {p['oracle']:.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {note} |"), p, (d, lo, hi)

    head = ("| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | "
            "d inst vs current, per scan (95% CI) | note |\n|---|---|---|---|---|---|---|---|---|---|")
    out = [f"# Lift sweep (CPU, {len(sids)} real-SAM 3 scans)", ""]
    reading = OUT / "reading.md"
    if reading.exists():
        out += [reading.read_text().rstrip(), ""]
    out += [f"Code `pipeline/eval/lift_sweep.py` (+ `pipeline/foodscan/labels.py`), raw numbers `results.json`. Total sweep runtime {R['runtime_s']:.0f} s on CPU "
            f"({len(sids)} scans in parallel). Union over the {len(sids)} scans, {base['n_items']} valid GT items, evaluated on the 2 mm canonical voxel scene as in `results/seg3d/merge`. "
            f"Deltas are the paired per-scan mean-inst-IoU difference vs `current` (2000-resample bootstrap over the {len(sids)} scans). "
            "`oracle-merge ceiling` = inst mIoU after merging every pair of lift ids whose GT-majority item (purity >= .5) is the same: what a perfect refiner/merge could add on top of that setting. "
            "fragmented = GT item with >= 2 lift ids holding >= 15 % each; no lift id = < 5 % of the item's points carry a food id.", "",
            "Pipeline replayed from the stored RAW per-frame SAM 3 maps: projection once, instance linking (as `segment.link_instances`, one-frame clusters dropped), weighting, decision, optional smoothing/superpoints; "
            "the flag-free default (`current`) = linking repair_ids 0.5, cos weights, min_nvotes 2, background as a class, argmax, second repair_ids pass, no smoothing.", "",
            "## Reference rows", "", head]
    for lab_, k in (("stored lift (V.lift labels in seg_points.npz; the 0.63 of the ablations)", "__stored"),
                    ("labels.lift_scan on the stored linked masks (masks_sam3)", "__lift_scan_stored_masks"),
                    ("current (this script's replay of today's defaults)", base_k)):
        out.append(row(lab_, [table[k][s] for s in sids])[0])
    out.append("")
    picks_all = {}
    for fam, keys in R["grids"].items():
        out += [f"## {fam}", "", head]
        best = (None, -1)
        for k in keys:
            c = json.loads(k)
            r, p, dd = row(name_of(c), [table[k][s] for s in sids])
            out.append(r)
        names = keys
        rows, picks = loso_pick(table, names, sids)
        r, p, dd = row("**LOSO-selected within family**", rows, "LOSO")
        out.append(r)
        out += ["", "LOSO picks: " + "; ".join(f"{s}: {name_of(json.loads(k))}" for s, k in zip(sids, picks)), ""]
        picks_all[fam] = picks
    out += ["## Superpoint ceiling diagnostic (GT-majority label per superpoint; uses GT, not a method)", "", head]
    for k in sorted(k for k in table if k.startswith("__sp_oracle")):
        out.append(row(k[2:], [table[k][s] for s in sids])[0])
    meta = [table["__meta"][s] for s in sids]
    out += ["", "Superpoints per scan (mean): " + ", ".join(f"{k}: {np.mean([m['n_sp'][k] for m in meta]):.0f}" for k in meta[0]["n_sp"]) + ".", ""]
    out += ["## Combined (cross of the family winners nominated on all scans; nomination is slightly optimistic, the LOSO row re-picks without the held-out scan)", "", head]
    ck = R["combos"]
    for k in ck:
        out.append(row(name_of(json.loads(k)), [table[k][s] for s in sids])[0])
    rows, picks = loso_pick(table, ck, sids)
    out.append(row("**LOSO-selected among combined**", rows, "LOSO")[0])
    out += ["", "LOSO picks: " + "; ".join(f"{s}: {name_of(json.loads(k))}" for s, k in zip(sids, picks)), ""]
    (OUT / "summary.md").write_text("\n".join(out))
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--procs", type=int, default=11)
    r.add_argument("--only", nargs="*")
    r.add_argument("--all40", action="store_true", help="all 20 hard_v12 + 20 hard_v13 real-vote scans")
    r.add_argument("--out", type=Path, help="output dir (default results/lift_sweep)")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("--out", type=Path)
    q.set_defaults(fn=lambda a: cmd_report())
    a = ap.parse_args()
    if getattr(a, "out", None):
        global OUT
        OUT = a.out
    a.fn(a)


if __name__ == "__main__":
    main()
