"""Real SAM 3 per-frame masks for synthetic scans -> real lifted votes for the 3D refiner.

Run from pipeline/ in the `foodseg` env (stage 1 needs the GPU, others are CPU):

    python -m eval.sam3_synth_votes run    --scans ../data/synth/hard_v12/scans [--variant names|generic] [--limit N]
    python -m eval.sam3_synth_votes relink --variant v2_image   # CPU: re-tune the 3D instance linking of an image-mode run
    python -m eval.sam3_synth_votes match  --scans ... [--variant names]      # SAM ids -> GT item ids
    python -m eval.sam3_synth_votes copy   --scans ... --dst ../data/synth/hard_v12_sam3/scans
    python -m eval.sam3_synth_votes export --dst ...          # worktree seg_export --votes sam3
    python -m eval.sam3_synth_votes stats  --scans ... --dst ... --out ../results/sam3_votes/summary.md

run: foodscan.segment.segment_scan (video predictor, per-concept text prompts, score-based
overlap, container/clutter aux prompts) with in-process Turing/fp16 patches for the 6 GB sm75
card (bf16 autocast -> fp16, FLASH-only SDPA -> efficient/math, detector backbone in fp16).
Variants: "names" = item names + categories from gt.json (oracle names) -> <scan>/masks_sam3;
"generic" = just "food" -> <scan>/masks_sam3_generic. Frames stay at full 1920x1440 for SAM;
label maps are stored at 960 px long side (segment_scan max_side).

match: SAM concept-instance ids are not GT item ids. Hungarian match of SAM food instances to
GT items by pooled 2-D IoU against gt/seg_ids over all frames (min IoU 0.1); unmatched SAM
instances get fresh ids K+1.. (they show up as splits / false positives), 253/254 are kept.
Writes <scan>/masks_sam3_gt[_generic]/ (+ match.json).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

CLUTTER, CONTAINER = 253, 254
WORKTREE = Path("/home/kyle/Desktop/FoodSeg/.claude/worktrees/synth-hard/pipeline")
VARIANTS = {"names": ("masks_sam3", "masks_sam3_gt"), "generic": ("masks_sam3_generic", "masks_sam3_gt_generic"),
            # v2 prompting: generic food@0.3 + simplified names (segment.py), video tracker vs per-frame image + 3D linking
            "v2_video": ("masks_sam3_v2", "masks_sam3_gt_v2"), "v2_image": ("masks_sam3_v2i", "masks_sam3_gt_v2i")}


def concepts_for(scan: Path, variant: str) -> list[str]:
    if variant == "generic":
        return ["food"]          # v2 variants take the oracle names; segment_scan simplifies them
    gt = json.loads((scan / "gt.json").read_text())
    out: list[str] = []
    for it in gt["items"]:
        for s in (it["name"], it.get("category") or ""):
            s = s.replace("_", " ").strip().lower()
            if s and s not in out:
                out.append(s)
    return out


# ------------------------------------------------------------------ stage 1: GPU

def turing_patches() -> None:
    """Must run before sam3 is imported."""
    import torch
    base = torch.autocast

    class _Autocast(base):
        def __init__(self, device_type, dtype=None, *a, **k):
            if device_type == "cuda" and dtype == torch.bfloat16:
                dtype = torch.float16
            super().__init__(device_type, dtype, *a, **k)

    torch.autocast = _Autocast
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import sam3.model.decoder as D
    D.sdpa_kernel = lambda _b: sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    import sam3.model_builder as MB
    orig = MB.build_sam3_video_model

    def build(*a, **k):
        k["device"] = "cpu"
        m = orig(*a, **k)
        m.detector.backbone.half()
        return m

    MB.build_sam3_video_model = build


class _Pred:
    """Delegates to the shared predictor (video or image backend) but hides shutdown() so segment_scan keeps it alive."""

    def __init__(self, p):
        self._p = p

    def __getattr__(self, k):
        if k == "shutdown":
            raise AttributeError(k)
        return getattr(self._p, k)


class _Hold:
    """Keeps the live predictor so an OOM can tear it down and build a fresh one (segment_scan retries once)."""

    def __init__(self, mk):
        self.mk, self.cur = mk, mk()

    def rebuild(self):
        import gc
        import torch
        try:
            self.cur.shutdown()
        except Exception:
            pass
        self.cur = None
        gc.collect()
        torch.cuda.empty_cache()
        self.cur = self.mk()
        return _Pred(self.cur)


def _gt_probe(scan: Path, rows: list):
    """Callback for image mode: record every raw detection (score, area, where it lands in the GT) for the aux study."""
    gtm = _gt_maps(scan)

    def probe(fi, prompt, kind, masks, scores):
        g = gtm.get(fi)
        if g is None:
            return
        gs = np.asarray(Image.fromarray(g).resize((masks.shape[2], masks.shape[1]), Image.NEAREST))
        food, cont, clut = (gs > 0) & (gs < 252), gs == CONTAINER, gs == CLUTTER
        for m, sc in zip(masks, scores):
            a = int(m.sum())
            if a == 0:
                continue
            rows.append({"scan": scan.name, "fi": fi, "prompt": prompt, "kind": kind, "score": round(float(sc), 3),
                         "area": a / m.size, "on_food": int((m & food).sum()) / a, "on_cont": int((m & cont).sum()) / a,
                         "on_clutter": int((m & clut).sum()) / a})
    return probe


def cmd_run(a) -> None:
    import torch
    image = a.variant == "v2_image"
    if not image:
        turing_patches()
    from foodscan.segment import Sam3ImageBackend, make_linker, segment_scan
    scans = sorted(p for p in a.scans.iterdir() if (p / "gt.json").exists())
    if a.only:
        scans = [s for s in scans if s.name in a.only]
    if a.limit:
        scans = scans[:a.limit]
    if a.sort_items:    # fewest items first: tracker memory and time grow with the object count
        scans.sort(key=lambda p: len(json.loads((p / "gt.json").read_text())["items"]))
    t0 = time.time()
    if image:
        mk = lambda: Sam3ImageBackend(min_thresh=a.floor)
    else:
        from sam3.model_builder import build_sam3_video_predictor
        mk = lambda: build_sam3_video_predictor(gpus_to_use=[0], async_loading_frames=True)
    hold = _Hold(mk)
    print(f"model load {time.time() - t0:.0f}s, gpu {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
    out_name = VARIANTS[a.variant][0]
    times = {}
    probe_rows: list = []
    for sc in scans:
        od = sc / out_name
        if (od / "concepts.json").exists() and not a.force:
            print(f"{sc.name}: done, skip", flush=True)
            continue
        names = concepts_for(sc, "names")
        t = time.time()
        kw = {}
        if image:
            z = np.load(sc / "gt" / "seg_points.npz")        # the cloud the export lifts onto: link in the same 3D frame
            kw = dict(mode="image", linker=make_linker(sc, z["xyz"].astype(np.float64), z["normal"].astype(np.float64)),
                      probe=_gt_probe(sc, probe_rows))
        try:
            meta = segment_scan(sc, names, out_dir=od, predictor=_Pred(hold.cur), rebuild=hold.rebuild, max_side=960,
                                reprompt_every=a.reprompt_every, food_thresh=a.food_thresh, name_thresh=a.name_thresh,
                                aux_thresh={"container": a.aux_thresh, "clutter": a.aux_thresh}, **kw)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if not (isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()):
                raise
            print(f"{sc.name}: CUDA OOM after retry, skipped (rebuilding predictor)", flush=True)
            hold.rebuild()
            continue
        times[sc.name] = time.time() - t
        print(f"{sc.name}: {len(names)} oracle names, {len(meta)} instances, {times[sc.name]:.0f}s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
        torch.cuda.empty_cache()
        if not a.limit:
            (a.scans / f"_times_{a.variant}.json").write_text(json.dumps(times, indent=1))
        if probe_rows:
            Path(a.probe_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.probe_out).write_text("\n".join(json.dumps(r) for r in probe_rows))
        if a.budget and time.time() - t0 > a.budget:
            print("time budget reached, stopping", flush=True)
            break
    print(f"total {time.time() - t0:.0f}s", flush=True)


def cmd_relink(a) -> None:
    """CPU: re-run the 3D instance linking of an image-mode run from its saved raw maps (no GPU)."""
    from foodscan.segment import load_raw, make_linker
    out_name = VARIANTS[a.variant][0]
    for sc in sorted(a.scans.iterdir()):
        od = sc / out_name
        if not (od / "raw" / "raw.json").exists() or (a.only and sc.name not in a.only):
            continue
        raw, npx, fs = load_raw(od)
        z = np.load(sc / "gt" / "seg_points.npz")
        link = make_linker(sc, z["xyz"].astype(np.float64), z["normal"].astype(np.float64),
                           overlap_thresh=a.overlap, min_frames=a.min_frames, min_support=a.min_support)
        maps, meta, fsc, st = link(raw, npx, fs)
        for fi, m in maps.items():
            Image.fromarray(m).save(od / f"{fi:06d}.png")
            (od / f"{fi:06d}.json").write_text(json.dumps({str(k): v for k, v in fsc[fi].items()}))
        (od / "instances.json").write_text(json.dumps({str(k): v for k, v in meta.items()}, indent=1))
        c = json.loads((od / "concepts.json").read_text())
        c["link"] = st
        for e in c["concepts"]:
            if e["kind"] in ("food", "name"):
                e["labels"] = sorted(g for g, v in meta.items() if v["name"] == e["name"])
        (od / "concepts.json").write_text(json.dumps(c, indent=1, default=str))
        print(f"{sc.name}: relinked {st['raw_ids']} raw ids -> {st['clusters']} instances", flush=True)


# ------------------------------------------------------------------ stage 2: id matching

def _gt_maps(scan: Path) -> dict[int, np.ndarray]:
    return {int(p.stem): np.asarray(Image.open(p)) for p in sorted((scan / "gt" / "seg_ids").glob("*.png"))}


def match_scan(scan: Path, variant: str, min_iou: float = 0.1) -> dict:
    from scipy.optimize import linear_sum_assignment
    src, dst = (scan / VARIANTS[variant][0]), (scan / VARIANTS[variant][1])
    gtm = _gt_maps(scan)
    sam = {int(p.stem): np.asarray(Image.open(p)) for p in sorted(src.glob("[0-9]*.png"))}
    inter: dict[tuple[int, int], int] = {}
    a_s: dict[int, int] = {}
    a_g: dict[int, int] = {}
    for fi, sm in sam.items():
        g = gtm.get(fi)
        if g is None:
            continue
        if sm.shape != g.shape:
            sm = np.asarray(Image.fromarray(sm).resize(g.shape[::-1], Image.NEAREST))
        food_s = (sm > 0) & (sm < CLUTTER)
        food_g = (g > 0) & (g < 252)
        ids, c = np.unique(sm[food_s], return_counts=True)
        for i, n in zip(ids, c):
            a_s[int(i)] = a_s.get(int(i), 0) + int(n)
        ids, c = np.unique(g[food_g], return_counts=True)
        for i, n in zip(ids, c):
            a_g[int(i)] = a_g.get(int(i), 0) + int(n)
        both = food_s & food_g
        pair, c = np.unique(np.stack([sm[both], g[both]], 1), axis=0, return_counts=True)
        for (s, gi), n in zip(pair, c):
            inter[(int(s), int(gi))] = inter.get((int(s), int(gi)), 0) + int(n)
    S, G = sorted(a_s), sorted(a_g)
    M = np.zeros((len(S), len(G)))
    for (s, g), n in inter.items():
        M[S.index(s), G.index(g)] = n / (a_s[s] + a_g[g] - n)
    mapping: dict[int, int] = {}
    if len(S) and len(G):
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] >= min_iou:
                mapping[S[i]] = G[j]
    nxt = max(G + [0]) + 1
    for s in S:
        if s not in mapping:
            mapping[s] = nxt
            nxt += 1
    if nxt > CLUTTER:
        raise RuntimeError(f"{scan.name}: too many ids")
    lut = np.arange(256, dtype=np.uint8)
    for s, g in mapping.items():
        lut[s] = g
    dst.mkdir(exist_ok=True)
    for p in src.glob("[0-9]*.png"):
        Image.fromarray(lut[np.asarray(Image.open(p))]).save(dst / p.name)
    for p in src.glob("[0-9]*.json"):
        sc = json.loads(p.read_text())
        (dst / p.name).write_text(json.dumps({str(mapping.get(int(k), int(k))): v for k, v in sc.items()}))
    for f in ("concepts.json", "instances.json"):
        shutil.copy(src / f, dst / f)
    info = {"sam_to_gt": {str(k): v for k, v in mapping.items()},
            "matched_iou": {str(s): float(M[S.index(s), G.index(g)]) for s, g in mapping.items() if g in G and (s, g) in inter},
            "unmatched_sam": [s for s in S if mapping[s] not in G],
            "unmatched_gt": [g for g in G if g not in mapping.values()],
            "n_sam": len(S), "n_gt": len(G)}
    (dst / "match.json").write_text(json.dumps(info, indent=1))
    return info


def cmd_match(a) -> None:
    for sc in sorted(a.scans.iterdir()):
        if (sc / VARIANTS[a.variant][0] / "concepts.json").exists():
            i = match_scan(sc, a.variant)
            print(f"{sc.name}: SAM {i['n_sam']} inst vs GT {i['n_gt']}, unmatched sam {len(i['unmatched_sam'])}, "
                  f"unmatched gt {len(i['unmatched_gt'])}", flush=True)


# ------------------------------------------------------------------ stage 3: copy + export

def cmd_copy(a) -> None:
    mdir = VARIANTS[a.variant][1]
    for sc in sorted(a.scans.iterdir()):
        if not (sc / mdir / "match.json").exists():
            continue
        d = a.dst / sc.name
        d.mkdir(parents=True, exist_ok=True)
        for n in ("gt", "depth", "conf", "smooth_depth"):
            if (sc / n).exists():
                if (d / n).exists():
                    shutil.rmtree(d / n)
                shutil.copytree(sc / n, d / n)
        for n in ("gt.json", "frames.jsonl", "meta.json"):
            if (sc / n).exists():
                shutil.copy(sc / n, d / n)
        if not (d / "rgb").exists():
            (d / "rgb").symlink_to((sc / "rgb").resolve())
        if (d / "masks_sam3").exists():
            shutil.rmtree(d / "masks_sam3")
        shutil.copytree(sc / mdir, d / "masks_sam3")
        print(f"copied {sc.name}", flush=True)


def cmd_export(a) -> None:
    sel = ",".join(a.only) if a.only else None
    for sc in sorted(a.dst.iterdir()):
        if a.only and sc.name not in a.only:
            continue
        code = ("import sys,json;from pathlib import Path;from synth.seg_export import export;"
                f"r=export(Path('{sc.resolve()}'),votes='sam3');print(json.dumps(r))")
        r = subprocess.run([sys.executable, "-c", code], cwd=WORKTREE, capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(WORKTREE)})
        print(sc.name, r.stdout.strip()[-300:] or r.stderr[-500:], flush=True)


# ------------------------------------------------------------------ stage 4: statistics

def lift_stats(npz: Path, frac: float = 0.2, min_pts: int = 200) -> dict:
    z = np.load(npz)
    lab, ll, ls = z["label"].astype(int), z["lift_label"].astype(int), z["lift_share"]
    food = (lab > 0) & (lab < 252)
    gids = [g for g in np.unique(lab[food]) if (lab == g).sum() >= min_pts]
    wrong = float((ll[food] != lab[food]).mean())
    voted = food & (ll > 0)
    ious = []
    for g in gids:
        a, b = lab == g, ll == g
        ious.append((a & b).sum() / max((a | b).sum(), 1))
    lids = [l for l in np.unique(ll[food]) if 0 < l < 252]
    cont = {(g, l): int(((lab == g) & (ll == l)).sum()) for g in gids for l in lids}
    merges = sum(1 for l in lids if sum(cont[(g, l)] >= frac * (lab == g).sum() for g in gids) >= 2)
    splits = sum(1 for g in gids if sum(cont[(g, l)] >= frac * (lab == g).sum() for l in lids) >= 2)
    return {"wrong": wrong, "share": float(ls[voted].mean()) if voted.any() else 0.0, "miou": float(np.mean(ious)) if ious else float("nan"),
            "merges": merges, "splits": splits, "n_food": int(food.sum()), "n_items": len(gids),
            "unvoted": float((ll[food] == 0).mean())}


def item_recalls(npz: Path, gt_json: Path, min_pts: int = 50) -> list[dict]:
    """Per GT item: share of its points whose lift_label equals its id (after SAM->GT matching), with kind/source."""
    z = np.load(npz)
    lab, ll = z["label"].astype(int), z["lift_label"].astype(int)
    out = []
    for it in json.loads(gt_json.read_text())["items"]:
        sel = lab == it["id"]
        if sel.sum() >= min_pts:
            src = it.get("source", "")
            out.append({"name": it["name"], "source": src,
                        "kind": {"metafood3d": "scan", "objaverse": "asset", "ycb": "asset"}.get(src, "procedural"),
                        "recall": float((ll[sel] == it["id"]).mean()), "unvoted": float((ll[sel] == 0).mean()),
                        "n": int(sel.sum())})
    return out


def recall_table(rows_by_scan: list[list[dict]]) -> list[str]:
    items = [i for r in rows_by_scan for i in r]
    L = ["", "Per-item lift recall (points of the GT item that get its own id) by kind / source:", "",
         "| group | items | mean recall | items with recall < 0.3 |", "|---|---|---|---|"]
    for key in ("kind", "source"):
        for g in sorted({i[key] for i in items}):
            v = [i for i in items if i[key] == g]
            L.append(f"| {key}={g} | {len(v)} | {np.mean([i['recall'] for i in v]):.3f} | {sum(i['recall'] < 0.3 for i in v)} |")
    worst = sorted(items, key=lambda i: i["recall"])[:12]
    L += ["", "Lowest-recall items: " + ", ".join(f"{i['name']} ({i['source']}, {i['recall']:.2f})" for i in worst)]
    return L


def cmd_stats(a) -> None:
    rows = []
    for d in sorted(a.dst.iterdir()):
        p, o = d / "gt" / "seg_points.npz", a.scans / d.name / "gt" / "seg_points.npz"
        if not (p.exists() and o.exists()) or not (d / "masks_sam3" / "match.json").exists() or (a.only and d.name not in a.only):
            continue
        r, s = lift_stats(p), lift_stats(o)
        m = json.loads((d / "masks_sam3" / "match.json").read_text())
        rows.append((d.name, r, s, m))
    if not rows:
        print("no scans")
        return
    f = lambda k, i: np.nanmean([x[i][k] for x in rows])
    t = lambda k, i: sum(x[i][k] for x in rows)
    L = [f"# SAM 3 real votes vs pseudo votes ({a.label})", "", f"{len(rows)} scans.", "",
         "| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r, s, m in rows:
        L.append(f"| {n} | {r['n_items']} | {r['wrong']:.3f} | {s['wrong']:.3f} | {r['share']:.3f} | {s['share']:.3f} | "
                 f"{r['miou']:.3f} | {s['miou']:.3f} | {r['merges']}/{s['merges']} | {r['splits']}/{s['splits']} | {r['unvoted']:.3f} |")
    L += ["", "| mean | real SAM 3 | pseudo |", "|---|---|---|",
          f"| food points with lift_label != label | {f('wrong', 1):.3f} | {f('wrong', 2):.3f} |",
          f"| food points with no vote | {f('unvoted', 1):.3f} | {f('unvoted', 2):.3f} |",
          f"| mean lift_share (voted food points) | {f('share', 1):.3f} | {f('share', 2):.3f} |",
          f"| per-scan instance mIoU | {f('miou', 1):.3f} | {f('miou', 2):.3f} |",
          f"| merges (total) | {t('merges', 1)} | {t('merges', 2)} |",
          f"| splits (total) | {t('splits', 1)} | {t('splits', 2)} |",
          f"| SAM instances with no GT match (total) | {sum(len(x[3]['unmatched_sam']) for x in rows)} | - |",
          f"| GT items with no SAM match (total) | {sum(len(x[3]['unmatched_gt']) for x in rows)} | - |", ""]
    L += recall_table([item_recalls(a.dst / x[0] / "gt" / "seg_points.npz", a.dst / x[0] / "gt.json") for x in rows])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(L)
    if a.append and a.out.exists():
        txt = a.out.read_text() + "\n\n" + txt
    a.out.write_text(txt)
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "relink", "match", "copy", "export", "stats"])
    ap.add_argument("--scans", type=Path, default=Path("../data/synth/hard_v12/scans"))
    ap.add_argument("--dst", type=Path, default=Path("../data/synth/hard_v12_sam3/scans"))
    ap.add_argument("--variant", choices=list(VARIANTS), default="names")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0, help="stop starting new scans after this many seconds")
    ap.add_argument("--reprompt-every", type=int, default=30)
    ap.add_argument("--overlap", type=float, default=0.5, help="relink: 3D overlap of the smaller mask to link two ids")
    ap.add_argument("--min-frames", type=int, default=2, help="relink: drop instances seen in fewer frames")
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--food-thresh", type=float, default=0.3)
    ap.add_argument("--name-thresh", type=float, default=0.2)
    ap.add_argument("--aux-thresh", type=float, default=0.3)
    ap.add_argument("--floor", type=float, default=0.15, help="image backend score floor (must be <= every threshold)")
    ap.add_argument("--probe-out", default="../results/sam3_votes/v2_probe.jsonl")
    ap.add_argument("--sort-items", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/sam3_votes/summary.md"))
    ap.add_argument("--label", default="oracle names")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "relink": cmd_relink, "match": cmd_match, "copy": cmd_copy, "export": cmd_export, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
