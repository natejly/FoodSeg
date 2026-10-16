"""Why does real SAM 3 miss food in synthetic renders? Single-image prompt/threshold probe.

Run from pipeline/ in the `foodseg` env (GPU stage under flock):

    python -m eval.sam3_gap run     --out ../results/sam3_gap     # 40 synthetic + 40 real frames
    python -m eval.sam3_gap report  --out ../results/sam3_gap     # tables + summary.md + qualitative crops (CPU)

Set A: per hard_v12 scan the most top-down and one oblique frame (1008 long side), GT from gt/seg_ids.
Set B: Nutrition5k overhead RGB whose ingredients map onto the synthetic categories; reference =
results/n5k/masks_sam3 (generic "food" at 0.4), the only real-side mask available.
Prompts: exact item names, simplified 1-2 word names (SIMPLE), generic food/dish/meal. The image
model is run once per prompt at score 0.15; the 0.3 / 0.5 thresholds filter the same detections.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCANS = ROOT / "data/synth/hard_v12/scans"
N5K = ROOT / "data/datasets/Nutrition5k"
N5K_MASKS = ROOT / "results/n5k/masks_sam3"
THRS = (0.5, 0.3, 0.15)
GENERIC = ("food", "dish", "meal")
LONG = 1008
HIT_IOU = 0.3

# item name (lower, spaces) -> simplified 1-2 word prompt
SIMPLE = {
    "black beans": "beans", "blueberries": "berries", "boiled egg": "egg", "bread roll": "bread",
    "brown rice": "rice", "cake slice": "cake", "cherry tomato": "tomato", "chicken breast": "chicken",
    "chicken thighs": "chicken", "cookies": "cookie", "couscous": "grain", "falafel": "fried ball",
    "gravy": "sauce", "green beans": "beans", "hummus": "dip", "kiwi": "fruit", "lettuce": "salad",
    "mashed potato": "potato", "melons": "melon", "melted cheese": "cheese", "mixed greens": "salad",
    "omelet": "egg", "papaya": "fruit", "pasta mixed dishes": "pasta", "pesto": "sauce",
    "pork chop": "meat", "radicchio": "lettuce", "salmon grill": "fish", "sausages": "sausage",
    "sour cream": "sauce", "squash": "vegetable", "tuna salad": "salad", "yeast bread": "bread",
}
# N5k ingredient keyword -> synthetic item name
REAL_KW = {
    "asparagus": "asparagus", "bacon": "bacon", "blueberr": "blueberries", "broccoli": "broccoli",
    "cauliflower": "cauliflower", "chicken breast": "chicken breast", "chicken": "chicken thighs",
    "cookie": "cookies", "corn": "corn", "cucumber": "cucumber", "boiled egg": "boiled egg",
    "omelet": "omelet", "falafel": "falafel", "grape": "grapes", "green bean": "green beans",
    "hummus": "hummus", "kiwi": "kiwi", "lettuce": "lettuce", "greens": "mixed greens",
    "spinach": "mixed greens", "arugula": "mixed greens", "melon": "melons", "cantaloupe": "melons",
    "mashed potato": "mashed potato", "olive": "olive", "papaya": "papaya", "pasta": "pasta mixed dishes",
    "pesto": "pesto", "pork": "pork chop", "rice": "rice", "salmon": "salmon grill", "sausage": "sausages",
    "sour cream": "sour cream", "squash": "squash", "strawberr": "strawberry", "sushi": "sushi",
    "tuna": "tuna salad", "bread": "yeast bread", "yogurt": "yogurt", "cherry tomato": "cherry tomato",
    "tomato": "cherry tomato", "cheese": "melted cheese", "gravy": "gravy", "couscous": "couscous",
    "cake": "cake slice", "ice cream": "ice cream", "egg": "boiled egg",
}


def simple_name(name: str) -> str:
    return SIMPLE.get(name, name)


def kind_of(source: str) -> str:
    return {"metafood3d": "scan", "objaverse": "asset", "ycb": "asset"}.get(source, "procedural")


def match_ingredient(ing: str) -> str | None:
    """Map an N5k ingredient string to a synthetic item name (longest keyword wins)."""
    ing = ing.lower()
    for kw in sorted(REAL_KW, key=len, reverse=True):
        if kw in ing:
            return REAL_KW[kw]
    return None


def mask_stats(masks: np.ndarray, scores: np.ndarray, ref: np.ndarray, thr: float) -> dict:
    """Best IoU of any mask with score>=thr against ref, union recall of ref, detection flag."""
    keep = scores >= thr
    out = {"any": bool(keep.any()), "best_score": float(scores.max()) if len(scores) else 0.0,
           "iou": 0.0, "recall": 0.0, "n": int(keep.sum())}
    if not keep.any() or ref.sum() == 0:
        return out
    ms = masks[keep]
    inter = (ms & ref).reshape(len(ms), -1).sum(1)
    union = ms.reshape(len(ms), -1).sum(1) + ref.sum() - inter
    out["iou"] = float((inter / np.maximum(union, 1)).max())
    out["recall"] = float((ms.any(0) & ref).sum() / ref.sum())
    return out


# ------------------------------------------------------------------ selection

def pick_frames(scan: Path) -> list[tuple[str, int]]:
    fr = [json.loads(l) for l in (scan / "frames.jsonl").read_text().splitlines() if l.strip()]
    down = np.array([f["transform"][1][2] for f in fr])      # +y of the forward-reversed z axis: larger = more top-down
    top = int(np.argmax(down))
    order = np.argsort(down)
    obl = None
    for i in order:
        a = np.asarray(Image.open(scan / "gt/seg_ids" / f"{i:06d}.png"))
        if ((a > 0) & (a < 253)).mean() > 0.01:
            obl = int(i)
            break
    return [("top", top), ("obl", obl if obl is not None else int(order[0]))]


def select_real(n: int = 40, per_kw_cap: int = 3) -> list[dict]:
    rows = []
    for f in sorted((N5K / "metadata").glob("dish_metadata_cafe*.csv")):
        for r in csv.reader(open(f, newline="")):
            d = r[0]
            ings = [r[i] for i in range(7, len(r), 7)]
            m = {}
            for ing in ings:
                c = match_ingredient(ing)
                if c and c not in m.values():
                    m[ing] = c
            if m and (N5K_MASKS / f"{d}.png").exists() and (N5K / "imagery/realsense_overhead" / d / "rgb.png").exists():
                ref = np.asarray(Image.open(N5K_MASKS / f"{d}.png")) > 0
                if 0.03 < ref.mean() < 0.9:
                    rows.append({"dish": d, "ings": m})
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    cnt: dict[str, int] = {}
    out = []
    # rare categories first, capped per category
    freq: dict[str, int] = {}
    for r in rows:
        for c in set(r["ings"].values()):
            freq[c] = freq.get(c, 0) + 1
    rows.sort(key=lambda r: min(freq[c] for c in r["ings"].values()))
    for r in rows:
        cs = set(r["ings"].values())
        if any(cnt.get(c, 0) >= per_kw_cap for c in cs):
            continue
        for c in cs:
            cnt[c] = cnt.get(c, 0) + 1
        out.append(r)
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------ GPU stage

def load_proc():
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(device="cpu")
    model.backbone.half()
    model = model.to("cuda")
    return Sam3Processor(model, device="cuda", confidence_threshold=min(THRS))


def detect(proc, state, prompt: str, cap: int = 60):
    import torch
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = proc.set_text_prompt(state=state, prompt=prompt)
    if not len(o["scores"]):
        return np.zeros((0, *state["masks"].shape[-2:]), bool), np.zeros(0, np.float32)
    m = o["masks"].reshape(-1, *o["masks"].shape[-2:]).bool().cpu().numpy()
    s = o["scores"].float().cpu().numpy()
    top = np.argsort(-s)[:cap]
    return m[top], s[top]


def save_best(store: dict, key: str, masks, scores, ref, thr):
    """Keep the best-IoU mask (score>=thr) for the qualitative figures."""
    keep = scores >= thr
    if not keep.any():
        return
    ms = masks[keep]
    inter = (ms & ref).reshape(len(ms), -1).sum(1)
    union = ms.reshape(len(ms), -1).sum(1) + ref.sum() - inter
    store[key] = ms[int(np.argmax(inter / np.maximum(union, 1)))]


def cmd_run(a) -> None:
    import torch
    out = a.out
    (out / "masks").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = load_proc()
    print(f"load {time.time() - t0:.0f}s", flush=True)
    rows = []
    cache: dict[str, tuple] = {}

    def run_prompt(state, p):
        if p not in cache:
            cache[p] = detect(proc, state, p)
        return cache[p]

    # ---- set A
    for scan in sorted(p for p in SCANS.iterdir() if (p / "gt.json").exists()):
        gt = json.loads((scan / "gt.json").read_text())
        items = {it["id"]: it for it in gt["items"]}
        for view, fi in pick_frames(scan):
            img = Image.open(scan / "rgb" / f"{fi:06d}.jpg").convert("RGB")
            sc = LONG / max(img.size)
            img = img.resize((round(img.width * sc), round(img.height * sc)), Image.BILINEAR)
            seg = Image.open(scan / "gt/seg_ids" / f"{fi:06d}.png").resize(img.size, Image.NEAREST)
            seg = np.asarray(seg)
            food = (seg > 0) & (seg < 253)
            tag = f"{scan.name}_{view}"
            img.save(out / "masks" / f"{tag}.jpg", quality=90)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                state = proc.set_image(img)
            cache.clear()
            store: dict[str, np.ndarray] = {}
            for iid, it in items.items():
                ref = seg == iid
                area = int(ref.sum())
                if area < 100:
                    continue
                name = it["name"].replace("_", " ").lower()
                for style, prompt in (("exact", name), ("simple", simple_name(name))) + tuple(("generic", g) for g in GENERIC):
                    ms, ss = run_prompt(state, prompt)
                    for t in THRS:
                        r = mask_stats(ms, ss, ref, t)
                        r.update(domain="synth", img=tag, view=view, item=iid, name=name, category=it["category"],
                                 kind=kind_of(it["source"]), source=it["source"], area=area, style=style,
                                 prompt=prompt, thr=t)
                        if style == "generic":
                            r.update(food_recall=float((np.any([m for m, s in zip(ms, ss) if s >= t], 0) & food).sum() / food.sum())
                                     if (ss >= t).any() else 0.0)
                        rows.append(r)
                    if style in ("exact", "simple"):
                        save_best(store, f"{style}_{iid}", ms, ss, ref, min(THRS))
                    if prompt == "food":
                        save_best(store, f"food_{iid}", ms, ss, ref, 0.3)
            # image-level food recall for the generic prompts
            for g in GENERIC:
                ms, ss = run_prompt(state, g)
                for t in THRS:
                    k = ss >= t
                    u = ms[k].any(0) if k.any() else np.zeros_like(food)
                    rows.append(dict(domain="synth", img=tag, view=view, item=-1, style="image", prompt=g, thr=t,
                                     food_recall=float((u & food).sum() / food.sum()),
                                     food_prec=float((u & food).sum() / max(u.sum(), 1)), any=bool(k.any())))
            np.savez_compressed(out / "masks" / f"{tag}.npz", **{k: np.packbits(v) for k, v in store.items()},
                                shape=np.array(food.shape))
            print(f"{tag}: {len(items)} items, {time.time() - t0:.0f}s", flush=True)
            torch.cuda.empty_cache()

    # ---- set B
    for r0 in select_real(a.n_real):
        d = r0["dish"]
        img = Image.open(N5K / "imagery/realsense_overhead" / d / "rgb.png").convert("RGB")
        ref = np.asarray(Image.open(N5K_MASKS / f"{d}.png")) > 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            state = proc.set_image(img)
        cache.clear()
        store = {}
        img.save(out / "masks" / f"real_{d}.jpg", quality=90)
        plan = [(ing, cat, "exact", ing.lower()) for ing, cat in r0["ings"].items()]
        plan += [(ing, cat, "simple", simple_name(cat)) for ing, cat in r0["ings"].items()]
        plan += [(None, None, "generic", g) for g in GENERIC]
        for ing, cat, style, prompt in plan:
            ms, ss = run_prompt(state, prompt)
            for t in THRS:
                r = mask_stats(ms, ss, ref, t)
                r.update(domain="real", img=f"real_{d}", item=ing, name=cat, category=cat, kind="real",
                         area=int(ref.sum()), style=style, prompt=prompt, thr=t)
                rows.append(r)
            save_best(store, f"{style}_{ing or prompt}", ms, ss, ref, min(THRS))
        np.savez_compressed(out / "masks" / f"real_{d}.npz", **{k: np.packbits(v) for k, v in store.items()},
                            shape=np.array(ref.shape))
        print(f"real {d}: {r0['ings']}, {time.time() - t0:.0f}s", flush=True)
    (out / "rows.json").write_text(json.dumps(rows))
    print(f"done {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/sam3_gap")
    ap.add_argument("--n-real", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        from eval.sam3_gap_report import main as rep
        rep(a)
