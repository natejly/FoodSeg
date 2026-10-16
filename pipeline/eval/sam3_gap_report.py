"""CPU report for eval.sam3_gap: tables -> summary.md, qualitative side-by-side crops."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from eval.sam3_gap import HIT_IOU, N5K_MASKS, THRS, simple_name


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    df = df.reset_index()
    cols = [str(c) if not isinstance(c, tuple) else " ".join(map(str, c)) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def overlay(img: Image.Image, mask: np.ndarray | None, color=(255, 40, 40)) -> Image.Image:
    im = np.asarray(img).astype(np.float32)
    if mask is not None and mask.shape == im.shape[:2]:
        im[mask] = 0.5 * im[mask] + 0.5 * np.array(color)
        edge = mask ^ (np.roll(mask, 1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 0) & np.roll(mask, -1, 1))
        im[edge] = color
    return Image.fromarray(im.clip(0, 255).astype(np.uint8))


def unpack(z, key):
    shape = tuple(z["shape"])
    return np.unpackbits(z[key])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def main(a) -> None:
    out = a.out
    df = pd.DataFrame(json.loads((out / "rows.json").read_text()))
    items = df[df["style"] != "image"].copy()
    items["category"] = items["name"].fillna("").str.lower()
    items.loc[items["domain"] == "synth", "category"] = items["name"]
    items["hit"] = items["iou"] >= HIT_IOU
    items["style"] = pd.Categorical(items["style"], ["exact", "simple", "generic"], ordered=True)
    S = []
    s = items.groupby(["domain", "style", "thr"], observed=True).agg(
        n=("any", "size"), detect=("any", "mean"), hit=("hit", "mean"), best_iou=("iou", "mean"),
        recall=("recall", "mean"))
    S.append("## 1. Detection by domain x prompt style x threshold\n\n"
             "detect = prompt returned any mask with score >= thr; hit = best mask IoU >= 0.3 with the GT item "
             "(synthetic) or with the n5k generic-food mask (real; named prompts are not expected to cover the "
             "whole dish, read real as a detectability reference); best_iou = mean over rows; recall = mean union "
             "recall of the item/reference.\n\n" + md(s))
    # generic per-prompt
    g = items[items["style"] == "generic"].groupby(["domain", "prompt", "thr"]).agg(
        detect=("any", "mean"), hit=("hit", "mean"), best_iou=("iou", "mean"), recall=("recall", "mean"))
    S.append("## 1b. Generic prompts separately\n\n" + md(g))
    img = df[(df["style"] == "image")].groupby(["prompt", "thr"]).agg(
        detect=("any", "mean"), food_recall=("food_recall", "mean"), food_precision=("food_prec", "mean"))
    S.append("## 2. Synthetic image-level food recall (union of masks vs all GT food pixels)\n\n" + md(img))
    syn = items[items["domain"] == "synth"].copy()
    best = syn[syn["style"].isin(["exact", "simple"])].groupby(["img", "item", "thr"]).agg(
        any=("any", "max"), hit=("hit", "max"), iou=("iou", "max")).reset_index()
    gen = syn[syn["style"] == "generic"].groupby(["img", "item", "thr"]).agg(
        any=("any", "max"), hit=("hit", "max"), iou=("iou", "max")).reset_index()
    # per-item: found by ANY of the named prompts / by ANY generic prompt / by either
    m = best.merge(gen, on=["img", "item", "thr"], suffixes=("_named", "_gen"))
    m["hit_either"] = m["hit_named"] | m["hit_gen"]
    m["any_either"] = m["any_named"] | m["any_gen"]
    s = m.groupby("thr").agg(n=("hit_named", "size"), any_named=("any_named", "mean"), hit_named=("hit_named", "mean"),
                              any_gen=("any_gen", "mean"), hit_gen=("hit_gen", "mean"),
                              any_either=("any_either", "mean"), hit_either=("hit_either", "mean"))
    S.append("## 3. Synthetic items: best of exact/simple names vs best of food/dish/meal\n\n" + md(s))
    # per category
    simp = syn[(syn["style"] == "simple")].copy()
    cat = simp.groupby(["category", "thr"]).agg(n=("any", "size"), detect=("any", "mean"), hit=("hit", "mean"),
                                                  iou=("iou", "mean")).reset_index()
    pc = cat[cat["thr"] == 0.15].set_index("category")[["n", "detect", "hit", "iou"]].add_suffix("@.15")
    pc = pc.join(cat[cat["thr"] == 0.5].set_index("category")[["detect", "hit"]].add_suffix("@.5"))
    gcat = syn[(syn["style"] == "generic") & (syn["thr"] == 0.3)].groupby(["category", "item", "img"]).agg(h=("hit", "max")).groupby("category").h.mean()
    pc["generic_hit@.3"] = gcat
    real = items[(items["domain"] == "real") & (items["style"] == "simple") & (items["thr"] == 0.15)]
    pc = pc.join(real.groupby("category").agg(real_n=("any", "size"), real_detect=("any", "mean")))
    pc = pc.sort_values("hit@.15")
    S.append("## 4. Per category, synthetic (simplified name; generic = best of food/dish/meal at 0.3)\n\n" + md(pc))
    never = pc[(pc["hit@.15"] == 0) & (pc["generic_hit@.3"].fillna(0) == 0)].index.tolist()
    S.append("Categories never hit (IoU >= 0.3) by the simple name at 0.15 nor by generic prompts at 0.3: "
             + (", ".join(never) or "none") + "\n")
    # size and kind
    named = syn[(syn["style"] == "simple") & syn["thr"].isin([0.5, 0.15])].copy()
    q = named["area"].quantile([1 / 3, 2 / 3]).values
    named["size"] = pd.cut(named["area"], [0, q[0], q[1], 1e9], labels=[f"small(<{q[0]:.0f}px)", "medium", f"large(>{q[1]:.0f}px)"])
    S.append("## 5. By item size (pixels at 1008 long side, terciles), simple name\n\n" + md(
        named.groupby(["size", "thr"], observed=True).agg(n=("any", "size"), detect=("any", "mean"), hit=("hit", "mean"), iou=("iou", "mean"))))
    S.append("## 6. By item kind (scan = MetaFood3D, asset = Objaverse/YCB, procedural = fills/pieces/leaves/primitives), simple name\n\n"
             + md(named.groupby(["kind", "thr"]).agg(n=("any", "size"), detect=("any", "mean"), hit=("hit", "mean"), iou=("iou", "mean"))))
    src = named[named["thr"] == 0.15].groupby("source").agg(n=("any", "size"), detect=("any", "mean"), hit=("hit", "mean"), iou=("iou", "mean"))
    S.append("By source at 0.15:\n\n" + md(src))
    view = named[named["thr"] == 0.15].groupby("view").agg(n=("any", "size"), detect=("any", "mean"), hit=("hit", "mean"))
    S.append("By view at 0.15:\n\n" + md(view))
    f = syn[(syn["prompt"] == "food")].copy()
    f["hit5"] = f["iou"] >= 0.5
    f["kind2"] = f["kind"]
    S.append("## 7. Generic 'food' on synthetic items by kind and threshold (hit = IoU>=0.3, hit5 = IoU>=0.5, recall = pixel recall)\n\n"
             + md(f.groupby(["kind", "thr"]).agg(n=("any", "size"), hit=("hit", "mean"), hit5=("hit5", "mean"), recall=("recall", "mean"))))
    ni = f[f["thr"] == 0.5].groupby("img").agg(items=("item", "nunique"), masks=("n", "first"))
    nj = f[f["thr"] == 0.3].groupby("img").agg(items=("item", "nunique"), masks=("n", "first"))
    S.append(f"Masks returned by 'food' per frame vs visible GT items: thr 0.5 mean {ni.masks.mean():.1f} masks / {ni['items'].mean():.1f} items; "
             f"thr 0.3 mean {nj.masks.mean():.1f} masks / {nj['items'].mean():.1f} items.\n")
    r = items[(items["domain"] == "real") & (items["style"] == "generic")].groupby(["prompt", "thr"]).agg(
        n=("any", "size"), detect=("any", "mean"), mean_masks=("n", "mean"), iou_vs_n5k_food=("iou", "mean"))
    S.append("## 8. Real N5k generic prompts (reference = n5k food mask from 'food'@0.4 + dish filter)\n\n" + md(r))
    (out / "tables.md").write_text("\n\n".join(S))
    qualitative(out, df, items)
    print("\n\n".join(S))


def qualitative(out, df, items) -> None:
    """Side-by-side: synthetic frame + best mask | real frame of the same category + best mask."""
    syn = items[(items["domain"] == "synth") & (items["style"] == "simple") & (items["thr"] == 0.15)]
    real = items[(items["domain"] == "real") & (items["style"] == "simple") & (items["thr"] == 0.15)]
    cats = [c for c in real["category"].unique() if c in set(syn["category"].map(lambda s: s))]
    cat_map = syn.assign(c=syn["name"]).groupby("c")
    pairs = []
    for name in real["name"].unique():
        s = syn[syn["name"] == name].sort_values("area", ascending=False)
        r = real[real["name"] == name]
        if len(s) and len(r):
            pairs.append((name, s.iloc[0], r.sort_values("iou", ascending=False).iloc[0]))
    # choose a mix: failures first (low iou synth), then successes
    pairs.sort(key=lambda p: p[1]["iou"])
    pick = pairs[:5] + pairs[-3:] if len(pairs) > 8 else pairs
    font = None
    for k, (name, sr, rr) in enumerate(pick):
        tiles = []
        for row, dom in ((sr, "synth"), (rr, "real")):
            tag = row["img"]
            im = Image.open(out / "masks" / f"{tag}.jpg").convert("RGB")
            z = np.load(out / "masks" / f"{tag}.npz")
            key = f"simple_{row['item']}"
            m = unpack(z, key) if key in z else None
            ov = overlay(im, m)
            if dom == "synth" and m is not None:
                ys, xs = np.where(m)
            else:
                ys, xs = (np.where(m) if m is not None else (np.array([0]), np.array([0])))
            tiles.append((im, ov, row))
        W = 420
        canvas = Image.new("RGB", (W * 4, int(W * 0.8) + 24), "white")
        for j, (im, ov, row) in enumerate(tiles):
            for jj, pic in enumerate((im, ov)):
                p = pic.copy()
                p.thumbnail((W, int(W * 0.8)))
                canvas.paste(p, ((2 * j + jj) * W, 24))
            ImageDraw.Draw(canvas).text((j * 2 * W + 4, 6),
                f"{'SYNTH' if j == 0 else 'REAL'} '{row['prompt']}' ({name}) best IoU {row['iou']:.2f} score {row['best_score']:.2f}", fill="black")
        canvas.save(out / f"qual_{k}_{name.replace(' ', '_')}.jpg", quality=88)
