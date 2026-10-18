"""Run the Nutrition5k Phase-2 ensemble (15 full-data rgbd_thick models) on an iPad FoodScan scan.

    python -m eval.real_n5k render SCAN_DIR OUT_DIR [--cam-h 0.319]      # CPU: overhead frame -> virtual N5k view
    python -m eval.real_n5k predict SCAN_DIR OUT_DIR --proj $PROJ        # GPU: SAM 3 mask, channels, 15 models

The N5k models need an overhead RealSense view (640x480, f = 465 px, camera ~35.9 cm above the counter, food on a
~4 cm scale) plus height-above-counter / food-thickness / food-mask channels. We pick the scan frame whose optical
axis is closest to gravity, back-project its (RGB-resolution) depth and re-render it from a virtual gravity-aligned
camera with N5k intrinsics, placed --cam-h above the table (0.319 = 0.359 - 0.04: N5k dishes sit on a ~4 cm scale,
so the table plays the scale top and food gets N5k's pixel size). Outputs go to OUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from foodscan.loader import Scan
from foodscan.volume import fit_support_plane

W0, H0 = 640, 480
F_PX = 0.359 / np.sqrt(5.957e-7)       # eval.nutrition5k.F_PX (463.8)
CX, CY = 319.5, 239.5


def pick_frame(scan: Scan):
    """Most overhead frame: optical axis (ARKit -Z in world; loader T_cw is OpenCV so +Z) vs world down."""
    best, best_s = None, -2
    for f in scan.frames:
        if f.tracking_state != "normal":
            continue
        s = float(f.T_cw[:3, 2] @ np.array([0, -1.0, 0]))
        if s > best_s:
            best, best_s = f, s
    return best, math.degrees(math.acos(min(1, best_s)))


def world_points(f):
    """(H,W,3) world points for every RGB pixel (depth upsampled to RGB res), valid mask, RGB."""
    rgb = f.rgb()
    h, w = rgb.shape[:2]
    d = f.depth()
    c = f.confidence()
    du = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    ok = np.isfinite(du) & (du > 0.05)
    if c is not None:
        ok &= cv2.resize(c, (w, h), interpolation=cv2.INTER_NEAREST) >= 1
    gy, gx = np.gradient(d)                       # drop flying pixels at depth edges
    edge = cv2.resize((np.hypot(gx, gy) > 0.02).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    ok &= ~cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    K = f.K_rgb
    v, u = np.mgrid[0:h, 0:w]
    z = du
    cam = np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], -1)
    R, t = f.T_cw[:3, :3], f.T_cw[:3, 3]
    return cam @ R.T + t, ok, rgb


def render(scan_dir: Path, out: Path, cam_h: float, rot: float = 0.0):
    out.mkdir(parents=True, exist_ok=True)
    scan = Scan(scan_dir, poses="arkit")
    f, tilt = pick_frame(scan)
    P, ok, rgb = world_points(f)
    pts, col = P[ok], rgb[ok]
    # table height: gravity-aligned support plane (up = world +y)
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 60000), replace=False)]
    plane = fit_support_plane(sub, dist_thresh=0.006, up_hint=np.array([0, 1.0, 0]))
    h = plane.height(pts)
    table_y = float(np.median(pts[np.abs(h) < 0.008][:, 1]))
    # food/dish centre: centroid of points 1-15 cm above the table within 0.35 m of the principal-ray hit
    ray_o, ray_d = f.T_cw[:3, 3], f.T_cw[:3, 2]
    hit = ray_o + ray_d * ((table_y - ray_o[1]) / ray_d[1])
    ab = pts[(pts[:, 1] > table_y + 0.01) & (pts[:, 1] < table_y + 0.15)]
    ab = ab[np.hypot(ab[:, 0] - hit[0], ab[:, 2] - hit[2]) < 0.30]
    ctr = ab.mean(0) if len(ab) > 500 else hit
    # virtual camera: looks along world -y, x = image right projected to horizontal
    zc = np.array([0, -1.0, 0])
    xc = f.T_cw[:3, 0] - (f.T_cw[:3, 0] @ zc) * zc
    xc /= np.linalg.norm(xc)
    if rot:
        a = math.radians(rot)
        xc = math.cos(a) * xc + math.sin(a) * np.cross(zc, xc)
    yc = np.cross(zc, xc)
    Rv = np.stack([xc, yc, zc], 0)                  # world -> virtual cam rotation
    C = np.array([ctr[0], table_y + cam_h, ctr[2]])
    pc = (pts - C) @ Rv.T
    z = pc[:, 2]
    u = np.round(F_PX * pc[:, 0] / z + CX).astype(int)
    v = np.round(F_PX * pc[:, 1] / z + CY).astype(int)
    m = (z > 0.05) & (u >= 0) & (u < W0) & (v >= 0) & (v < H0)
    u, v, z, c = u[m], v[m], z[m], col[m]
    order = np.argsort(-z)                          # far first, near overwrites
    depth = np.zeros((H0, W0), np.float32)
    img = np.zeros((H0, W0, 3), np.uint8)
    depth[v[order], u[order]] = z[order]
    img[v[order], u[order]] = c[order]
    # fill pinholes (disocclusion, 1-3 px) only
    hole = (depth == 0).astype(np.uint8)
    small = hole & ~cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    img = cv2.inpaint(img, small, 3, cv2.INPAINT_TELEA)
    dm = cv2.inpaint(depth, small, 3, cv2.INPAINT_TELEA)
    info = dict(scan=scan_dir.name, frame=f.index, tilt_from_vertical_deg=round(tilt, 1), table_y=table_y,
                cam_h=cam_h, centre_xz=[float(ctr[0]), float(ctr[2])], valid_frac=float((dm > 0).mean()),
                median_table_depth=float(np.median(dm[dm > 0])), n_frames_scan=len(scan))
    cv2.imwrite(str(out / "overhead_frame.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "virtual_rgb.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.save(out / "virtual_depth.npy", dm)
    json.dump(info, open(out / "render_info.json", "w"), indent=1)
    print(info, flush=True)
    return img, dm.astype(np.float64), info


def colorize(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def predict(scan_dir: Path, out: Path, proj: Path, cam_h: float):
    import torch
    from eval.n5k_cache import thickness
    from eval.n5k_predict import train_ids
    from eval.n5k_train import EXTRA, MEAN, STD, TARGETS, build_model, flips
    from eval.nutrition5k import read_metadata
    from PIL import Image

    img, depth, info = render(scan_dir, out, cam_h)
    # --- SAM 3 food mask on the virtual view, generic prompts as in eval.n5k_masks_sam3
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(device="cpu")
    model.backbone.half()
    model = model.to("cuda")
    proc = Sam3Processor(model, device="cuda", confidence_threshold=0.4)

    def masks(st, prompt):
        o = proc.set_text_prompt(state=st, prompt=prompt)
        m = o["masks"]
        return m.reshape(-1, *m.shape[-2:]).bool().cpu().numpy() if len(o["scores"]) else np.zeros((0, H0, W0), bool)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        st = proc.set_image(Image.fromarray(img))
        food_i, dish_i = masks(st, "food"), np.concatenate([masks(st, "plate"), masks(st, "bowl")])
    food = np.zeros((H0, W0), bool)
    kept = 0
    for fm in food_i:
        iou = [(fm & q).sum() / max((fm | q).sum(), 1) for q in dish_i]
        if not iou or max(iou) < 0.6:
            food |= fm
            kept += 1
    print(f"SAM3: {len(food_i)} food / {len(dish_i)} dish masks, kept {kept}, food frac {food.mean():.3f}", flush=True)
    del model, proc
    torch.cuda.empty_cache()
    info.update(sam_food_instances=len(food_i), sam_dish_instances=len(dish_i), sam_kept=kept, food_frac=float(food.mean()))

    # --- channels exactly as eval.n5k_cache.one
    hc, th = thickness(depth, food) if food.sum() >= 50 else (np.zeros_like(depth), np.zeros_like(depth))
    rs = lambda a: cv2.resize(a, (448, 336), interpolation=cv2.INTER_NEAREST)   # noqa: E731
    rgb448 = cv2.resize(img, (448, 336), interpolation=cv2.INTER_AREA)
    hc448, th448, mk448 = rs(hc.astype(np.float16)), rs(th.astype(np.float16)), rs(food.astype(np.uint8))
    x = torch.from_numpy(rgb448).permute(2, 0, 1)[None].float() / 255
    x = (x - MEAN) / STD
    ex = [torch.from_numpy(a.astype(np.float32))[None, None] * s for a, s in ((hc448, 10.0), (th448, 20.0), (mk448, 1.0))]
    x = torch.cat([x] + ex, 1).cuda().contiguous(memory_format=torch.channels_last)

    # --- 15 models
    root = proj / "datasets/Nutrition5k"
    meta = read_metadata(root)
    ids = (proj / "data/cache/n5k_448_sam3b/ids.txt").read_text().split()
    data = SimpleNamespace(index={d: i for i, d in enumerate(ids)})
    groups = {"A1_cnx_log": "cnx_log_full", "A2_cnl_raw": "cnl_raw_full", "B1_cnx80_log": "cnx80_log_full"}
    preds = {}
    for g, p in groups.items():
        for s in range(5):
            run = proj / f"results/n5k/phase2/{p}_s{s}"
            a = json.load(open(run / "summary.json"))["args"]
            y = torch.tensor([[math.log1p(meta[d][t]) for t in TARGETS] for d in train_ids(a, root, data, meta)])
            mu, sd = y.mean(0), y.std(0)
            net = build_model(a["arch"], EXTRA[a["channels"]], float(a["drop_path"]), (336, 448), len(TARGETS), pretrained=False)
            net.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
            net = net.cuda().to(memory_format=torch.channels_last).eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z = torch.stack([net(v).float()[:, :len(TARGETS)] for v in flips(x)]).mean(0).cpu()
            preds[run.name] = torch.expm1(z * sd + mu).clamp(min=0)[0].numpy()
            print(run.name, dict(zip(TARGETS, np.round(preds[run.name], 1))), flush=True)
    allp = np.array(list(preds.values()))
    ens = allp.mean(0)
    res = dict(scan=scan_dir.name, ensemble="15 Phase-2 full-data models, equal weight, 4-flip TTA", cam_h=cam_h,
               pred=dict(zip(TARGETS, map(float, ens))), per_model_sd=dict(zip(TARGETS, map(float, allp.std(0)))),
               groups={g: dict(zip(TARGETS, map(float, np.mean([v for k, v in preds.items() if k.startswith(p)], 0))))
                       for g, p in groups.items()},
               per_model={k: dict(zip(TARGETS, map(float, v))) for k, v in preds.items()}, render=info)
    json.dump(res, open(out / "prediction.json", "w"), indent=1)

    # --- visuals of the model input
    ov = img.copy()
    ov[food] = (0.5 * ov[food] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cv2.imwrite(str(out / "model_input_rgb.png"), cv2.cvtColor(rgb448, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "model_input_mask_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    v = depth[depth > 0]
    panels = [rgb448, colorize(cv2.resize(depth.astype(np.float32), (448, 336), interpolation=cv2.INTER_NEAREST),
                               np.percentile(v, 1), np.percentile(v, 99)),
              colorize(hc448.astype(np.float32), 0, 0.1), colorize(th448.astype(np.float32), 0, 0.05),
              np.repeat(mk448[..., None] * 255, 3, 2).astype(np.uint8)]
    names = ["rgb 448x336", "virtual depth (m)", "h_counter 0-10cm", "thick 0-5cm", "food mask"]
    for p, n in zip(panels, names):
        cv2.putText(p, n, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "model_input_panels.png"), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
    print("DONE", json.dumps(res["pred"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "predict"])
    ap.add_argument("scan", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cam-h", type=float, default=0.319)
    ap.add_argument("--proj", type=Path)
    a = ap.parse_args()
    if a.stage == "render":
        render(a.scan, a.out, a.cam_h)
    else:
        predict(a.scan, a.out, a.proj, a.cam_h)
