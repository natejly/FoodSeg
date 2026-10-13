"""Grade a FoodScan capture: is a bad reconstruction the scan technique or the sensor?

    python -m foodscan.scan_report SCAN_DIR [SCAN_DIR ...] [--json]

Reads only frames.jsonl, conf/ and rgb/ (no fusion). Each metric is compared with the
target from docs/collection_protocol.md and flagged when it misses. Technique problems
(coverage, speed, light, distance) are fixable by rescanning. A scan that passes them all
and still reconstructs badly points at the sensor or the fusion.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .loader import CONF_HIGH, Scan
from .run_meal import ARKIT_UP, look_at

# (metric, comparison, target, what to do about a miss)
TARGETS = [
    ("seconds", ">=", 30, "scan longer: 45-90 s"),
    ("orbit_coverage", ">=", 0.9, "circle the whole plate"),
    ("max_elevation_deg", ">=", 70, "add a pass from directly above"),
    ("speed_cm_s_p90", "<=", 12, "move slower"),
    ("blur_px_median", "<=", 4, "slow down and/or add light"),
    ("exposure_ms_median", "<=", 12, "add light (exposure is near ARKit's 16.7 ms cap)"),
    ("distance_cm_median", ">=", 25, "back off: LiDAR bias grows below ~20-25 cm"),
    ("distance_cm_median", "<=", 50, "move closer: the plate covers too few depth pixels"),
    ("conf_high_frac", ">=", 0.7, "LiDAR confidence is low: check glare, dark/glossy plate, light"),
    ("tracking_normal_frac", ">=", 0.95, "tracking dropped: more texture, slower motion"),
]


def report(scan_dir: str, n_sharp: int = 40) -> dict:
    scan = Scan(scan_dir)
    frames = scan.frames
    good = scan.good_frames()
    if len(good) < 2:
        raise RuntimeError(f"{scan_dir}: fewer than 2 frames with normal tracking")
    raw = [json.loads(line) for line in open(scan.dir / "frames.jsonl") if line.strip()]
    exposure = {r["index"]: r.get("exposureDuration", np.nan) for r in raw}

    t = np.array([f.timestamp for f in good])
    C = np.array([f.T_cw[:3, 3] for f in good])
    centre = look_at(scan)
    v = C - centre
    dist = np.linalg.norm(v, axis=1)
    up = v @ ARKIT_UP
    elev = np.degrees(np.arcsin(np.clip(up / dist, -1, 1)))
    horiz = v - np.outer(up, ARKIT_UP)
    ref = np.cross(ARKIT_UP, [1.0, 0, 0])   # any horizontal axis; ARKit world is gravity-aligned
    az = np.arctan2(horiz @ np.cross(ARKIT_UP, ref), horiz @ ref)
    bins = np.unique(((az[elev < 75] + np.pi) / (2 * np.pi) * 36).astype(int) % 36)

    dt = np.diff(t)
    ok = dt > 0
    speed = np.linalg.norm(np.diff(C, axis=0), axis=1)[ok] / dt[ok]
    exp_s = np.array([exposure[f.index] for f in good])[1:][ok]
    fx = good[0].K_rgb[0, 0]
    blur = speed * exp_s * fx / dist[1:][ok]

    conf_high = []
    for f in good[:: max(1, len(good) // 40)]:
        c = f.confidence()
        if c is not None:
            h, w = c.shape   # central half of the frame, where the plate should be
            conf_high.append((c[h // 4: 3 * h // 4, w // 4: 3 * w // 4] >= CONF_HIGH).mean())

    sample = good[:: max(1, len(good) // n_sharp)]
    sharp = np.array([f.sharpness() for f in sample])

    m = {
        "scan": scan.dir.name,
        "device": scan.meta.get("device"),
        "frames": len(frames),
        "seconds": float(t[-1] - t[0]),
        "tracking_normal_frac": len(good) / max(1, len(frames)),
        "orbit_coverage": len(bins) / 36,
        "min_elevation_deg": float(elev.min()),
        "max_elevation_deg": float(elev.max()),
        "distance_cm_median": float(100 * np.median(dist)),
        "speed_cm_s_median": float(100 * np.median(speed)),
        "speed_cm_s_p90": float(100 * np.percentile(speed, 90)),
        "exposure_ms_median": float(1000 * np.nanmedian(exp_s)),
        "blur_px_median": float(np.nanmedian(blur)),
        "blur_px_p90": float(np.nanpercentile(blur, 90)),
        "conf_high_frac": float(np.mean(conf_high)) if conf_high else float("nan"),
        "blurry_frame_frac": float((sharp < 0.5 * np.median(sharp)).mean()) if sharp.any() else 0.0,
    }
    m["flags"] = [f"{k} {m[k]:.3g} (target {op} {tgt}): {fix}" for k, op, tgt, fix in TARGETS
                  if np.isfinite(m[k]) and not (m[k] >= tgt if op == ">=" else m[k] <= tgt)]
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dirs", nargs="+")
    ap.add_argument("--json", action="store_true", help="print one JSON object per scan")
    args = ap.parse_args()
    for d in args.scan_dirs:
        m = report(d)
        if args.json:
            print(json.dumps(m))
            continue
        print(f"== {m['scan']} ({m['device']}) ==")
        for k, v in m.items():
            if k not in ("scan", "device", "flags"):
                print(f"  {k:22s} {v:.3g}" if isinstance(v, float) else f"  {k:22s} {v}")
        print("  " + ("OK: technique within targets" if not m["flags"] else
                      "\n  ".join(["FLAGS:"] + m["flags"])))


if __name__ == "__main__":
    main()
