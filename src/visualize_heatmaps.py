#!/usr/bin/env python3
"""
visualize_heatmaps.py

Generate heatmap overlay images for top-N anomaly frames from results folders.

Usage:
    python src/visualize_heatmaps.py --results_dir ./results --top_n 250

Outputs saved to:
    ./results/heatmap_overlay/<TestXXX>_top<NN>_<YYYYmmdd-HHMMSS>/
"""

import os
import cv2
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

def make_folder(path):
    os.makedirs(path, exist_ok=True)
    return path

def overlay_heatmap_on_frame(frame_bgr, score_norm, colormap=cv2.COLORMAP_JET, alpha_frame=0.6, alpha_heat=0.4):
    """
    frame_bgr: uint8 HxWx3 BGR
    score_norm: float 0..1
    returns: overlay (uint8 BGR)
    """
    # ensure frame is HxW x3
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be HxWx3 uint8 BGR image")

    h, w = frame_bgr.shape[:2]
    heat_val = np.clip(int(score_norm * 255), 0, 255)
    heatmap_single = np.full((h, w), heat_val, dtype=np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_single, colormap)  # BGR

    # blend
    overlay = cv2.addWeighted(frame_bgr, alpha_frame, heatmap_color, alpha_heat, 0)
    return overlay

def safe_imwrite(path, img):
    """Write image and return True on success, False otherwise."""
    try:
        ok = cv2.imwrite(path, img)
        return bool(ok)
    except Exception as e:
        print(f"[ERROR] Failed to write {path}: {e}")
        return False

def process_one_test(test_path, out_root, top_n=250, colormap=cv2.COLORMAP_JET, keep_existing=False):
    """
    test_path: folder containing frames and anomaly_scores.csv (e.g. ./results/Test001)
    out_root: base folder to save overlays (./results/heatmap_overlay)
    """
    basename = os.path.basename(test_path.rstrip(os.sep))
    csv_path = os.path.join(test_path, "anomaly_scores.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV not found: {csv_path}")
        return dict(name=basename, saved=0, skipped=0, reason="no_csv")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARNING] Cannot read CSV {csv_path}: {e}")
        return dict(name=basename, saved=0, skipped=0, reason="csv_read_error")

    # required columns
    if 'frame' not in df.columns or 'score' not in df.columns:
        print(f"[WARNING] CSV missing required columns 'frame'/'score': {csv_path}")
        return dict(name=basename, saved=0, skipped=0, reason="bad_columns")

    # Normalize scores 0..1 (per-video)
    scores = df['score'].astype(float).values
    mn, mx = scores.min(), scores.max()
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    df['score_norm'] = (scores - mn) / denom

    # Sort and pick top N
    df_top = df.sort_values('score_norm', ascending=False).head(top_n)

    # Prepare output folder with timestamp to avoid overwrite unless requested
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_folder_name = f"{basename}_top{top_n}_{ts}"
    out_dir = os.path.join(out_root, out_folder_name)
    make_folder(out_dir)

    saved = 0
    skipped = 0
    failed_writes = 0

    print(f"[INFO] Processing {basename}, top {top_n} anomalies -> saving to {out_dir}")
    for _, row in tqdm(df_top.iterrows(), total=len(df_top), desc=f"{basename}"):
        try:
            frame_num = int(row['frame'])
            score_norm = float(row['score_norm'])
        except Exception:
            skipped += 1
            continue

        # frame filename convention: 00001.jpg etc
        frame_filename = f"{frame_num:05d}.jpg"
        frame_path = os.path.join(test_path, frame_filename)
        if not os.path.exists(frame_path):
            # try alternative: skip leading zeros mismatch
            alt = None
            for ext in (".jpg", ".png", ".jpeg"):
                p = os.path.join(test_path, f"{frame_num}{ext}")
                if os.path.exists(p):
                    alt = p
                    break
            if alt is None:
                skipped += 1
                continue
            frame_path = alt

        frame = cv2.imread(frame_path)
        if frame is None:
            skipped += 1
            continue

        # produce overlay (BGR)
        try:
            overlay = overlay_heatmap_on_frame(frame, score_norm, colormap=colormap)
        except Exception as e:
            print(f"[ERROR] Could not create overlay for {frame_path}: {e}")
            skipped += 1
            continue

        # filename includes score to avoid name collisions within same run
        out_name = f"{frame_num:05d}_{score_norm:.4f}.jpg"
        out_path = os.path.join(out_dir, out_name)

        # if file exists and keep_existing False -> overwrite, else if keep_existing True and exists -> skip
        if os.path.exists(out_path) and keep_existing:
            skipped += 1
            continue

        ok = safe_imwrite(out_path, overlay)
        if ok:
            saved += 1
        else:
            failed_writes += 1

    return dict(name=basename, saved=saved, skipped=skipped, failed_writes=failed_writes, out_dir=out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results", help="root results folder that contains TestXXX subfolders")
    parser.add_argument("--top_n", type=int, default=250, help="top N anomalies to visualize per test folder")
    parser.add_argument("--out", type=str, default=None, help="output heatmap directory (default: ./results/heatmap_overlay)")
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap name (e.g. JET, HOT, VIRIDIS)")
    parser.add_argument("--keep_existing", action="store_true", help="if set and output exists, do not overwrite existing files")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"[ERROR] results_dir not found: {results_dir}")
        return

    heatmap_root = args.out if args.out else os.path.join(results_dir, "heatmap_overlay")
    make_folder(heatmap_root)

    # map colormap string to cv2 constant
    cmap_str = args.colormap.upper()
    cmap_map = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS if hasattr(cv2, 'COLORMAP_VIRIDIS') else cv2.COLORMAP_JET,
        "PLASMA": cv2.COLORMAP_PLASMA if hasattr(cv2, 'COLORMAP_PLASMA') else cv2.COLORMAP_JET,
    }
    colormap = cmap_map.get(cmap_str, cv2.COLORMAP_JET)

    # find candidate test folders (folders that have anomaly_scores.csv & frame images)
    candidates = []
    for entry in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, entry)
        if os.path.isdir(full):
            # require anomaly_scores.csv
            if os.path.exists(os.path.join(full, "anomaly_scores.csv")):
                candidates.append(full)

    if not candidates:
        print(f"[WARNING] No result subfolders with anomaly_scores.csv found under {results_dir}")
        return

    summary = []
    for test_path in candidates:
        res = process_one_test(test_path, heatmap_root, top_n=args.top_n, colormap=colormap, keep_existing=args.keep_existing)
        summary.append(res)

    # print summary
    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"{s['name']}: saved={s.get('saved',0)}, skipped={s.get('skipped',0)}, failed_writes={s.get('failed_writes',0)}, out={s.get('out_dir','-')}")
    print(f"[INFO] All heatmaps saved under {heatmap_root}")

if __name__ == "__main__":
    main()
