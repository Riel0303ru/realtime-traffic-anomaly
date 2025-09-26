# src/make_videos_from_folders.py
import os
import glob
import cv2
from tqdm import tqdm
import argparse

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_image_list(folder):
    # Accept common image extensions, case-insensitive
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    imgs = []
    for p in patterns:
        imgs.extend(glob.glob(os.path.join(folder, p)))
        imgs.extend(glob.glob(os.path.join(folder, p.upper())))
    imgs = sorted(imgs)
    return imgs

def make_video_from_folder(img_folder, out_file, fps=10):
    imgs = get_image_list(img_folder)
    if not imgs:
        print(f"[WARN] no imgs in {img_folder}")
        return False

    # Read first frame to get size
    first = cv2.imread(imgs[0])
    if first is None:
        print(f"[ERROR] unable to read first image {imgs[0]}")
        return False
    h, w = first.shape[:2]

    make_folder(os.path.dirname(out_file))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_file, fourcc, fps, (w, h))

    for p in tqdm(imgs, desc=f"Writing {os.path.basename(out_file)}", unit="frame"):
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] can't read {p}, skipping")
            continue
        # if image has single channel, convert to BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # if image size mismatches, resize to first frame size
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        vw.write(img)

    vw.release()
    return True

def main(results_dir, heatmap_subfolder="heatmap_overlay", out_videos="results/videos", fps=10, overwrite=False):
    heatmap_root = os.path.join(results_dir, heatmap_subfolder)
    if not os.path.isdir(heatmap_root):
        print(f"[ERROR] Heatmap root not found: {heatmap_root}")
        return

    make_folder(out_videos)

    # each subfolder in heatmap_root -> create a video
    subfolders = sorted([os.path.join(heatmap_root, d) for d in os.listdir(heatmap_root)
                         if os.path.isdir(os.path.join(heatmap_root, d))])

    if not subfolders:
        print(f"[WARN] No subfolders found under {heatmap_root}")
        return

    for sub in subfolders:
        folder_name = os.path.basename(sub.rstrip(os.sep))
        safe_name = folder_name.replace(" ", "_")
        out_file = os.path.join(out_videos, f"{safe_name}.mp4")
        if os.path.exists(out_file) and not overwrite:
            print(f"[SKIP] {out_file} already exists (use --overwrite to replace)")
            continue
        ok = make_video_from_folder(sub, out_file, fps=fps)
        if ok:
            print(f"[OK] wrote {out_file}")
        else:
            print(f"[FAIL] couldn't make video for {sub}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make mp4 videos from folders of frames")
    parser.add_argument("--results_dir", type=str, default=".", help="project root where results/ lives (default '.')")
    parser.add_argument("--heatmap_subfolder", type=str, default="results/heatmap_overlay", help="relative path (under results_dir) to heatmap folders")
    parser.add_argument("--out_videos", type=str, default="results/videos", help="output folder for generated mp4s (relative to results_dir)")
    parser.add_argument("--fps", type=int, default=10, help="video frames per second")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing mp4s")
    args = parser.parse_args()

    # Normalize paths relative to current working dir
    results_dir = os.path.abspath(args.results_dir)
    # If user passed default string for heatmap_subfolder that already includes 'results/', support it:
    heatmap_subfolder = args.heatmap_subfolder
    # If heatmap_subfolder is an absolute path, use it directly
    if os.path.isabs(heatmap_subfolder):
        heatmap_subfolder_abs = heatmap_subfolder
    else:
        heatmap_subfolder_abs = os.path.join(results_dir, heatmap_subfolder)

    out_videos_abs = args.out_videos
    if not os.path.isabs(out_videos_abs):
        out_videos_abs = os.path.join(results_dir, out_videos_abs)

    main(results_dir, heatmap_subfolder=heatmap_subfolder_abs, out_videos=out_videos_abs, fps=args.fps, overwrite=args.overwrite)
