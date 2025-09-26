import os
import cv2
import numpy as np
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    tifffile = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
FRAMES_DIR = os.path.join(BASE_DIR, "..", "data", "frames")

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frames_from_video(video_path, output_dir, start_count=1, skip_frame=1):
    make_folder(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return start_count

    count = 0
    saved = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip_frame == 0:
            frame_name = f"{str(start_count).zfill(5)}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            start_count += 1
            saved += 1
        count += 1

    cap.release()
    print(f"[INFO] Extracted {saved} frames from {os.path.basename(video_path)}")
    return start_count

def copy_tif_sequence(tif_path, output_dir, start_count=1):
    if tifffile is None:
        print("[ERROR] Install 'tifffile' to handle .tif sequences: pip install tifffile")
        return start_count

    make_folder(output_dir)
    imgs = tifffile.imread(tif_path)
    if len(imgs.shape) == 2:  # single frame
        imgs = imgs[np.newaxis, ...]
    elif len(imgs.shape) == 3 and imgs.shape[0] < 10:  # sometimes tif stack dim [H,W,N]
        imgs = np.transpose(imgs, (2,0,1))

    saved = 0
    for i in tqdm(range(len(imgs)), desc=f"Processing {os.path.basename(tif_path)}"):
        img = imgs[i]
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frame_name = f"{str(start_count).zfill(5)}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), img)
        start_count += 1
        saved += 1

    print(f"[INFO] Copied {saved} frames from {os.path.basename(tif_path)}")
    return start_count

def main(skip_frame=1):
    make_folder(FRAMES_DIR)
    # Looping per subfolder video/TIF
    for root, dirs, files in os.walk(RAW_DIR):
        rel_root = os.path.relpath(root, RAW_DIR)
        if rel_root == ".":  # skip root
            continue
        output_folder = os.path.join(FRAMES_DIR, rel_root)
        make_folder(output_folder)

        # Mulai counter frame dari 1 untuk setiap folder
        frame_counter = 1
        for f in sorted(files):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, f)
                frame_counter = extract_frames_from_video(video_path, output_folder, start_count=frame_counter, skip_frame=skip_frame)
            elif f.lower().endswith(".tif"):
                tif_path = os.path.join(root, f)
                frame_counter = copy_tif_sequence(tif_path, output_folder, start_count=frame_counter)

        print(f"[INFO] All frames from folder '{rel_root}' saved under {output_folder}")

if __name__ == "__main__":
    main(skip_frame=1)
