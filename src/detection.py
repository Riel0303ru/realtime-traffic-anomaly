# src/detection.py
"""
Detection script:
- Loads YOLO via ultralytics if available, else tries torch.hub('ultralytics/yolov5','yolov5s').
- Scans frames directory tree (data/frames/*/Test/*) and runs detection per frame folder.
- Saves CSV: frame,bbox_x1,y1,x2,y2,conf,class_name
- Optionally saves overlayed images.
"""

import os
import sys
import argparse
from tqdm import tqdm
import cv2
import csv

# try to import model backends
YOLO = None
use_ultralytics = False
use_torchhub = False
try:
    from ultralytics import YOLO as UltralyticsYOLO  # if user installed ultralytics
    YOLO = UltralyticsYOLO
    use_ultralytics = True
except Exception:
    try:
        import torch
        # we'll use torch.hub to fetch yolov5s (requires network the first run)
        use_torchhub = True
        torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
    except Exception:
        pass

CLASS_FILTER = ["person", "car", "motorbike", "bicycle", "scooter", "bus", "truck"]  # filter of interest

def make_folder(p):
    os.makedirs(p, exist_ok=True)

def discover_video_folders(frames_root):
    # expects frames_root like data/frames/UCSDped2/Test
    out = []
    for root, dirs, files in os.walk(frames_root):
        imgs = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
        if imgs:
            out.append(root)
    return sorted(out)

def load_model(device):
    if use_ultralytics:
        print("[INFO] Using ultralytics YOLO model from 'ultralytics' package.")
        model = YOLO("yolov8n.pt")  # small network
        return model
    elif use_torchhub:
        import torch
        print("[INFO] Using torch.hub('ultralytics/yolov5','yolov5s')")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        model.eval()
        return model
    else:
        raise RuntimeError("No YOLO backend available. Install 'ultralytics' or enable internet for torch.hub.")

def detect_on_image(model, img, device, conf_thresh=0.25):
    """
    Returns list of detections: (x1,y1,x2,y2,conf,label)
    Works for ultralytics and yolov5 hub outputs.
    """
    results = []
    if use_ultralytics:
        res = model(img)  # ultralytics returns result object
        r = res[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return results
        for b in boxes:
            conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else float(b.conf)
            if conf < conf_thresh:
                continue
            cls_id = int(b.cls.cpu().numpy()) if hasattr(b, "cls") else int(b.cls)
            label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)

            if not hasattr(b, "xyxy"):
                continue
            xyxy = b.xyxy.cpu().numpy().astype(int).flatten().tolist()
            if len(xyxy) < 4:
                continue

            x1, y1, x2, y2 = xyxy[:4]
            results.append((x1, y1, x2, y2, conf, label))
        return results

    else:
        out = model(img)  # torch.hub yolov5 model
        preds = out.xyxy[0]  # tensor Nx6: x1,y1,x2,y2,conf,class
        if preds is None or len(preds) == 0:
            return results
        for p in preds.cpu().numpy():
            if len(p) < 6:
                continue
            x1, y1, x2, y2, conf, cls = p.tolist()
            if conf < conf_thresh:
                continue
            label = model.names[int(cls)]
            results.append((int(x1), int(y1), int(x2), int(y2), float(conf), label))
        return results

def visualize_detections(img, dets):
    for (x1,y1,x2,y2,conf,label) in dets:
        color = (0,255,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        txt = f"{label}:{conf:.2f}"
        cv2.putText(img, txt, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img

def run_detection(frames_root, results_root, device="cpu", conf_thresh=0.25, save_vis=True, filter_classes=None):
    model = load_model(device)
    folders = discover_video_folders(frames_root)
    if not folders:
        print("[WARN] No folders with frames found under", frames_root)
        return

    for vf in folders:
        rel = os.path.relpath(vf, frames_root)
        out_dir = os.path.join(results_root, "detections", rel)
        out_vis = os.path.join(results_root, "detections_visual", rel)
        make_folder(out_dir)
        if save_vis:
            make_folder(out_vis)
        csv_path = os.path.join(out_dir, "detections.csv")
        print(f"[INFO] Running detection on {rel} -> {csv_path}")

        frames = sorted([f for f in os.listdir(vf) if f.lower().endswith((".jpg",".jpeg",".png"))])
        if not frames:
            print("  [SKIP] no images in", vf); continue

        with open(csv_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["frame", "x1","y1","x2","y2","conf","label"])
            for fname in tqdm(frames, desc=f"Detect {rel}", unit="frame"):
                fpath = os.path.join(vf, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                dets = detect_on_image(model, img, device, conf_thresh=conf_thresh)
                if filter_classes:
                    dets = [d for d in dets if d[5] in filter_classes]
                frame_idx = int(os.path.splitext(fname)[0].lstrip("0") or 0)
                for (x1,y1,x2,y2,conf,label) in dets:
                    writer.writerow([frame_idx, x1, y1, x2, y2, f"{conf:.4f}", label])
                if save_vis:
                    vis = img.copy()
                    vis = visualize_detections(vis, dets)
                    cv2.imwrite(os.path.join(out_vis, fname), vis)

        print(f"[OK] detections saved to {csv_path}; visualizations to {out_vis if save_vis else '(disabled)'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", default="data/frames/UCSDped2/Test", help="root containing Test folders")
    p.add_argument("--results_root", default="results", help="root to write results/*")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--no_vis", action="store_true", help="don't save visualization images")
    p.add_argument("--filter", nargs="*", default=CLASS_FILTER, help="class names to keep (default common ones)")
    args = p.parse_args()

    frames_root = os.path.abspath(args.frames_root)
    results_root = os.path.abspath(args.results_root)
    run_detection(frames_root, results_root, device=args.device, conf_thresh=args.conf,
                  save_vis=(not args.no_vis), filter_classes=args.filter)
