# src/realtime_detection.py
"""
Robust realtime detection script:
- Works with ultralytics YOLO (yolov8) or torch.hub yolov5.
- Tries cv2.VideoCapture first; if gagal, fallback to ffmpeg pipe.
- Robust parsing of detection outputs to avoid IndexError.
- Press 'q' to quit.
"""

import os, sys, shutil, subprocess, time
from pathlib import Path
import numpy as np
import cv2
import torch

# ------------- CONFIG -------------
VIDEO_SOURCE = "https://atcs.tasikmalayakota.go.id/camera/simpanglima.m3u8"
CONF_THRESH = 0.25
MODEL_PREF = "ultralytics"   # "ultralytics" or "yolov5" (tries ultralytics first)
# -----------------------------------

def find_ffmpeg():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    # common windows locations (optional)
    for p in [r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
        if Path(p).exists():
            return str(p)
    return None

def probe_resolution(source):
    """Try ffprobe to get resolution, return (w,h) or None."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", source]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=8)
        s = out.decode().strip()
        if "," in s:
            w,h = s.split(",")
            return int(w), int(h)
    except Exception:
        return None
    return None

# ------- Load model (try ultralytics then yolov5) -------
def load_model(device):
    # try ultralytics YOLO (yolov8)
    try:
        from ultralytics import YOLO as UltralyticsYOLO
        print("[INFO] Loading Ultralytics YOLO (yolov8n)...")
        model = UltralyticsYOLO("yolov8n.pt")  # will download if missing
        return ("ultralytics", model)
    except Exception as e:
        print("[WARN] ultralytics not available:", e)

    # fallback to torch.hub yolov5
    try:
        print("[INFO] Loading YOLOv5 via torch.hub (yolov5s)...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        model.eval()
        return ("yolov5", model)
    except Exception as e:
        print("[ERROR] Failed to load any YOLO backend:", e)
        return (None, None)

# ------- Robust detection parsing & drawing -------
def parse_and_draw_ultralytics(result_obj, frame, conf_thresh=0.25, keep_classes=None):
    """
    result_obj: output of model(frame) for ultralytics (list-like)
    frame: BGR numpy array (we draw on it)
    """
    if result_obj is None:
        return frame
    # result_obj is usually a list, one item per image
    try:
        r = result_obj[0]
    except Exception:
        return frame
    boxes = getattr(r, "boxes", None)
    names = getattr(r, "names", None) or {}
    if boxes is None:
        return frame
    # boxes might be a Boxes object; iterate safely
    try:
        # iterate over boxes attribute if available
        for b in boxes:
            # some boxes may not have .xyxy depending on version; guard it
            xyxy = None
            if hasattr(b, "xyxy"):
                try:
                    arr = b.xyxy.cpu().numpy()
                    if arr.size:
                        # b.xyxy may be an array Nx4; pick first row if needed
                        if arr.ndim == 2:
                            xyxy = arr[0].astype(int).tolist()
                        else:
                            xyxy = arr.astype(int).tolist()
                except Exception:
                    xyxy = None
            # fallback: try .xyxy property as list
            if xyxy is None:
                try:
                    xyxy = list(map(int, b.xyxy))
                except Exception:
                    xyxy = None
            if not xyxy or len(xyxy) < 4:
                continue
            x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
            conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else float(getattr(b,"conf",0.0))
            cls_id = int(b.cls.cpu().numpy()) if hasattr(b, "cls") else int(getattr(b,"cls",0))
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            if conf < conf_thresh:
                continue
            if keep_classes and label not in keep_classes:
                continue
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (x1, max(10,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    except Exception as e:
        # don't crash drawing step
        print("[WARN] exception parsing ultralytics result:", e)
    return frame

def parse_and_draw_yolov5(result_obj, frame, conf_thresh=0.25, keep_classes=None):
    """
    result_obj: output from model(frame) for yolov5 hub
    """
    if result_obj is None:
        return frame
    try:
        preds = result_obj.xyxy[0]  # tensor Nx6
        if preds is None or len(preds)==0:
            return frame
        names = getattr(result_obj, "names", None) or {}
        for p in preds.cpu().numpy():
            if len(p) < 6:
                continue
            x1,y1,x2,y2,conf,cls = p.tolist()
            if conf < conf_thresh:
                continue
            label = names[int(cls)] if isinstance(names, (list,dict)) and int(cls) in names else str(int(cls))
            if keep_classes and label not in keep_classes:
                continue
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (int(x1), max(10,int(y1)-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    except Exception as e:
        print("[WARN] exception parsing yolov5 result:", e)
    return frame

# ------- ffmpeg frame generator -------
def ffmpeg_frame_generator(ffmpeg_bin, source, width, height):
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", source,
        "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bytes_per_frame = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(bytes_per_frame)
            if len(raw) != bytes_per_frame:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            yield frame
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        if proc.stdout: proc.stdout.close()
        if proc.stderr: proc.stderr.close()

# ------- main loop -------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend, model = load_model(device)
    if backend is None:
        print("[ERROR] No model backend available. Install ultralytics or allow torch.hub.")
        return

    # first try cv2.VideoCapture (works for many m3u8)
    print("[INFO] Trying cv2.VideoCapture(...) for source:", VIDEO_SOURCE)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if cap.isOpened():
        print("[OK] VideoCapture opened source.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] cv2 read returned no frame; breaking.")
                    break
                # model inference and robust parsing
                if backend == "ultralytics":
                    res = model(frame)
                    vis = parse_and_draw_ultralytics(res, frame, CONF_THRESH)
                else:
                    out = model(frame)
                    vis = parse_and_draw_yolov5(out, frame, CONF_THRESH)
                cv2.imshow("Realtime Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    # fallback to ffmpeg pipe
    print("[WARN] VideoCapture couldn't open the source; trying ffmpeg pipe.")
    ffmpeg_bin = find_ffmpeg()
    if ffmpeg_bin is None:
        print("[ERROR] ffmpeg not found in PATH. Install ffmpeg or add it to PATH.")
        return

    res = probe_resolution(VIDEO_SOURCE)
    if res:
        width, height = res
        print(f"[INFO] probed resolution: {width}x{height}")
    else:
        print("[WARN] Could not probe resolution; using fallback 640x360.")
        width, height = 640, 360

    try:
        for frame in ffmpeg_frame_generator(ffmpeg_bin, VIDEO_SOURCE, width, height):
            if frame is None:
                continue
            if backend == "ultralytics":
                res = model(frame)
                vis = parse_and_draw_ultralytics(res, frame, CONF_THRESH)
            else:
                out = model(frame)
                vis = parse_and_draw_yolov5(out, frame, CONF_THRESH)
            cv2.imshow("Realtime Detection (ffmpeg)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("[ERROR] ffmpeg pipeline error:", e)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
