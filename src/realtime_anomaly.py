# src/realtime_anomaly.py
import os
import time
import argparse
import cv2
import numpy as np
from collections import deque

# Backend detection flags (ultralytics preferred)
USE_ULTRALYTICS = False
USE_TORCHHUB = False
try:
    from ultralytics import YOLO as UltralyticsYOLO
    USE_ULTRALYTICS = True
except Exception:
    try:
        import torch
        torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
        USE_TORCHHUB = True
    except Exception:
        pass

# ----- CONFIG -----
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "motorbike", "scooter", "van"}
PERSON_CLASS_NAMES = {"person"}
DEFAULT_MODEL_ULTRA = "yolov8n.pt"
DEFAULT_MODEL_YOLOV5 = "yolov5s"

def make_folder(p):
    os.makedirs(p, exist_ok=True)

def load_model(device):
    if USE_ULTRALYTICS:
        print("[INFO] Loaded ultralytics YOLO backend")
        model = UltralyticsYOLO(DEFAULT_MODEL_ULTRA)
        return "ultralytics", model
    if USE_TORCHHUB:
        import torch
        print("[INFO] Loaded YOLOv5 via torch.hub")
        model = torch.hub.load("ultralytics/yolov5", DEFAULT_MODEL_YOLOV5, pretrained=True)
        model.to(device)
        model.eval()
        return "yolov5_hub", model
    raise RuntimeError("No YOLO backend available. Install 'ultralytics' or enable torch.hub internet.")

def parse_ultralytics_results(res, conf_thresh):
    dets = []
    try:
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return dets
        xyxy_all = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        names = getattr(res.model, "names", None) or getattr(res, "names", {})
        for i in range(len(xyxy_all)):
            xy = xyxy_all[i]
            conf = float(confs[i])
            if conf < conf_thresh:
                continue
            cls_id = int(cls_ids[i])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            x1, y1, x2, y2 = map(int, xy.tolist())
            dets.append((x1,y1,x2,y2,conf,label))
    except Exception:
        pass
    return dets

def parse_yolov5_hub_results(out, conf_thresh):
    dets = []
    try:
        preds = out.xyxy[0]
        if preds is None or preds.shape[0] == 0:
            return dets
        names = out.names
        for p in preds.cpu().numpy():
            x1,y1,x2,y2,conf,cls = p.tolist()
            if conf < conf_thresh:
                continue
            label = names[int(cls)] if isinstance(names, (list, dict)) else str(int(cls))
            dets.append((int(x1), int(y1), int(x2), int(y2), float(conf), label))
    except Exception:
        pass
    return dets

def detect_and_draw(backend_model_tuple, frame, conf_thresh=0.25, class_filter=None):
    backend, model = backend_model_tuple
    try:
        if backend == "ultralytics":
            res_list = model(frame)
            if not res_list:
                return frame, []
            res = res_list[0]
            dets = parse_ultralytics_results(res, conf_thresh)
        else:
            out = model(frame)
            dets = parse_yolov5_hub_results(out, conf_thresh)

        if class_filter:
            dets = [d for d in dets if d[5] in class_filter]

        vis = frame.copy()
        box_color = (0,0,0)   # black boxes
        text_color = (0,0,0)  # black text
        for (x1,y1,x2,y2,conf,label) in dets:
            cv2.rectangle(vis, (x1,y1), (x2,y2), box_color, 2)
            txt = f"{label} {conf:.2f}"
            cv2.putText(vis, txt, (max(0,x1), max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        return vis, dets
    except Exception as e:
        print(f"[ERROR] detection failed: {e}")
        return frame, []

# --------------------
# New: time-window based anomaly history
# --------------------
class TimeWindowAnomaly:
    def __init__(self, window_seconds=60.0):
        self.window_seconds = float(window_seconds)
        self.storage = deque()  # each entry: (timestamp, flag_int)

    def add(self, flag_bool):
        now = time.time()
        self.storage.append((now, 1 if flag_bool else 0))
        self._purge_old(now)

    def _purge_old(self, now=None):
        if now is None:
            now = time.time()
        cutoff = now - self.window_seconds
        while self.storage and self.storage[0][0] < cutoff:
            self.storage.popleft()

    def get_stats(self):
        self._purge_old()
        total = len(self.storage)
        if total == 0:
            return 0.0, 0
        s = sum(f for (_,f) in self.storage)
        pct = 100.0 * s / total
        return pct, total

    def set_window(self, seconds):
        self.window_seconds = float(seconds)
        self._purge_old()

# ----- MAIN -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video source (0 for webcam or RTSP/m3u8 URL or file)")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--save_alerts", action="store_true", help="save alert frames to results/alerts")
    p.add_argument("--filter", nargs="*", default=list(VEHICLE_CLASSES.union(PERSON_CLASS_NAMES)))
    p.add_argument("--debounce", type=float, default=2.0, help="seconds between saved alerts")
    p.add_argument("--anomaly_window_sec", type=float, default=60.0, help="time window (sec) to compute anomaly percent")
    p.add_argument("--debug", action="store_true", help="print debug info")
    args = p.parse_args()

    src = args.source
    device = args.device
    conf_thresh = args.conf
    class_filter = set(args.filter)

    try:
        backend, model = load_model(device)
    except Exception as e:
        print("Model load error:", e)
        return

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("❌ Gagal membuka sumber video:", src)
        print("  - Pastikan URL valid, OpenCV/FFmpeg mendukung m3u8/rtsp.")
        return

    if args.save_alerts:
        alerts_dir = os.path.join("results", "alerts")
        make_folder(alerts_dir)

    anomaly_tracker = TimeWindowAnomaly(window_seconds=args.anomaly_window_sec)
    last_saved = 0
    fps_deque = deque(maxlen=30)
    frame_idx = 0
    last_debug_time = 0

    print("[INFO] Starting. press 'q' to quit.")
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_idx += 1
        vis, dets = detect_and_draw((backend, model), frame, conf_thresh=conf_thresh, class_filter=class_filter)

        n_person = sum(1 for d in dets if d[5] in PERSON_CLASS_NAMES)
        n_vehicle = sum(1 for d in dets if d[5] in VEHICLE_CLASSES)
        # rule: vehicle present and no person => anomaly
        anomaly_flag = (n_vehicle > 0 and n_person == 0)
        anomaly_tracker.add(anomaly_flag)
        anomaly_pct, sample_count = anomaly_tracker.get_stats()

        # draw stats panel (semi-transparent)
        panel_w, panel_h = 420, 100
        overlay = vis.copy()
        cv2.rectangle(overlay, (5,5), (5+panel_w,5+panel_h), (255,255,255), -1)
        alpha = 0.25
        vis = cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0)

        txt_x = 15
        cv2.putText(vis, f"Persons: {n_person}", (txt_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Vehicles: {n_vehicle}", (txt_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        state_text = "ANOMALY" if anomaly_flag else "normal"
        cv2.putText(vis, f"State: {state_text}", (txt_x+240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Anom% (last {int(args.anomaly_window_sec)}s, n={sample_count}): {anomaly_pct:.1f}%", (txt_x+240, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        # save alert (debounced)
        if args.save_alerts and anomaly_flag:
            now = time.time()
            if now - last_saved > args.debounce:
                fname = f"alert_{int(now)}_f{frame_idx}_p{n_person}_v{n_vehicle}.jpg"
                cv2.imwrite(os.path.join(alerts_dir, fname), vis)
                last_saved = now
                print("[ALERT] saved:", fname)

        # FPS
        t1 = time.time()
        fps_deque.append(1.0 / (t1 - t0 + 1e-9))
        fps = sum(fps_deque) / len(fps_deque)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow("Realtime Anomaly", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # debug printing every 2 seconds
        if args.debug and time.time() - last_debug_time > 2.0:
            print(f"[DEBUG] frame={frame_idx}, persons={n_person}, vehicles={n_vehicle}, anomaly_flag={anomaly_flag}, anom%={anomaly_pct:.1f}, samples={sample_count}")
            last_debug_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
