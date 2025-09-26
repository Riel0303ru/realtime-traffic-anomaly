"""
Realtime detection + SORT tracker
- Uses ultralytics YOLOv8 if installed, else torch.hub yolov5 fallback
- SORT implemented with a simple Kalman (state: [x,y,s,r, vx,vy,vs])
- Hungarian assignment via scipy.optimize.linear_sum_assignment
- Blue boxes, white text, FPS, Anom% window, unique counts
"""

import os
import time
import argparse
from collections import deque, defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Try ultralytics first, else torch.hub
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

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "motorbike", "scooter", "van"}
PERSON_CLASSES = {"person"}

# ---------------------
# Utilities
# ---------------------
def iou(bb_test, bb_gt):
    # bb: [x1,y1,x2,y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1])
    o = inter / (area1 + area2 - inter + 1e-6)
    return o

# ---------------------
# Minimal Kalman for SORT (adapted)
# State: [cx, cy, s, r, vx, vy, vs]
# where s = scale = area, r = aspect ratio
# ---------------------
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = w * h
        r = w / float(h + 1e-6)
        # state vector
        self.x = np.array([cx, cy, s, r, 0., 0., 0.]).reshape((7,1))
        # covariance
        self.P = np.eye(7) * 10.
        # process & measurement matrices
        dt = 1.0
        # F: state transition
        self.F = np.eye(7)
        self.F[0,4] = dt
        self.F[1,5] = dt
        self.F[2,6] = dt
        # Q: process noise
        self.Q = np.eye(7) * 0.01
        # measurement matrix H
        self.H = np.zeros((4,7))
        self.H[0,0] = 1.0
        self.H[1,1] = 1.0
        self.H[2,2] = 1.0
        self.H[3,3] = 1.0
        # R: measurement noise
        self.R = np.eye(4) * 1.0
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history = []
        self.last_bbox = bbox

    def predict(self):
        # x' = F x
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        # return predicted bbox
        return self._get_bbox()

    def update(self, bbox):
        # measurement z from bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = w*h
        r = w/(h+1e-6)
        z = np.array([cx, cy, s, r]).reshape((4,1))
        # Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(7)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_bbox = bbox
        self.history = []

    def _get_bbox(self):
        cx, cy, s, r = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        w = np.sqrt(max(s,1e-6) * max(r,1e-6))
        h = max(s,1e-6) / (w + 1e-6)
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        return [x1, y1, x2, y2]

# ---------------------
# SORT manager
# ---------------------
class Sort:
    def __init__(self, max_age=8, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        detections: Nx5 array-like [[x1,y1,x2,y2,score,label], ...]
        returns: list of dicts {'id','bbox','score','label'}
        """
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # remove nan trackers
        for idx in reversed(to_del):
            self.trackers.pop(idx)
        dets = np.array([d[:4] for d in detections]) if len(detections)>0 else np.empty((0,4))
        scores = [d[4] for d in detections] if len(detections)>0 else []
        labels = [d[5] for d in detections] if len(detections)>0 else []
        N = len(trks)
        M = len(dets)
        matches, unmatched_trk, unmatched_det = [], list(range(N)), list(range(M))
        if N>0 and M>0:
            # compute IoU cost matrix (we solve assignment to maximize IoU -> minimize 1-ioU)
            iou_mat = np.zeros((N,M))
            for i in range(N):
                for j in range(M):
                    iou_mat[i,j] = iou(trks[i], dets[j])
            cost = 1 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            matches = []
            for r,c in zip(row_ind, col_ind):
                if iou_mat[r,c] < self.iou_threshold:
                    continue
                matches.append((r,c))
            matched_trks = [m[0] for m in matches]
            matched_dets = [m[1] for m in matches]
            unmatched_trk = [i for i in range(N) if i not in matched_trks]
            unmatched_det = [j for j in range(M) if j not in matched_dets]
        # update matched trackers
        for trk_idx, det_idx in matches:
            bbox = detections[det_idx][:4]
            self.trackers[trk_idx].update(bbox)
        # create trackers for unmatched detections
        for idx in unmatched_det:
            bbox = detections[idx][:4]
            new_trk = KalmanBoxTracker(bbox)
            self.trackers.append(new_trk)
        # remove old trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
        # prepare returned active tracks
        output = []
        for trk in self.trackers:
            if (trk.hit_streak >= self.min_hits) or (trk.time_since_update <= 1):
                bbox = trk._get_bbox()
                output.append({'id': trk.id,
                               'bbox': [int(b) for b in bbox],
                               'score': float(0.0)})
        return output

# ---------------------
# Detection helpers
# ---------------------
def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return np.array([])

def parse_ultralytics(res, model, conf_thresh):
    dets = []
    if res is None:
        return dets
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return dets
    xyxy = _to_numpy(boxes.xyxy)
    confs = _to_numpy(boxes.conf)
    cls = _to_numpy(boxes.cls)
    names = getattr(model, "names", None)
    if names is None:
        names = getattr(getattr(model, "model", None), "names", None)
    for i in range(len(xyxy)):
        x1,y1,x2,y2 = [int(v) for v in xyxy[i][:4]]
        conf = float(confs[i]) if i < len(confs) else 0.0
        if conf < conf_thresh:
            continue
        cls_id = int(cls[i]) if i < len(cls) else -1
        label = str(names.get(cls_id, cls_id)).lower() if isinstance(names, dict) else str(cls_id)
        dets.append((x1,y1,x2,y2,conf,label))
    return dets

def parse_yolov5(out, conf_thresh):
    dets = []
    try:
        preds = out.xyxy[0]
        arr = _to_numpy(preds)
        names = getattr(out, "names", None)
        if arr is None or arr.size == 0:
            return dets
        for row in arr:
            x1,y1,x2,y2,conf,cls = row[:6]
            if float(conf) < conf_thresh:
                continue
            label = str(names[int(cls)]).lower() if names is not None else str(int(cls))
            dets.append((int(x1),int(y1),int(x2),int(y2),float(conf),label))
    except Exception:
        pass
    return dets

def load_model(device, ultra_name="yolov8n.pt", y5_name="yolov5s"):
    if USE_ULTRALYTICS:
        print("[INFO] Using ultralytics YOLOv8:", ultra_name)
        model = UltralyticsYOLO(ultra_name)
        return "ultralytics", model
    if USE_TORCHHUB:
        import torch
        print("[INFO] Using torch.hub yolov5:", y5_name)
        model = torch.hub.load('ultralytics/yolov5', y5_name, pretrained=True)
        model.to(device)
        model.eval()
        return "yolov5_hub", model
    raise RuntimeError("No YOLO backend available. Install ultralytics or allow torch.hub download.")

def detect_frame(backend_model, frame, conf_thresh=0.25, imgsz=640):
    backend, model = backend_model
    dets = []
    try:
        if backend == "ultralytics":
            try:
                res_list = model.predict(source=frame, conf=conf_thresh, imgsz=imgsz, verbose=False)
                if res_list:
                    dets = parse_ultralytics(res_list[0], model, conf_thresh)
            except TypeError:
                res = model(frame)
                if res:
                    dets = parse_ultralytics(res[0], model, conf_thresh)
        else:
            try:
                model.conf = conf_thresh
                out = model(frame, size=imgsz)
            except Exception:
                out = model(frame)
            dets = parse_yolov5(out, conf_thresh)
    except Exception as e:
        print("[ERROR] detection failed:", e)
        dets = []
    return dets

# ---------------------
# Drawing
# ---------------------
def draw_boxes(frame, tracks, class_names, color=(255, 255, 255)):
    """
    Gambar bounding box + label jenis kendaraan + ID tracking.
    """
    for track in tracks:
        track_id = track.track_id  # ID unik dari tracker
        l, t, r, b = track.to_ltrb()  # koordinat box (left, top, right, bottom)

        cls_id = int(track.cls) if hasattr(track, "cls") else -1
        cls_name = class_names[cls_id] if cls_id in range(len(class_names)) else "unknown"

        # Teks: nama + nomor
        label = f"{cls_name} #{track_id}"

        # Gambar kotak
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)

        # Teks di atas kotak
        cv2.putText(frame, label, (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# ---------------------
# Anomaly window
# ---------------------
class TimeWindowAnomaly:
    def __init__(self, window_seconds=60.0):
        self.window_seconds = float(window_seconds)
        self.storage = deque()
    def add(self, flag):
        now = time.time()
        self.storage.append((now, 1 if flag else 0))
        cutoff = now - self.window_seconds
        while self.storage and self.storage[0][0] < cutoff:
            self.storage.popleft()
    def stats(self):
        s = sum(v for (_,v) in self.storage)
        n = len(self.storage)
        pct = 100.0 * s / n if n>0 else 0.0
        return pct, n

# ---------------------
# Main loop
# ---------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save_alerts", action="store_true")
    p.add_argument("--anomaly_window_sec", type=float, default=60.0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    try:
        backend_model = load_model(args.device)
    except Exception as e:
        print("[ERROR] Model load:", e)
        return

    # open source
    try:
        src_int = int(args.source)
        cap = cv2.VideoCapture(src_int)
    except Exception:
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("❌ Gagal membuka source:", args.source)
        return

    if args.save_alerts:
        os.makedirs("results/alerts", exist_ok=True)

    sort = Sort(max_age=8, min_hits=2, iou_threshold=0.3)
    tw = TimeWindowAnomaly(args.anomaly_window_sec)
    unique_ids_per_class = defaultdict(set)
    fps_q = deque(maxlen=30)
    frame_idx = 0

    print("[INFO] Starting SORT tracker realtime. press q to quit.")
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue
        frame_idx += 1
        dets = detect_frame(backend_model, frame, conf_thresh=args.conf, imgsz=args.imgsz)
        # keep classes of interest
        dets = [(x1,y1,x2,y2,conf,label.lower()) for (x1,y1,x2,y2,conf,label) in dets if label.lower() in (VEHICLE_CLASSES.union(PERSON_CLASSES))]
        # update sort: convert to required form
        dets_for_sort = []
        for d in dets:
            x1,y1,x2,y2,conf,label = d
            dets_for_sort.append([float(x1), float(y1), float(x2), float(y2), float(conf), label])
        tracked = sort.update(dets_for_sort)
        # update counts by checking IoU between tracked bbox and last detections' labels
        persons = 0
        vehicles = 0
        # try attach labels: match tracked bbox to dets by IoU to read label
        for t in tracked:
            tb = t['bbox']
            best_label = None
            best_iou = 0.0
            for d in dets:
                i = iou(tb, d[:4])
                if i > best_iou:
                    best_iou = i
                    best_label = d[5]
            if best_label is None:
                continue
            t['label'] = best_label
            if best_label in PERSON_CLASSES:
                persons += 1
                unique_ids_per_class['person'].add(t['id'])
            if best_label in VEHICLE_CLASSES:
                vehicles += 1
                unique_ids_per_class[best_label].add(t['id'])

        anomaly_flag = (vehicles > 0 and persons == 0)
        tw.add(anomaly_flag)
        anom_pct, sample_count = tw.stats()
        # Draw tracked boxes and labels
        vis = frame.copy()
        for t in tracked:
            bbox = t['bbox']
            track_id = t['id']
            label = t.get('label', 'rider')
            color = (255, 0, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{label} #{track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # black stats panel (white text)
        cv2.rectangle(vis, (6,6), (430,120), (0,0,0), -1)
        cv2.putText(vis, f"FPS: --", (12,26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Persons (cur): {persons}", (12,52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Vehicles (cur): {vehicles}", (12,82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Unique persons: {len(unique_ids_per_class['person'])}", (240,52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Unique vehicles: {sum(len(unique_ids_per_class[c]) for c in VEHICLE_CLASSES)}", (240,82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Anom% {int(args.anomaly_window_sec)}s (n={sample_count}): {anom_pct:.1f}%", (12,108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        if args.save_alerts and anomaly_flag:
            fname = f"results/alerts/alert_{int(time.time())}_f{frame_idx}_p{persons}_v{vehicles}.jpg"
            cv2.imwrite(fname, vis)

        t1 = time.time()
        fps_q.append(1.0 / max(1e-6, (t1 - t0)))
        fps = sum(fps_q)/len(fps_q) if fps_q else 0.0
        cv2.putText(vis, f"FPS: {fps:.1f}", (12,26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Realtime SORT Tracker", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if args.debug and frame_idx % 200 == 0:
            print(f"[DEBUG] frame {frame_idx} persons={persons} vehicles={vehicles} unique_p={len(unique_ids_per_class['person'])}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
