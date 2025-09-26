# src/anomaly.py

import os
import numpy as np
from collections import defaultdict
import cv2

TRACKING_DIR = os.path.join("output", "tracking")
OUTPUT_DIR = os.path.join("output", "anomalies")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_speed(positions):
    """
    Hitung kecepatan per frame dari posisi bounding box.
    positions = list of (x, y, w, h)
    """
    speeds = []
    for i in range(1, len(positions)):
        x1, y1, _, _ = positions[i-1]
        x2, y2, _, _ = positions[i]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        speeds.append(dist)
    return speeds

def detect_anomalies(obj_positions, speed_threshold=20, frame_size=(640,480)):
    """
    Rule-based anomaly detection:
    - speed > speed_threshold -> abnormal
    - posisi di luar trotoar (misal x < 0 atau x > frame width) -> abnormal
    """
    anomalies = defaultdict(list)
    for obj_id, pos_list in obj_positions.items():
        speeds = compute_speed(pos_list)
        for i, speed in enumerate(speeds):
            x, y, w, h = pos_list[i+1]
            abnormal = False
            if speed > speed_threshold:
                abnormal = True
            if x < 0 or x + w > frame_size[0] or y < 0 or y + h > frame_size[1]:
                abnormal = True
            anomalies[obj_id].append(abnormal)
    return anomalies

def save_anomaly_log(anomalies, output_file):
    with open(output_file, "w") as f:
        for obj_id, flags in anomalies.items():
            f.write(f"Object {obj_id}: {flags}\n")
    print(f"[INFO] Anomaly log saved to {output_file}")

def main():
    video_folders = [f for f in os.listdir(TRACKING_DIR) if os.path.isdir(os.path.join(TRACKING_DIR, f))]
    for vid in video_folders:
        track_path = os.path.join(TRACKING_DIR, vid)
        # load obj_positions (dummy: from tracking.py bisa save pickle nanti)
        import pickle
        with open(os.path.join(track_path, "obj_positions.pkl"), "rb") as f:
            obj_positions = pickle.load(f)
        
        anomalies = detect_anomalies(obj_positions, speed_threshold=20)
        output_file = os.path.join(OUTPUT_DIR, f"{vid}_anomalies.txt")
        save_anomaly_log(anomalies, output_file)

if __name__ == "__main__":
    main()
