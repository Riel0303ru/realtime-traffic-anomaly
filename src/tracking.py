# src/tracking.py

import os
import cv2
from detection import detect_frames, FRAMES_DIR, OUTPUT_DIR
from collections import defaultdict

def track_objects(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
    
    trackers = cv2.MultiTracker_create()  # buat multi object
    object_ids = {}  # map tracker idx ke ID
    next_id = 1
    obj_positions = defaultdict(list)  # simpan posisi tiap object per frame

    for idx, f in enumerate(frame_files):
        frame_path = os.path.join(input_folder, f)
        frame = cv2.imread(frame_path)

        if idx == 0:
            # Inisialisasi tracker dari deteksi awal
            model = None  # kalau mau bisa pake detect_frames() buat deteksi pertama
            # misal manual, ambil semua box dari folder visual
            print("[INFO] Init tracking with first frame detections")
            visual_path = os.path.join(OUTPUT_DIR, os.path.basename(input_folder))
            if os.path.exists(visual_path):
                boxes = []
                # disini asumsi YOLO sudah generate overlay, kita bisa parse box dari file lain
                # sementara dummy: ambil seluruh frame sebagai satu object
                h, w, _ = frame.shape
                boxes.append((0,0,w,h))  # full frame dummy
                for b in boxes:
                    tracker = cv2.TrackerCSRT_create()
                    trackers.add(tracker, frame, b)
                    object_ids[len(object_ids)] = next_id
                    next_id += 1

        else:
            success, boxes = trackers.update(frame)
            for i, box in enumerate(boxes):
                x, y, w, h = map(int, box)
                obj_id = object_ids[i]
                obj_positions[obj_id].append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame, f"ID {obj_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        out_path = os.path.join(output_folder, f)
        cv2.imwrite(out_path, frame)
    
    print(f"[INFO] Tracking saved to {output_folder}")
    return obj_positions

def main():
    video_folders = [f for f in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, f))]
    for vid in video_folders:
        input_folder = os.path.join(FRAMES_DIR, vid)
        output_folder = os.path.join(OUTPUT_DIR, "tracking", vid)
        track_objects(input_folder, output_folder)

if __name__ == "__main__":
    main()
