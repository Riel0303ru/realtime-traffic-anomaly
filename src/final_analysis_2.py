import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
RESULTS_DIR = "./results"
HEATMAP_DIR = os.path.join(RESULTS_DIR, "heatmap_overlay")
TOP_N = 250  # jumlah top anomalies yang mau di-visualize

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --------------------------
# Fungsi overlay heatmap
# --------------------------
def overlay_heatmap(frame, score, colormap=cv2.COLORMAP_JET):
    """
    frame: np.uint8 HxWx3
    score: float 0..1 (anomaly score)
    """
    heatmap_val = np.uint8(score * 255)
    heatmap = cv2.applyColorMap(np.full((frame.shape[0], frame.shape[1]), heatmap_val, dtype=np.uint8), colormap)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay

# --------------------------
# Main visualisasi
# --------------------------
def main():
    # Loop tiap folder TestXXX
    for test_folder in sorted(os.listdir(RESULTS_DIR)):
        test_path = os.path.join(RESULTS_DIR, test_folder)
        if not os.path.isdir(test_path):
            continue

        csv_path = os.path.join(test_path, "anomaly_scores.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV not found: {csv_path}")
            continue

        # Load CSV
        df = pd.read_csv(csv_path)
        if 'frame' not in df.columns or 'score' not in df.columns:
            print(f"[WARNING] CSV missing required columns: {csv_path}")
            continue

        # Normalisasi score 0-1
        df['score_norm'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min() + 1e-8)

        # Pilih top N anomalies
        df_top = df.sort_values('score_norm', ascending=False).head(TOP_N)

        # Folder hasil overlay
        out_dir = os.path.join(HEATMAP_DIR, test_folder)
        make_folder(out_dir)

        print(f"[INFO] Processing {test_folder}, top {TOP_N} anomalies...")
        for _, row in tqdm(df_top.iterrows(), total=len(df_top)):
            frame_num = int(row['frame'])
            score = float(row['score_norm'])
            frame_file = os.path.join(test_path, f"{frame_num:05d}.jpg")
            if not os.path.exists(frame_file):
                continue

            frame = cv2.imread(frame_file)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = overlay_heatmap(frame_rgb, score)
            save_path = os.path.join(out_dir, f"{frame_num:05d}_{score:.4f}.jpg")
            cv2.imwrite(save_path, overlay)

        print(f"[INFO] Heatmap overlay saved in {out_dir}")

if __name__ == "__main__":
    main()
