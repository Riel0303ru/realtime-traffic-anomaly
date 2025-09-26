import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Config
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------
# Loop semua folder Test
# --------------------------
video_folders = sorted([f for f in os.listdir(RESULTS_DIR) if f.lower().startswith("test")])

for vf in video_folders:
    folder_path = os.path.join(RESULTS_DIR, vf)
    csv_path = os.path.join(folder_path, "anomaly_scores.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV not found in {vf}")
        continue

    # Load CSV
    df = pd.read_csv(csv_path)
    if 'score' not in df.columns:
        print(f"[WARNING] 'score' column not found in {vf}")
        continue

    scores = df['score'].values
    frames = df['frame'].values

    # --------------------------
    # Statistik dasar & threshold
    # --------------------------
    mean_score = scores.mean()
    std_score = scores.std()
    threshold = mean_score + 3*std_score

    print(f"Video: {vf}")
    print(f"Mean score: {mean_score:.6f}, Std: {std_score:.6f}, Recommended threshold: {threshold:.6f}")

    # Top 5 anomaly frames
    top_idx = scores.argsort()[-5:][::-1]
    print("Top 5 anomaly frames:", frames[top_idx], "with scores:", scores[top_idx])

    # --------------------------
    # Plot histogram
    # --------------------------
    plt.figure(figsize=(6,4))
    sns.histplot(scores, bins=50, kde=True, color='skyblue')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.title(f"Anomaly Score Distribution - {vf}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{vf}_histogram.png"))
    plt.close()

    # --------------------------
    # Plot trend per frame
    # --------------------------
    plt.figure(figsize=(10,4))
    plt.plot(frames, scores, color='blue', label='Anomaly Score')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.scatter(frames[top_idx], scores[top_idx], color='orange', label='Top anomalies', zorder=5)
    plt.title(f"Anomaly Score Trend - {vf}")
    plt.xlabel("Frame")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{vf}_trend.png"))
    plt.close()

    print(f"[INFO] Plots saved for {vf} in {PLOTS_DIR}\n")

print("[INFO] Analysis completed for all videos.")
