import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------
# Loop tiap folder Test
# ------------------------------
for folder in sorted(os.listdir(RESULTS_DIR)):
    folder_path = os.path.join(RESULTS_DIR, folder)
    if not os.path.isdir(folder_path) or not folder.lower().startswith("test"):
        continue

    csv_path = os.path.join(folder_path, "anomaly_scores.csv")
    if not os.path.isfile(csv_path):
        print(f"[WARNING] No CSV found in {folder}")
        continue

    df = pd.read_csv(csv_path)
    if 'score' not in df.columns:
        print(f"[WARNING] 'score' column not found in {csv_path}")
        continue

    plt.figure(figsize=(12, 4))
    plt.plot(df['frame'], df['score'], label='Anomaly Score', color='red')
    plt.title(f"Anomaly Scores - {folder}")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(PLOTS_DIR, f"{folder}_score_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved plot: {save_path}")

print("[INFO] All plots generated!")
