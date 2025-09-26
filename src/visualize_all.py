import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Config
# --------------------------
RESULTS_DIR = os.path.join(os.getcwd(), "results")
HEATMAP_DIR = os.path.join(RESULTS_DIR, "heatmap_overlay")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------------
# Gabungkan semua CSV
# --------------------------
all_data = []

for video_folder in sorted(os.listdir(RESULTS_DIR)):
    video_path = os.path.join(RESULTS_DIR, video_folder)
    csv_file = os.path.join(video_path, "anomaly_scores.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df["video"] = video_folder  # tandai nama video
        all_data.append(df)

if not all_data:
    raise FileNotFoundError("No CSV found in any video folder!")

all_df = pd.concat(all_data, ignore_index=True)
all_df.to_csv(os.path.join(PLOT_DIR, "anomaly_scores_global.csv"), index=False)
print(f"[INFO] Global CSV saved: {os.path.join(PLOT_DIR, 'anomaly_scores_global.csv')}")

# --------------------------
# Plot global anomaly score
# --------------------------
plt.figure(figsize=(18,6))
for video_name in sorted(all_df["video"].unique()):
    df_vid = all_df[all_df["video"] == video_name]
    plt.plot(df_vid["frame"], df_vid["score"], label=video_name)

plt.xlabel("Frame")
plt.ylabel("Anomaly Score")
plt.title("Anomaly Score per Frame (All Videos)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "anomaly_scores_global.png"))
plt.show()
print(f"[INFO] Global plot saved: {os.path.join(PLOT_DIR, 'anomaly_scores_global.png')}")
