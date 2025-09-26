# generate_overlay_summary.py
import os, csv, glob
OUT = "results/heatmap_overlay_summary.csv"
rows = []
for folder in sorted(glob.glob("results/heatmap_overlay/*")):
    for f in glob.glob(os.path.join(folder, "*.jpg")):
        name = os.path.basename(f)
        parts = name.split("_")
        frame = parts[0]
        score = parts[1].split(".")[0] if len(parts) > 1 else ""
        rows.append([os.path.basename(folder), frame, score, f])

with open(OUT, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["test_folder","frame","score","path"])
    writer.writerows(rows)

print("Saved summary to", OUT)
