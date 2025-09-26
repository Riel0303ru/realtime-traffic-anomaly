import os

frames_dir = "data/frames/UCSDped2"
for root, dirs, files in os.walk(frames_dir):
    jpg_files = [f for f in files if f.endswith(".jpg")]
    if jpg_files:
        print(root, len(jpg_files), "frames")
