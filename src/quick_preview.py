# quick_preview.py
import cv2, glob, os
p = sorted(glob.glob("results/heatmap_overlay/*/*"))[:10]
for f in p:
    img = cv2.imread(f)
    cv2.imshow(os.path.basename(f), img)
    cv2.waitKey(500)  # 500ms per image
cv2.destroyAllWindows()
