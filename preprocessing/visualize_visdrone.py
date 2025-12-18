import cv2
import pandas as pd

# Paths
img_path = "data/raw/visdrone/images/train/0000002_00448_d_0000015.jpg"
ann_path = "data/raw/visdrone/annotations/train/0000002_00448_d_0000015.txt"

# Column names for VisDrone annotations
cols = ["x", "y", "w", "h", "score", "category", "trunc", "occ"]

# Read annotations
df = pd.read_csv(ann_path, header=None, names=cols)

# Read image
img = cv2.imread(img_path)

# Draw ALL bounding boxes (no filtering)
for _, row in df.iterrows():
    if row["category"] in [1,4,5,6,9]:
        x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show image
cv2.imshow("VisDrone Sample", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
