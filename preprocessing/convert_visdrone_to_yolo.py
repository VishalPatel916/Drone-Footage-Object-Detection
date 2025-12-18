import os
import cv2
import pandas as pd

# Mapping VisDrone category IDs to YOLO class IDs
VISDRONE_TO_YOLO = {
    1: 0,  # pedestrian
    4: 1,  # car
    5: 2,  # van
    6: 3,  # truck
    9: 4   # bus
}

COLS = ["x", "y", "w", "h", "score", "category", "trunc", "occ"]

def convert_split(split):
    img_dir = f"data/raw/visdrone/images/{split}"
    ann_dir = f"data/raw/visdrone/annotations/{split}"

    out_img_dir = f"data/processed/yolo/images/{split}"
    out_lbl_dir = f"data/processed/yolo/labels/{split}"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for ann_file in os.listdir(ann_dir):
        ann_path = os.path.join(ann_dir, ann_file)
        img_name = ann_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w, _ = img.shape

        df = pd.read_csv(ann_path, header=None, names=COLS)

        yolo_labels = []

        for _, row in df.iterrows():
            cat = row["category"]
            if cat not in VISDRONE_TO_YOLO:
                continue

            x, y, w, h = row["x"], row["y"], row["w"], row["h"]

            # Convert to YOLO format (normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            yolo_labels.append(
                f"{VISDRONE_TO_YOLO[cat]} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )

        if yolo_labels:
            cv2.imwrite(os.path.join(out_img_dir, img_name), img)
            with open(os.path.join(out_lbl_dir, ann_file), "w") as f:
                f.write("\n".join(yolo_labels))

def main():
    convert_split("train")
    convert_split("val")
    print("✅ VisDrone → YOLO conversion completed")

if __name__ == "__main__":
    main()
