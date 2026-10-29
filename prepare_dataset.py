"""
PCB Dataset Preparation Script
Converts PKU PCB Defect Dataset (XML) to YOLO format,
creates train/val split (80/20), and generates data.yaml
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ============ CONFIGURATION ============
PKU_ROOT = "PCB_DATASET/PCB_DATASET"
OUTPUT_DIR = "pcb_project/datasets/pcb_merged"
SEED = 42
TRAIN_RATIO = 0.8

# Class mapping (folder name -> class id)
CLASS_MAP = {
    "Missing_hole": 0,
    "Mouse_bite": 1,
    "Open_circuit": 2,
    "Short": 3,
    "Spur": 4,
    "Spurious_copper": 5,
}

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def convert_voc_to_yolo(xml_path, class_id):
    """Convert Pascal VOC XML annotation to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Clamp values to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def prepare_dataset():
    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    all_samples = []  # list of (image_path, yolo_lines, class_name)

    print("Converting PKU PCB Defect Dataset...")
    img_dir = os.path.join(PKU_ROOT, "images")
    ann_dir = os.path.join(PKU_ROOT, "Annotations")

    for class_folder, class_id in CLASS_MAP.items():
        img_folder = os.path.join(img_dir, class_folder)
        ann_folder = os.path.join(ann_dir, class_folder)

        if not os.path.exists(img_folder):
            print(f"  WARNING: {img_folder} not found, skipping")
            continue
        if not os.path.exists(ann_folder):
            print(f"  WARNING: {ann_folder} not found, skipping")
            continue

        img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        converted = 0
        skipped = 0

        for img_file in img_files:
            stem = os.path.splitext(img_file)[0]
            xml_file = os.path.join(ann_folder, stem + ".xml")

            if not os.path.exists(xml_file):
                skipped += 1
                continue

            try:
                yolo_lines = convert_voc_to_yolo(xml_file, class_id)
                if yolo_lines:
                    img_path = os.path.join(img_folder, img_file)
                    all_samples.append((img_path, yolo_lines, class_folder))
                    converted += 1
            except Exception as e:
                print(f"  ERROR processing {xml_file}: {e}")
                skipped += 1

        print(f"  {class_folder}: {converted} converted, {skipped} skipped")

    print(f"\nTotal samples: {len(all_samples)}")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Copy files
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        for i, (img_path, yolo_lines, class_name) in enumerate(samples):
            # Use unique filename to avoid collisions
            ext = os.path.splitext(img_path)[1]
            stem = os.path.splitext(os.path.basename(img_path))[0]
            new_name = f"{class_name}_{stem}"

            # Copy image
            dst_img = f"{OUTPUT_DIR}/images/{split_name}/{new_name}{ext}"
            shutil.copy2(img_path, dst_img)

            # Write YOLO label
            dst_label = f"{OUTPUT_DIR}/labels/{split_name}/{new_name}.txt"
            with open(dst_label, "w") as f:
                f.write("\n".join(yolo_lines))

    # Count per class
    print("\nPer-class distribution:")
    for class_name, class_id in CLASS_MAP.items():
        count = sum(1 for _, _, c in all_samples if c == class_name)
        print(f"  {class_id}: {class_name} = {count}")

    # Create data.yaml
    os.makedirs("pcb_project", exist_ok=True)
    yaml_path = "pcb_project/data.yaml"
    abs_path = os.path.abspath(OUTPUT_DIR)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\ndata.yaml saved to {yaml_path}")
    print(f"Dataset ready at {OUTPUT_DIR}/")
    print("\nNext step: run training with:")
    print(f"  yolo task=detect mode=train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640 batch=16 device=0")


if __name__ == "__main__":
    prepare_dataset()
