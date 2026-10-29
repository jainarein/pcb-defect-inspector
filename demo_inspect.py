"""
PCB Inspection Demo — YOLO Defect Detection + Anomaly Module
Runs on CPU. Takes an image, detects known defects with YOLO,
checks for unknown anomalies, and displays pass/fail verdict.
"""

import cv2
import numpy as np
import pickle
import os
import time
from ultralytics import YOLO

# ============ CONFIG ============
YOLO_MODEL = "runs/detect/pcb_project/runs/yolo11n/weights/best.onnx"
ANOMALY_MODEL = "anomaly_model.pkl"
CONFIDENCE_THRESHOLD = 0.4
IMG_SIZE_ANOMALY = 224

CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper"
]

COLORS = [
    (0, 0, 255),    # red
    (0, 165, 255),  # orange
    (0, 255, 255),  # yellow
    (255, 0, 0),    # blue
    (255, 0, 255),  # magenta
    (0, 255, 0),    # green
]


def extract_features(img):
    """Same feature extraction as training"""
    img = cv2.resize(img, (IMG_SIZE_ANOMALY, IMG_SIZE_ANOMALY))
    hist_features = []
    for c in range(3):
        hist = cv2.calcHist([img], [c], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist_features.extend(hist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    grid = 8
    cell_h, cell_w = IMG_SIZE_ANOMALY // grid, IMG_SIZE_ANOMALY // grid
    edge_features = []
    for i in range(grid):
        for j in range(grid):
            cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            edge_features.append(cell.mean() / 255.0)

    texture = cv2.Laplacian(gray, cv2.CV_64F)
    for i in range(grid):
        for j in range(grid):
            cell = texture[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            edge_features.append(cell.var())

    return np.array(hist_features + edge_features, dtype=np.float32)


def inspect_board(image_path):
    """Run full inspection on a single board image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        return

    display = img.copy()
    h, w = img.shape[:2]
    defects_found = []

    # --- YOLO Detection ---
    print("\n" + "="*50)
    print(f"Inspecting: {os.path.basename(image_path)}")
    print("="*50)

    model = YOLO(YOLO_MODEL)
    t1 = time.time()
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
    yolo_time = (time.time() - t1) * 1000

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id]
            color = COLORS[cls_id]

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            defects_found.append((cls_name, conf))

    print(f"\nYOLO inference: {yolo_time:.1f} ms")
    if defects_found:
        print(f"Defects detected: {len(defects_found)}")
        for name, conf in defects_found:
            print(f"  - {name} ({conf:.2f})")
    else:
        print("No known defects detected")

    # --- Anomaly Detection ---
    with open(ANOMALY_MODEL, "rb") as f:
        anom = pickle.load(f)

    t2 = time.time()
    feat = extract_features(img)
    feat_norm = (feat - anom["mean"]) / anom["std"]
    distances, _ = anom["nn_model"].kneighbors(feat_norm.reshape(1, -1))
    anomaly_score = distances.mean()
    anomaly_time = (time.time() - t2) * 1000

    is_anomaly = anomaly_score > anom["threshold"]

    print(f"\nAnomaly score: {anomaly_score:.2f} (threshold: {anom['threshold']:.2f})")
    print(f"Anomaly inference: {anomaly_time:.1f} ms")

    # --- Verdict ---
    total_time = yolo_time + anomaly_time

    if defects_found or is_anomaly:
        verdict = "FAIL"
        verdict_color = (0, 0, 255)  # red
        reasons = []
        if defects_found:
            reasons.append(f"{len(defects_found)} defect(s)")
        if is_anomaly:
            reasons.append("anomaly detected")
        reason_text = " + ".join(reasons)
    else:
        verdict = "PASS"
        verdict_color = (0, 200, 0)  # green
        reason_text = "no issues"

    # Draw verdict banner
    banner_h = 60
    cv2.rectangle(display, (0, 0), (w, banner_h), verdict_color, -1)
    cv2.putText(display, f"{verdict} - {reason_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Draw timing info
    info = f"YOLO: {yolo_time:.0f}ms | Anomaly: {anomaly_time:.0f}ms | Total: {total_time:.0f}ms"
    cv2.putText(display, info, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    print(f"\n>>> VERDICT: {verdict} ({reason_text})")
    print(f">>> Total time: {total_time:.1f} ms")

    # Save result
    out_path = f"result_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, display)
    print(f"\nResult saved to {out_path}")

    # Show if display available
    try:
        cv2.imshow("PCB Inspection", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # Default: pick a random validation image
        val_dir = "pcb_project/datasets/pcb_merged/images/val"
        if os.path.exists(val_dir):
            files = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))]
            if files:
                import random
                test_img = os.path.join(val_dir, random.choice(files))
                inspect_board(test_img)
            else:
                print("No images found in val dir")
        else:
            print("Usage: python demo_inspect.py <image_path>")
    else:
        inspect_board(sys.argv[1])
