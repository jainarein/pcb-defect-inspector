# PCB Defect Inspector — Low-Cost Edge AI Automated Optical Inspection

A complete, low-cost PCB defect inspection system combining quantized YOLO11n object detection with nearest-neighbor anomaly detection, designed to run on a Raspberry Pi.

---

## The Problem

Small and medium electronics manufacturers inspect PCBs manually — slow, inconsistent, and prone to fatigue. Commercial AOI machines cost ₹5,00,000+, out of reach for most MSMEs.

## The Solution

A Raspberry Pi + camera + ring light that inspects each board in under 3 seconds, entirely offline, for under ₹15,000 in hardware.

---

## How It Works

The system runs two independent checks on every board image:

### Check 1 — YOLO Defect Detector (Known Defects)

A YOLO11n model scans the board for 6 trained defect types: missing hole, mouse bite, open circuit, short, spur, and spurious copper. It draws bounding boxes with confidence scores. Any detection above 0.4 confidence triggers a FAIL.

### Check 2 — Anomaly Detector (Unknown/New Boards)

When a factory starts making a new board, YOLO only knows the 6 defects it was trained on — it has no idea what the new board should look like. The anomaly module fills this gap.

It learns what a "good" board looks like from just 10–20 defect-free samples. For each image it extracts a 224-dimensional feature vector (color histograms + edge density + texture variance), builds a memory bank, and uses k-nearest-neighbor distance to flag boards that look different from known-good ones.

### Verdict

```
IF (any YOLO detection > 0.4) OR (anomaly score > threshold):
    → FAIL
ELSE:
    → PASS
```

---

## Results

### Detection Accuracy

| Metric | YOLOv8n | YOLO11n |
|--------|---------|---------|
| mAP@50 | 91.7% | **92.6%** |
| mAP@50-95 | 48.3% | **49.2%** |
| Precision | 94.7% | **95.3%** |
| Recall | 87.8% | 87.7% |
| Parameters | 3.13M | **2.60M** |
| Model size | 6.2 MB | **5.3 MB** |
| GFLOPs | 8.6 | **6.4** |
| CPU inference (ONNX) | **54.3 ms** | 70.4 ms |

### Per-Class Performance (YOLO11n)

| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| missing_hole | 0.985 | 0.974 | 0.981 |
| mouse_bite | 0.959 | 0.869 | 0.933 |
| open_circuit | 0.933 | 0.898 | 0.942 |
| short | 0.993 | 0.909 | 0.970 |
| spur | 0.967 | 0.732 | 0.798 |
| spurious_copper | 0.941 | 0.889 | 0.945 |

### Anomaly Detection

| Metric | Value |
|--------|-------|
| AUROC | 0.918 |
| Best F1 | 0.930 |
| Accuracy | 91.8% |

---

## Hardware

| Component | Specification |
|-----------|--------------|
| Compute | Raspberry Pi 4/5 (4 GB) |
| Camera | RPi Camera Module 3 |
| Storage + PSU | 32 GB microSD + USB-C |
| Lighting | LED ring light |
| Enclosure | 3D-printed / acrylic |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/jainarein/pcb-defect-inspector.git
cd pcb-defect-inspector
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python scikit-learn onnxruntime
```

### 2. Download datasets

- PKU PCB Defect Dataset: https://www.kaggle.com/datasets/akhatova/pcb-defects
- DeepPCB: `git clone https://github.com/tangsanli5201/DeepPCB.git`

### 3. Prepare dataset

```bash
python prepare_dataset.py
```

### 4. Train detectors

```bash
yolo task=detect mode=train model=yolov8n.pt data=pcb_project/data.yaml epochs=100 imgsz=640 batch=16 device=0 project=runs name=baseline
yolo task=detect mode=train model=yolo11n.pt data=pcb_project/data.yaml epochs=100 imgsz=640 batch=16 device=0 project=runs name=yolo11n
```

### 5. Export to ONNX

```bash
python export_models.py
```

### 6. Train anomaly module

```bash
python setup_anomaly_data.py
python fix_anomaly_data.py
python train_anomaly.py
```

### 7. Run inspection

```bash
python demo_inspect.py                    # random validation image
python demo_inspect.py path/to/board.jpg  # specific image
```

---

## Project Structure

```
pcb-defect-inspector/
├── prepare_dataset.py          # PKU XML → YOLO format + train/val split
├── setup_anomaly_data.py       # Prepare anomaly training data
├── fix_anomaly_data.py         # Add DeepPCB templates to good set
├── train_anomaly.py            # Train nearest-neighbor anomaly detector
├── export_models.py            # Export YOLO models to ONNX
├── demo_inspect.py             # Full inspection pipeline → PASS/FAIL
├── anomaly_model.pkl           # Trained anomaly model
├── pcb_project/
│   ├── data.yaml               # YOLO dataset config
│   └── datasets/pcb_merged/    # Merged dataset in YOLO format
├── runs/                       # Training outputs
├── inhouse_boards/             # Anomaly detection data (good + defective)
└── docs/                       # Result images
```

---

## Training Pipeline

```
PKU Dataset (XML) → prepare_dataset.py → YOLO format (693 images, 6 classes)
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                               YOLOv8n              YOLO11n
                                    │                   │
                                    ▼                   ▼
                               ONNX export         ONNX export
                                    └─────────┬─────────┘
                                              ▼
                                    CPU / Pi benchmark

PKU + DeepPCB → setup scripts → good/defective split → train_anomaly.py → anomaly_model.pkl
```

---

## Tech Stack

- Ultralytics YOLO (YOLOv8n, YOLO11n)
- ONNX Runtime
- scikit-learn (NearestNeighbors)
- OpenCV
- Training: NVIDIA T550, Intel i7-13620H, 32 GB RAM
- Deployment: Raspberry Pi 4/5

---

## Future Work

- Raspberry Pi 5 benchmarks
- INT8 TFLite quantization
- Deep feature extraction (PatchCore with ResNet backbone)
- Spatial anomaly localization with heatmaps
- Real factory board dataset
- Web dashboard for inspection history

---

## Datasets

- [PKU PCB Defect Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) — 693 images, 6 classes
- [DeepPCB](https://github.com/tangsanli5201/DeepPCB) — 1500+ template images

## License

MIT

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PKU Robotics Lab](https://robotics.pkusz.edu.cn/resources/dataset/)
- [DeepPCB](https://github.com/tangsanli5201/DeepPCB) by Tang et al.
- [PatchCore](https://arxiv.org/abs/2106.08265) (Roth et al., CVPR 2022) for anomaly detection inspiration
