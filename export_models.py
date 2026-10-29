from ultralytics import YOLO
import os

base = "runs/detect/pcb_project/runs"

m8 = YOLO(f"{base}/baseline/weights/best.pt")
m8.export(format="onnx")
print("YOLOv8n ONNX done")

m11 = YOLO(f"{base}/yolo11n/weights/best.pt")
m11.export(format="onnx")
print("YOLO11n ONNX done")

# Print file sizes
for name, path in [
    ("YOLOv8n .pt", f"{base}/baseline/weights/best.pt"),
    ("YOLOv8n .onnx", f"{base}/baseline/weights/best.onnx"),
    ("YOLO11n .pt", f"{base}/yolo11n/weights/best.pt"),
    ("YOLO11n .onnx", f"{base}/yolo11n/weights/best.onnx"),
]:
    size = os.path.getsize(path) / (1024*1024)
    print(f"{name}: {size:.1f} MB")
