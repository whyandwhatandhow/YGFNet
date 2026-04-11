# train_yolov8l.py
from ultralytics import YOLO
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ===== 1. 加载 YOLOv8l =====
model = YOLO("yolov8l.yaml")   # ⚠️ 用官方结构

print(model.info())

# ===== 2. 训练 =====
model.train(
    data="/home/dell/ymd/YGFNet/ultralytics/datasets/exdark.yaml",
    epochs=220,
    imgsz=640,
    batch=2,
    multi_scale=0.2,
    close_mosaic=10,
    optimizer="AdamW",
    lr0=0.001,
    cache=True,
    workers=8,
    device=0,



    project="/home/dell/ymd/YGFNet/runs",
    name=f"yolov8l_exdark_{timestamp}",
)