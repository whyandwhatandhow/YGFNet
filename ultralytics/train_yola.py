# train_yola.py
from ultralytics import YOLO
import datetime

from yola_trainer import YOLATrainer

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model = YOLO("/home/dell/ymd/YGFNet/ultralytics/ultralytics/models/yolav8.yaml")
model.load("yolov8l.pt")

print(model.info())

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
    ii=5.0,
    ii_warmup=10,
    trainer=YOLATrainer,
    project="/home/dell/ymd/YGFNet/runs",
    name=f"yola_exdark_{timestamp}",
)