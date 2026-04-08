# test_yola.py
from ultralytics import YOLO

# ===== 1. 加载训练好的权重 =====
model = YOLO("/home/dell/ymd/YGFNet/runs/detect/yola_exdark_xxx/weights/best.pt")

# ===== 2. 在测试集上评估 =====
metrics = model.val(
    data="/home/dell/ymd/YGFNet/ultralytics/datasets/exdark.yaml",
    split="test",     # 如果没有test，就改成 val
    imgsz=640,
    batch=2,
    device=0,
    save_json=True,   # COCO格式评估（可选）
    project="/home/dell/ymd/YGFNet/runs",
    name="yola_exdark_test"
)

print(metrics)