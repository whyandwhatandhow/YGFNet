import os
from pathlib import Path

# 替换成你yaml里train/val的实际图片文件夹
train_dir = Path("/home/dell/ymd/YGFNet/ultralytics/datasets/exdark/images/train")
val_dir   = Path("/home/dell/ymd/YGFNet/ultralytics/datasets/exdark/images/val")

train_imgs = {p.name for p in train_dir.rglob("*.jpg")} | {p.name for p in train_dir.rglob("*.png")}
val_imgs   = {p.name for p in val_dir.rglob("*.jpg")}   | {p.name for p in val_dir.rglob("*.png")}

overlap = train_imgs & val_imgs
print(f"Train图片数: {len(train_imgs)}")
print(f"Val图片数:   {len(val_imgs)}")
print(f"重叠图片数:   {len(overlap)}")
if overlap:
    print("泄露图片示例:", list(overlap)[:10])
else:
    print("✅ 无明显文件名重叠（但仍建议检查路径是否完全独立）")