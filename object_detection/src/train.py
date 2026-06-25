"""
用 YOLOv8n 训练文字检测模型
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.parent
YAML_PATH = BASE_DIR / "dataset" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs"


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(YAML_PATH),
        epochs=25,
        imgsz=640,
        batch=16,
        project=str(RUNS_DIR),
        name="text_det",
        exist_ok=True,
        patience=15,       # 15轮无提升则早停
        optimizer="AdamW",
        lr0=1e-3,
        verbose=True,
    )

    best = RUNS_DIR / "text_det" / "weights" / "best.pt"
    print(f"\nTraining done. Best weights: {best}")


if __name__ == "__main__":
    main()
