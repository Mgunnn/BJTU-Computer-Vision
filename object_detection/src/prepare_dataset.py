"""
将 LabelMe JSON 标注转换为 YOLO 格式，并划分训练/验证集
"""
import os
import json
import base64
import shutil
import random
from pathlib import Path
from PIL import Image
import io

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"

TRAIN_RATIO = 0.8
SEED = 42

for split in ["train", "val"]:
    (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def extract_image(json_data, json_path):
    """从 json 里找到对应图片，优先读文件，缺失则用内嵌 base64 解码"""
    img_name = json_data.get("imagePath", "")
    img_path = DATA_DIR / img_name
    if img_path.exists():
        return Image.open(img_path).convert("RGB"), img_name

    # 用内嵌 base64
    img_data = json_data.get("imageData")
    if img_data:
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")
        # 用 json 文件名作为图片名
        img_name = json_path.stem + ".jpg"
        return img, img_name

    return None, None


def convert_one(json_path, split):
    with open(json_path) as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        return False

    img, img_name = extract_image(data, json_path)
    if img is None:
        return False

    W, H = img.size

    lines = []
    for shape in shapes:
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape["points"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        # 防止坐标顺序颠倒
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        # 裁剪到 [0,1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not lines:
        return False

    # 保存图片
    img_save_name = Path(img_name).stem + ".jpg"
    img.save(DATASET_DIR / "images" / split / img_save_name, quality=95)

    # 保存标签
    label_path = DATASET_DIR / "labels" / split / (Path(img_name).stem + ".txt")
    label_path.write_text("\n".join(lines))

    return True


def main():
    json_files = sorted(DATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    random.seed(SEED)
    random.shuffle(json_files)
    n_train = int(len(json_files) * TRAIN_RATIO)
    splits = {"train": json_files[:n_train], "val": json_files[n_train:]}

    for split, files in splits.items():
        ok = 0
        for jf in files:
            if convert_one(jf, split):
                ok += 1
        print(f"{split}: {ok}/{len(files)} converted")

    # 写 data.yaml
    yaml_content = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val
nc: 1
names: ['TEXT']
"""
    (DATASET_DIR / "data.yaml").write_text(yaml_content)
    print(f"\ndata.yaml saved to {DATASET_DIR / 'data.yaml'}")
    print("Dataset ready!")


if __name__ == "__main__":
    main()
