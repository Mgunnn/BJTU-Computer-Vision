"""
联合展示"检索 + 检测"结果
每类 landmark 挑 2 张 query，展示：query图 | 检索Top1 | 检测结果（带框）
共 12 × 2 = 24 组
"""
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict

# 把 image_retrieval/src 加入路径，复用特征文件
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "image_retrieval" / "src"))

BASE_DIR = Path(__file__).parent.parent
RETRIEVAL_DIR = ROOT / "image_retrieval"
FEATURE_DIR = RETRIEVAL_DIR / "features"
VIZ_DIR = BASE_DIR / "results" / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
SAMPLES_PER_LM = 2
CONF_THRESH = 0.25


def get_label(path):
    name = Path(path).stem
    prefix = name.split("-")[0].lower()
    return prefix if prefix in LANDMARKS else None


def load_img_pil(path, size=(300, 300)):
    try:
        return Image.open(path).convert("RGB").resize(size)
    except Exception:
        return Image.new("RGB", size, (220, 220, 220))


def draw_boxes(img_pil, boxes, scores):
    """用 PIL 直接在图上画框，避免 matplotlib canvas 渲染尺寸问题"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = f"{score:.2f}"
        draw.rectangle([x1, max(0, y1 - 16), x1 + len(label) * 8, y1], fill=(0, 0, 0))
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill=(0, 255, 0))
    return img


def main():
    # 加载检索特征
    base_feats = np.load(FEATURE_DIR / "base_features.npy")
    query_feats = np.load(FEATURE_DIR / "query_features.npy")
    base_paths = (FEATURE_DIR / "base_paths.txt").read_text().strip().split("\n")
    query_paths = (FEATURE_DIR / "query_paths.txt").read_text().strip().split("\n")
    scores_matrix = query_feats @ base_feats.T

    # 加载检测模型
    best_pt = BASE_DIR / "runs" / "text_det" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"找不到模型权重: {best_pt}")
        print("请先运行 train.py")
        return

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    from ultralytics import YOLO
    det_model = YOLO(str(best_pt))

    # 按 landmark 分组 query
    lm_to_queries = defaultdict(list)
    for qi, qp in enumerate(query_paths):
        lbl = get_label(qp)
        if lbl:
            lm_to_queries[lbl].append(qi)

    for lm in LANDMARKS:
        queries = lm_to_queries.get(lm, [])
        if not queries:
            print(f"No query for {lm}, skip")
            continue
        samples = random.sample(queries, min(SAMPLES_PER_LM, len(queries)))

        for s_idx, qi in enumerate(samples):
            qpath = query_paths[qi]

            # Top-1 检索结果
            top1_idx = int(np.argmax(scores_matrix[qi]))
            top1_path = base_paths[top1_idx]

            # 对 query 做检测（在原图上检测）
            orig_img = Image.open(qpath).convert("RGB")
            orig_w, orig_h = orig_img.size
            det_results = det_model(qpath, conf=CONF_THRESH, verbose=False)[0]
            boxes = det_results.boxes.xyxy.cpu().numpy() if det_results.boxes else []
            confs = det_results.boxes.conf.cpu().numpy() if det_results.boxes else []

            # 缩放框坐标到 300×300
            scale_x, scale_y = 300 / orig_w, 300 / orig_h
            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                scaled_boxes.append([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y])

            q_img = load_img_pil(qpath)
            top1_img = load_img_pil(top1_path)
            det_img = draw_boxes(q_img.copy(), scaled_boxes, confs)

            # 拼图
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            fig.suptitle(f"{lm.upper()} — Sample {s_idx + 1}", fontsize=12)

            axes[0].imshow(q_img);   axes[0].set_title("Query", fontsize=9);        axes[0].axis("off")
            axes[1].imshow(top1_img); axes[1].set_title("Top-1 Retrieved", fontsize=9); axes[1].axis("off")
            axes[2].imshow(det_img);  axes[2].set_title("Text Detection", fontsize=9);  axes[2].axis("off")

            plt.tight_layout()
            save_path = VIZ_DIR / f"{lm}_sample{s_idx + 1}.png"
            plt.savefig(save_path, dpi=120)
            plt.close()
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
