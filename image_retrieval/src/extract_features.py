"""
离线提取 base + query 所有图片的特征，保存为 .npy 文件
"""
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_BJTU_DIR = os.path.join(BASE_DIR, "base", "BJTU")
BASE_UTIL_DIR = os.path.join(BASE_DIR, "base", "util_pic")
QUERY_DIR = os.path.join(BASE_DIR, "query")
FEATURE_DIR = os.path.join(BASE_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # 去掉最后的分类层，取 2048 维全局平均池化特征
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval().to(DEVICE)
    return model


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return transform(img)
    except Exception:
        return None


def extract(model, paths, desc=""):
    features, valid_paths = [], []
    batch_imgs, batch_paths = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            t = torch.stack(batch_imgs).to(DEVICE)
            feat = model(t).squeeze(-1).squeeze(-1).cpu().numpy()
        features.append(feat)
        valid_paths.extend(batch_paths)
        batch_imgs.clear()
        batch_paths.clear()

    for p in tqdm(paths, desc=desc):
        img = load_image(p)
        if img is None:
            continue
        batch_imgs.append(img)
        batch_paths.append(p)
        if len(batch_imgs) == BATCH_SIZE:
            flush()
    flush()

    return np.vstack(features), valid_paths


def get_all_images(directory):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def main():
    model = build_model()
    print(f"Using device: {DEVICE}")

    # base = BJTU + util_pic
    base_paths = get_all_images(BASE_BJTU_DIR) + get_all_images(BASE_UTIL_DIR)
    print(f"Base images: {len(base_paths)}")
    base_feats, base_paths_valid = extract(model, base_paths, desc="Base")
    # L2 归一化，方便后续用点积代替余弦相似度
    base_feats /= np.linalg.norm(base_feats, axis=1, keepdims=True) + 1e-8
    np.save(os.path.join(FEATURE_DIR, "base_features.npy"), base_feats)
    with open(os.path.join(FEATURE_DIR, "base_paths.txt"), "w") as f:
        f.write("\n".join(base_paths_valid))
    print(f"Saved base features: {base_feats.shape}")

    # query
    query_paths = get_all_images(QUERY_DIR)
    print(f"Query images: {len(query_paths)}")
    query_feats, query_paths_valid = extract(model, query_paths, desc="Query")
    query_feats /= np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8
    np.save(os.path.join(FEATURE_DIR, "query_features.npy"), query_feats)
    with open(os.path.join(FEATURE_DIR, "query_paths.txt"), "w") as f:
        f.write("\n".join(query_paths_valid))
    print(f"Saved query features: {query_feats.shape}")


if __name__ == "__main__":
    main()
