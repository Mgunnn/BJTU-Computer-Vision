"""
可视化检索结果：每个 landmark 挑 2 张 query，展示 Top-5 检索结果
"""
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURE_DIR = os.path.join(BASE_DIR, "features")
RESULT_DIR = os.path.join(BASE_DIR, "results", "visualizations")
os.makedirs(RESULT_DIR, exist_ok=True)

LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
TOP_N = 5
SAMPLES_PER_LM = 2


def get_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    prefix = name.split("-")[0].lower()
    return prefix if prefix in LANDMARKS else None


def load_img(path, size=(150, 150)):
    try:
        return Image.open(path).convert("RGB").resize(size)
    except Exception:
        return Image.new("RGB", size, (200, 200, 200))


def main():
    base_feats = np.load(os.path.join(FEATURE_DIR, "base_features.npy"))
    query_feats = np.load(os.path.join(FEATURE_DIR, "query_features.npy"))

    with open(os.path.join(FEATURE_DIR, "base_paths.txt")) as f:
        base_paths = [l.strip() for l in f.readlines()]
    with open(os.path.join(FEATURE_DIR, "query_paths.txt")) as f:
        query_paths = [l.strip() for l in f.readlines()]

    base_labels = [get_label(p) for p in base_paths]

    # 按 landmark 分组 query
    lm_to_queries = defaultdict(list)
    for qi, qp in enumerate(query_paths):
        lbl = get_label(qp)
        if lbl:
            lm_to_queries[lbl].append(qi)

    scores = query_feats @ base_feats.T

    for lm in LANDMARKS:
        queries = lm_to_queries.get(lm, [])
        if not queries:
            continue
        samples = random.sample(queries, min(SAMPLES_PER_LM, len(queries)))

        for s_idx, qi in enumerate(samples):
            sorted_idx = np.argsort(-scores[qi])[:TOP_N]

            fig, axes = plt.subplots(1, TOP_N + 1, figsize=(3 * (TOP_N + 1), 3.5))
            fig.suptitle(f"{lm.upper()} — Query {s_idx + 1}", fontsize=12)

            # query 图
            axes[0].imshow(load_img(query_paths[qi]))
            axes[0].set_title("Query", fontsize=9)
            axes[0].axis("off")

            # top-N 检索结果
            for rank, bi in enumerate(sorted_idx):
                img = load_img(base_paths[bi])
                retrieved_lbl = base_labels[bi]
                correct = retrieved_lbl == lm
                border_color = "green" if correct else "red"
                axes[rank + 1].imshow(img)
                axes[rank + 1].set_title(
                    f"#{rank+1} {retrieved_lbl or '?'}\n{'✓' if correct else '✗'}",
                    fontsize=8,
                    color=border_color,
                )
                for spine in axes[rank + 1].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                axes[rank + 1].axis("off")

            plt.tight_layout()
            save_path = os.path.join(RESULT_DIR, f"{lm}_sample{s_idx + 1}.png")
            plt.savefig(save_path, dpi=120)
            plt.close()
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
