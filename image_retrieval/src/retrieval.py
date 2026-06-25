"""
检索 + 计算 P@K + 绘图
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURE_DIR = os.path.join(BASE_DIR, "features")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

K_VALUES = [20, 40, 60]
LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]


def get_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    prefix = name.split("-")[0].lower()
    return prefix if prefix in LANDMARKS else None


def precision_at_k(retrieved_labels, query_label, k):
    top_k = retrieved_labels[:k]
    hits = sum(1 for lbl in top_k if lbl == query_label)
    return hits / k


def main():
    base_feats = np.load(os.path.join(FEATURE_DIR, "base_features.npy"))
    query_feats = np.load(os.path.join(FEATURE_DIR, "query_features.npy"))

    with open(os.path.join(FEATURE_DIR, "base_paths.txt")) as f:
        base_paths = [l.strip() for l in f.readlines()]
    with open(os.path.join(FEATURE_DIR, "query_paths.txt")) as f:
        query_paths = [l.strip() for l in f.readlines()]

    base_labels = [get_label(p) for p in base_paths]

    # 余弦相似度矩阵 (num_query, num_base)
    scores = query_feats @ base_feats.T  # 已L2归一化，点积=余弦相似度

    # 按地点分组统计 P@K
    # landmark -> {k: [precision values per query]}
    pk_per_landmark = {lm: {k: [] for k in K_VALUES} for lm in LANDMARKS}

    for qi, qpath in enumerate(query_paths):
        qlabel = get_label(qpath)
        if qlabel is None:
            continue
        sorted_idx = np.argsort(-scores[qi])  # 降序
        retrieved_labels = [base_labels[i] for i in sorted_idx]
        for k in K_VALUES:
            p = precision_at_k(retrieved_labels, qlabel, k)
            pk_per_landmark[qlabel][k].append(p)

    # 汇总结果
    summary = {}
    for lm in LANDMARKS:
        summary[lm] = {}
        for k in K_VALUES:
            vals = pk_per_landmark[lm][k]
            summary[lm][k] = round(np.mean(vals), 4) if vals else 0.0

    print("\n=== P@K Results ===")
    print(f"{'Landmark':<8}", "  ".join(f"P@{k}" for k in K_VALUES))
    for lm in LANDMARKS:
        row = "  ".join(f"{summary[lm][k]:.4f}" for k in K_VALUES)
        print(f"{lm:<8} {row}")

    with open(os.path.join(RESULT_DIR, "pk_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 画图：每个 landmark 一张 P@K 图
    for lm in LANDMARKS:
        fig, ax = plt.subplots(figsize=(5, 4))
        pk_vals = [summary[lm][k] for k in K_VALUES]
        ax.bar([str(k) for k in K_VALUES], pk_vals, color="steelblue")
        ax.set_ylim(0, 1)
        ax.set_xlabel("K")
        ax.set_ylabel("Precision@K")
        ax.set_title(f"P@K — {lm.upper()}")
        for i, v in enumerate(pk_vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"pk_{lm}.png"), dpi=120)
        plt.close()
        print(f"Saved pk_{lm}.png")

    print(f"\nAll results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
