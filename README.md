# BJTU Computer Vision Lab 2

北京交通大学《计算机视觉基础》实验二：**基于预训练特征的图像检索 + 文字区域检测**

---

## 任务概览

| 任务 | 方法 | 评价指标 | 结果 |
|------|------|----------|------|
| 图像检索 | ResNet50 预训练特征 + 余弦相似度 | P@K (K=20,40,60) | 多数地点 P@20 > 0.85 |
| 文字检测 | YOLOv8n 微调（25 epochs） | mAP50 / Precision / Recall | mAP50 = 0.742 |

---

## 项目结构

```
lab2/
├── image_retrieval/
│   ├── src/
│   │   ├── extract_features.py   # ResNet50 特征提取，保存为 .npy
│   │   ├── retrieval.py          # 余弦相似度检索，计算 P@K
│   │   └── visualize_retrieval.py# 检索结果可视化（Query + Top-5）
│   └── results/
│       ├── pk_*.png              # 各地点 P@K 柱状图
│       ├── pk_results.json       # P@K 数值汇总
│       └── visualizations/       # 检索可视化图片
│
├── object_detection/
│   ├── src/
│   │   ├── convert_annotations.py# LabelMe JSON → YOLO txt 格式转换
│   │   ├── train.py              # YOLOv8n 微调训练
│   │   └── visualize_detection.py# 联合检索+检测可视化
│   ├── runs/text_det/weights/
│   │   └── best.pt               # 训练好的模型权重
│   └── results/visualizations/   # 检测可视化图片
│
├── generate_report.js            # 生成实验报告 .docx（需要 node）
├── gen_arch_diagram.py           # 生成方案架构图
├── arch_diagram.png              # 方案架构图
├── 实验报告.docx                  # 最终实验报告
├── package.json
└── .gitignore
```

---

## 环境依赖

**Python**
```bash
pip install torch torchvision tqdm pillow matplotlib ultralytics
```

**Node.js**（仅用于生成报告）
```bash
npm install
```

---

## 运行步骤

### 任务一：图像检索

> 需要自行准备数据集，放置于 `image_retrieval/base/`（base库）和 `image_retrieval/query/`（查询集）

```bash
# 1. 提取特征
python image_retrieval/src/extract_features.py

# 2. 检索评测（计算 P@K，生成柱状图）
python image_retrieval/src/retrieval.py

# 3. 可视化检索结果
python image_retrieval/src/visualize_retrieval.py
```

### 任务二：文字检测

> 需要自行准备 LabelMe 标注数据，放置于 `object_detection/data/`

```bash
# 1. 转换标注格式
python object_detection/src/convert_annotations.py

# 2. 训练（或直接使用 runs/text_det/weights/best.pt）
python object_detection/src/train.py

# 3. 联合可视化
python object_detection/src/visualize_detection.py
```

### 生成实验报告

```bash
node generate_report.js
# 输出：实验报告.docx
```

---

## 数据集说明

数据集由课程提供，未包含在本仓库中。

| 数据 | 路径 | 说明 |
|------|------|------|
| 检索 base 库 | `image_retrieval/base/` | 7728 张校园图片（12类地点） |
| 检索 query 集 | `image_retrieval/query/` | 135 张查询图片 |
| 文字检测标注 | `object_detection/data/` | 1494 张 LabelMe JSON 标注 |

---

## 实验结果

### P@K 检索精度

| 地点 | P@20 | P@40 | P@60 |
|------|------|------|------|
| fhy（芳华园） | 0.910 | 0.890 | 0.840 |
| nm（南门）    | 0.903 | 0.860 | 0.814 |
| zx（知行楼）  | 0.904 | 0.880 | 0.845 |
| kx（科学楼）  | 0.175 | 0.138 | 0.100 |

### 文字检测（YOLOv8n，25 epochs）

| Precision | Recall | mAP50 |
|-----------|--------|-------|
| 0.818 | 0.744 | 0.742 |

---

## 参考文献

- He et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Jocher et al. *Ultralytics YOLOv8*. 2023.
