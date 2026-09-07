# Week 2 Assignment: Environment Setup Guide

# Note for Week 2 Assignment

## Environment Information

- **IDE**: PyCharm 2024.3

- **Python Version**: 3.12

- **OS**: Windows 11

- **Hardware**:

    - Processor: AMD Ryzen 9 7940HX with Radeon Graphics (2.40 GHz)

    - RAM: 32.0 GB

## Dependencies Installation

Install the required packages using the following commands:

```Bash

# Install PyTorch (CPU version, stable and compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other required libraries
pip install ultralytics albumentations opencv-python matplotlib pillow pandas
```

## Dataset Structure

Please ensure the VisDrone-DET 2019 dataset is placed as follows:

```Plain Text

project_folder/
├── datasets/
│   └── VisDrone2019/
│       ├── train/
│       │   ├── images/
│       │   └── annotations/
│       └── val/
│           ├── images/
│           └── annotations/
└── m1w2_visdrone.py
```
Note that due to the large size of the VisDrone2019 dataset, it cannot be uploaded to the Git repository,
Download link:https://github.com/VisDrone/VisDrone-Dataset
## How to Run

1. Open the project.

2. Set the Python interpreter to **Python 3.12**.

3. Install all dependencies listed above.

4. Run the code directly; it will load the dataset and display 5 annotated images with bounding boxes.

m1w2_visdrone.py
```Bash

import os
import cv2
import matplotlib.pyplot as plt

# ===================== Path Setting =====================
BASE_PATH = "./datasets/VisDrone2019"
SPLIT = "train"  # or "val"

IMAGE_DIR = os.path.join(BASE_PATH, SPLIT, "images")
ANNOT_DIR = os.path.join(BASE_PATH, SPLIT, "annotations")

# Get image list
img_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])

# ===================== Parse Annotations =====================
def parse_annot(anno_path):
    boxes = []
    with open(anno_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            x1 = int(float(data[0]))
            y1 = int(float(data[1]))
            w = int(float(data[2]))
            h = int(float(data[3]))
            x2 = x1 + w
            y2 = y1 + h
            boxes.append([x1, y1, x2, y2])
    return boxes

# ===================== Draw Bounding Boxes =====================
def draw_boxes(img, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

# ===================== Display 5 Images =====================
plt.figure(figsize=(20, 12))

for i in range(5):
    # Read image
    img_path = os.path.join(IMAGE_DIR, img_files[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read annotations
    name = os.path.splitext(img_files[i])[0]
    anno_path = os.path.join(ANNOT_DIR, name + ".txt")
    boxes = parse_annot(anno_path)

    # Draw boxes
    img = draw_boxes(img, boxes)

    # Display image
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image {i+1}")

plt.tight_layout()
plt.show()
```


next
# 中文理解版

## 1. 当前研究目标

目前论文主要研究**无人机航拍图像中轻量目标检测模型的小目标检测和复杂环境鲁棒性**。

结合无人机巡检实际场景，重点考虑：

- 高空或远距离拍摄造成的小目标问题；
- 低分辨率输入导致小目标细节丢失；
- 过曝和欠曝；
- 无人机运动造成的运动模糊和噪声；
- 通用预训练模型与 VisDrone 专门训练模型之间的差异。

计划在 **2026 年 9 月完成主要实验和论文，并准备投稿 IEEE Access**。

---

## 2. 研究计划的主要改进

### YOLOv8s 与 YOLO26s 对比

论文主要选择 **YOLOv8s 和 YOLO26s**。

YOLOv8s 作为比较成熟的基线，YOLO26s 作为目前最新一代 YOLO。

选择 YOLO26s 不只是因为它比较新，还因为它增加了针对小目标学习的设计。因此实验主要想验证：

**这些新的设计在无人机小目标场景中是否真的能够带来提升。**

---

### 预训练模型与 Fine-Tuned 模型对比

两个模型都分别测试：

- 原始预训练模型；
- 在 VisDrone 上 Fine-Tuned 后的模型。

最终主要比较：

- YOLOv8s Pretrained
- YOLOv8s Fine-Tuned
- YOLO26s Pretrained
- YOLO26s Fine-Tuned

这样不仅可以看到 Fine-Tuning 后精度提升多少，还可以进一步研究：

**训练后的模型在复杂光照、运动模糊和噪声环境中是不是也更加鲁棒。**

论文会把：

**绝对检测精度**

和

**相对于正常环境的性能保持率**

分开分析。

---

### 640 与 1280 小目标分析

增加 **640 和 1280 输入分辨率对比**。

重点不是简单证明 1280 的整体 mAP 更高，而是分析：

**1280 的提升是不是主要来自 tiny/small objects。**

这与无人机巡检比较符合，因为航拍中的目标或者缺陷往往只占很少的像素。

---

### 保留 3×3 复合扰动实验

继续使用目前的 3×3 实验：

三种光照：

- 正常
- 过曝
- 欠曝

三种运动/图像质量：

- 无模糊
- 中等运动模糊
- 严重运动模糊 + 噪声

共 9 种情况。

主要目的是模拟更加接近真实无人机巡检的复杂情况，而不是只在正常图片或者单一扰动条件下测试。

---

## 3. 论文的四个核心贡献

1. **基于实际 UAV 巡检的复合鲁棒性评价方法**  
   使用光照、运动模糊和噪声建立 3×3 测试方案。

2. **小目标分辨率分析**  
   研究 640 提高到 1280 后，提升是否主要集中在 tiny/small objects。

3. **Domain Adaptation 鲁棒性分析**  
   对比预训练模型和 VisDrone Fine-Tuned 模型，分析 Fine-Tuning 对正常精度和复杂环境鲁棒性的影响。

4. **新旧两代检测模型对比**  
   对比 YOLOv8s 和 YOLO26s，验证新一代针对小目标的设计在 UAV 场景中是否真正有效。

---

## 4. 目前正在进行的工作

目前正在 **优化和重新整理实验代码**，主要包括：

- 统一训练和测试配置；
- 支持 YOLOv8s 和 YOLO26s；
- 支持 Pretrained / Fine-Tuned 对比；
- 自动测试 640 / 1280；
- 自动完成 9 种扰动条件；
- 使用真实 mAP50 和 mAP50:95；
- 增加 tiny/small/medium/large 分析；
- 增加各类别检测结果；
- 计算 robustness retention；
- 自动保存 CSV / JSON；
- 统一随机种子和实验日志，保证实验可重复。

代码完成以后，就开始重新跑完整实验，并根据最终数据更新论文的 Results 和 Discussion。

---

## 5. 9 月目标

9 月主要完成：

1. 完成实验代码；
2. Fine-Tune YOLOv8s 和 YOLO26s；
3. 完成 Pretrained 与 Fine-Tuned 对比；
4. 完成 640 与 1280 小目标实验；
5. 完成 3×3 鲁棒性测试；
6. 生成最终表格、图片和统计结果；
7. 完成论文；
8. 准备投稿 **IEEE Access**。

总体上不继续增加很多模型和复杂模块，而是集中把 **小目标、分辨率、Fine-Tuning、复杂环境鲁棒性和最新 YOLO 模型验证** 这几部分做好。
