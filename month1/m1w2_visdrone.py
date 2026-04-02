import os
import cv2
import matplotlib.pyplot as plt

# ===================== 路径设置 =====================
BASE_PATH = "./datasets/VisDrone2019"
SPLIT = "train"  # 或者 val

IMAGE_DIR = os.path.join(BASE_PATH, SPLIT, "images")
ANNOT_DIR = os.path.join(BASE_PATH, SPLIT, "annotations")

# 获取图片列表
img_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])

# ===================== 解析标注 =====================
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

# ===================== 绘制框 =====================
def draw_boxes(img, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

# ===================== 显示 5 张图 =====================
plt.figure(figsize=(20, 12))

for i in range(5):
    # 读取图片
    img_path = os.path.join(IMAGE_DIR, img_files[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取标注
    name = os.path.splitext(img_files[i])[0]
    anno_path = os.path.join(ANNOT_DIR, name + ".txt")
    boxes = parse_annot(anno_path)

    # 画图
    img = draw_boxes(img, boxes)

    # 显示
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image {i+1}")

plt.tight_layout()
plt.show()
