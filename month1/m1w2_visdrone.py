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
