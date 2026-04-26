import os
import glob

# ===================== 你只需要改这个路径 =====================
BASE_DIR = r"E:\ForStudy\code\VisDrone_Detection\month1\m1w3\datasets\visdrone_val_aug_9conditions"
# ==============================================================

CONDITIONS = ["S1T1", "S1T2", "S1T3", "S2T1", "S2T2", "S2T3", "S3T1", "S3T2", "S3T3"]

# VisDrone 类别映射 → YOLO 0-9
CLASS_MAP = {
    0: 0,   # pedestrian
    1: 1,   # person
    2: 2,   # car
    3: 3,   # van
    4: 4,   # bus
    5: 5,   # truck
    6: 6,   # motor
    7: 7,   # bicycle
    8: 8,   # awning-tricycle
    9: 9,   # tricycle
}

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def convert_one(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 8:
            continue

        try:
            x1 = float(parts[0])
            y1 = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])
            category = int(parts[4])
        except:
            continue

        if category not in CLASS_MAP:
            continue

        # 归一化
        x_center = (x1 + w / 2) / IMG_WIDTH
        y_center = (y1 + h / 2) / IMG_HEIGHT
        norm_w = w / IMG_WIDTH
        norm_h = h / IMG_HEIGHT

        cls = CLASS_MAP[category]
        yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(yolo_lines)

def run_convert():
    for cond in CONDITIONS:
        label_dir = os.path.join(BASE_DIR, cond, "labels")
        if not os.path.exists(label_dir):
            print(f"跳过 {cond}（无labels文件夹）")
            continue

        txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
        print(f"\n正在转换 {cond} → {len(txt_files)} 个标签文件")

        for txt in txt_files:
            convert_one(txt)

    print("\n✅ 全部转换完成！现在 YOLO 可以正常读取标签了")

if __name__ == "__main__":
    run_convert()