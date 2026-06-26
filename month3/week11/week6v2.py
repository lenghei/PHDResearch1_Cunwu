import os
import cv2
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==============================================================================
# 1. 路径与矩阵布局配置 (对齐你的数据集结构)
# ==============================================================================
DATASET_ROOT = "./visdrone_val_aug_9conditions"

MATRIX_LAYOUT = [
    ["S1T1", "S1T2", "S1T3"],
    ["S2T1", "S2T2", "S2T3"],
    ["S3T1", "S3T2", "S3T3"]
]


def visualize_model_predictions(model_path="yolov8n.pt", conf_threshold=0.15):
    """
    在 3x3 恶劣天气/光照矩阵中，抽取同一张无人机航拍图片，
    让模型进行真实推理，把检测成功的图片拼成矩阵画布，自动保存并弹窗。
    """
    # 从基准环境 S1T1 中随机抽取一张图片名
    s1t1_img_dir = os.path.join(DATASET_ROOT, "S1T1", "images")
    if not os.path.exists(s1t1_img_dir):
        print(f"[!] 找不到基准测试目录: {s1t1_img_dir}，请检查路径。")
        return

    images = [f for f in os.listdir(s1t1_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("[!] S1T1 文件夹内没有找到有效的图片。")
        return

    # 锁定单张样本，追踪它在 9 种退化条件下的表现
    target_img_name = random.choice(images)
    print(f"🎯 已选中样本图片进行全矩阵推理追踪: {target_img_name}")

    # 加载指定的检测模型
    print(f"[*] 正在载入学术评估模型: {model_path}")
    model = YOLO(model_path)

    # 创建 3x3 图像大画布
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(
        f"UAV Detection Predictions Matrix (Model: {os.path.basename(model_path)} | Sample: {target_img_name})\n"
        f"Columns: Temporal Degradation (T1->T3) | Rows: Spatial Illumination (S1->S3)",
        fontsize=14, fontweight='bold', y=0.96
    )

    for row_idx in range(3):
        for col_idx in range(3):
            cond = MATRIX_LAYOUT[row_idx][col_idx]
            img_path = os.path.join(DATASET_ROOT, cond, "images", target_img_name)
            ax = axes[row_idx, col_idx]

            if os.path.exists(img_path):
                # 执行模型预测推理
                # verbose=False 关闭控制台冗余输出；device='cpu' 或 0
                results = model.predict(source=img_path, conf=conf_threshold, verbose=False)[0]

                # 使用 Ultralytics 原生 plot() 函数直接渲染预测框和标签（返回 BGR 矩阵）
                plot_img_bgr = results.plot()

                # 转换通道 BGR -> RGB，防止 Matplotlib 颜色颠倒（如蓝脸变红脸）
                plot_img_rgb = cv2.cvtColor(plot_img_bgr, cv2.COLOR_BGR2RGB)

                ax.imshow(plot_img_rgb)
                ax.set_title(f"{cond} (Detections: {len(results.boxes)})", fontsize=11, fontweight='bold',
                             color='darkblue')
            else:
                ax.text(0.5, 0.5, f"Missing:\n{cond}", ha='center', va='center',
                        fontsize=12, color='crimson', fontweight='bold')
                ax.set_facecolor('#ffeeee')

            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # 【自动保存】高清成果图，便于放进你的 PhD 进展报告
    save_output_path = f"matrix_predictions_output.png"
    plt.savefig(save_output_path, dpi=300, bbox_inches='tight')
    print(f"[+Done] 预测成果矩阵图片已成功保存至本地: {save_output_path}")

    # 【弹窗显示】直接在屏幕上跳出交互式窗口
    print("[*] 正在唤起图形界面弹窗...")
    plt.show()


if __name__ == "__main__":
    # 提示：如果使用官方未微调权重，由于类别不匹配，可能需要将 conf 调得极低（如 0.01）才能勉强看到乱抓的框。
    # 建议换成你在 VisDrone 上训练出来的最佳模型权重路径，例如: "runs/detect/train/weights/best.pt"
    visualize_model_predictions(model_path="yolov8n.pt", conf_threshold=0.15)