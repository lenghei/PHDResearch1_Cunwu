# YOLOv8s Baseline Evaluation on VisDrone2019-DET-val
## Reproduction Instructions

## 1. Environment Information
- IDE: PyCharm 2024.3
- Python Version: 3.12
- OS: Windows 11
- Hardware:
  - Processor: AMD Ryzen 9 7940HX with Radeon Graphics (2.40 GHz)
  - RAM: 32.0 GB
  - GPU: Not used (CPU inference for stable reproduction)
- CUDA/cuDNN: Not required

## 2. Dependencies Installation
Run the following commands in the terminal to install required packages:

```bash
# Install PyTorch (CPU version, stable and compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core libraries for detection and evaluation
pip install ultralytics opencv-python pandas pycocotools matplotlib
```

## 3. Dataset Preparation
The experiment uses the **VisDrone2019-DET validation set** for baseline evaluation.

### Dataset Structure
Please place the dataset in the following structure:
```
project_folder/
├── data/
│   └── VisDrone2019-DET-val/
│       ├── images/        # validation images (.jpg)
│       └── annotations/   # validation annotations (.txt)
└── evaluate.py
```

### Dataset Download
The dataset is not included in the repository due to its large size.
Official download link:
https://github.com/VisDrone/VisDrone-Dataset

## 4. Experiment Settings
- Model: YOLOv8s (official COCO pre-trained weights)
- No training or fine-tuning is performed
- Evaluation metrics: AP@0.5, AP@0.5:0.95 (per-class and overall)
- Inference size: 640
- Confidence threshold: 0.001
- NMS IOU threshold: 0.65

## 5. How to Run
1. Open the project in PyCharm 2024.3
2. Set the Python interpreter to Python 3.12
3. Install all dependencies listed above
4. Confirm the dataset path is correctly organized
5. Run `evaluate_m1w4.py` directly
6. Wait for inference and evaluation to finish

## 6. Expected Outputs
After running the code, the following outputs will be generated:
1. Terminal log: full evaluation summary and metrics
2. `yolov8s_visdrone_baseline.csv`: per-class and overall evaluation results
3. `visdrone_coco_gt.json`: COCO-format annotation file for evaluation

## 7. Reproducibility Description
All parameters are fixed. No random operations are used during inference or evaluation. Results are consistent and reproducible across multiple runs in the same environment.
