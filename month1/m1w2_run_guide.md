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
