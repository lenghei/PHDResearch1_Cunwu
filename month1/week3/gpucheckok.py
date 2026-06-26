import torch
import sys

print("================ PyTorch 硬件环境诊断 ================")
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用 (is_available): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"PyTorch 内部编译的 CUDA 版本: {torch.version.cuda}")
    print(f"当前可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前默认 GPU 索引: {torch.cuda.current_device()}")
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    print("\n[🎉 恭喜] 你的 PyTorch CUDA 环境完全正常！可以直接在代码中使用 device=0 加速。")
else:
    print("\n[❌ 警报] PyTorch 目前只能看到 CPU 驱动。")
    print("这通常是因为通过单纯的 'pip install torch' 误装了 CPU 版本的 PyTorch 轮子。")
print("======================================================")