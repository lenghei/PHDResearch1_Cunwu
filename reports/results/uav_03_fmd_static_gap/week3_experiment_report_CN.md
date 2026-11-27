# UAV-GAP 第三周实验结果说明（更新版）

## 1. 本周任务

第三周要求建立检测器特征模型 FM_D，记录实际检测器属性，并逐个解释 FM_P 退化叶与默认 YOLO 训练增强之间的静态覆盖关系。第三周不要求执行第4–6周的500组正式P-D评价，也不要求重新训练模型。

## 2. FM_D 配置空间

原始 `fm_d.uvl` 保留不变：34个叶子特征、4条跨树约束。独立穷举导出得到 **635,904** 个合法配置；按训练阶段特征去重得到 **844** 个训练变体。其余配置差异来自推理阶段，例如输入尺寸、FullFrame/SAHI、后处理、精度、预处理和TTA。

本次生成的 `configuration_validation.csv` 有 **635,904条数据记录（不含表头）和67列**。其中34列逐个表示FM_D叶特征，另外33列记录配置ID、训练变体ID、模型类别信息、合法性、约束状态及测量状态。每一行代表一个合法D配置。CSV中所有行均通过独立枚举器的四条约束检查；配置ID与34叶特征向量均唯一；844个训练变体映射完整。

**重要限制：** 本次运行环境没有安装FlamaPy/PySAT，因此本次完整CSV验证是独立Python穷举器对原UVL四条约束的复核，不冒称已经在此环境重新运行FlamaPy。包内提供 `flamapy_count.py`，需要在你本机的 `.venv_flamapy` 再做一次独立求解器确认。

## 3. 五种实际GPU属性测量

测量使用同一个 YOLOv8s 第76轮 best checkpoint，参数量 **11,139,470**；GPU为 NVIDIA GeForce RTX 4060 Laptop GPU。使用同一组32张 VisDrone val 图像，每种设置2遍，固定3次预热。监督墙钟30.45秒，最高温度63°C。

| 推理设置 | GFLOPs (2×MAC) | 推理均值 (ms/img) | 缓存PNG至取框 (ms/img) | 峰值allocated (MiB) |
|---|---:|---:|---:|---:|
| 640 FP32 | 28.46 | 11.15 | 33.06 | 78.50 |
| 960 FP32 | 64.03 | 17.17 | 38.35 | 168.83 |
| 1280 FP32 | 113.83 | 26.92 | 48.28 | 275.23 |
| 640 FP16 | 28.46 | 12.13 | 33.67 | 172.60 |
| 1280 FP16 | 113.83 | 25.04 | 46.39 | 128.90 |

推理均值包括图像预处理、模型前向和NMS。缓存PNG至取框还包括读取PNG、SHA校验、解码和取框；不包括预测文件写入和COCO评价。峰值为PyTorch allocated；原记录还保存reserved，但本次收到的摘要没有提供reserved具体数值，因此CSV中留空。以上是小样本属性测量，不是AP测试，也不能代表完整548张图的正式评价。

五种 profile 没有被强行关联到635,904行中的某一条唯一配置，因为现有交付材料没有把该checkpoint训练时的所有增强叶（尤其Albu_Default是否实际激活）完整绑定到权重。原始 profile 保存在 `measured_detector_profiles.csv`，主配置矩阵中的实测字段保持空白，避免错误关联。

## 4. Static Gap：三个覆盖口径

FM_P有46,368个合法退化配置。复算得到：

| 口径 | Covered | Gap | Gap比例 | 解释 |
|---|---:|---:|---:|---|
| 严格恒等 | 1 | 46,367 | 99.9978% | 只把Normal视为完全相同 |
| 亮度中间口径 | 3 | 46,365 | 99.9935% | Normal、轻度过曝、轻度欠曝；HSV仅近似gain |
| 宽松代理口径 | 18 | 46,350 | 99.9612% | 3种亮度×2种失焦状态×3种缩放状态，均是弱代理假设 |

宽松18的计算与已有18个covered configuration ID逐一一致。它不是模型检测成功率，也不是46,350种图像都实测失败。特别是：HSV亮度乘法不能完全复现加性bias；普通Blur/MedianBlur只是小半径Defocus的代理；scale/Mosaic不等同于真实无人机高度变化。历史checkpoint的实际增强覆盖仍为 `UNKNOWN`，因为需要把保存的训练参数、实际安装版本和Albumentations激活记录与checkpoint准确绑定。

逐个退化叶的初始代理状态见 `static_gap_leaf_rules.csv`。其中 `EXACT` 只用于Normal恒等；弱代理标记 `APPROXIMATE`；默认增强未覆盖的类别标记 `NOT_COVERED`；历史checkpoint实际覆盖统一保留 `UNKNOWN`。

## 5. SAHI 与 TTA

原始 FM_D 继续保留 `SAHI_Slicing => !TTA`，以保证原模型和已冻结评价组合不变。独立扩展文件 `fm_d_sahi_tta.uvl` 单独开放一个受限SAHI+TTA设置，历史FlamaPy/PySAT记录为635,905个配置。已提供的复现说明称：在SAHI 0.11.36和一个YOLOv8s CPU样例上，只有将 `augment=True` 传到底层Ultralytics模型覆盖项时，四个切片才进入TTA增强分支，合并框坐标检查通过。这只证明该测试版本和该调用方式可以运行，不证明AP有提升，也不应把扩展配置混入原635,904。

## 6. 本次可复现验证结果

- FM_D叶特征数：34。
- FM_D跨树约束数：4。
- 合法配置：635,904。
- 训练变体：844。
- CSV：635,904行、67列。
- configuration_id唯一：通过。
- 34叶配置向量唯一：通过。
- 四条约束全量复核：通过。
- 训练变体ID数量：844，通过。
- 误填实测指标：0行。
- 静态覆盖复算：1 / 3 / 18，通过；18个ID与提供的历史结果一致。
- 单元测试：5/5通过。
- FlamaPy/PySAT在本次容器中：未运行（依赖缺失），需在本机隔离环境补做。

## 7. 尚未完成的证据

1. 本机FlamaPy独立重新计数及dead-feature检查。
2. 将五种GPU profile的checkpoint SHA、原始measurement CSV和reserved显存数值一起归档；当前说明文件只提供摘要数字。
3. 将历史checkpoint的真实训练args、Ultralytics安装源码版本及Albumentations实际激活状态准确绑定，以确定真实训练覆盖；目前不能把8.4.33计划默认值直接当作历史checkpoint实际执行值。
4. SAHI+TTA目前只有单checkpoint、单裁剪CPU兼容性证据，没有精度评价。

## 8. 复现命令

从 `E:\ForStudy\code\uavgap`：

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/uav_03_fmd_static_gap/export_configuration_validation.py
& .\.venv\Scripts\python.exe -B main/experiments/uav_03_fmd_static_gap/validate_configuration_validation.py
& .\.venv\Scripts\python.exe -B main/experiments/uav_03_fmd_static_gap/verify_static_gap.py
& .\.venv\Scripts\python.exe -B -m unittest discover -s main/experiments/uav_03_fmd_static_gap/tests -p 'test_*.py' -v
& .\.venv_flamapy\Scripts\python.exe -B main/experiments/uav_03_fmd_static_gap/flamapy_count.py --model main/experiments/uav_03_fmd_static_gap/fm_d.uvl
```

不要为了重新生成报告而重复GPU测量，不启动模型训练，也不启动第4–6周500组评价。
