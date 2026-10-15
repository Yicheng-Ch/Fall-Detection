# YOLOv13-N-HRNet-DCVN-for-Student-Fall-Detection

> 官方 PyTorch 实现 | 论文处于投刊阶段，标题：《Context-Aware Deep Learning for Fall Detection of College Students in Physical Education Activities》
> 提出融合 HRNet 高分辨率表征与 DCVN 多尺度上下文增强的实时检测模型，实现大学体育课复杂场景下站立与跌倒行为的高精度、低延迟分类，为校园边缘端安全监测提供可靠方案。

## 1. 研究背景与模型定位

跌倒是全球性公共安全与健康威胁，在高校体育教学场景中，学生因激烈对抗、器械失误或场地湿滑引发的意外跌倒频繁发生。此类场景具有**人群密集、运动幅度大、肢体遮挡严重、光照多变**等特点，对检测系统的精度与实时性提出双重挑战。现有视觉方法所采用的逐级下采样骨干在严重遮挡和快速运动下易丢失空间细节，高分辨率表征与低计算开销之间的矛盾仍制约边缘设备的实时高精度检测。

本文提出 **YOLOv13-N-HRNet-DCVN**，以 YOLOv13-N 为基线，通过定制化 HRNet 并行高分辨率骨干全程维护空间细节，并嵌入 DCVN（Dense Context-aware Voting Network）多尺度卷积注意力增强模块强化遮挡场景判别能力，最终在自建大学体育课跌倒数据集（SPFD-4213）上实现 **99.82% mAP@50、13.35 M 参数量、2.0 ms 单帧推理**的综合性能，为边缘计算设备在校园安全监测中的高可靠部署提供有效方案。

## 2. 核心创新点

### 2.1 HRNet 高分辨率骨干网络



* 以并行多分辨率子网替代 YOLOv13-N 原有串行下采样结构，全程保持 1/4 高分辨率特征图；

* 两级 HRModule 含四条并行分支（步长 1/2/4/8，对应 1/4 至 1/32 分辨率），低分辨率分支经上采样后与高分辨率分支逐元素相加，实现跨尺度信息融合；

* 有效保留人体关节点位置与躯干倾角等关键空间细节，显著提升小尺度目标与遮挡场景下的空间定位精度。

### 2.2 DCVN 多尺度上下文增强模块



* 选用 3×3、5×5、7×7 三种卷积核构建并行分支，分别提取摔倒局部细节、中尺度肢体姿态与全局场景上下文；

* 多尺度输出与原始输入进行密集拼接（Dense Concatenation），最大程度保留有效信息；

* 通过自适应全局平均池化 + 轻量级全连接网络生成通道注意力权重，对摔倒关键特征通道加权强化，抑制操场地面、运动器材、无关跑动学生等背景干扰；

* 仅增加约 4.2 M 参数与 0.2 ms 额外延迟，即驱动 mAP@\[0.5:0.95] 提升 2 个百分点以上。

### 2.3 精度–速度的最优权衡



* 在 13.35 M 轻量化参数规模下实现 99.82% mAP@50 与 2.0 ms 推理速度；

* 宽度倍率由 0.25 提升至 0.5，DSConv/A2C2f/HyperACE 模块通道等比扩展，增强特征表达与抗过拟合能力；

* FullPAD 颈部网络精简冗余卷积堆叠，参数量降低约 30%，适配校园边缘计算设备（如嵌入式 NVR、边缘盒子、教学终端）。

## 3. 实验数据集

### 3.1 自建数据集：SPFD-4213

本研究基于**自建大学生体育课跌倒检测数据集（Student Physical-education Fall Detection Dataset, SPFD-4213）**，数据采集于河南省多所高校，采用 Vivo S16e（5000 万像素后置主摄）统一拍摄，覆盖不同光照条件、拍摄角度与运动速度。



| 数据集       | 类别                    | 样本总数 | 场景          | 划分（训练：验证：测试）    |
| --------- | --------------------- | ---- | ----------- | --------------- |
| SPFD-4213 | 站立（Standing）、跌倒（Fall） | 4213 | 跑道、篮球场、室内场景 | 7 : 2 : 1（随机划分） |

场景覆盖说明：



* **跑道（Runway）**：行走与奔跑过程中的正常姿态及意外跌倒；

* **篮球场（Basketball court）**：跑动、跳跃及剧烈运动中的失衡状态；

* **室内场景（Indoor scene）**：武术、瑜伽、力量训练及室内步行等多种活动形式。

### 3.2 公开数据集（跨域泛化验证）

为验证模型跨场景泛化能力，引入两个公开数据集：



| 公开数据集                      | 来源                                             | 场景特征           |
| -------------------------- | ---------------------------------------------- | -------------- |
| le2i-with-upright-and-fall | Le2i Fall Detection Dataset（Roboflow Universe） | 多种室内场景下的站立与跌倒  |
| falldown-detection-f8xtb   | 网络图片标注（Roboflow Universe）                      | 体育赛事等户外场景的摔倒行为 |

### 3.3 数据集目录结构



```
datasets/
├── SPFD-4213/                  # 自建大学生体育课跌倒数据集
│   ├── images/
│   │   ├── train/              # 训练集图像（70%）
│   │   ├── val/                # 验证集图像（20%）
│   │   └── test/               # 测试集图像（10%）
│   ├── labels/                 # YOLO 格式标注（class_id x_center y_center w h）
│   └── datasets.yaml           # 类别与路径配置
├── le2i-with-upright-and-fall/  # 室内公开数据集
└── falldown-detection-f8xtb/   # 室外公开数据集
```

## 4. 实验环境配置

### 4.1 依赖安装

推荐使用 Anaconda 创建独立虚拟环境，确保依赖版本兼容（适配 PyTorch 2.x 与 CUDA 11.8+）：



```
# 1. 创建并激活虚拟环境
conda create -n yolo-hrnet-dcvn python=3.10
conda activate yolo-hrnet-dcvn

# 2. 安装 PyTorch 与 TorchVision（按 CUDA 版本选择，示例为 CUDA 12.1；CPU 用户可替换为 cpu 版本）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装目标检测与数据处理依赖
pip install ultralytics~=8.3.0
pip install numpy~=2.1.0 matplotlib~=3.9.0 opencv-python~=4.10.0.84
pip install pandas~=2.2.3 pillow~=11.0.0 tqdm~=4.66.0
pip install pyyaml~=6.0.2 seaborn~=0.13.2
```

### 4.2 硬件要求



* **GPU**：推荐 NVIDIA GPU（显存 ≥ 8 GB，如 RTX 3060/4060，支持 CUDA 11.8+），完整训练 300 轮耗时约 3–5 小时，显存占用峰值 ≤ 8 GB；

* **CPU**：支持推理测试（单张图像推理约 5–10 ms），但完整训练耗时显著增加（约 40–60 小时），不推荐；

* **边缘部署**：模型可经 TensorRT / ONNX 优化后部署至 Jetson 系列或嵌入式边缘盒子，满足校园实时监测需求。

## 5. 实验结果

### 5.1 核心性能（SPFD-4213 测试集）



| 指标              | 数值         |
| --------------- | ---------- |
| 参数量（Parameters） | 13.35 M    |
| mAP@50          | **99.82%** |
| mAP@\[0.5:0.95] | **95.01%** |
| 单帧推理速度          | **2.0 ms** |

### 5.2 跨域泛化性能（公开数据集）



| 数据集                        | 场景 | 召回率（Recall） | 精确率（Precision） | mAP@50 |
| -------------------------- | -- | ----------- | -------------- | ------ |
| le2i-with-upright-and-fall | 室内 | 96.24%      | 98.32%         | 98.79% |
| falldown-detection-f8xtb   | 室外 | 94.33%      | 97.68%         | 98.43% |

> 注：室内外核心指标差异 < 2%，验证了 HRNet 与 DCVN 协同机制在抑制复杂背景噪声、增强小目标特征判别方面的鲁棒性。

### 5.3 与主流方法对比

在自建数据集上，所提模型与近年主流跌倒检测方法对比如下：



| 方法                           | 年份       | mAP@50     |
| ---------------------------- | -------- | ---------- |
| 改进 YOLOv5s（Yu 等）             | 2026     | 93.70%     |
| YOLOv5-SIoU-SimAM（Wu 等）      | 2023     | 96.70%     |
| FIDC-YOLOv5（Wang 等）          | 2026     | 97.00%     |
| YOLOv8m（Pereira 等）           | 2024     | 97.10%     |
| **YOLOv13-N-HRNet-DCVN（本文）** | **2026** | **99.82%** |

> 注：相较自建数据集上 mAP 最高的 YOLOv8m 方法，本文模型提升 2.72 个百分点；相较 mAP 最低的改进 YOLOv5s，提升 6.12 个百分点。混淆矩阵分析显示，所提模型对站立与跌倒两类的误判仅 5/7 例，显著低于 YOLOv11（7/8 例）、YOLOv10（8/12 例）与 EfficientDet（10/11 例）。

### 5.4 消融实验



| 模型变体                   | HRNet | DCVN  | 参数量         | 推理速度       | mAP@50         |
| ---------------------- | ----- | ----- | ----------- | ---------- | -------------- |
| YOLOv13-N（基线，宽度 0.25）  | ✗     | ✗     | 2.46 M      | —          | 较低             |
| YOLOv13-N（宽度扩展 0.5）    | ✗     | ✗     | 9.84 M      | 4.5 ms     | 中              |
| + DCVN                 | ✗     | ✓     | \~14.0 M    | +0.2 ms    | Precision 显著提升 |
| + HRNet                | ✓     | ✗     | —           | —          | Recall 显著提升    |
| **+ HRNet + DCVN（最终）** | **✓** | **✓** | **13.35 M** | **2.0 ms** | **99.82%**     |

## 6. 代码使用说明

### 6.1 模型训练

运行 `train.py` 启动训练，支持通过参数调整配置（适配 SPFD-4213 数据集）：



```
python train.py \
  --data ./datasets/SPFD-4213/datasets.yaml \
  --model yolov13n-hrnet-dcvn.yaml \
  --epochs 300 \
  --batch_size 16 \
  --img_size 640 \
  --lr0 1e-3 \
  --weight_decay 5e-4 \
  --save_dir ./runs/train \
  --device cuda:0 \
  --cache
```

#### 关键参数说明：



| 参数名            | 含义                            | 默认值                                  |
| -------------- | ----------------------------- | ------------------------------------ |
| `--data`       | 数据集 YAML 配置路径                 | `./datasets/SPFD-4213/datasets.yaml` |
| `--model`      | 模型结构配置文件（含 HRNet 骨干与 DCVN 定义） | `yolov13n-hrnet-dcvn.yaml`           |
| `--epochs`     | 训练轮数                          | 300                                  |
| `--batch_size` | 批次大小（根据 GPU 显存调整，8/16/32）     | 16                                   |
| `--img_size`   | 输入图像尺寸                        | 640                                  |
| `--lr0`        | 初始学习率                         | 1e-3                                 |
| `--device`     | 训练设备（`cuda:0` 或 `cpu`）        | `cuda:0`                             |

#### 训练输出：



* 训练过程自动保存 **验证集 mAP@50 最优**的权重至 `runs/train/weights/best.pt`；

* 训练曲线（分类损失、边界框回归损失、mAP@50、mAP@\[0.5:0.95]）实时可视化并保存至 `runs/train/`；

* 混淆矩阵、PR 曲线、F1 曲线等评估图自动生成。

### 6.2 模型验证

在测试集上评估已训练模型：



```
python val.py \
  --data ./datasets/SPFD-4213/datasets.yaml \
  --weights ./runs/train/weights/best.pt \
  --device cuda:0
```

### 6.3 模型预测

对单张或单段视频进行跌倒检测：



```
python predict.py \
  --source ./examples/basketball_fall.jpg \
  --weights ./runs/train/weights/best.pt \
  --img_size 640 \
  --conf_threshold 0.5 \
  --device cuda:0
```

#### 预测输出示例：



```
输入图像路径：./examples/basketball_fall.jpg
检测目标 1：跌倒（Fall），置信度 0.987，边界框 [x1, y1, x2, y2]
检测目标 2：站立（Standing），置信度 0.962，边界框 [x1, y1, x2, y2]
```

### 6.4 预训练权重

提供基于 SPFD-4213 数据集训练完成的最优权重，可直接用于推理或微调：



* 本地权重文件：`runs/train/weights/best.pt`；

* 百度网盘分享：链接与提取码将在论文录用后公开；

* 适用场景：仅针对 "站立 / 跌倒" 二分类，若需扩展其他体育异常行为（如晕厥、冲撞），建议基于此权重微调（冻结 HRNet 浅层，仅训练检测头与 DCVN 输出层，可减少 50% 以上训练数据量）。

## 7. 项目文件结构



```
yolov13n-hrnet-dcvn-fall-detection/
├── datasets/                          # 数据集目录
│   ├── SPFD-4213/                     # 自建大学生体育课跌倒数据集
│   ├── le2i-with-upright-and-fall/    # 室内公开数据集
│   └── falldown-detection-f8xtb/      # 室外公开数据集
├── models/                            # 模型定义
│   ├── yolov13n_hrnet_dcvn.yaml       # 整体模型结构配置
│   ├── hrnet_backbone.py              # 定制化 HRNet 高分辨率骨干（并行多分支 + 跨尺度融合）
│   ├── dcvn_module.py                 # DCVN 多尺度上下文增强模块（3×3/5×5/7×7 并行 + 通道注意力）
│   └── neck_head.py                   # FullPAD 颈部网络与一体化检测头
├── examples/                           # 预测示例图像
│   ├── runway_fall.jpg
│   ├── basketball_fall.jpg
│   └── indoor_fall.jpg
├── utils/                             # 工具函数
│   ├── dataset.py                     # 数据集加载与预处理（Mosaic、MixUp、随机仿射等增强）
│   ├── metrics.py                     # mAP、Precision、Recall、F1 评估
│   └── visualize.py                   # 训练曲线、混淆矩阵、PR 曲线可视化
├── train.py                           # 模型训练脚本
├── val.py                             # 模型验证脚本
├── predict.py                         # 模型推理脚本
└── README.md                          # 项目说明文档（本文档）
```

## 8. 已知问题与注意事项



1. **类别扩展**：当前模型与权重仅针对 "站立 / 跌倒" 二分类，若新增异常行为类别（如晕厥、冲撞、运动损伤），需补充对应标注数据并重新训练（建议每类样本量 ≥ 500 张）；

2. **体育项目覆盖**：数据集目前覆盖跑道、篮球场与室内常规运动，未包含游泳、击剑、攀岩、乒乓球、羽毛球、武术、跆拳道等项目，直接迁移至上述场景前建议补充对应数据微调；

3. **极端天气**：模型未在雾天、暴雨、积水反光等极端户外条件下充分验证，雨天 / 雾天部署前应通过数据增强模拟上述场景以提升鲁棒性；

4. **输入分辨率**：图像默认 resize 至 640×640，若原始分辨率过低（< 320×320），远端小尺度目标特征可能丢失，建议监控摄像头输出分辨率 ≥ 720p；

5. **CUDA 兼容**：若 PyTorch 安装时出现 CUDA 不兼容，可切换 CPU 版本（所有脚本 `--device` 改为 `cpu`），但推理延迟将升至 5–10 ms / 帧，训练耗时大幅增加；

6. **损伤类型细分**：当前模型仅检测跌倒事件，未结合损伤类型（骨折、扭伤、擦伤）细分，后续可引入骨骼关键点与损伤特征关联模型实现伤情预判。

## 9. 引用与联系方式

### 9.1 引用方式

论文处于投刊阶段，正式发表后将更新 BibTeX 引用格式，当前可临时引用：



```
@article{chen2026fall,
  title={Context-Aware Deep Learning for Fall Detection of College Students in Physical Education Activities},
  author={Chen, Yicheng and [其他作者，待发表时补充]},
  journal={[期刊名称，待录用后补充]},
  year={2026},
  note={Manuscript submitted for publication}
}
```

### 9.2 联系方式

若遇到代码运行问题或学术交流需求，请联系：



* 邮箱：chenyicheng@huuc.edu.cn

* GitHub Issue：直接在本仓库提交 Issue，将在 1–3 个工作日内回复。