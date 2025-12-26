# WaveDSNet: Wavelet Dynamic Convolution with Dual-Stream Synergistic Fusion Network

这是论文 **"WaveDSNet: Wavelet Dynamic Convolution with Dual-Stream Synergistic Fusion Network for SAR Water Change Detection"** 的官方 PyTorch 实现代码。

该项目提出了一种端到端的 SAR 水体变化检测网络，并在新建的大规模基准数据集 **XDU-SWCD** 上取得了 SOTA 性能 。

## 🏗️ 网络架构 (Architecture)

![architecture](assert\architecture.png)

WaveDSNet 采用孪生编码器结构，在每个 Stage 嵌入 WNS 模块，随后通过 CSDI 进行多尺度特征融合，最后由 BASE 模块输出变化图和边缘图。

## 🚀 简介 (Introduction)

SAR 水体变化检测面临着斑点噪声干扰、语义交互不足以及弱边界模糊等挑战。为了解决这些问题，我们提出了 **WaveDSNet**。该网络包含以下核心组件：

- **WTCSwinTransformer Backbone**: 结合小波变换（Wavelet Transform）与 CSwin Transformer，利用 **WNS (Wavelet-based Noise Suppression)** 模块在频域中解耦斑点噪声与结构语义 。

  <img src="assert\WTCSWinTransformer.png" alt="image-20251226142349516" style="zoom: 25%;" />

- **CSDI (Complementary Semantic Difference Interaction)**: 通过双向注意力机制，促进语义特征与差异特征的互补融合 。

  <img src="assert\CSDI.png" alt="CSDI" style="zoom: 25%;" />

- **BASE (Boundary-Aware Supervised Extraction)**: 施加显式的几何约束，优化边界定位精度 。

  <img src="D:\Master\Research\工作代码开源\WaveDSNet\assert\BASE.png" alt="BASE" style="zoom: 25%;" />

此外，我们构建了 **XDU-SWCD** 数据集，这是目前用于 SAR 水体变化检测的大规模高分辨率基准 。



## 🌟 主要特性 (Main Features)

- **频域去噪**: 引入小波动态卷积，针对 SAR 图像的乘性噪声进行自适应处理。

- **双流协同**: 语义流与差异流的深度交互，有效减少误报（如稻田、阴影）。

- **高精度边界**: 专门的边缘监督分支，解决水陆交界处的模糊问题。

- **SOTA 性能**: 在 XDU-SWCD 及公共数据集上均显著优于现有方法（如 SFEARNet, DDRL 等）。

  

## 📊 数据集 (XDU-SWCD Dataset)

为了解决数据稀缺问题，我们构建了 **XDU-SWCD** 数据集 。

|     **区域**      |        **时间**        | **传感器** | **尺寸**  | **分辨率** | **波段** | **极化方式** | **类别数** | **裁剪块数** |
| :---------------: | :--------------------: | :--------: | :-------: | :--------: | :------: | :----------: | :--------: | :----------: |
|     **西安**      | 2019-01-05  2019-06-29 |  **GF-3**  | 7199×7516 |     3m     |    C     |      HH      |     2      |     870      |
| **西安**&**咸阳** | 2019-06-29  2020-09-23 |  **GF-3**  | 5490×8250 |     3m     |    C     |      HH      |     2      |     726      |
|     **咸阳**      | 2019-06-29  2019-11-04 |  **GF-3**  | 6219×6473 |     3m     |    C     |      HH      |     2      |     650      |
| **咸阳**&**铜川** | 2019-12-03  2020-09-23 |  **GF-3**  | 7393×8266 |     3m     |    C     |      HH      |     2      |     957      |

- **传感器**: GF-3 (C-波段 SAR)
- **分辨率**: 3米 (HH 极化)
- **规模**: 3,203 对样本 (覆盖西安、咸阳、铜川等流域)
- **挑战**: 包含枯水/丰水期变化，以及具有迷惑性的“难负样本”（如稻田、灌溉区）。

下载链接: [XDU-SWCD Download Link](https://www.google.com/search?q=https://github.com/gosling123456/WaveDSNet) *(XXXXX)*

## 🛠️ 安装与环境 (Installation)

本项目基于 PyTorch 开发。

```
# 克隆项目
git clone https://github.com/gosling123456/WaveDSNet.git
cd WaveDSNet

# 创建虚拟环境
conda create -n waved python=3.10
conda activate waved

# 安装依赖
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## ⚡ 快速开始 (Usage)

### 1. 数据准备

请按照以下结构组织数据：

```
data/
├── XDU-SWCD/
│   ├── A/
│   ├── B/
│   ├── label/
│   ├── list/
│   │   ├── train.txt/
│   │   ├── test.txt/
│   │   ├── val.txt/
```

### 2. 训练 (Training)

Bash

```
python train.py --dataset XDU-SWCD --batch_size 16 --lr 1e-3 --epochs 100
```



*(注：超参数参考论文 Exper A 部分：LR=1e-3, Batch=16, GPUs=2x RTX 5090, Optimizer=AdamW)* 



### 3. 测试 (Testing)

Bash

```
python test.py --checkpoint checkpoints/best_model.pth --output_dir results/
```

## 📈 实验结果 (Results)

WaveDSNet 在 **XDU-SWCD** 数据集上的性能对比（变化类指标）：

| **Method**           | **IoU**   | **F1-score** | **Precision (PA1)** | **Recall** |
| -------------------- | --------- | ------------ | ------------------- | ---------- |
| FC-EF                | 39.26     | 56.38        | 50.08               | 64.51      |
| SFEARNet (SOTA)      | 41.20     | 25.94*       | 93.50               | 26.42      |
| DDRL (SOTA)          | 47.95     | 31.54        | 71.07               | 36.18      |
| **WaveDSNet (Ours)** | **56.16** | **71.92**    | **77.75**           | **66.91**  |



## 🔗 引用 (Citation)

如果您觉得本项目对您的研究有所帮助，请考虑引用我们的论文：

```
###
```

## 📄 开源协议 (License)

本项目遵循 [MIT License](https://www.google.com/search?q=LICENSE).

------



**联系方式**: 如有问题，请联系 `zlren@xidian.edu.cn` 或提 Issue 。
