# WaveDSNet: Wavelet Dynamic Convolution with Dual-Stream Synergistic Fusion Network

这是论文 **"WaveDSNet: Wavelet Dynamic Convolution with Dual-Stream Synergistic Fusion Network for SAR Water Change Detection"** 的官方 PyTorch 实现代码。

该项目提出了一种端到端的 SAR 水体变化检测网络，并在新建的大规模基准数据集 **XDU-SWCD** 上取得了 SOTA 性能 。

## 🏗️ 网络架构 (Architecture)

<img src="assert\architecture.png" alt="image-20251226142349516" style="zoom: 25%;" />
WaveDSNet 采用孪生编码器结构，

## 🚀 简介 (Introduction)

SAR 水体变化检测面临着斑点噪声干扰、语义交互不足以及弱边界模糊等挑战。为了解决这些问题，我们提出了 **WaveDSNet**。该网络包含以下核心组件：

- **WTCSwinTransformer Backbone**: 结合小波变换（Wavelet Transform）与 CSwin Transformer，利用 **WNS (Wavelet-based Noise Suppression)** 模块在频域中解耦斑点噪声与结构语义 。

  <img src="assert\WTCSWinTransformer.png" alt="image-20251226142349516" style="zoom: 25%;" />

  **工作流：**首先，使用CSwin Transformer块对输入图像进行处理。该块采用十字形窗口注意力机制，能捕获长距离依赖关系，理解整个场景的语义（如水域、陆地）；随后，特征图进入小波去噪模块（WNS）。这里，应用离散小波变换将特征图分解为四个子带：1个低频子带：代表图像的整体轮廓和均匀区域（如平静水面）和3个高频子带：代表细节和边缘，但也包含噪声。针对每个子带，网络生成自适应的卷积核。在低频区域，卷积核类似平滑器，抑制随机噪声；在高频边界区域，卷积核增强边缘响应。这种“内容感知”机制允许网络在不同区域采用不同策略，避免一刀切去噪导致的细节丢失。处理后的子带通过逆小波变换合并，得到去噪后的特征图。这一步确保噪声被抑制，同时重要结构（如细小河流）得以保留。

- **CSDI (Complementary Semantic Difference Interaction)**: 通过双向注意力机制，促进语义特征与差异特征的互补融合 。

  <img src="assert\CSDI.png" alt="CSDI" style="zoom: 25%;" />

  工作流：输入多级语义特征和差异特征，分别通过卷积层生成查询、键和值向量。这相当于将特征转换为适合注意力计算的形式。为降低计算量，特征图被划分为局部窗口，注意力在窗口内计算，保持效率。后经过双向注意力计算：

  ​	方向一（语义到差异）：用语义特征的查询向量与差异特征的键向量交互，评估语义上下文如何影响变化区域。

  ​	方向二（差异到语义）：用差异特征的查询向量与语义特征的键向量交互，评估变化信号如何修正语义理解。

  两个方向的输出被拼接，然后通过门控机制（基于卷积和权重生成）自适应融合。门控权重决定每个位置更依赖语义还是差异信息，例如，在边界区域赋予差异更高权重以增强灵敏度。融合后的特征强调变化相关区域，减少噪声引起的误报（如季节性强反射）。

- **BASE (Boundary-Aware Supervised Extraction)**: 施加显式的几何约束，优化边界定位精度 。

  <img src="assert\BASE.png" alt="BASE" style="zoom: 25%;" />
  
  主分支专注于像素级分类（变化/未变化）。它使用卷积层和上采样操作，逐步恢复分辨率，输出变化概率图。辅助分支专门捕获边界。为避免细节丢失，它早期应用上采样放大特征图，再用卷积提取边缘响应（如梯度变化）。两个分支共享特征，但各有侧重。变化分支确保整体区域准确性，边缘分支强制网络学习边界拓扑（如连续性）。训练时，损失函数结合变化损失和边缘损失，通过权重平衡两者贡献。最终变化图融合了分支结果，边界更光滑、连贯。例如，在弱边界处（如浅水区），边缘监督提供额外约束，减少断裂。
  
  

## 🌟 主要特性 (Main Features)

- **频域去噪**: 引入小波动态卷积，针对 SAR 图像的乘性噪声进行自适应处理。

- **双流协同**: 语义流与差异流的深度交互，有效减少误报（如稻田、阴影）。

- **高精度边界**: 专门的边缘监督分支，解决水陆交界处的模糊问题。

- **SOTA 性能**: 在 XDU-SWCD 及公共数据集上均显著优于现有方法（如 SFEARNet, DDRL 等）。

  

## 📊 数据集 (XDU-SWCD Dataset)

为了解决数据稀缺问题，我们构建了 **XDU-SWCD** 数据集 。

<img src="assert\dataset.png" alt="dataset" style="zoom: 25%;" />

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

下载链接：[百度网盘](https://pan.baidu.com/s/1lzr1wbQqFq-dxxWaW1NHhg?pwd=vf6e)  | [Google Drive](https://drive.google.com/file/d/1YcoNoKniScIT8QeggGRZXJCP6zRP-fp_/view?usp=drive_link)
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

### 1. 在西安数据集上的测试结果

![xian](assert\xian.png)



![Xian_result](assert\Xian_result.png)

| **Method**           | **IoU**   | **F1-score** | **Precision (PA1)** | **Recall** |
| -------------------- | --------- | ------------ | ------------------- | ---------- |
| FC-EF                | 39.26     | 56.38        | 50.08               | 64.51      |
| SFEARNet (SOTA)      | 41.20     | 25.94*       | 93.50               | 26.42      |
| DDRL (SOTA)          | 47.95     | 31.54        | 71.07               | 36.18      |
| **WaveDSNet (Ours)** | **56.16** | **71.92**    | **77.75**           | **66.91**  |

### 2. 在公开数据集上的零样本泛化性测试结果

![public](assert\public.png)



![public_result](assert\public_result.png)

## 🔗 引用 (Citation)

如果您觉得本项目对您的研究有所帮助，请考虑引用我们的论文：

```
###
```

## 📄 开源协议 (License)

本项目遵循 [MIT License](https://www.google.com/search?q=LICENSE).

------



**联系方式**: 如有问题，请联系 `zlren@xidian.edu.cn` 或提 Issue 。
