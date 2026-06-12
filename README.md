# WaveDSNet: Wavelet Dynamic Convolution with Dual-Stream Synergistic Fusion Network

This is the official PyTorch implementation of the paper **"WaveDSNet: Wavelet Dynamic Convolution with Dual-Stream Synergistic Fusion Network for SAR Water Change Detection"**.

This project proposes an end-to-end SAR water change detection network and achieves state-of-the-art performance on the newly constructed large-scale benchmark dataset **XDU-SWCD**.

## 🏗️ Network Architecture

<img src="assert\architecture.png" alt="image-20251226142349516" style="zoom: 25%;" />
WaveDSNet adopts a siamese encoder structure that processes bi-temporal SAR images through three synergistic modules to address specific challenges in SAR change detection.

## 🚀 Introduction

SAR water change detection faces challenges including speckle noise interference, insufficient semantic interaction, and weak boundary ambiguity. To address these issues, we propose **WaveDSNet**, which contains the following core components:

- **WTCSwinTransformer Backbone**: Combining Wavelet Transform with CSwin Transformer, this backbone utilizes the **WNS (Wavelet-based Noise Suppression)** module to decouple speckle noise from structural semantics in the frequency domain.

  <img src="assert\WTCSWinTransformer.png" alt="image-20251226142349516" style="zoom: 25%;" />

  **Workflow**: First, the CSwin Transformer block processes input images using cross-shaped window attention mechanism to capture long-range dependencies and understand scene semantics. Then, feature maps enter the WNS module where Discrete Wavelet Transform decomposes them into four subbands: one low-frequency subband representing overall contours and uniform areas, and three high-frequency subbands containing details, edges, and noise. For each subband, adaptive convolutional kernels are generated - acting as smoothers in low-frequency regions to suppress random noise, and edge enhancers in boundary regions. This content-aware mechanism avoids detail loss from uniform denoising. Processed subbands are merged through inverse wavelet transform, ensuring noise suppression while preserving important structures.

- **CSDI (Complementary Semantic Difference Interaction)**: This module promotes complementary fusion between semantic features and difference features through bidirectional attention mechanism.

  <img src="assert\CSDI.png" alt="CSDI" style="zoom: 25%;" />

  **Workflow**: Input multi-level semantic features and difference features are projected into query, key, and value vectors through convolutional layers. For computational efficiency, feature maps are divided into local windows where attention is computed. Bidirectional attention calculation includes:

  - Direction 1 (semantic to difference): Interaction between semantic queries and difference keys evaluates how semantic context influences change regions.
  - Direction 2 (difference to semantic): Interaction between difference queries and semantic keys assesses how change signals modify semantic understanding. 

  Outputs from both directions are concatenated and adaptively fused through gating mechanism, where gate weights determine the reliance on semantic or difference information at each position. The fused features emphasize change-relevant regions while reducing false alarms.

- **BASE (Boundary-Aware Supervised Extraction)**: This module imposes explicit geometric constraints to optimize boundary localization accuracy.

  <img src="assert\BASE.png" alt="BASE" style="zoom: 25%;" />
  
  The main branch focuses on pixel-level classification (changed/unchanged) using convolutional layers and upsampling operations to gradually restore resolution and output change probability maps. The auxiliary branch specifically captures boundaries by applying early upsampling to amplify feature maps, then using convolutional layers to extract edge responses. Both branches share features but have different focuses - the change branch ensures overall region accuracy while the edge branch forces the network to learn boundary topology. During training, the loss function combines change loss and edge loss with balanced weights. The final change map integrates results from both branches, producing smoother and more continuous boundaries.
  
  

## 🌟 Main Features

- **Frequency Domain Denoising**: Introduces wavelet dynamic convolution for adaptive processing of multiplicative noise in SAR images

- **Dual-Stream Synergy**: Deep interaction between semantic and difference streams effectively reduces false alarms.

- **High-Precision Boundaries**: Specialized edge supervision branch resolves ambiguity in water-land interfaces.

- **SOTA Performance**: Significantly outperforms existing methods on both XDU-SWCD and public datasets

  

## 📊 XDU-SWCD Dataset

To address data scarcity, we constructed the **XDU-SWCD** dataset containing diverse hydrological scenarios.

<img src="assert\dataset.png" alt="dataset" style="zoom: 25%;" />

|        **Region**        |    **Time Period**     |  Sensor  |   Size    | Resolution | Band | Polarization | Classes | Samples |
| :----------------------: | :--------------------: | :------: | :-------: | :--------: | :--: | :----------: | :-----: | :-----: |
|        **Xi'an**         | 2019-01-05  2019-06-29 | **GF-3** | 7199×7516 |     3m     |  C   |      HH      |    2    |   870   |
|   **Xi'an & Xianyang**   | 2019-06-29  2020-09-23 | **GF-3** | 5490×8250 |     3m     |  C   |      HH      |    2    |   726   |
|       **Xianyang**       | 2019-06-29  2019-11-04 | **GF-3** | 6219×6473 |     3m     |  C   |      HH      |    2    |   650   |
| **Xianyang & Tongchuan** | 2019-12-03  2020-09-23 | **GF-3** | 7393×8266 |     3m     |  C   |      HH      |    2    |   957   |

- **Sensor**: GF-3 (C-band SAR)
- **Resolution**: 3 meters (HH polarization)
- **Scale**: 3,203 sample pairs (covering river basins in Xi'an, Xianyang, Tongchuan)**Challenges**: Includes dry/wet season changes and challenging negative samples (like paddy fields)

Download:：[Baidu Netdisk](https://pan.baidu.com/s/1lzr1wbQqFq-dxxWaW1NHhg?pwd=vf6e)  | [Google Drive](https://drive.google.com/file/d/1YcoNoKniScIT8QeggGRZXJCP6zRP-fp_/view?usp=drive_link)

## 🛠️ Installation

This project is developed based on PyTorch.

```bash
# Clone repository
git clone https://github.com/gosling123456/WaveDSNet.git
cd WaveDSNet

# Create virtual environment
conda create -n waved python=3.10
conda activate waved

# Install dependencies
pip install -r requirements.txt
```

## ⚡  Quick Start

### 1.  Data Preparation

Organize data in the following structure:

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

### 2. Training

```bash
python train.py --dataset XDU-SWCD --batch_size 16 --lr 1e-3 --epochs 100
```

*Note: Hyperparameters reference Experiment A section - LR=1e-3, Batch=16, GPUs=2×RTX 5090, Optimizer=AdamW*

)

### 3. Testing

```bash
python test.py --checkpoint checkpoints/best_model.pth --output_dir results/
```

## 📈 Experimental Results

### 1. Performance on Xi'an Dataset

<img src="assert\xian.png" alt="xian_result" style="zoom: 25%;" />

<img src="assert\Xian_result.png" alt="Xian_result" style="zoom: 100%;" />



### 2.  Zero-shot Generalization on Public Datasets

<img src="assert\public.png" alt="public" style="zoom: 25%;" />



<img src="assert\public_result.png" alt="public_result" style="zoom: 100%;" />

## 🔗 Citation

If you find this project helpful for your research, please consider citing our paper:

```
###
```
