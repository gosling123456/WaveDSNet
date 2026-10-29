# WaveDSNet: A Wavelet-Enhanced Discrepancy Semantic Network for Water Change Detection in SAR Imagery

This is the official PyTorch implementation of the paper **[WaveDSNet: A Wavelet‑Enhanced Discrepancy Semantic Network for Water Change Detection in SAR Imagery](https://ieeexplore.ieee.org/document/11703712)**.

## News
- **29 OCt 2026**: The dataset of this study is publicly released.
- **30 Sep 2026**: Current manuscript version updated.
- **22 Sep 2026**: Paper published online.
- **17 Sep 2026**: Manuscript accepted.
- **08 Sep 2026**: Submission of the second revised manuscript.
- **16 Aug 2026**: Submission of the first revised manuscript.
- **27 Apr 2026**: Initial manuscript submitted.


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

This project proposes an end-to-end SAR water change detection network and achieves state-of-the-art performance on the newly constructed large-scale benchmark dataset!
Download:：[Baidu Netdisk](https://pan.baidu.com/s/1lzr1wbQqFq-dxxWaW1NHhg?pwd=vf6e)  | [Google Drive](https://drive.google.com/file/d/1YcoNoKniScIT8QeggGRZXJCP6zRP-fp_/view?usp=drive_link)


## 🛠️ Installation

This project is developed based on PyTorch.

```bash
# Clone repository
git clone https://github.com/gosling123456/WaveDSNet.git
cd WaveDSNet

# Create virtual environment
conda create -n wavedsnet python=3.10
conda activate wavedsnet

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
python main.py
```


## 🔗 Citation

If you find this project helpful for your research, please consider citing our paper:

```
@article{gao2026wavedsnet,
 author={Gao, Shulin and Ren, Zhongle and Wang, Kai and Feng, Guangyao and Li, Xi and Zang, Weicheng and Hou, Biao and Li, Weibin and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={WaveDSNet: A Wavelet-Enhanced Discrepancy Semantic Network for Water Change Detection in SAR Imagery}, 
  year={2026},
  volume={64},
  number={},
  pages={5216517-5216517},
  doi={10.1109/TGRS.2026.3736079}}
```
