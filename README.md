#  DSFFNet

##  Overview

**DSFFNet** is a high-precision defect segmentation framework designed for industrial surface inspection.  
The network leverages dual-stream feature extraction and frequency-aware fusion mechanisms to enhance boundary perception and improve segmentation accuracy.

**Copper strip defect segmentation diagram**：
<p align="center">
  <img src="https://github.com/user-attachments/assets/fdac1dd5-283e-4ccf-9aaa-4251c20e802c" width="500"/>
</p>

---

##  Key Features

- Dual-Stream Architecture  
- Frequency-Aware Fusion  
- High-Precision Segmentation  
- Lightweight & Scalable  

---

##  1. Hardware & Software Requirements

###  Hardware Requirements

| Component | Recommended | Minimum |
|----------|------------|--------|
| GPU | ≥ 8GB VRAM | ≥ 4GB |
| CPU | 8+ cores | 4 cores |
| RAM | 32 GB | 16 GB |
| Storage | ≥ 50 GB SSD | ≥ 20 GB SSD |

---

### 💻 Software Requirements

- Python 3.11
- CUDA 11.8
- PyTorch 2.2.2

---

##  2. Installation & Environment Setup

### 2.1 Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2.2 Create Environment

```bash
conda create -n dsffnet python=3.11 -y
conda activate dsffnet
```

### 2.3 Install Dependencies

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -r requirements.txt
```

---

## 📁 3. Project Structure

```bash
DSFFNet/
├── component/
├── structure/
├── train.py
├── conv.py
├── app.py
├── DCTcatch module.py
└── requirements.txt
```

---

##  4. Training

```bash
python train.py --config config.yaml
```

---

##  License
This README provides clear instructions for setting up the environment with your specific package versions, including the custom `flash_attn` wheel file. Let me know if you need any adjustments!

