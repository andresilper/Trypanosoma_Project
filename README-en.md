# 🔬 Automated Detection of *Trypanosoma cruzi* in Microscopy Images

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Automatic detection of *Trypanosoma cruzi* parasites in microscopy images using Deep Learning with Transfer Learning (VGG16 and MobileNetV2).

---

## 📋 Table of Contents

- [About](#about)
- [Problem & Motivation](#problem--motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Challenges & Solutions](#challenges--solutions)
- [Future Work](#future-work)
- [Author](#author)

---

## 🎯 About

This project implements a **binary classification system** to automatically detect the presence of *Trypanosoma cruzi* parasites in microscopy images. *T. cruzi* is the causative agent of Chagas disease, a neglected tropical disease affecting millions of people in the Americas.

The goal is to assist healthcare professionals in rapid and accurate diagnosis through **artificial intelligence**, reducing manual analysis time and increasing detection accuracy.

### ✨ Highlights

- 🏆 **AUC of 0.989** on validation set
- 📊 **Average accuracy of 93.8%** on real test data
- 🎯 **Sensitivity of 94.2%** (parasite detection rate)
- ✅ **Specificity of 92.7%** (low false positive rate)
- 🚀 Comparison between VGG16 and MobileNetV2
- 💪 Robust overfitting solution through regularization techniques

---

## 🔍 Problem & Motivation

### Clinical Challenge

Traditional *T. cruzi* detection requires:
- ⏱️ Time-consuming manual analysis by specialized microscopists
- 👁️ High level of attention and expertise
- 🔬 Identification of small parasites (~20μm) in large samples
- ⚠️ Risk of false negatives in low parasitemia cases

### Proposed Solution

A deep learning model that:
- ✅ Automates initial slide screening
- ✅ Reduces analysis time
- ✅ Maintains high sensitivity to avoid missing positive cases
- ✅ Provides decision support for healthcare professionals

---

## 📊 Dataset

### Characteristics

- **Resolution:** 224×224 pixels
- **Classes:** 
  - `Positive (1)`: Presence of *T. cruzi*
  - `Negative (0)`: Absence of parasite
- **Split:**
  - 🏋️ Train: ~1,600 images
  - 📝 Validation: ~700 images
  - 🧪 Test: 5 independent slides (18, 19, 20, 23, 24)

### Preprocessing

```python
# Normalization with ImageNet statistics
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Data Augmentation (On-the-fly)

To increase model robustness, we apply random transformations during training:

- ↔️ Horizontal and vertical flip (p=0.5)
- 🔄 Random rotation (±15°)
- 📐 Affine transformation (translation, scale, shear)
- 🎨 Color jitter (brightness, contrast, saturation, hue)
- ✂️ Random crop with scale 0.8-1.0

---

## 🧠 Methodology

### 1. Transfer Learning

We use ImageNet pre-trained models as feature extractors:
- **VGG16**: 138M parameters, classic and robust architecture
- **MobileNetV2**: 3.5M parameters, efficient for mobile devices

### 2. Feature Freezing

```python
# Freeze convolutional layers
for param in model.features.parameters():
    param.requires_grad = False
```

**Rationale:** With only ~1,600 training images, training all layers would cause severe overfitting.

### 3. Classifier Architecture

```python
model.classifier = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(256, 1)
)
```

**Simplified design** to prevent overfitting on small datasets.

### 4. Optimization & Regularization

| Technique | Value | Rationale |
|---------|-------|-----------|
| **Optimizer** | AdamW | Better than Adam for regularization |
| **Learning Rate** | 5×10⁻⁵ | Balance between convergence and stability |
| **Weight Decay** | 1×10⁻⁴ | L2 regularization |
| **Dropout** | 0.5 | Prevents neuron co-adaptation |
| **Batch Size** | 32 | Compromise between memory and convergence |

### 5. Training Strategies

- **Early Stopping:** Patience of 7 epochs to prevent overfitting
- **Learning Rate Scheduler:** ReduceLROnPlateau (reduces LR when val_loss plateaus)
- **Loss Function:** BCEWithLogitsLoss (numerically stable)

---

## 🏗️ Model Architectures

### VGG16

**Features:**
- 📦 138 million parameters
- 🎯 Classic and well-established architecture
- 🔧 Deep convolutional layers (13 conv + 3 FC)

**Advantages:**
- ✅ High learning capacity
- ✅ Robust features for classification
- ✅ Extensively studied and tested

**Disadvantages:**
- ⚠️ Heavy model (>500MB)
- ⚠️ Slower inference

### MobileNetV2

**Features:**
- 📦 3.5 million parameters
- 🚀 Optimized for efficiency
- 🔧 Depthwise separable convolutions

**Advantages:**
- ✅ Lightweight model (~14MB)
- ✅ Fast inference
- ✅ Ideal for mobile devices

**Disadvantages:**
- ⚠️ Lower capacity than VGG16
- ⚠️ May have slightly lower performance

---

## 📈 Results

### VGG16 - Validation Metrics

- **AUC-ROC:** 0.989
- **Best Val Loss:** 0.2763
- **Epochs trained:** 29 (early stopping)

### VGG16 - Real Test Performance

| Slide | Samples | Accuracy | Sensitivity | Specificity | TP | FP | TN | FN |
|-------|---------|----------|-------------|-------------|----|----|----|----|
| **18** | 320 | **86.3%** | 75.6% | **98.7%** | 130 | 2 | 146 | 42 |
| **19** | 167 | **94.6%** | **98.9%** | 90.0% | 86 | 8 | 72 | 1 |
| **20** | 248 | **93.2%** | 96.9% | 89.1% | 125 | 13 | 106 | 4 |
| **23** | 230 | **97.8%** | **100%** | 95.8% | 112 | 5 | 113 | 0 |
| **24** | 936 | **97.4%** | **99.8%** | 95.2% | 457 | 23 | 455 | 1 |
| **Average** | - | **93.8%** | **94.2%** | **92.7%** | - | - | - | - |

### 📊 Results Interpretation

**🎯 Sensitivity (Recall) - 94.2%:**
- Model detects **94 out of 100** parasites present
- Crucial for diagnosis: **few false negatives**
- Slide 23: 100% detection rate!

**✅ Specificity - 92.7%:**
- Model correctly identifies **93 out of 100** negative samples
- Reduces manual review workload for false positives
- Slide 18: 98.7% - excellent reliability

**📈 Overall Accuracy - 93.8%:**
- Consistent performance across 4 out of 5 slides (>93%)
- Slide 18: 86.3% (possibly different characteristics)

### 🏆 Highlights by Slide

- **Slide 23:** Perfect performance (100% sensitivity, 0 false negatives)
- **Slide 24:** Largest dataset (936 samples), maintained 99.8% sensitivity
- **Slide 18:** 98.7% specificity (only 2 false positives)

### MobileNetV2 - Results

> 🚧 **Under development** - Results will be added soon

---

## 📁 Project Structure

```
Projeto-Trypanossoma/
│
├── README.md                          # This file
├── requirements.txt                   # Project dependencies
├── LICENSE                            # MIT License
│
├── data/                              # Data (10MB, included in repo)
│   ├── train/
│   │   ├── positivo/
│   │   └── negativo/
│   ├── val/
│   │   ├── positivo/
│   │   └── negativo/
│   └── test/
│       ├── lamina_18/
│       ├── lamina_19/
│       ├── lamina_20/
│       ├── lamina_23/
│       └── lamina_24/
│
├── models/                            # Trained models
│   ├── vgg16_best_model.pth
│   └── mobilenetv2_best_model.pth
│
├── notebooks/                         # Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_vgg16_training.ipynb
│   └── 03_mobilenetv2_training.ipynb
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── dataset.py                     # Dataset and DataLoader
│   ├── models.py                      # Model definitions
│   ├── train.py                       # Training loop
│   ├── evaluate.py                    # Evaluation and metrics
│   └── utils.py                       # Utility functions
│
├── results/                           # Results and visualizations
│   ├── training_curves/
│   ├── confusion_matrices/
│   └── roc_curves/
│
└── docs/                              # Additional documentation
    └── methodology.md
```

---

## 🚀 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Projeto-Trypanossoma.git
cd Projeto-Trypanossoma
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset

The complete dataset (10MB) is **included in this repository** for reproducibility.

### 4. Train the Model

#### VGG16

```bash
python src/train.py --model vgg16 --epochs 50 --batch-size 32 --lr 5e-5
```

#### MobileNetV2

```bash
python src/train.py --model mobilenetv2 --epochs 50 --batch-size 32 --lr 5e-5
```

### 5. Evaluate on Test Set

```bash
python src/evaluate.py --model vgg16 --checkpoint models/vgg16_best_model.pth --test-dir data/test/
```

### 6. Make Predictions on New Images

```python
from src.models import load_model
from src.utils import predict_image

model = load_model('vgg16', 'models/vgg16_best_model.pth')
result = predict_image(model, 'path/to/image.jpg')

print(f"Prediction: {'Positive' if result > 0.5 else 'Negative'}")
print(f"Confidence: {result:.2%}")
```

---

## 📦 Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=9.5.0
tqdm>=4.65.0
```

**System:**
- Python 3.8+
- CUDA 11.8+ (optional, for GPU)
- 8GB RAM minimum (16GB recommended)

---

## 💡 Challenges & Solutions

### 🔥 Challenge 1: Severe Overfitting

**Symptoms:**
- Train loss decreasing, but validation loss increasing
- Growing gap between training and validation

**Identified Causes:**
1. ❌ Pre-generated augmented dataset (fixed images)
2. ❌ BatchNorm with small dataset causing noise
3. ❌ Learning rate too high (1e-4)
4. ❌ All VGG layers trainable

**Implemented Solutions:**
1. ✅ On-the-fly data augmentation (infinite variations)
2. ✅ Removed BatchNorm from classifier
3. ✅ Reduced learning rate to 5e-5
4. ✅ Froze VGG features
5. ✅ Simplified classifier (512 → 256 features)
6. ✅ Dropout of 0.5
7. ✅ Weight decay of 1e-4

**Result:**
- Stable val loss converging with train loss
- Minimal gap between curves
- AUC of 0.989

### 🐛 Challenge 2: Label Misalignment

**Symptom:**
- Inconsistent and unexpected results

**Cause:**
- Labels and images in different orders

**Solution:**
```python
# Use DataFrames to ensure alignment
df = pd.DataFrame({
    'filename': sorted(image_paths),
    'label': corresponding_labels
})
```

### ⚡ Challenge 3: Slow Convergence

**Symptom:**
- Model not improving after several epochs

**Cause:**
- Learning rate too low (1e-5) with BatchNorm

**Solution:**
- Learning rate scheduler (ReduceLROnPlateau)
- Start with higher LR (5e-5), reduce when plateauing

---

## 🔮 Future Work

- [ ] Implement and compare MobileNetV2
- [ ] Test other architectures (ResNet, EfficientNet)
- [ ] Implement model ensemble
- [ ] Create web interface with Gradio/Streamlit
- [ ] Parasite segmentation (exact localization)
- [ ] Automatic parasitemia quantification
- [ ] Detection of other protozoa
- [ ] Mobile device deployment (TFLite/ONNX)
- [ ] Explainability with Grad-CAM
- [ ] Dataset augmentation with GAN techniques

---

## 📚 References

1. World Health Organization. (2023). Chagas disease (American trypanosomiasis)
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition
3. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks
4. He, K., et al. (2016). Deep Residual Learning for Image Recognition

---

## 👨‍💻 Author

**[Your Name]**

- 🎓 PhD Candidate in [Your Field]
- 💼 LinkedIn: [your-linkedin](https://linkedin.com/in/your-profile)
- 📧 Email: your.email@example.com
- 🐙 GitHub: [@your-username](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset provided by [Institution/Laboratory]
- Computational infrastructure: [GPU/Cloud provider]
- Advisor: [Advisor name]

---

## 📊 Project Status

![Status](https://img.shields.io/badge/Status-Active-success)

**Last update:** January 2026

---

## 🇧🇷 Portuguese Version

[Documentação em português disponível aqui](README-pt.md)

---

<div align="center">

**⭐ If this project was useful to you, please consider giving it a star!**

Made with ❤️ and 🐍 Python

</div>
