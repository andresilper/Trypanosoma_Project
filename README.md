# 🔬 Classificação Automática de *Trypanosoma cruzi* em Imagens de Microscopia

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Detecção automática do parasita *Trypanosoma cruzi* em imagens de microscopia utilizando Deep Learning com Transfer Learning (VGG16 e MobileNetV2).

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Problema e Motivação](#problema-e-motivação)
- [Dataset](#dataset)
- [Metodologia](#metodologia)
- [Arquiteturas Comparadas](#arquiteturas-comparadas)
- [Resultados](#resultados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Usar](#como-usar)
- [Requisitos](#requisitos)
- [Desafios e Soluções](#desafios-e-soluções)
- [Trabalhos Futuros](#trabalhos-futuros)
- [Autor](#autor)

---

## 🎯 Sobre o Projeto

Este projeto implementa um **sistema de classificação binária** para detectar automaticamente a presença do parasita *Trypanosoma cruzi* em imagens de microscopia. O *T. cruzi* é o agente causador da Doença de Chagas, uma doença tropical negligenciada que afeta milhões de pessoas nas Américas.

O objetivo é auxiliar profissionais de saúde no diagnóstico rápido e preciso através de **inteligência artificial**, reduzindo o tempo de análise manual e aumentando a acurácia na detecção.

### ✨ Destaques

- 🏆 **AUC de 0.989** no conjunto de validação
- 📊 **Acurácia média de 93.8%** em dados de teste reais
- 🎯 **Sensibilidade de 94.2%** (detecção de parasitas)
- ✅ **Especificidade de 92.7%** (baixo índice de falsos positivos)
- 🚀 Comparação entre VGG16 e MobileNetV2
- 💪 Solução robusta de overfitting através de técnicas de regularização

---

## 🔍 Problema e Motivação

### Desafio Clínico

A detecção de *T. cruzi* tradicionalmente requer:
- ⏱️ Análise manual demorada por microscopistas especializados
- 👁️ Alto nível de atenção e experiência
- 🔬 Identificação de parasitas pequenos (~20μm) em grandes amostras
- ⚠️ Risco de falsos negativos em casos de baixa parasitemia

### Solução Proposta

Um modelo de deep learning que:
- ✅ Automatiza a triagem inicial de lâminas
- ✅ Reduz tempo de análise
- ✅ Mantém alta sensibilidade para não perder casos positivos
- ✅ Fornece suporte à decisão para profissionais de saúde

---

## 📊 Dataset

### Características

- **Resolução:** 224×224 pixels
- **Classes:** 
  - `Positivo (1)`: Presença de *T. cruzi*
  - `Negativo (0)`: Ausência do parasita
- **Divisão:**
  - 🏋️ Treino: ~1.600 imagens
  - 📝 Validação: ~700 imagens
  - 🧪 Teste: 5 lâminas independentes (18, 19, 20, 23, 24)

### Pré-processamento

```python
# Normalização com estatísticas da ImageNet
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Data Augmentation (On-the-fly)

Para aumentar a robustez do modelo, aplicamos transformações aleatórias durante o treinamento:

- ↔️ Flip horizontal e vertical (p=0.5)
- 🔄 Rotação aleatória (±15°)
- 📐 Transformação afim (translação, escala, cisalhamento)
- 🎨 Variação de brilho, contraste, saturação e matiz
- ✂️ Random crop com escala 0.8-1.0

---

## 🧠 Metodologia

### 1. Transfer Learning

Utilizamos modelos pré-treinados na ImageNet como extratores de features:
- **VGG16**: 138M parâmetros, arquitetura clássica e robusta
- **MobileNetV2**: 3.5M parâmetros, eficiente para dispositivos móveis

### 2. Feature Freezing

```python
# Congelamento das camadas convolucionais
for param in model.features.parameters():
    param.requires_grad = False
```

**Justificativa:** Com apenas ~1.600 imagens de treino, treinar todas as camadas causaria overfitting severo.

### 3. Arquitetura do Classificador

```python
model.classifier = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(256, 1)
)
```

**Design simplificado** para evitar overfitting em datasets pequenos.

### 4. Otimização e Regularização

| Técnica | Valor | Justificativa |
|---------|-------|---------------|
| **Optimizer** | AdamW | Melhor que Adam para regularização |
| **Learning Rate** | 5×10⁻⁵ | Balanceamento entre convergência e estabilidade |
| **Weight Decay** | 1×10⁻⁴ | Regularização L2 |
| **Dropout** | 0.5 | Previne co-adaptação de neurônios |
| **Batch Size** | 32 | Compromisso entre memória e convergência |

### 5. Estratégias de Treinamento

- **Early Stopping:** Patience de 7 épocas para evitar overfitting
- **Learning Rate Scheduler:** ReduceLROnPlateau (reduz LR quando val_loss estagnar)
- **Loss Function:** BCEWithLogitsLoss (estável numericamente)

---

## 🏗️ Arquiteturas Comparadas

### VGG16

**Características:**
- 📦 138 milhões de parâmetros
- 🎯 Arquitetura clássica e bem estabelecida
- 🔧 Camadas convolucionais profundas (13 conv + 3 FC)

**Vantagens:**
- ✅ Alta capacidade de aprendizado
- ✅ Features robustas para classificação
- ✅ Muito estudada e testada

**Desvantagens:**
- ⚠️ Pesada (>500MB)
- ⚠️ Inferência mais lenta

### MobileNetV2

**Características:**
- 📦 3.5 milhões de parâmetros
- 🚀 Otimizada para eficiência
- 🔧 Depthwise separable convolutions

**Vantagens:**
- ✅ Modelo leve (~14MB)
- ✅ Inferência rápida
- ✅ Ideal para dispositivos móveis

**Desvantagens:**
- ⚠️ Menor capacidade que VGG16
- ⚠️ Pode ter performance levemente inferior

---

## 📈 Resultados

### VGG16 - Métricas de Validação

- **AUC-ROC:** 0.989
- **Melhor Val Loss:** 0.2763
- **Épocas treinadas:** 29 (early stopping)

### VGG16 - Performance em Teste Real

| Lâmina | Amostras | Acurácia | Sensibilidade | Especificidade | TP | FP | TN | FN |
|--------|----------|----------|---------------|----------------|----|----|----|----|
| **18** | 320 | **86.3%** | 75.6% | **98.7%** | 130 | 2 | 146 | 42 |
| **19** | 167 | **94.6%** | **98.9%** | 90.0% | 86 | 8 | 72 | 1 |
| **20** | 248 | **93.2%** | 96.9% | 89.1% | 125 | 13 | 106 | 4 |
| **23** | 230 | **97.8%** | **100%** | 95.8% | 112 | 5 | 113 | 0 |
| **24** | 936 | **97.4%** | **99.8%** | 95.2% | 457 | 23 | 455 | 1 |
| **Média** | - | **93.8%** | **94.2%** | **92.7%** | - | - | - | - |

### 📊 Interpretação dos Resultados

**🎯 Sensibilidade (Recall) - 94.2%:**
- O modelo detecta **94 de cada 100** parasitas presentes
- Crucial para diagnóstico: **poucos falsos negativos**
- Lâmina 23: 100% de detecção!

**✅ Especificidade - 92.7%:**
- O modelo corretamente identifica **93 de cada 100** amostras negativas
- Reduz trabalho de revisão manual de falsos positivos
- Lâmina 18: 98.7% - excelente confiabilidade

**📈 Acurácia Geral - 93.8%:**
- Performance consistente em 4 de 5 lâminas (>93%)
- Lâmina 18: 86.3% (possivelmente características diferentes)

### 🏆 Destaques por Lâmina

- **Lâmina 23:** Desempenho perfeito (100% sensibilidade, 0 falsos negativos)
- **Lâmina 24:** Maior volume de dados (936 amostras), manteve 99.8% sensibilidade
- **Lâmina 18:** Especificidade de 98.7% (apenas 2 falsos positivos)

### MobileNetV2 - Resultados

> 🚧 **Em desenvolvimento** - Resultados serão adicionados em breve

---

## 📁 Estrutura do Projeto

```
Projeto-Trypanossoma/
│
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências do projeto
├── LICENSE                            # Licença MIT
│
├── data/                              # Dados (não incluídos no repositório)
│   ├── train/
│   ├── val/
│   └── test/
│       ├── lamina_18/
│       ├── lamina_19/
│       ├── lamina_20/
│       ├── lamina_23/
│       └── lamina_24/
│
├── models/                            # Modelos treinados
│   ├── vgg16_best_model.pth
│   └── mobilenetv2_best_model.pth
│
├── notebooks/                         # Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_vgg16_training.ipynb
│   └── 03_mobilenetv2_training.ipynb
│
├── src/                               # Código fonte
│   ├── __init__.py
│   ├── dataset.py                     # Dataset e DataLoader
│   ├── models.py                      # Definição dos modelos
│   ├── train.py                       # Loop de treinamento
│   ├── evaluate.py                    # Avaliação e métricas
│   └── utils.py                       # Funções auxiliares
│
├── results/                           # Resultados e visualizações
│   ├── training_curves/
│   ├── confusion_matrices/
│   └── roc_curves/
│
└── docs/                              # Documentação adicional
    ├── methodology.md
    └── experiment_log.md
```

---

## 🚀 Como Usar

### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/Projeto-Trypanossoma.git
cd Projeto-Trypanossoma
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Preparar os Dados

Organize suas imagens seguindo a estrutura:

```
data/
├── train/
│   ├── positivo/
│   └── negativo/
├── val/
│   ├── positivo/
│   └── negativo/
└── test/
    └── lamina_XX/
```

### 4. Treinar o Modelo

#### VGG16

```bash
python src/train.py --model vgg16 --epochs 50 --batch-size 32 --lr 5e-5
```

#### MobileNetV2

```bash
python src/train.py --model mobilenetv2 --epochs 50 --batch-size 32 --lr 5e-5
```

### 5. Avaliar no Teste

```bash
python src/evaluate.py --model vgg16 --checkpoint models/vgg16_best_model.pth --test-dir data/test/
```

### 6. Fazer Predições em Novas Imagens

```python
from src.models import load_model
from src.utils import predict_image

model = load_model('vgg16', 'models/vgg16_best_model.pth')
result = predict_image(model, 'path/to/image.jpg')

print(f"Predição: {'Positivo' if result > 0.5 else 'Negativo'}")
print(f"Confiança: {result:.2%}")
```

---

## 📦 Requisitos

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

**Sistema:**
- Python 3.8+
- CUDA 11.8+ (opcional, para GPU)
- 8GB RAM mínimo (16GB recomendado)

---

## 💡 Desafios e Soluções

### 🔥 Problema 1: Overfitting Severo

**Sintoma:**
- Train loss caindo, mas validation loss subindo
- Gap crescente entre treino e validação

**Causas Identificadas:**
1. ❌ Dataset aumentado pré-gerado (imagens fixas)
2. ❌ BatchNorm com dataset pequeno causando ruído
3. ❌ Learning rate muito alto (1e-4)
4. ❌ Todas as camadas da VGG treináveis

**Soluções Implementadas:**
1. ✅ Data augmentation on-the-fly (variações infinitas)
2. ✅ Remoção do BatchNorm do classificador
3. ✅ Redução do learning rate para 5e-5
4. ✅ Congelamento das features da VGG
5. ✅ Classificador simplificado (512 → 256 features)
6. ✅ Dropout de 0.5
7. ✅ Weight decay de 1e-4

**Resultado:**
- Val loss estável e convergindo junto com train loss
- Gap mínimo entre as curvas
- AUC de 0.989

### 🐛 Problema 2: Desalinhamento de Labels

**Sintoma:**
- Resultados inconsistentes e estranhos

**Causa:**
- Labels e imagens em ordens diferentes

**Solução:**
```python
# Usar DataFrames para garantir alinhamento
df = pd.DataFrame({
    'filename': sorted(image_paths),
    'label': corresponding_labels
})
```

### ⚡ Problema 3: Convergência Lenta

**Sintoma:**
- Modelo não melhorando após várias épocas

**Causa:**
- Learning rate muito baixo (1e-5) com BatchNorm

**Solução:**
- Learning rate scheduler (ReduceLROnPlateau)
- Começa com LR maior (5e-5), reduz quando estagnar

---

## 🔮 Trabalhos Futuros

- [ ] Implementar e comparar MobileNetV2
- [ ] Testar outras arquiteturas (ResNet, EfficientNet)
- [ ] Implementar ensemble de modelos
- [ ] Criar interface web com Gradio/Streamlit
- [ ] Segmentação de parasitas (localização exata)
- [ ] Quantificação automática de parasitemia
- [ ] Detecção de outros protozoários
- [ ] Deploy em dispositivo móvel (TFLite/ONNX)
- [ ] Explicabilidade com Grad-CAM
- [ ] Aumento do dataset com técnicas de GAN

---

## 📚 Referências

1. World Health Organization. (2023). Chagas disease (American trypanosomiasis)
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition
3. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks
4. He, K., et al. (2016). Deep Residual Learning for Image Recognition

---

## 👨‍💻 Autor

**[Seu Nome]**

- 🎓 Doutorando em [Sua Área]
- 💼 LinkedIn: [seu-linkedin](https://linkedin.com/in/seu-perfil)
- 📧 Email: seu.email@exemplo.com
- 🐙 GitHub: [@seu-usuario](https://github.com/seu-usuario)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🙏 Agradecimentos

- Dataset fornecido por [Instituição/Laboratório]
- Infraestrutura computacional: [GPU/Cloud provider]
- Orientação: [Nome do orientador]

---

## 📊 Status do Projeto

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

**Última atualização:** Janeiro 2026

---

<div align="center">

**⭐ Se este projeto foi útil para você, considere dar uma estrela!**

Made with ❤️ and 🐍 Python

</div>
