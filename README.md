# 🧠 Artificial Neural Network on GPU — Fashion MNIST Classifier

> A fully connected **Artificial Neural Network (ANN)** built with PyTorch to classify clothing images from the Fashion MNIST dataset, with GPU acceleration support.

---

## 📌 Overview

| Property | Value |
|----------|-------|
| **Type** | ANN / Multi-Layer Perceptron (MLP) |
| **Dataset** | Fashion MNIST |
| **Input Size** | 784 pixels (28×28) |
| **Output Classes** | 10 clothing categories |
| **Optimizer** | SGD |
| **Loss Function** | CrossEntropyLoss |
| **Epochs** | 100 |
| **Batch Size** | 32 |
| **Device** | GPU (CUDA) / CPU fallback |

---

## � Dataset

The training data file (`Data/fashion-mnist_train.csv`) is **not included** in this repository due to file size limits.

Download it from Kaggle: **[Fashion MNIST — Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)**

Place the downloaded `fashion-mnist_train.csv` inside a `Data/` folder at the project root before running the notebook.

---

## �👗 What is Fashion MNIST?

Fashion MNIST is a dataset of **60,000 grayscale images** (28×28 pixels), each belonging to one of 10 clothing categories:

| Label | Category | Label | Category |
|-------|----------|-------|----------|
| 0 | 👕 T-shirt/top | 5 | 👡 Sandal |
| 1 | 👖 Trouser | 6 | 👔 Shirt |
| 2 | 🧥 Pullover | 7 | 👟 Sneaker |
| 3 | 👗 Dress | 8 | 👜 Bag |
| 4 | 🧣 Coat | 9 | 👢 Ankle boot |

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT LAYER                       │
│               784 neurons (28×28)                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              HIDDEN LAYER 1                         │
│    Linear(784 → 128)  +  ReLU activation            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              HIDDEN LAYER 2                         │
│    Linear(128 → 64)   +  ReLU activation            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               OUTPUT LAYER                          │
│          Linear(64 → 10)  →  10 class scores        │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Training Pipeline

```
Raw CSV Data
     │
     ▼
  Load with pandas
     │
     ▼
  Train/Test Split (80% / 20%)
     │
     ▼
  Normalize (÷ 255)
     │
     ▼
  CustomDataset  →  DataLoader (batch_size=32)
     │
     ▼
  Model → Forward Pass → Loss → Backward Pass → Update Weights
     │
     ▼ (repeat × 100 epochs)
  Trained Model
     │
     ▼
  Evaluate on Test Set → Accuracy %
```

---

## 📖 Step-by-Step Explanation

### 1. 📦 Importing Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

| Library | Purpose |
|---------|---------|
| `pandas` | Load the CSV data file |
| `sklearn` | Split data into train/test sets |
| `torch` | Core PyTorch library |
| `Dataset / DataLoader` | Manage and batch data efficiently |
| `nn` | Build neural network layers |
| `optim` | Optimization algorithms (SGD) |
| `matplotlib` | Visualize images |

---

### 2. 🎲 Setting a Random Seed

```python
torch.manual_seed(42)
```

Ensures **reproducibility** — running the notebook multiple times produces identical results.

---

### 3. ⚡ Checking for GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Automatically selects GPU if available, otherwise falls back to CPU.

> **Why GPU?** GPUs have thousands of cores that can process matrix operations (the core of neural network math) in parallel — making training **10–50x faster** than CPU.

---

### 4. 📂 Loading the Dataset

```python
df = pd.read_csv('../Data/fashion-mnist_train.csv')
```

Each row in the CSV represents one image:

```
| label | pixel1 | pixel2 | ... | pixel784 |
|-------|--------|--------|-----|----------|
|   5   |   0    |  128   | ... |    255   |
```

- **Column 0** → clothing label (0–9)
- **Columns 1–784** → pixel values (28×28 = 784 pixels flattened)

---

### 5. 🖼️ Visualizing the Images

```python
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
```

Displays the **first 16 images** in a 4×4 grid. Each pixel array is reshaped from (784,) back to (28, 28) for display.

---

### 6. ✂️ Train / Test Split

```python
X = df.iloc[:, 1:].values   # pixel values → features
y = df.iloc[:, 0].values    # clothing label → target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```
Total: 60,000 samples
├── Training:  48,000 (80%) ← model learns from this
└── Testing:   12,000 (20%) ← model is evaluated on this
```

---

### 7. 📏 Normalizing the Data

```python
X_train = X_train / 255.0
X_test  = X_test  / 255.0
```

Pixel values range **0–255**. After normalization they become **0.0–1.0**.

> **Why normalize?** Smaller, consistent input values help gradients flow better, resulting in faster and more stable training.

---

### 8. 🗂️ Custom Dataset Class

```python
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

PyTorch requires data wrapped in a `Dataset`. This class:
- Converts NumPy arrays → **PyTorch tensors**
- `__len__` → returns total number of samples
- `__getitem__` → returns one `(features, label)` pair by index

---

### 9. 🔀 DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| `batch_size` | 32 | Process 32 images at a time |
| `shuffle=True` | Train only | Prevents the model from memorizing order |
| `shuffle=False` | Test only | Order doesn't matter for evaluation |

---

### 10. 🏛️ Defining the Neural Network

```python
class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
```

- **`nn.Linear`** — Fully connected layer; every input connects to every output neuron
- **`nn.ReLU`** — Activation function: outputs `max(0, x)`, adding non-linearity
- **`nn.Sequential`** — Chains layers so data flows through them in order
- Final layer outputs **10 raw scores** (one per clothing class), called *logits*

---

### 11. ⚙️ Hyperparameters & Setup

```python
learning_rate = 0.1
epochs = 100

model     = MyNN(X_train.shape[1]).to(device)  # move to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

| Component | Choice | Purpose |
|-----------|--------|---------|
| `learning_rate` | 0.1 | Step size for weight updates |
| `epochs` | 100 | Full passes over training data |
| `CrossEntropyLoss` | — | Standard loss for multi-class classification |
| `SGD` | — | Updates weights using gradient descent |

---

### 12. 🔁 Training Loop

```python
for epoch in range(epochs):
    for batch_features, batch_labels in train_loader:

        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        outputs = model(batch_features)           # 1. Forward pass
        loss    = criterion(outputs, batch_labels) # 2. Compute loss

        optimizer.zero_grad()  # 3. Clear old gradients
        loss.backward()        # 4. Backpropagation
        optimizer.step()       # 5. Update weights
```

Each epoch performs these steps for every batch:

```
① Forward Pass   → model makes predictions
② Loss           → measure how wrong predictions are
③ Zero Gradients → reset before computing new ones
④ Backprop       → compute how much each weight contributed to the error
⑤ Update Weights → nudge weights in the direction that reduces loss
```

---

### 13. 📊 Evaluation

```python
model.eval()

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()

print(f"Accuracy: {(correct / total) * 100:.2f}%")
```

| Code | Purpose |
|------|---------|
| `model.eval()` | Switch to evaluation mode (disables dropout, etc.) |
| `torch.no_grad()` | Disable gradient tracking to save memory |
| `torch.max(outputs, 1)` | Pick the class with the highest score as prediction |
| `correct / total * 100` | Final accuracy as a percentage |

---

## 📚 Key Concepts Summary

| Concept | What It Does |
|---------|-------------|
| `device` | Runs model on GPU if available |
| `Dataset` | Wraps data in a PyTorch-compatible format |
| `DataLoader` | Batches & shuffles data for training |
| `nn.Linear` | Fully connected layer |
| `nn.ReLU` | Activation function — adds non-linearity |
| `nn.Sequential` | Stacks layers into a clean pipeline |
| `CrossEntropyLoss` | Measures prediction error for classification |
| `SGD` | Optimizer — updates model weights |
| `loss.backward()` | Computes gradients via backpropagation |
| `optimizer.step()` | Applies gradients to update weights |
| `.to(device)` | Moves tensors / model to GPU |

---

## 💡 Notes

> **ANN vs CNN for images:**
> This model is an **ANN (fully connected)**. It treats each pixel independently and ignores spatial relationships between neighboring pixels.
> A **CNN (Convolutional Neural Network)** would perform better on image data because it's designed to detect local patterns like edges and shapes.

> **`pin_memory=True` in DataLoader:**
> Speeds up CPU → GPU data transfer by storing batches in pinned (page-locked) memory.
