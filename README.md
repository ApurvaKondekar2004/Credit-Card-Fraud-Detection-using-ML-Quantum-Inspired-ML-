# Credit Card Fraud Detection: Traditional vs Quantum-Inspired SVM

## Overview
This project detects fraudulent credit card transactions using:
- **Traditional SVM (RBF kernel)**
- **Quantum-Inspired SVM (Custom Polynomial Kernel)**

It compares performance between the two methods on a highly imbalanced dataset from Kaggle.

## Dataset
**Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Rows:** 284,807 transactions  
- **Fraud cases:** 492 (0.17%)  
- **Features:** 30 anonymized PCA features (`V1`-`V28`), `Time`, `Amount`  
- **Target:** `Class` (0 = Non-Fraud, 1 = Fraud)

## Workflow

### 1. Data Preprocessing
- Kept **all fraud cases**
- Randomly sampled **29,500 non-fraud cases** for efficiency
- Applied **SMOTE** to balance classes (fraud to non-fraud ratio = 1:5)
- Standardized features using `StandardScaler`

### 2. Feature Importance
- Trained **Linear SVM** on a subset (5000 rows)
- Used **SHAP (SHapley Additive Explanations)** to find most influential features

### 3. Dimensionality Reduction
- Applied **PCA (5 components)** for visualization

### 4. Model Training
#### Traditional SVM:
- Kernel: **RBF**
- Standard Scikit-learn `SVC` implementation

#### Quantum-Inspired SVM:
- Custom **Polynomial Kernel**
- Implemented in batches to reduce memory usage
- Passed as `precomputed` kernel to `SVC`

### 5. Evaluation
- Metrics:
  - **Confusion Matrix**
  - **Precision, Recall, F1-score**
  - **ROC Curve & AUC**
  - **Scatter Plots** (2D visual separation)
- Compared both models on fraud detection performance

## Results

| Model                  | Precision (Fraud) | Recall (Fraud) | Accuracy | AUC     |
|------------------------|-------------------|----------------|----------|---------|
| Traditional SVM (RBF)  | 0.99              | 0.96           | 99.29%   | ~0.99   |
| Quantum-Inspired SVM   | 0.97              | 0.99           | 99.36%   | ~0.99   |

**Key Finding:**  
Quantum-Inspired SVM improved **recall** for fraud detection (fewer false negatives), making it more effective for catching fraud cases.Even though the overall accuracy is similar, the quantum-inspired SVM catches more fraudulent transactions. In fraud detection, reducing false negatives is far more important than slightly improving overall accuracy.

