# Boosting Ensemble Learning

A comprehensive implementation of **Boosting** ensemble learning algorithms including **AdaBoost**, **Gradient Boosting**, **XGBoost**, and **LightGBM** with detailed comparisons, visualizations, and examples.

## 🎯 Overview

This project provides production-ready implementations and comparisons of major boosting algorithms:

- **AdaBoost** (Adaptive Boosting) - The original boosting algorithm
- **Gradient Boosting** - Generalized boosting with gradient descent
- **XGBoost** - Extreme Gradient Boosting (modern, high-performance)
- **LightGBM** - Light Gradient Boosting Machine (fast, efficient)

### Key Features

- **Modular architecture** with clean separation of concerns
- **Multiple execution modes** (single algorithm, comparison, experimentation)
- **Comprehensive metrics** (accuracy, precision, recall, F1, ROC-AUC)
- **Rich visualizations** (confusion matrices, feature importance, learning curves)
- **Multiple datasets** (Breast Cancer, Wine, Synthetic)
- **Parameter experimentation** capabilities
- **Production-ready code** with proper documentation

## 📊 What is Boosting?

Boosting is an ensemble technique that trains models **sequentially**, where each new model focuses on correcting the errors of previous models.

### Key Concepts

1. **Sequential Training**: Models are added one at a time
2. **Error Correction**: Each new model focuses on previous mistakes
3. **Weighted Voting**: Models vote with different weights based on performance
4. **Bias Reduction**: Reduces both bias and variance

### Algorithm Comparison

| Algorithm | Year | Key Innovation | Speed | Accuracy | Memory |
|-----------|------|----------------|-------|----------|---------|
| **AdaBoost** | 1996 | Adaptive sample weighting | Medium | Good | Low |
| **Gradient Boosting** | 2001 | Gradient descent optimization | Slow | Better | Medium |
| **XGBoost** | 2014 | Regularization + parallelization | Fast | Best | Medium |
| **LightGBM** | 2017 | Histogram-based + leaf-wise | **Fastest** | Best | **Lowest** |

### How Each Algorithm Works

#### AdaBoost
```
1. Initialize sample weights uniformly
2. For each iteration:
   - Train weak learner on weighted samples
   - Calculate learner's error
   - Update sample weights (increase for misclassified)
   - Calculate learner's voting weight
3. Final prediction: weighted majority vote
```

#### Gradient Boosting
```
1. Initialize with constant prediction
2. For each iteration:
   - Calculate negative gradient (pseudo-residuals)
   - Fit weak learner to gradients
   - Add to ensemble with learning rate
3. Final prediction: sum of all learners
```

#### XGBoost
```
- Adds regularization terms (L1, L2)
- Uses second-order gradients (Newton-Raphson)
- Built-in cross-validation
- Parallel tree construction
- Handles missing values automatically
```

#### LightGBM
```
- Histogram-based splitting (faster)
- Leaf-wise tree growth (deeper trees)
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Optimal for large datasets
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Navigate to the project directory:

```bash
cd boosting
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start

Train a single algorithm with visualizations:

```bash
python main.py --mode single --algorithm adaboost --visualize
```

Compare all boosting algorithms:

```bash
python main.py --mode compare --visualize --show-info
```

Experiment with different n_estimators:

```bash
python main.py --mode experiment --algorithm xgboost --visualize
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `single` | Execution mode: `single`, `compare`, or `experiment` |
| `--algorithm` | str | `adaboost` | Algorithm: `adaboost`, `gradient_boosting`, `xgboost`, `lightgbm` |
| `--dataset` | str | `breast_cancer` | Dataset: `breast_cancer`, `wine`, `synthetic` |
| `--visualize` | flag | False | Generate and display visualizations |
| `--show-info` | flag | False | Display dataset information |

### Usage Examples

#### Train AdaBoost
```bash
python main.py --mode single --algorithm adaboost --visualize
```

#### Train XGBoost with Wine dataset
```bash
python main.py --mode single --algorithm xgboost --dataset wine --visualize
```

#### Compare all algorithms
```bash
python main.py --mode compare --visualize --show-info
```

#### Experiment with Gradient Boosting
```bash
python main.py --mode experiment --algorithm gradient_boosting --visualize
```

#### Train LightGBM on synthetic data
```bash
python main.py --mode single --algorithm lightgbm --dataset synthetic --visualize
```

## 📁 Project Structure

```
boosting/
├── adaboost.py              # Legacy AdaBoost script
├── gradientboosting.py      # Legacy Gradient Boosting script
├── main.py                  # Modern CLI interface
├── config.py                # Configuration parameters
├── data_loader.py           # Data loading utilities
├── models.py                # Model creation and evaluation
├── visualization.py         # Plotting functions
├── requirements.txt         # Dependencies
├── README.md               # This file
├── LICENSE                 # License
└── plots/                  # Generated visualizations (auto-created)
```

## 🔧 Configuration

Modify `config.py` to customize algorithm parameters:

```python
# AdaBoost configuration
ADABOOST_CONFIG = {
    'n_estimators': 50,
    'learning_rate': 1.0,
    'random_state': 23
}

# XGBoost configuration
XGBOOST_CONFIG = {
    'n_estimators': 50,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Experiment ranges
EXPERIMENTS = {
    'n_estimators_range': [5, 10, 25, 50, 100, 200],
    'learning_rate_range': [0.01, 0.05, 0.1, 0.5, 1.0]
}
```

## 📈 Features

### 1. Comprehensive Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Probabilistic**: ROC-AUC, Log Loss
- **Visual**: Confusion Matrix, Classification Report

### 2. Rich Visualizations

- **Confusion Matrix** - Detailed prediction breakdown
- **Algorithm Comparison** - Side-by-side performance
- **Feature Importance** - Top contributing features
- **Learning Curves** - Performance vs n_estimators
- **Multi-Algorithm Feature Comparison** - Feature importance across algorithms

### 3. Multiple Datasets

- **Breast Cancer Wisconsin** - Binary classification (569 samples, 30 features)
- **Wine Recognition** - Multi-class classification (178 samples, 13 features)
- **Synthetic** - Generated classification data (1000 samples, 20 features)

### 4. Three Execution Modes

#### Single Mode
Train and evaluate one algorithm:
```bash
python main.py --mode single --algorithm xgboost --visualize
```

#### Compare Mode
Compare all algorithms simultaneously:
```bash
python main.py --mode compare --visualize
```

#### Experiment Mode
Analyze effect of n_estimators:
```bash
python main.py --mode experiment --algorithm gradient_boosting --visualize
```

## 📊 Expected Results

### Breast Cancer Dataset

| Algorithm | Accuracy | Precision | Recall | F1-Score | Training Time |
|-----------|----------|-----------|--------|----------|---------------|
| AdaBoost | ~96-97% | ~95-96% | ~97-98% | ~96-97% | Fast |
| Gradient Boosting | ~96-98% | ~95-97% | ~97-99% | ~96-98% | Medium |
| XGBoost | **~97-99%** | **~96-98%** | **~98-99%** | **~97-99%** | Fast |
| LightGBM | **~97-99%** | **~96-98%** | **~98-99%** | **~97-99%** | **Fastest** |

*XGBoost and LightGBM typically achieve the best performance.*

## 🎓 When to Use Each Algorithm

### AdaBoost
✅ **Use when:**
- You have small to medium datasets
- You want interpretable results
- You need a simple, proven algorithm
- Computational resources are limited

❌ **Avoid when:**
- You have noisy data with outliers
- Dataset is very large
- You need the absolute best performance

### Gradient Boosting
✅ **Use when:**
- You need strong predictive performance
- You can afford longer training time
- You want fine-grained control over loss function
- Feature importance is critical

❌ **Avoid when:**
- Training time is a constraint
- Dataset is extremely large (>1M samples)
- You need real-time predictions

### XGBoost
✅ **Use when:**
- You want state-of-the-art performance
- You have structured/tabular data
- You need built-in regularization
- You want parallel processing
- **Recommended for most production use cases**

❌ **Avoid when:**
- Interpretability is more important than accuracy
- You have extremely limited memory

### LightGBM
✅ **Use when:**
- You have very large datasets (>100K samples)
- Training speed is critical
- Memory is constrained
- You need real-time model updates
- **Best for big data scenarios**

❌ **Avoid when:**
- Dataset is very small (<1K samples)
- You need maximum interpretability

## 🔬 Python API Usage

```python
from data_loader import load_data
from models import create_xgboost_model, train_model, evaluate_model
import config

# Load data
x_train, x_test, y_train, y_test = load_data('breast_cancer')

# Create and train model
model = create_xgboost_model(config.XGBOOST_CONFIG)
train_model(model, x_train, y_train)

# Evaluate
results = evaluate_model(model, x_train, y_train, x_test, y_test)
print(f"Test Accuracy: {results['test']['accuracy']:.4f}")

# Get feature importance
from models import get_feature_importance
from data_loader import get_dataset_info

dataset_info = get_dataset_info('breast_cancer')
importance = get_feature_importance(model, dataset_info['feature_names'])
```

## 📚 Key Differences: Boosting vs Bagging

| Aspect | Boosting | Bagging |
|--------|----------|---------|
| **Training** | Sequential | Parallel |
| **Focus** | Error correction | Variance reduction |
| **Sampling** | Weighted | Random with replacement |
| **Combination** | Weighted vote | Equal vote/average |
| **Overfitting Risk** | Higher (can overfit) | Lower (reduces overfitting) |
| **Bias/Variance** | Reduces both | Mainly reduces variance |
| **Examples** | AdaBoost, XGBoost | Random Forest, Bagging |

## 🤝 Contributing

Enhancement ideas:
- CatBoost implementation
- Hyperparameter optimization (Optuna, GridSearch)
- Cross-validation support
- Early stopping visualization
- SHAP values for interpretability
- Multi-output classification
- Regression support

## 📖 Mathematical Background

### AdaBoost Weight Update

**Sample weight update:**
```
w_i^(t+1) = w_i^(t) * exp(α_t * I(y_i ≠ h_t(x_i)))
```

**Learner weight:**
```
α_t = 0.5 * ln((1 - ε_t) / ε_t)
```

where ε_t is the weighted error rate.

### Gradient Boosting Update

**Model update:**
```
F_m(x) = F_(m-1)(x) + η * h_m(x)
```

where h_m fits the negative gradient:
```
h_m = argmin Σ [−g_i − h(x_i)]²
g_i = ∂L(y_i, F(x_i)) / ∂F(x_i)
```

## 🔍 References

- Freund, Y., & Schapire, R. E. (1997). "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting"
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

## 💡 Example Output

```
================================================================================
BOOSTING ENSEMBLE LEARNING
================================================================================

Loading breast_cancer dataset...
Training samples: 426
Test samples: 143
Features: 30

[Mode: Algorithm Comparison]

Comparing AdaBoost, Gradient Boosting, XGBoost, and LightGBM...
Training AdaBoost...
Training Gradient Boosting...
Training XGBoost...
Training LightGBM...

================================================================================
XGBOOST RESULTS
================================================================================

Training Set Performance:
  Accuracy:  1.0000
  Precision: 1.0000
  Recall:    1.0000
  F1-Score:  1.0000
  ROC-AUC:   1.0000

Test Set Performance:
  Accuracy:  0.9790
  Precision: 0.9792
  Recall:    0.9792
  F1-Score:  0.9790
  ROC-AUC:   0.9956

================================================================================
Best Algorithm: XGBoost
Test Accuracy: 0.9790
================================================================================
```

## ⚠️ Important Notes

1. **XGBoost and LightGBM** require separate installation
2. **Feature scaling** is not required for tree-based boosting
3. **Overfitting** can occur with too many estimators
4. **Learning rate** and **n_estimators** have inverse relationship
5. **Early stopping** can prevent overfitting (not implemented in basic version)

---

**Happy Boosting! 🚀**

*From AdaBoost to LightGBM: The Evolution of Ensemble Learning*
