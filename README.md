# Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)](https://scikit-learn.org/)

## Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
  - [Softmax Regression](#1-softmax-regressionipynb)
  - [Support Vector Machine Regression](#2-support-vector-machine-regressionipynb)
  - [Voting Classifier](#3-voting-classifiercombined-with-decision_tree--random_forest--adaboost_classifieripynb)
  - [k-Means Clustering with PCA](#4-k-means-clustering-with-pcaipynb)


## Overview

| Notebook | Topic | Key Techniques |
|----------|-------|----------------|
| `Softmax Regression.ipynb` | Multi-class digit classification | From-scratch gradient descent, early stopping |
| `Support Vector Machine Regression.ipynb` | Kernel regression on synthetic data | SVR, kernel comparison, `GridSearchCV` |
| `Voting Classifier(...).ipynb` | Heart disease prediction | Decision Tree, Random Forest, AdaBoost, soft voting |
| `k-Means clustering with PCA.ipynb` | Face image clustering | PCA, eigenfaces, k-means, silhouette analysis |

## Notebooks

### 1. [`Softmax Regression.ipynb`](Softmax%20Regression.ipynb)

**Goal:** Classify handwritten digits on the MNIST dataset using multinomial logistic regression.

**Workflow:**
- Load 70,000 MNIST images via `sklearn.datasets.fetch_openml`
- Stratified train / validation / test split (70% / 15% / 15%)
- Implement **Softmax Regression from scratch** with NumPy (full-batch gradient descent, cross-entropy loss, early stopping)
- Compare against `sklearn.linear_model.LogisticRegression` on the held-out test set

**Highlights:**
- Custom implementation achieves ~**90.65%** test accuracy, closely matching scikit-learn (~**91.28%**)
- Confusion matrices and per-class precision / recall / F1 analysis

---

### 2. [`Support Vector Machine Regression.ipynb`](Support%20Vector%20Machine%20Regression.ipynb)

**Goal:** Investigate how SVM kernels and hyperparameters affect regression on polynomial synthetic data.

**Workflow:**
- Implement `generate_polynomial_dataset(degree, n_samples)` for \(y = \sum_{k=0}^{n} a_k x^k + \epsilon\)
- Build linear, quadratic, and cubic datasets (1,000 samples each; 80/20 train–test split)
- Fit `sklearn.svm.SVR` with **linear**, **polynomial**, and **RBF** kernels
- Tune hyperparameters on the cubic dataset via **5-fold `GridSearchCV`** (8 combinations × 5 folds)
- Evaluate the optimal model on the test set (\(R^2\), MSE, RMSE, MAE)

**Highlights:**
- Best tuned model: RBF kernel, \(C = 10\), \(\gamma = 0.1\) → test \(R^2 \approx 0.946\)

---

### 3. [`Voting Classifier(Combined with Decision_Tree & Random_Forest & AdaBoost_classifier).ipynb`](Voting%20Classifier%28Combined%20with%20Decision_Tree%20%26%20Random_Forest%20%26%20AdaBoost_classifier%29.ipynb)

**Goal:** Predict heart disease presence using ensemble classifiers on the UCI Cleveland dataset.

**Workflow:**
- Load and clean `processed.cleveland.data` (drop rows with `?` missing markers; binarise severity 0–4 → absent/present)
- Stratified 80/20 train–test split
- Train and evaluate:
  - `DecisionTreeClassifier` (baseline, unpruned)
  - `RandomForestClassifier` (500 trees)
  - `AdaBoostClassifier` (500 estimators)
  - `VotingClassifier` with soft voting (`voting="soft"`)
- Compare weighted precision / recall / F1, confusion matrices, and Random Forest feature importances

**Highlights:**
- Random Forest achieves the strongest hold-out performance among individual models
- Feature importances align with clinical intuition (`cp`, `thal`, `thalach`, `oldpeak`, `ca`)

---

### 4. [`k-Means clustering with PCA.ipynb`](k-Means%20clustering%20with%20PCA.ipynb)

**Goal:** Cluster face images and assess whether PCA improves k-means clustering quality.

**Workflow:**
- Load the **Olivetti Faces** dataset (400 images, 40 subjects × 10 images; 4,096 features per image)
- Visualise sample gallery, eigenfaces (first 10 principal components), and 2D PCA scatter
- Run **k-means** (\(k = 40\)) in raw pixel space
- Reduce dimensionality with PCA (retain components explaining 95% variance), then re-run k-means
- Compare **silhouette scores** between pixel-space and PCA-reduced representations

**Highlights:**
- PCA pre-processing improves mean silhouette score (positive Δ vs. raw pixels)
- Discussion of trade-offs: curse of dimensionality vs. linearity assumption and variance ≠ identity separability

