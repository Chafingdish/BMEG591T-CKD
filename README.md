# Resilience of ANN, KNN, and LR Under Feature Sparsity and Data Decay in CKD Diagnosis

**BMEG 591T — ML in Medicine**
**Flora Deng · 14085211**

---

## Overview

This project stress-tests three machine learning classifiers (Artificial Neural Network (ANN), K-Nearest Neighbours (KNN), and Logistic Regression (LR)) on the UCI Chronic Kidney Disease dataset under two real-world data degradation conditions. Rather than evaluating peak accuracy alone, the goal is to measure which model remains most reliable when clinical data quality degrades.

## Research Question

> Which model remains most reliable as clinical data quality degrades?

## Dataset

- **Source:** [UCI Machine Learning Repository — Chronic Kidney Disease (id=336)](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Size:** 400 patients, 24 features + binary target (CKD / not CKD)
- **Missing values:** 0.5% (htn, dm) to 38% (rbc) per feature
- Downloaded automatically via the `ucimlrepo` Python package

## Files

- Figures: Inlucded all generated plots
- ckd.py: code file