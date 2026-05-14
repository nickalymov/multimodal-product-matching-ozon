# 📦 Ozon Multimodal Product Matching (Top 7 Solution)

<p align="center">
  <img src="./docs/img/project_demo.gif" alt="Project Demo" width="900">
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Rank-7th%20out%20of%20110-gold.svg" alt="Rank">
  <img src="https://img.shields.io/badge/Metric-ROC--AUC%200.9216-orange.svg" alt="Metric">
  <img src="https://img.shields.io/badge/Python-3.10-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/ML-Multimodal%20Learning-blue.svg" alt="ML">
</p>

## 📌 Project Overview
This repository contains a high-ranking solution for the **Ozon Tech ML Challenge**. The task was to solve the **Entity Resolution** problem: determining whether two different product listings represent the same physical item using a combination of textual descriptions, technical attributes, and product images.

**Key Achievement:** Ranked **7th** solo place out of 110 teams.

## 🧠 Multimodal Approach
The core of the solution lies in effective feature fusion from three distinct data sources:

1.  **Textual Data (NLP):** 
    *   Calculated string similarities (Levenshtein, Jaccard) on product names and descriptions.
    *   Utilized pre-trained **BERT embeddings** to capture semantic meaning.
    *   Engineered "trick" features: matching specific numeric patterns (e.g., dimensions, volumes) using Regex.
2.  **Visual Data (Computer Vision):**
    *   Processed **ResNet-50 embeddings** for main and additional product images.
    *   Computed Cosine and Euclidean distances between image vectors.
    *   Analyzed embedding entropy to detect image complexity and similarity.
3.  **Structured Attributes:**
    *   Parsed complex JSON attribute mappings.
    *   Implemented category-specific feature engineering (Top-N most frequent attributes per category).
    *   Calculated Jaccard similarity for vital vs. minor attribute keys.

## 🚀 Modeling Pipeline
I implemented two distinct modeling strategies:

*   **AutoGluon Stacked Ensemble (Primary):** 
    *   Used `best_quality` preset with 1 bag set and multi-layer stacking.
    *   Achieved a peak **ROC-AUC of 0.9216**.
    *   Models included: LightGBM, CatBoost, XGBoost, and Neural Networks.
*   **HistGradientBoosting + Optuna (Lightweight):**
    *   A production-friendly alternative optimized for fast inference.
    *   Hyperparameter tuning performed via **Optuna** (TPE Sampler).

## 📂 Project Structure
*   `1_main_features.py` — **Core Feature Engineering**: Logic for NLP similarities, ResNet distances, and initial attribute parsing.
*   `2_add_cat_features.py` — **Advanced Attribute Matching**: Category-specific processing and Top-N frequent attribute extraction.
*   `autogluon.ipynb` — **Primary Model Training**: End-to-end pipeline for AutoGluon Stacked Ensembling.
*   `training_hgb.py` — **Lightweight Model Training**: Training HGB model with **Optuna** optimization.
*   `ag_inference.py` — **AutoGluon Inference**: Script for generating final predictions.
*   `hgb_inference.py` — **HGB Inference**: Fast inference pipeline for the lightweight model.
*   `requirements.txt` — Project dependencies.

## 🛠 Tech Stack
*   **Frameworks:** `AutoGluon`, `Scikit-learn`, `Optuna`
*   **Data Science:** `Pandas`, `NumPy`, `SciPy`, `Tqdm`
*   **NLP & CV:** `Levenshtein`, `Regex`, `BERT`, `ResNet`
*   **Environment:** `PyCharm`, `Git`, `Parquet/PyArrow`

## 🏁 Results & Analysis
The solution demonstrated that while AutoML (AutoGluon) provides a massive boost through ensembling, the **quality of Feature Engineering** (especially handling JSON attributes and Regex-based text parsing) was the decisive factor in reaching the Top 10.

---
*Developed by <a href="https://github.com/nickalymov" target="_blank">Nick Alymov</a>*
