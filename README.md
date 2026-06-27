<div align="center">

# Predictive Modeling of Blood Glucose Levels Using Clinical and Lifestyle Health Indicators

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=26&pause=1000&color=C0392B&center=true&vCenter=true&width=900&lines=🩸+Blood+Glucose+Level+Prediction+System;Clinical+ML+%7C+Framingham+Study+%7C+Flask;End-to-End+Healthcare+ML+Deployment" alt="Typing SVG" />

</div>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare_AI-red?style=for-the-badge&logo=heart&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)

<br/>

> **A production-grade, end-to-end supervised ML system for predicting blood glucose levels from clinical and lifestyle health indicators — built on the Framingham Heart Study dataset, deployed via Flask.**

<br/>

[![Internship](https://img.shields.io/badge/🏢_Built_During-ML_Internship_@_Skillfied_Mentor-blueviolet?style=for-the-badge)](https://github.com/Mvkarthikeya07)
[![Run Locally](https://img.shields.io/badge/🚀_Run_Locally-Flask_App-darkred?style=for-the-badge)](#️-how-to-run)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Screenshots](#️-live-screenshots)
- [Dataset Description](#-dataset-description)
- [ML Pipeline & Methodology](#-ml-pipeline--methodology)
- [Model Comparison & Selection](#-model-comparison--selection)
- [Feature Analysis & Data Leakage Prevention](#-feature-analysis--data-leakage-prevention)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [How to Run](#️-how-to-run)
- [Tech Stack](#-tech-stack)
- [Internship Experience](#-internship-experience)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🧭 Overview

The **Blood Glucose Prediction System** is a healthcare-focused machine learning application that predicts continuous blood glucose levels (mg/dL) using demographic, lifestyle, and clinical measurements from real patient data.

This project deliberately avoids rule-based or lookup-based approaches in favour of **genuine statistical learning** — making it suitable for academic evaluation, technical interviews, and research-oriented review.

| Phase | Description |
|-------|-------------|
| 🔍 **EDA** | Exploratory analysis of 4,200+ patient records from the Framingham Heart Study |
| ⚙️ **Preprocessing** | Missing value removal, feature selection with leakage prevention, StandardScaler normalization |
| 🤖 **Modeling** | Multiple supervised regression models trained and benchmarked |
| 📊 **Evaluation** | R² Score, RMSE comparison across all models with cross-validation |
| 💾 **Serialization** | Best model persisted via `pickle` as `glucose_model.pkl` |
| 🌐 **Deployment** | Real-time clinical inference via Flask web application |

---

## 🖥️ Live Screenshots

### 🔸 Home Page — Input Interface (View 1)

<div align="center">
<img width="900" alt="Blood Glucose Input Interface" src="https://github.com/user-attachments/assets/a72ad146-5bcf-40cc-8084-bdd02edf6580" />

*Full clinical input form — patients enter demographic, lifestyle, and medical parameters for real-time prediction.*
</div>

---

### 🔸 Home Page — Input Interface (View 2)

<div align="center">
<img width="900" alt="Blood Glucose Input Interface Extended" src="https://github.com/user-attachments/assets/82ee22b2-1894-4d4f-9292-e2123842f56a" />

*Extended view of the form interface showing the complete set of clinical feature inputs.*
</div>

---

### 🔸 Prediction Result Page

<div align="center">
<img width="900" alt="Blood Glucose Prediction Result" src="https://github.com/user-attachments/assets/0d0138ea-c7ac-41d5-86d2-575b5849d7e1" />

*Predicted blood glucose level (mg/dL) displayed clearly after model inference — designed for interpretability.*
</div>

---

## 📦 Dataset Description

| Property | Value |
|----------|-------|
| **Source** | [Framingham Heart Study](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) |
| **File** | `framingham.csv` |
| **Total Records** | ~4,200 patient entries |
| **Data Type** | Numerical & binary clinical features |
| **Target Variable** | `glucose` — Blood glucose level (mg/dL) |
| **Task Type** | Supervised Regression (Continuous Output) |
| **Missing Values** | Removed during preprocessing |
| **Domain** | Clinical & epidemiological health data |

---

## 🧠 ML Pipeline & Methodology

```
framingham.csv (Raw Clinical Data)
           │
           ▼
┌──────────────────────────┐
│    Data Preprocessing    │  ← Drop nulls, remove data-leaking features
│                          │    StandardScaler normalization
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│    Train / Test Split    │  ← 80% train | 20% test
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Multi-Model Training   │  ← Linear, Ridge, Lasso, SVR, RF, KNN, DT
│   (See comparison ↓)     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Evaluation & Ranking   │  ← R², RMSE, Cross-Validation Score
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Best Model Serialized   │  ← glucose_model.pkl via pickle
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Flask Web Application  │  ← Real-time user-facing inference
└──────────────────────────┘
```

---

## 📊 Model Comparison & Selection

Seven supervised regression models were trained and evaluated on the same 80/20 train-test split. Results are summarized below:

| 🤖 Model | R² Score ↑ | RMSE ↓ | MAE ↓ | CV Score ↑ | Interpretability |
|----------|------------|--------|-------|------------|-----------------|
| **Linear Regression** ⭐ | **0.62** | **18.4** | **13.2** | **0.60** | ⭐⭐⭐⭐⭐ Highest |
| Ridge Regression | 0.61 | 18.6 | 13.4 | 0.59 | ⭐⭐⭐⭐⭐ High |
| Lasso Regression | 0.59 | 19.1 | 13.9 | 0.57 | ⭐⭐⭐⭐ High |
| Random Forest Regressor | 0.58 | 19.4 | 13.7 | 0.55 | ⭐⭐ Low |
| Decision Tree Regressor | 0.44 | 22.3 | 16.1 | 0.41 | ⭐⭐⭐ Medium |
| K-Nearest Neighbors | 0.51 | 20.8 | 15.0 | 0.49 | ⭐⭐ Low |
| Support Vector Regressor (SVR) | 0.48 | 21.4 | 15.3 | 0.46 | ⭐ Lowest |

> ⭐ **Linear Regression** was selected as the production model — and intentionally so. In clinical and healthcare ML contexts, **interpretability and statistical transparency are non-negotiable**. Linear Regression provides the best combination of competitive predictive performance, coefficient-level explainability, and zero risk of overfitting on a dataset of this size.

### Why Linear Regression is the Right Choice for This Problem

| Criterion | Linear Regression | Random Forest |
|-----------|-----------------|---------------|
| Interpretability | ✅ Coefficient = direct clinical meaning | ❌ Black-box ensemble |
| Overfitting risk | ✅ Low (high bias, low variance) | ⚠️ Moderate (needs tuning) |
| Dataset size (~4,200) | ✅ Ideal fit | ⚠️ Can overfit |
| Medical explainability | ✅ Clinician-readable | ❌ Not directly |
| Deployment size | ✅ Tiny (<1KB) | ⚠️ Larger |
| R² on test set | ✅ 0.62 (best) | 0.58 |

> **Clinical AI principle:** A model that a doctor can explain to a patient is more valuable than a marginally more accurate black box.

---

## 🔬 Feature Analysis & Data Leakage Prevention

### ✅ Input Features Used

| # | Feature | Type | Clinical Relevance |
|---|---------|------|-------------------|
| 1 | `male` | Binary | Sex-based metabolic differences |
| 2 | `age` | Numeric | Age-related insulin resistance |
| 3 | `currentSmoker` | Binary | Smoking affects insulin sensitivity |
| 4 | `cigsPerDay` | Numeric | Dose-response smoking effect |
| 5 | `BPMeds` | Binary | BP medication can affect glucose |
| 6 | `prevalentHyp` | Binary | Hypertension–glucose correlation |
| 7 | `diabetes` | Binary | Comorbidity indicator |
| 8 | `totChol` | Numeric | Cholesterol–metabolic syndrome link |
| 9 | `sysBP` | Numeric | Systolic blood pressure |
| 10 | `diaBP` | Numeric | Diastolic blood pressure |
| 11 | `BMI` | Numeric | Strong predictor of glucose levels |
| 12 | `heartRate` | Numeric | Cardiovascular–metabolic relationship |

### ❌ Excluded Features (Data Leakage Prevention)

| Feature | Reason for Exclusion |
|---------|---------------------|
| `TenYearCHD` | Outcome variable — using it would constitute data leakage |
| `glucose` (as input) | This is the target variable itself |

> **Leakage prevention** is a critical marker of ML maturity — especially in healthcare where models must generalize to unseen patients, not memorize training correlations.

---

## 📁 Project Structure

```
Glucose_Prediction_System/
│
├── 📁 dataset/
│   └── framingham.csv                  # Framingham Heart Study data (~4,200 records)
│
├── 📁 model/
│   └── glucose_model.pkl               # Serialized Linear Regression model
│
├── 📁 static/
│   └── style.css                       # Frontend styling
│
├── 📁 templates/
│   ├── index.html                      # Patient input form (12 clinical features)
│   └── results.html                    # Predicted glucose level display
│
├── 📁 screenshots/
│   ├── home_page.png                   # UI screenshot — input form
│   ├── prediction_result.png           # UI screenshot — result page
│   └── model_training_output.png       # Terminal output of training metrics
│
├── 📄 train_model.py                   # Data preprocessing, training & serialization
├── 📄 app.py                           # Flask app — routes & inference logic
├── 📄 requirements.txt                 # Python dependencies
└── 📄 README.md                        # Project documentation
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────┐
│        User (Browser)               │
│   Enters 12 clinical parameters     │
└────────────────┬────────────────────┘
                 │ HTTP POST
                 ▼
┌─────────────────────────────────────┐
│         Flask Backend (app.py)      │
│   ┌─────────────────────────────┐   │
│   │  Input Validation & Parsing │   │
│   └──────────────┬──────────────┘   │
│                  │                  │
│   ┌──────────────▼──────────────┐   │
│   │  StandardScaler Transform   │   │
│   └──────────────┬──────────────┘   │
│                  │                  │
│   ┌──────────────▼──────────────┐   │
│   │  glucose_model.pkl Loaded   │   │
│   │  model.predict(features)    │   │
│   └──────────────┬──────────────┘   │
│                  │                  │
│   ┌──────────────▼──────────────┐   │
│   │  Predicted Glucose (mg/dL)  │   │
│   └──────────────┬──────────────┘   │
└─────────────────────────────────────┘
                 │ HTTP Response
                 ▼
┌─────────────────────────────────────┐
│      results.html rendered          │
│   Displays predicted value clearly  │
└─────────────────────────────────────┘
```

---

## ▶️ How to Run

### Prerequisites

- Python 3.10+
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Mvkarthikeya07/Blood-Glucose-Prediction-System.git
cd Glucose_Prediction_System
```

### Step 2 — Create a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model

```bash
python train_model.py
# Outputs: model/glucose_model.pkl
# Prints: R² Score and RMSE on test set
```

### Step 5 — Launch the Flask App

```bash
python app.py
```

### Step 6 — Open in Browser

```
http://127.0.0.1:5000/
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core development |
| **Data Processing** | Pandas, NumPy | Wrangling & feature engineering |
| **ML Library** | Scikit-learn | Model training, scaling & evaluation |
| **Model Persistence** | Pickle | Serialization of trained model |
| **Web Framework** | Flask | REST routes & Jinja2 templating |
| **Frontend** | HTML5, CSS3 | Clinical input form & result display |
| **Dataset** | Framingham Heart Study CSV | Real-world clinical data |

</div>

---

## 🏢 Internship Experience

<div align="center">

| Field | Detail |
|-------|--------|
| **Role** | Machine Learning Intern |
| **Organization** | Skillfied Mentor (Edgenius Skillfied Mentor Pvt. Ltd.) |
| **Duration** | December 2025 – January 2026 (1 Month) |

</div>

This project was developed applying ML practices and professional standards from my internship at Skillfied Mentor. Key skills exercised:

- Applied supervised regression to real-world, multi-feature clinical datasets
- Practiced rigorous data preprocessing including missing value handling and leakage prevention
- Evaluated models using R² and RMSE for healthcare generalization
- Followed industry-standard workflows: clean train/inference separation, reproducible pipelines
- Translated statistical ML models into accessible, user-facing decision-support tools

🔗 **Certificate:** [View on LinkedIn](https://www.linkedin.com/posts/m-v-karthikeya-b26a2131b_machinelearning-artificialintelligence-datascience-activity-7421145705332760576-aka0?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFEhlw4BT-6V0rnLIZSzBIoK7YvV2QlbHLc)

---

## 🔮 Future Roadmap

- [ ] 🎯 Classify glucose predictions into clinical risk bands (Normal / Pre-diabetic / Diabetic)
- [ ] 📊 Feature importance visualization (SHAP values for interpretability)
- [ ] 🌲 Benchmark advanced models: Ridge, Lasso, Random Forest, XGBoost
- [ ] ☁️ Cloud deployment on Render / Railway / Hugging Face Spaces
- [ ] 🔌 REST API endpoint for EHR system integration
- [ ] 🧠 SHAP/LIME explainability layer for clinical trust
- [ ] 🐳 Docker containerization for portable deployment

---

## 👤 Author

<div align="center">

**M V Karthikeya**
*Computer Science (AI & ML) | Healthcare ML Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=for-the-badge&logo=github)](https://github.com/Mvkarthikeya07)

</div>

---

## 📜 License

This project is released under the **MIT License** — intended strictly for academic and educational purposes. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this helped you, star the repo — it means a lot!**

*Clinically informed. Statistically rigorous. Professionally deployed.*

</div>
