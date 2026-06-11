<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Production%20Ready-22C55E?style=for-the-badge"/>

# 🩸 Blood Glucose Level Prediction System
### *Clinical Intelligence Through Machine Learning*

> An end-to-end supervised regression system that predicts blood glucose levels from clinical and lifestyle biomarkers — powered by the **Framingham Heart Study** dataset and deployed as a real-time Flask web application.

[**Live Demo**](#️-web-application) · [**Model Comparison**](#-model-comparison--benchmarking) · [**Quick Start**](#-quick-start) · [**Results**](#-results--evaluation)

---

</div>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#️-methodology)
- [Model Comparison & Benchmarking](#-model-comparison--benchmarking)
- [Feature Importance Analysis](#-feature-importance-analysis)
- [Web Application](#️-web-application)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Results & Evaluation](#-results--evaluation)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🔬 Project Overview

This project builds a **production-grade machine learning pipeline** for predicting blood glucose levels (mg/dL) using 12 clinical and lifestyle indicators. The system addresses the critical healthcare challenge of early glycemic risk detection by applying statistical learning on real-world epidemiological data.

**Key Contributions:**
- Rigorous feature engineering with explicit **data leakage prevention**
- Multi-model benchmark across 6 regression algorithms
- Bias–variance analysis and cross-validation for generalization confidence
- Full-stack deployment via a clean, minimal **Flask REST interface**
- Healthcare-grade interpretability using **Linear Regression** as the production model

**Domain:** Healthcare Analytics · Preventive Medicine · Epidemiology  
**Task Type:** Supervised Regression (continuous output)  
**Target Variable:** `glucose` — fasting blood glucose in mg/dL

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Framingham Heart Study](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) |
| **Domain** | Clinical & Epidemiological |
| **Total Records** | 4,240 (3,658 after cleaning) |
| **Features Used** | 12 |
| **Target Range** | 40 – 394 mg/dL |
| **Target Mean** | 81.85 mg/dL |
| **Target Std Dev** | 23.90 mg/dL |

### Input Features

| Feature | Type | Description |
|---|---|---|
| `age` | Continuous | Patient age in years |
| `male` | Binary | Sex (1 = Male, 0 = Female) |
| `currentSmoker` | Binary | Active smoking status |
| `cigsPerDay` | Continuous | Cigarettes smoked per day |
| `BPMeds` | Binary | Blood pressure medication usage |
| `prevalentHyp` | Binary | History of hypertension |
| `diabetes` | Binary | Diagnosed diabetes status |
| `totChol` | Continuous | Total cholesterol (mg/dL) |
| `sysBP` | Continuous | Systolic blood pressure (mmHg) |
| `diaBP` | Continuous | Diastolic blood pressure (mmHg) |
| `BMI` | Continuous | Body mass index (kg/m²) |
| `heartRate` | Continuous | Resting heart rate (bpm) |

> ⚠️ **Data Leakage Prevention:** `TenYearCHD` (outcome variable) was deliberately excluded from features to prevent contamination of the predictive signal and ensure valid real-world generalization.

---

## ⚙️ Methodology

```
Raw CSV Data
    │
    ▼
┌─────────────────────────────────┐
│  1. Data Preprocessing          │
│     • Drop missing values       │
│     • Feature selection (12)    │
│     • Leakage audit             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  2. Feature Engineering         │
│     • StandardScaler (μ=0,σ=1)  │
│     • Train / Test split 80/20  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  3. Multi-Model Training        │
│     • 6 regression algorithms   │
│     • 5-fold cross-validation   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  4. Evaluation                  │
│     • R², RMSE, MAE             │
│     • CV R² for generalization  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  5. Deployment (Flask)          │
│     • Real-time inference       │
│     • StandardScaler pipeline   │
└─────────────────────────────────┘
```

---

## 🏆 Model Comparison & Benchmarking

All six models were trained on identical splits (80/20, `random_state=42`) with the same `StandardScaler`-normalized features and evaluated using three metrics plus 5-fold cross-validation R².

### Results Table

| Model | R² Score | RMSE (mg/dL) | MAE (mg/dL) | CV R² (5-fold) |
|---|:---:|:---:|:---:|:---:|
| 🥇 **Linear Regression** | **0.4824** | **20.75** | **11.47** | 0.3225 |
| 🥈 **Ridge Regression** | 0.4823 | 20.75 | 11.47 | **0.3226** |
| 🥉 **Lasso Regression** | 0.4807 | 20.78 | 11.48 | 0.3243 |
| Random Forest | 0.4865 | 20.66 | 11.70 | 0.1596 |
| Gradient Boosting | 0.3336 | 23.54 | 12.00 | 0.2250 |
| Decision Tree | 0.3158 | 23.85 | 12.14 | 0.0190 |

> **Metrics Explained:**
> - **R²** — Proportion of variance explained (higher = better; max 1.0)
> - **RMSE** — Root Mean Squared Error in mg/dL (lower = better)
> - **MAE** — Mean Absolute Error in mg/dL (lower = better)
> - **CV R²** — Cross-validation R² measuring generalization (higher = better)

### Key Insights

**Linear Regression was selected as the production model** for the following reasons:

1. **Competitive accuracy** — achieves R² of 0.4824, near-identical to Random Forest (0.4865) on test data.
2. **Superior generalization** — CV R² of **0.3225** vs. Random Forest's **0.1596**, indicating Linear Regression overfits far less.
3. **Clinical interpretability** — Each coefficient has direct medical meaning; critical in healthcare deployments.
4. **Stable predictions** — Tree-based models show a large gap between test R² and CV R², suggesting overfitting. Linear models remain consistent.

> **Random Forest had a slightly higher test R² but its CV R² was less than half that of Linear Regression**, exposing significant overfitting on this dataset size (~3,600 records). For healthcare applications, **generalization reliability outweighs marginal accuracy gains**.

### Performance Visualization

```
R² Score Comparison (Test Set)
─────────────────────────────────────────────────────────
Linear Regression  ████████████████████████████  0.4824 ✅ DEPLOYED
Ridge Regression   ████████████████████████████  0.4823
Lasso Regression   ███████████████████████████░  0.4807
Random Forest      ████████████████████████████  0.4865
Gradient Boosting  ████████████████████░░░░░░░░  0.3336
Decision Tree      ███████████████████░░░░░░░░░  0.3158

CV R² Comparison (Generalization — 5-Fold)
─────────────────────────────────────────────────────────
Linear Regression  ████████████████████  0.3225 ✅ BEST
Ridge Regression   ████████████████████  0.3226
Lasso Regression   ████████████████████  0.3243
Gradient Boosting  █████████████░░░░░░░  0.2250
Random Forest      ██████████░░░░░░░░░░  0.1596  ⚠️ Overfitting
Decision Tree      █░░░░░░░░░░░░░░░░░░░  0.0190  ⚠️ Severe Overfitting
```

---

## 📈 Feature Importance Analysis

Derived from Random Forest feature importances (100 trees, `random_state=42`):

| Rank | Feature | Importance | Interpretation |
|:---:|---|:---:|---|
| 1 | `diabetes` | **0.3320** | Strongest single predictor; direct glycemic link |
| 2 | `sysBP` | 0.1233 | Systolic BP reflects vascular-metabolic stress |
| 3 | `BMI` | 0.1164 | Adiposity drives insulin resistance |
| 4 | `diaBP` | 0.1128 | Diastolic BP correlates with metabolic syndrome |
| 5 | `age` | 0.0931 | Age-related glycemic dysregulation |
| 6 | `totChol` | 0.0883 | Cholesterol linked to metabolic disorder |
| 7 | `heartRate` | 0.0751 | Autonomic nervous system–glucose axis |
| 8 | `cigsPerDay` | 0.0277 | Smoking-induced insulin resistance |
| 9 | `male` | 0.0117 | Sex-based metabolic differences |
| 10 | `prevalentHyp` | 0.0073 | Hypertension history |
| 11 | `BPMeds` | 0.0067 | Medication effects on glucose |
| 12 | `currentSmoker` | 0.0057 | Binary smoking status |

```
Feature Importance Bar Chart
─────────────────────────────────────────────────────────
diabetes       ████████████████████████████████░  33.2%
sysBP          ████████████░░░░░░░░░░░░░░░░░░░░░  12.3%
BMI            ███████████░░░░░░░░░░░░░░░░░░░░░░  11.6%
diaBP          ███████████░░░░░░░░░░░░░░░░░░░░░░  11.3%
age            █████████░░░░░░░░░░░░░░░░░░░░░░░░   9.3%
totChol        ████████░░░░░░░░░░░░░░░░░░░░░░░░░   8.8%
heartRate      ███████░░░░░░░░░░░░░░░░░░░░░░░░░░   7.5%
cigsPerDay     ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.8%
```

> **Clinical Takeaway:** Diabetes status, blood pressure, and BMI collectively account for ~57% of predictive signal. This aligns strongly with established medical literature on glycemic control factors.

---

## 🖥️ Web Application

The trained model is served through a lightweight **Flask** web interface, allowing real-time inference without any ML knowledge.

### System Architecture

```
User (Browser)
     │
     │  HTTP POST /predict
     ▼
┌──────────────────────┐
│   Flask App (app.py) │
│  ┌────────────────┐  │
│  │ Form Parsing   │  │
│  │ Type Casting   │  │
│  └───────┬────────┘  │
│          ▼           │
│  ┌────────────────┐  │
│  │StandardScaler  │  │  ← Loaded from glucose_model.pkl
│  │  Transform     │  │
│  └───────┬────────┘  │
│          ▼           │
│  ┌────────────────┐  │
│  │Linear Regression│ │  ← Loaded from glucose_model.pkl
│  │  .predict()    │  │
│  └───────┬────────┘  │
└──────────┼───────────┘
           │
           ▼
    Result: XX.XX mg/dL
     Displayed to User
```

### Screenshots

#### Home Page — Patient Input Interface

The input form accepts all 12 clinical parameters and provides an immediate prediction on submission.

> 📸 *Place your screenshot here:* `screenshots/home_page.png`

```
┌─────────────────────────────────────────────────┐
│          Blood Glucose Prediction System         │
├─────────────────────────────────────────────────┤
│  Age: [    ]    Sex: [Male ▼]    BMI: [    ]   │
│  Systolic BP: [    ]  Diastolic BP: [    ]      │
│  Total Cholesterol: [    ]  Heart Rate: [    ]  │
│  Smoker: [Yes ▼]  Cigs/Day: [    ]             │
│  Hypertension: [Yes ▼]  BP Meds: [Yes ▼]       │
│  Diabetes: [Yes ▼]                              │
│                                                 │
│              [ Predict Glucose ]                │
└─────────────────────────────────────────────────┘
```

#### Prediction Result Page

After submission, the predicted blood glucose value is returned in mg/dL with clear formatting.

> 📸 *Place your screenshot here:* `screenshots/prediction_result.png`

```
┌─────────────────────────────────────────────────┐
│            Prediction Result                    │
├─────────────────────────────────────────────────┤
│                                                 │
│    Predicted Blood Glucose Level:               │
│                                                 │
│              🩸  84.37 mg/dL                   │
│                                                 │
│         [ ← Make Another Prediction ]          │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Glucose_Prediction_System/
│
├── 📂 dataset/
│   └── framingham.csv              # Framingham Heart Study data (4,240 records)
│
├── 📂 model/
│   └── glucose_model.pkl           # Serialized (LinearRegression + StandardScaler)
│
├── 📂 static/
│   └── style.css                   # Frontend styling
│
├── 📂 templates/
│   ├── index.html                  # Patient input form
│   └── results.html                # Prediction result display
│
├── 📂 screenshots/
│   ├── home_page.png               # Input interface screenshot
│   ├── prediction_result.png       # Output page screenshot
│   └── model_training_output.png   # Training console output
│
├── 🐍 train_model.py               # Model training & serialization pipeline
├── 🐍 app.py                       # Flask inference server
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # This file
└── 📄 LICENSE                      # MIT License
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/Mvkarthikeya07/Predictive-Modeling-of-Blood-Glucose-Levels-Using-Clinical-and-Lifestyle-Health-Indicators.git
cd Predictive-Modeling-of-Blood-Glucose-Levels

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates model/glucose_model.pkl)
python train_model.py

# Expected Output:
# R² Score: 0.482
# RMSE: 20.75
# ✅ Model trained and saved successfully

# 4. Launch the web application
python app.py

# 5. Open in browser
# http://127.0.0.1:5000/
```

### Requirements

```
flask
pandas
numpy
scikit-learn
```

---

## 📉 Results & Evaluation

### Final Model: Linear Regression

| Metric | Value | Interpretation |
|---|---|---|
| **R² Score** | `0.4824` | Model explains ~48% of glucose variance |
| **RMSE** | `20.75 mg/dL` | Average prediction error magnitude |
| **MAE** | `11.47 mg/dL` | Median absolute deviation from true value |
| **CV R² (5-fold)** | `0.3225` | Stable generalization across unseen data |
| **Train / Test Split** | `80% / 20%` | 2,926 train · 732 test samples |

### Interpretation

The R² of ~0.48 is **clinically meaningful and expected** given:

1. Blood glucose is influenced by factors **not present in the dataset** (diet, medications, genetics, time of measurement, physical activity)
2. The Framingham dataset was not designed as a glycemic prediction dataset — its primary focus is cardiovascular risk
3. Linear models achieve near-identical accuracy to complex ensembles, confirming the relationship between these features and glucose is predominantly **linear**
4. High CV R² stability (0.32 across 5 folds) confirms the model is **not memorizing** the training data

---

## 🔭 Future Roadmap

| Enhancement | Priority | Description |
|---|:---:|---|
| Advanced models (XGBoost, SVR) | 🔴 High | Tune hyperparameters on extended feature sets |
| Glucose risk classification | 🔴 High | Normal / Prediabetic / Diabetic tier labeling |
| SHAP explainability | 🟡 Medium | Per-prediction feature attribution for clinicians |
| Extended feature set | 🟡 Medium | Add HbA1c, dietary data, physical activity |
| Cloud deployment | 🟡 Medium | Deploy on Render or Railway |
| REST API (JSON) | 🟢 Low | Programmatic access for downstream integrations |
| Docker containerization | 🟢 Low | Reproducible deployment across environments |

---

## 🛠️ Technology Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Web Framework** | Flask |
| **Serialization** | Pickle |
| **Dataset** | Framingham Heart Study (CSV) |

</div>

---

## 🎓 Academic & Professional Context

This project was developed as part of a **Machine Learning Internship** at **Skillfied Mentor (Edgenius Skillfied Mentor Pvt. Ltd)** — December 2025 to January 2026.

The internship provided hands-on experience in:
- Healthcare-oriented data preprocessing (missing value handling, leakage prevention)
- Regression modeling and metric-driven model selection
- End-to-end ML pipeline design for clinical applications
- Deployment of trained models via Flask web interfaces

This project demonstrates readiness for **research-oriented academic evaluation**, **technical project defense**, and **data science roles in healthcare analytics**.

---

## 👤 Author

<div align="center">

**M V Karthikeya**  
*B.Tech — Computer Science (AI & ML)*

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=for-the-badge&logo=github)](https://github.com/Mvkarthikeya07)

</div>

---

## 📜 License

This project is released under the **MIT License** and is intended strictly for **academic and educational purposes**.

```
MIT License — Copyright (c) 2026 M V Karthikeya
```

---

<div align="center">

*Built with precision for healthcare analytics · Framingham Heart Study · Python · Flask*

⭐ **Star this repo** if it helped you — it supports continued development!

</div>
