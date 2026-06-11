<div align="center">

<br/>

```
```

### 🩸 Predictive Modeling of Blood Glucose Levels
#### *Clinical Intelligence · Machine Learning · Real-Time Deployment*

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Deployed-6366F1?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-E11D48?style=for-the-badge)

<br/>

> **"Transforming clinical data into actionable glucose predictions — bridging the gap between statistical learning and healthcare intelligence."**

<br/>

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#-project-overview) |
| 2 | [System Architecture](#-system-architecture) |
| 3 | [Dataset & Features](#-dataset--features) |
| 4 | [ML Methodology](#-ml-methodology) |
| 5 | [Performance Metrics](#-performance-metrics) |
| 6 | [Web Application](#-web-application) |
| 7 | [Project Structure](#-project-structure) |
| 8 | [Quick Start](#-quick-start) |
| 9 | [Technology Stack](#-technology-stack) |
| 10 | [Internship Experience](#-internship-experience) |
| 11 | [Future Roadmap](#-future-roadmap) |
| 12 | [Author](#-author) |

---

## 🎯 Project Overview

This project presents a **production-grade, end-to-end Machine Learning system** for predicting blood glucose levels (mg/dL) using clinical and lifestyle features derived from the **Framingham Heart Study** — one of the most influential longitudinal cardiovascular datasets in medical research history.

### What makes this system stand out:

| Aspect | Implementation |
|--------|---------------|
| **Data Integrity** | Rigorous feature selection to eliminate data leakage |
| **Clinical Relevance** | Features chosen based on medically validated correlates of glucose |
| **Production Mindset** | Clean separation of training and inference logic |
| **Accessibility** | Real-time predictions via a lightweight Flask web interface |
| **Transparency** | Linear Regression selected for interpretability — critical in clinical AI |

### Core Objectives

- ✅ Analyze real-world clinical data to identify key glucose-influencing factors
- ✅ Design and train a supervised regression model for continuous glucose prediction
- ✅ Rigorously evaluate model performance with standard statistical metrics
- ✅ Deploy the trained model as an accessible, user-friendly web application

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLOOD GLUCOSE PREDICTION SYSTEM              │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
    ┌──────▼──────┐   ┌───────▼──────┐  ┌───────▼──────┐
    │  DATA LAYER │   │  MODEL LAYER │  │   WEB LAYER  │
    │─────────────│   │──────────────│  │──────────────│
    │ framingham  │   │ train_model  │  │   Flask App  │
    │    .csv     │──▶│    .py       │─▶│    app.py    │
    │             │   │              │  │              │
    │  ~4,200     │   │ glucose_     │  │  index.html  │
    │  patients   │   │  model.pkl   │  │ results.html │
    └─────────────┘   │ scaler.pkl   │  └──────▲───────┘
                      └──────────────┘         │
                                               │
                              ┌────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    END USER INPUT  │
                    │  (Web Interface)   │
                    │                   │
                    │  Age, BMI, BP,     │
                    │  Cholesterol...    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  StandardScaler    │
                    │  (Feature Scaling) │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Linear Regression │
                    │   Trained Model    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Predicted Glucose│
                    │     (mg/dL)        │
                    └────────────────────┘
```

---

## 📊 Dataset & Features

### Source
> **Framingham Heart Study** — A landmark longitudinal cohort study tracking cardiovascular health since 1948, conducted by the National Heart, Lung, and Blood Institute (NHLBI).

| Property | Detail |
|----------|--------|
| **Records** | ~4,200 patients |
| **Domain** | Clinical epidemiology |
| **Data Types** | Numerical + Binary medical features |
| **Target Variable** | `glucose` — Blood glucose level (mg/dL) |

### Feature Set

```
┌──────────────────────┬────────────────────────────────────────┬────────────┐
│ Feature              │ Description                            │ Category   │
├──────────────────────┼────────────────────────────────────────┼────────────┤
│ male                 │ Sex (1 = Male, 0 = Female)             │ Demographic│
│ age                  │ Age of patient (years)                 │ Demographic│
├──────────────────────┼────────────────────────────────────────┼────────────┤
│ currentSmoker        │ Whether patient currently smokes       │ Lifestyle  │
│ cigsPerDay           │ Cigarettes smoked per day              │ Lifestyle  │
├──────────────────────┼────────────────────────────────────────┼────────────┤
│ BPMeds               │ Currently on blood pressure medication │ Medical Hx │
│ prevalentHyp         │ History of hypertension                │ Medical Hx │
│ diabetes             │ Diabetes diagnosis                     │ Medical Hx │
├──────────────────────┼────────────────────────────────────────┼────────────┤
│ totChol              │ Total cholesterol (mg/dL)              │ Clinical   │
│ sysBP                │ Systolic blood pressure (mmHg)         │ Clinical   │
│ diaBP                │ Diastolic blood pressure (mmHg)        │ Clinical   │
│ BMI                  │ Body Mass Index (kg/m²)                │ Clinical   │
│ heartRate            │ Resting heart rate (bpm)               │ Clinical   │
└──────────────────────┴────────────────────────────────────────┴────────────┘
```

> ⚠️ **Data Leakage Prevention**: Outcome variables such as `TenYearCHD` were **intentionally excluded** to ensure valid model generalization and prevent contamination of training signal.

---

## 🧠 ML Methodology

### Pipeline Overview

```
Raw Data (framingham.csv)
        │
        ▼
┌───────────────────┐
│  1. Preprocessing │  → Drop nulls, select features, validate types
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. Scaling       │  → StandardScaler (zero mean, unit variance)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. Train/Test    │  → 80% Train │ 20% Test (stratified split)
│     Split         │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. Model         │  → Linear Regression (OLS)
│     Training      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  5. Evaluation    │  → R² Score · RMSE
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  6. Serialization │  → glucose_model.pkl (joblib)
└───────────────────┘
```

### Why Linear Regression?

Linear Regression was the deliberate model choice — not a limitation:

| Reason | Explanation |
|--------|-------------|
| **Interpretability** | Coefficients are clinically meaningful and auditable |
| **Transparency** | Critical requirement in medical decision-support systems |
| **Generalization** | Avoids overfitting on a dataset of ~4,200 records |
| **Baseline Validity** | Establishes a rigorous performance benchmark before complex models |
| **Regulatory Alignment** | Simpler models align better with healthcare AI explainability standards |

---

## 📈 Performance Metrics

| Metric | Description | Significance |
|--------|-------------|--------------|
| **R² Score** | Proportion of variance explained by the model | Measures fit quality (1.0 = perfect) |
| **RMSE** | Root Mean Squared Error in mg/dL | Measures average prediction error |

> The model demonstrates **stable predictive performance without overfitting**, confirming an appropriate bias–variance trade-off for this clinical dataset size and feature space.

---

## 🌐 Web Application

A **production-ready Flask web application** serves the trained model with zero inference latency for end-users.

### Interface Flow

```
[ Home Page ]
    User inputs:
    • Age, Sex, BMI
    • Smoking status, Cigarettes/day
    • Blood pressure values (Sys/Dia)
    • Total cholesterol, Heart rate
    • Medical history flags
         │
         ▼ POST /predict
[ Prediction Result Page ]
    Displays:
    ✦ Predicted Glucose Level: XX.XX mg/dL
```

### Application Design Principles

- **Modular backend** — training and inference are fully decoupled
- **Stateless inference** — each prediction is independent and reproducible
- **Input validation** — sanitized form data before model ingestion
- **Extensible** — ready for REST API conversion or cloud deployment

### Screenshots

**Home Page — Input Interface**

> Users enter demographic, lifestyle, and clinical parameters through a clean form interface.

**Prediction Result Page**

> The system returns the predicted blood glucose level in mg/dL with clear result display.

---

## 📁 Project Structure

```
Glucose_Prediction_System/
│
├── 📂 dataset/
│   └── framingham.csv              # Framingham Heart Study data (~4,200 records)
│
├── 📂 model/
│   └── glucose_model.pkl           # Serialized trained Linear Regression model
│
├── 📂 static/
│   └── style.css                   # Web application styling
│
├── 📂 templates/
│   ├── index.html                  # Input form — home page
│   └── results.html                # Prediction result display page
│
├── 📂 screenshots/
│   ├── home_page.png
│   ├── prediction_result.png
│   └── model_training_output.png
│
├── 🐍 train_model.py               # Data preprocessing + model training pipeline
├── 🐍 app.py                       # Flask web application + inference logic
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # Project documentation
```

---

## ⚡ Quick Start

### Prerequisites

```bash
Python >= 3.8
pip package manager
```

### Installation & Execution

```bash
# Step 1 — Clone the repository
git clone https://github.com/Mvkarthikeya07/Glucose_Prediction_System.git
cd Glucose_Prediction_System

# Step 2 — Install dependencies
pip install -r requirements.txt

# Step 3 — Train the model
#   Preprocesses data, fits LinearRegression, serializes model + scaler
python train_model.py

# Step 4 — Launch the web application
python app.py

# Step 5 — Open in your browser
# Navigate to: http://127.0.0.1:5000/
```

### Output Specification

| Property | Value |
|----------|-------|
| **Output Type** | Continuous numerical value |
| **Unit** | mg/dL (milligrams per deciliter) |
| **Purpose** | Blood glucose estimation for analytical and educational use |
| **Latency** | < 50ms per prediction (local deployment) |

---

## 🛠 Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                     │
├─────────────────┬───────────────────────────────────────┤
│ Language        │ Python 3.10+                          │
├─────────────────┼───────────────────────────────────────┤
│ Data Processing │ Pandas · NumPy                        │
├─────────────────┼───────────────────────────────────────┤
│ Machine Learning│ Scikit-learn (LinearRegression,       │
│                 │ StandardScaler, train_test_split)     │
├─────────────────┼───────────────────────────────────────┤
│ Serialization   │ Joblib / Pickle (.pkl)                │
├─────────────────┼───────────────────────────────────────┤
│ Web Framework   │ Flask 2.x                             │
├─────────────────┼───────────────────────────────────────┤
│ Frontend        │ HTML5 · CSS3                          │
├─────────────────┼───────────────────────────────────────┤
│ Dataset         │ Framingham Heart Study (NHLBI)        │
└─────────────────┴───────────────────────────────────────┘
```

---

## 🏢 Internship Experience

<div>

### Machine Learning Intern
**Skillfied Mentor** (Edgenius Skillfied Mentor Pvt. Ltd)
`December 2025 – January 2026` · *1 Month*

</div>

This project was designed and built applying professional ML practices and industry standards cultivated during my internship at **Skillfied Mentor**, with a strong focus on healthcare-oriented data analysis and regression modeling.

#### Key Skills Applied from Internship

```
Supervised Regression          ████████████████████  Advanced
Healthcare Data Preprocessing  ██████████████████░░  Proficient
Feature Engineering            ████████████████░░░░  Proficient
Model Evaluation (R², RMSE)    ██████████████████░░  Proficient
Flask Deployment               ████████████████░░░░  Proficient
ML Code Architecture           ██████████████░░░░░░  Developing
```

| Area | Internship Learning Applied |
|------|----------------------------|
| **Data Integrity** | Handling missing medical values; preventing data leakage through disciplined feature selection |
| **Preprocessing** | StandardScaler applied to normalize clinical measurements across varying ranges |
| **Model Evaluation** | R² and RMSE used to assess predictive performance and generalization capacity |
| **ML Workflow** | Industry-standard separation of training and inference; reproducible pipelines |
| **Deployment** | Production-oriented Flask application design with modular architecture |
| **Documentation** | Professional-grade ML project structure for academic and industrial review |

> The internship experience directly shaped the **design philosophy** of this project — emphasizing data integrity, model transparency, and real-world applicability in healthcare-focused ML systems.

---

## 🚀 Future Roadmap

```
v1.0  ✅  Linear Regression baseline · Flask deployment · Framingham dataset
          └── Current release

v1.1  🔲  Risk classification layer (Normal / Pre-diabetic / Diabetic ranges)
          └── Categorical output overlay on continuous prediction

v1.2  🔲  Advanced model comparison
          └── Ridge · Lasso · Random Forest · Gradient Boosting

v1.3  🔲  Explainability layer
          └── SHAP values · LIME explanations · Feature importance visualization

v2.0  🔲  Cloud deployment
          └── Render / Railway / AWS EC2

v2.1  🔲  REST API
          └── JSON endpoint for EHR system integration

v3.0  🔲  Deep learning extension
          └── Neural network regressor with uncertainty quantification
```

---

## 🎓 Academic & Professional Relevance

This project demonstrates:

| Competency | Evidence |
|------------|----------|
| ML Fundamentals | Supervised regression, bias-variance trade-off, generalization |
| Real-world Data Handling | Missing values, feature selection, leakage prevention |
| Clinical Domain Awareness | Medically validated features, interpretable modeling |
| End-to-End System Design | Data → Model → Deployment pipeline |
| Software Engineering | Modular code, clean separation of concerns, version-ready structure |

**Ideal for:**
- 📚 Research-oriented academic evaluation and project defense
- 💼 Technical interviews and ML portfolio demonstration
- 🏥 Healthcare AI prototyping and clinical analytics showcases

---

## 👤 Author

<div align="center">

### M V Karthikeya

*Computer Science (AI & ML)*

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=for-the-badge&logo=github)](https://github.com/Mvkarthikeya07)

</div>

---

## 📜 License

```
MIT License — Open for academic and educational use.

Copyright (c) 2025 M V Karthikeya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
subject to the standard MIT License conditions.

This project is intended strictly for academic, educational, and research
purposes. It is NOT intended for clinical diagnosis or medical decision-making.
```

---

<div align="center">

**⭐ If this project helped you, consider starring the repository!**

*Built with precision · Powered by data · Deployed for impact*

</div>
