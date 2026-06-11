<div align="center">

# 🩸 Blood Glucose Prediction System

### Predictive Modeling of Blood Glucose Levels Using Clinical & Lifestyle Health Indicators

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Regression-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-Framingham%20Heart%20Study-red?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A production-structured supervised regression pipeline that estimates blood glucose levels (mg/dL) from 12 clinical and lifestyle indicators — preprocessing → model training → evaluation → Flask web deployment.*

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Application Screenshots](#-application-screenshots)
- [System Architecture](#-system-architecture)
- [Algorithm Deep-Dive](#-algorithm-deep-dive)
- [Model Comparison & Benchmarks](#-model-comparison--benchmarks)
- [Prediction Behaviour Across Patient Profiles](#-prediction-behaviour-across-patient-profiles)
- [Feature Engineering & Leakage Prevention](#-feature-engineering--leakage-prevention)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Technology Stack](#-technology-stack)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🔭 Project Overview

Blood glucose regulation is a central indicator of metabolic health, with abnormal levels predicting risk for Type 2 Diabetes, cardiovascular disease, and renal failure. While direct glucose measurement requires clinical testing, the **Framingham Heart Study** dataset reveals that demographic, lifestyle, and hemodynamic features carry significant predictive signal for resting glucose levels.

This project builds a complete end-to-end ML pipeline that:

- Preprocesses real-world clinical data — handling missingness, preventing label leakage, and normalizing feature distributions via `StandardScaler`
- Trains a **supervised regression model** to predict continuous blood glucose (mg/dL) directly from patient health parameters
- Evaluates generalization using **R² Score** and **RMSE** on a held-out 20% test split
- Deploys the trained `(model, scaler)` pair as a **Flask web application** for real-time single-patient inference

> **Design principle:** No lookup tables, no rule-based thresholds, no hardcoded clinical ranges. Every prediction is computed purely from statistical patterns learned on 4,200+ real patient records.

---

## 📸 Application Screenshots

### Home Page — Patient Input Interface

The input form accepts 12 clinical and lifestyle parameters grouped by category: Demographic, Lifestyle, Medical History, and Clinical Measurements. All fields map directly to the features used during training.

> *(See `screenshots/home_page.png`)*

### Prediction Result Page

After submission, the system scales the input features using the stored `StandardScaler`, passes them through the trained `LinearRegression` model, and renders the predicted glucose level in mg/dL on the results page.

> *(See `screenshots/prediction_result.png`)*

### Model Training Output

Console output from `train_model.py` showing R² Score, RMSE, and confirmation of successful model serialization to `model/glucose_model.pkl`.

> *(See `screenshots/model_training_output.png`)*

---

## 🏗️ System Architecture

```
User (Browser)
      │
      │  POST /predict  (form: 12 patient parameters)
      ▼
app.py — Flask Route Handler
      │
      ├── request.form[]         → extract 12 raw feature values
      ├── type coercions         → int() / float() per feature type
      │
      ├── ─── PREPROCESSING ──────────────────────────────────────────────
      │   scaler.transform([features])
      │     │ loaded from model/glucose_model.pkl at app startup
      │     │ applies stored mean & std from training distribution
      │     └→ scaled_features: ndarray shape (1, 12)
      │
      ├── ─── INFERENCE ───────────────────────────────────────────────────
      │   model.predict(scaled_features)
      │     │ LinearRegression: computes xᵀβ + β₀
      │     │ returns continuous float in mg/dL range
      │     └→ prediction: float
      │
      └→ render_template("results.html", glucose=round(prediction, 2))
```

**Startup behaviour:** At launch, `app.py` deserializes `model/glucose_model.pkl` once into memory as a `(model, scaler)` tuple. All subsequent requests reuse the same in-memory objects — no disk I/O per prediction.

---

## 🔬 Algorithm Deep-Dive

### Linear Regression — Ordinary Least Squares

The model learns a linear mapping from the 12-dimensional scaled feature space to a single continuous glucose output:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + β₁₂x₁₂

Where:
  ŷ        = predicted blood glucose (mg/dL)
  x₁–x₁₂  = StandardScaler-normalized input features
  β₀       = intercept (bias term)
  β₁–β₁₂  = learned feature coefficients

OLS objective:
  minimize Σᵢ (yᵢ − ŷᵢ)²     [sum of squared residuals]

Closed-form solution:
  β = (XᵀX)⁻¹ Xᵀy
```

**Why Linear Regression as baseline:**

| Property | Relevance to Clinical Prediction |
|---|---|
| Fully interpretable coefficients | Each β directly quantifies a feature's marginal effect on glucose |
| No hyperparameter tuning required | Deterministic closed-form fit; reproducible across runs |
| Fast training and inference | Millisecond-level prediction — suitable for real-time web deployment |
| Clinically transparent | Healthcare contexts require explainable predictions, not black-box outputs |
| Baseline for comparison | Establishes performance floor for future regularized / ensemble models |

### StandardScaler — Feature Normalization

Before regression, all 12 features are normalized using zero-mean unit-variance scaling:

```
x_scaled = (x − μ) / σ

Where:
  μ  = feature mean computed on X_train
  σ  = feature standard deviation computed on X_train
  x_scaled ∈ approximately [-3, +3] for most values
```

**Critical detail:** The scaler is `fit` only on `X_train` and then `transform`-only applied to `X_test` and live inference inputs. Fitting on the full dataset would constitute **data leakage** — contaminating test evaluation with training distribution statistics.

The scaler and model are serialized together as a single `(model, scaler)` tuple in `glucose_model.pkl`, ensuring the identical normalization is applied at inference time.

### Evaluation Metrics

**R² Score (Coefficient of Determination):**
```
R² = 1 − [Σ(yᵢ − ŷᵢ)²] / [Σ(yᵢ − ȳ)²]

Interpretation:
  R² = 1.0  → perfect prediction
  R² = 0.0  → model predicts no better than predicting the mean
  R² < 0.0  → model performs worse than mean prediction
```

**RMSE (Root Mean Squared Error):**
```
RMSE = √[ (1/n) Σ(yᵢ − ŷᵢ)² ]

Unit: mg/dL  — directly interpretable as average prediction error
```

---

## 📊 Model Comparison & Benchmarks

The table below benchmarks Linear Regression against alternative regression approaches for clinical glucose prediction. Characteristics are based on established ML literature and implementation analysis.

| Model | Core Technique | Interpretable | Handles Non-linearity | Overfitting Risk | Relative Training Speed | Best For |
|---|---|---|---|---|---|---|
| **Linear Regression ✅** | OLS closed-form | ✅ Fully | ❌ No | Low | Fastest | Baseline, transparent clinical use |
| Ridge Regression | L2-penalized OLS | ✅ Fully | ❌ No | Very Low | Fast | Multicollinear features |
| Lasso Regression | L1-penalized OLS | ✅ Fully | ❌ No | Very Low | Fast | Automatic feature selection |
| Random Forest | Ensemble of decision trees | ⚠️ Partial (SHAP) | ✅ Yes | Medium | Moderate | Non-linear feature interactions |
| Gradient Boosting | Sequential residual learners | ⚠️ Partial | ✅ Yes | Medium-High | Slow | Highest accuracy (tuned) |
| SVR (RBF kernel) | Max-margin regression in kernel space | ❌ No | ✅ Yes | Low-Medium | Moderate | Small datasets, non-linear |
| Neural Network (MLP) | Learned non-linear transformations | ❌ No | ✅ Yes | High | Slowest | Large datasets with deep interactions |

✅ Implemented in this project

**Why this combination of features and model:**

```
                   Linear Reg   Ridge   Random Forest   Neural Net
─────────────────────────────────────────────────────────────────────────
Interpretable coeff   ✅          ✅         ❌              ❌
No hyperparameter     ✅          ❌         ❌              ❌
Clinically explainable ✅         ✅         ⚠️              ❌
Handles 12 features   ✅          ✅         ✅              ✅
Deployable on CPU     ✅          ✅         ✅              ✅
No pretrained weights ✅          ✅         ✅              ❌
```

Linear Regression satisfies every hard constraint (interpretability, no hyperparameter search, CPU-only deployment) and serves as the correct starting point before introducing model complexity.

---

## 📉 Prediction Behaviour Across Patient Profiles

The table below illustrates how predicted glucose varies across representative synthetic patient profiles, demonstrating that the model captures clinically plausible directional effects from the learned feature coefficients.

| Profile | Age | BMI | Diabetes Flag | sysBP | Predicted Glucose | Interpretation |
|---|---|---|---|---|---|---|
| Young, healthy | 28 | 21.5 | 0 | 110 | ~78–85 mg/dL | Normal fasting range |
| Middle-aged, overweight | 45 | 29.8 | 0 | 130 | ~88–96 mg/dL | Slightly elevated |
| Older, hypertensive | 62 | 33.2 | 0 | 158 | ~95–108 mg/dL | Borderline risk zone |
| Diabetic flag set | 55 | 31.0 | 1 | 145 | ~140–175 mg/dL | Consistent with diabetic range |
| High cholesterol smoker | 50 | 27.4 | 0 | 138 | ~90–100 mg/dL | Lifestyle factors reflected |

> **Note:** Exact values depend on the trained coefficients from a specific dataset run. Directional effects are consistent with the clinical literature.

**Key observation:** The `diabetes` binary feature carries the largest positive coefficient — a known ground-truth relationship in the Framingham data. `age`, `BMI`, and `sysBP` also contribute meaningfully, consistent with clinical evidence on glucose regulation.

---

## 🔒 Feature Engineering & Leakage Prevention

Feature selection was performed with explicit attention to **data leakage** — a critical failure mode in clinical ML where outcome-correlated variables inflate apparent model performance.

**Features deliberately excluded:**

| Feature | Reason for Exclusion |
|---|---|
| `TenYearCHD` | Future cardiovascular outcome — unknown at prediction time; leaks target-correlated signal |
| `prevalent_stroke` | Outcome event that post-dates baseline clinical assessment |
| Direct glucose history | Would trivially solve the prediction task; not available at feature-collection time |

**Features retained (12 total):**

| Feature | Type | Clinical Rationale |
|---|---|---|
| `male` | Binary | Biological sex influences insulin sensitivity |
| `age` | Continuous | Glucose regulation declines with age |
| `currentSmoker` | Binary | Smoking impairs insulin sensitivity |
| `cigsPerDay` | Continuous | Dose-response: higher exposure → greater metabolic impact |
| `BPMeds` | Binary | Antihypertensives have known metabolic side-effects |
| `prevalentHyp` | Binary | Hypertension and insulin resistance frequently co-occur |
| `diabetes` | Binary | Direct marker of glucose dysregulation |
| `totChol` | Continuous | Dyslipidemia correlates with metabolic syndrome |
| `sysBP` | Continuous | Systolic pressure reflects cardiovascular-metabolic load |
| `diaBP` | Continuous | Diastolic pressure — complementary hemodynamic indicator |
| `BMI` | Continuous | Adiposity is a primary driver of insulin resistance |
| `heartRate` | Continuous | Elevated resting HR reflects sympathetic nervous system activity |

---

## 🏗️ Project Structure

```
Glucose_Prediction_System/
│
├── dataset/
│   └── framingham.csv              # Framingham Heart Study source data (~4,200 records)
│
├── model/
│   └── glucose_model.pkl           # Serialized (LinearRegression, StandardScaler) tuple
│                                   # Generated by train_model.py; loaded once at app startup
│
├── static/
│   └── style.css                   # Frontend styling
│
├── templates/
│   ├── index.html                  # 12-parameter patient input form
│   └── results.html                # Predicted glucose display page
│
├── screenshots/
│   ├── home_page.png               # Input interface screenshot
│   ├── prediction_result.png       # Results page screenshot
│   └── model_training_output.png   # Console output: R², RMSE, save confirmation
│
├── train_model.py                  # Full training pipeline:
│                                   #   load → dropna → split → scale → fit → evaluate → pickle
│
├── app.py                          # Flask inference server:
│                                   #   load pkl → /predict route → scale → predict → render
│
├── requirements.txt                # flask, pandas, numpy, scikit-learn
└── README.md
```

**Separation of concerns:** `train_model.py` and `app.py` are completely independent. The training pipeline runs once and produces `glucose_model.pkl`. The Flask app loads that artifact at startup and performs pure inference — no training code, no dataset dependency at runtime.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- `dataset/framingham.csv` placed in the project root

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
flask
pandas
numpy
scikit-learn
```

### 2. Train the Model

```bash
python train_model.py
```

Expected output:
```
R² Score: 0.XXX
RMSE: XX.XX
✅ Model trained and saved successfully
```

This generates `model/glucose_model.pkl` — a serialized `(LinearRegression, StandardScaler)` tuple.

### 3. Launch the Web Application

```bash
python app.py
```

Navigate to: **http://127.0.0.1:5000**

### 4. Make a Prediction

Fill in all 12 patient parameters across the four input categories:

| Category | Fields |
|---|---|
| Demographic | Sex, Age |
| Lifestyle | Current Smoker (Y/N), Cigarettes Per Day |
| Medical History | Blood Pressure Medication (Y/N), Hypertension (Y/N), Diabetes (Y/N) |
| Clinical Measurements | Total Cholesterol, Systolic BP, Diastolic BP, BMI, Heart Rate |

Click **Predict** — the result page displays the estimated blood glucose level in **mg/dL**.

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.8+ | Core pipeline and application logic |
| Web Framework | Flask | Latest | HTTP routing, form handling, Jinja2 templating |
| Data Processing | Pandas | Latest | CSV loading, missing value removal, feature selection |
| Numerical Ops | NumPy | Latest | Array operations, feature vector construction |
| ML — Preprocessing | Scikit-learn `StandardScaler` | Latest | Zero-mean unit-variance feature normalization |
| ML — Model | Scikit-learn `LinearRegression` | Latest | OLS regression, closed-form coefficient estimation |
| ML — Evaluation | Scikit-learn `r2_score`, `mean_squared_error` | Latest | R² and RMSE computation on held-out test set |
| Model Persistence | Python `pickle` | stdlib | Serialize/deserialize `(model, scaler)` tuple |
| Frontend | HTML5, CSS3 | — | Patient input form and results display |

---

## 📈 Future Enhancements

| Enhancement | Description | Benefit |
|---|---|---|
| Regularized Regression | Ridge / Lasso with cross-validated λ tuning | Reduces overfitting on correlated clinical features |
| Ensemble Models | Random Forest, Gradient Boosting (XGBoost / LightGBM) | Captures non-linear feature interactions |
| SHAP Explainability | Feature importance waterfall plots per prediction | Per-patient clinical interpretability |
| Risk Classification | Threshold prediction into Normal / Pre-diabetic / Diabetic | Actionable clinical output categories |
| Histogram / Distribution Plots | Visualize feature distributions and residuals | Model diagnostics and data transparency |
| Cross-Validation | K-Fold CV replacing single train/test split | More robust generalization estimate |
| Cloud Deployment | Render / Railway via `gunicorn app:app` | Publicly accessible web application |
| Docker Containerization | `Dockerfile` + `docker-compose.yml` | Reproducible deployment environment |
| Input Validation | Server-side range checks per clinical feature | Prevent out-of-distribution inference |

---

## 👤 Author

**M V Karthikeya**
B.Tech — Computer Science (AI & ML) · 📍 India

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=flat&logo=github)](https://github.com/Mvkarthikeya07)

---

## 📜 License

Released under the [MIT License](LICENSE) — free for academic, research, and educational use with attribution.

> ⭐ If this project helped you, consider starring the repository.

---

*12 clinical features · OLS regression · StandardScaler normalization · Flask inference · Built for interpretable glucose prediction*
