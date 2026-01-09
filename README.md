🩸 A-Machine-Learning-Based-Framework-for-Blood-Glucose-Level-Prediction-Using-Clinical-Health-Data

📌 Project Overview

This project presents an end-to-end Machine Learning system for predicting blood glucose levels using clinical and lifestyle data derived from the Framingham Heart Study dataset. The system integrates data preprocessing, supervised regression modeling, performance evaluation, and web-based deployment, demonstrating the practical application of Machine Learning in healthcare analytics.

The project emphasizes statistical learning, proper feature selection, and model generalization, deliberately avoiding rule-based or lookup-based approaches. A lightweight Flask web application enables real-time glucose prediction, making the system accessible to non-technical users.

🎯 Objectives

To analyze real-world clinical data and identify key factors influencing blood glucose levels

To design and train a supervised regression model for continuous glucose prediction

To evaluate model performance using standard statistical metrics

To deploy the trained model as a user-friendly web application

🧠 Dataset Description

Dataset Source: Framingham Heart Study

Domain: Clinical and epidemiological health data

Number of Records: ~4,200 patients

Data Type: Numerical and binary medical features

🎯 Target Variable

glucose — Blood glucose level (mg/dL)

🔢 Input Features Used

Demographic: male, age

Lifestyle: currentSmoker, cigsPerDay

Medical history: BPMeds, prevalentHyp, diabetes

Clinical measurements: totChol, sysBP, diaBP, BMI, heartRate

Outcome-based variables such as TenYearCHD were intentionally excluded to prevent data leakage and ensure valid generalization.

⚙️ Methodology
1️⃣ Data Preprocessing

Removal of missing values

Selection of medically relevant and non-leaking features

Feature scaling using StandardScaler to normalize input distributions

2️⃣ Model Selection

Linear Regression was selected as the baseline model due to:

Interpretability in medical contexts

Suitability for continuous outcome prediction

Statistical transparency and simplicity

3️⃣ Training & Evaluation

Dataset split: 80% training / 20% testing

Evaluation metrics:

R² Score (explained variance)

Root Mean Squared Error (RMSE)

The model demonstrates stable predictive performance without overfitting, indicating an appropriate bias–variance trade-off.

🌐 System Architecture
User Input (Web Interface)
        ↓
Feature Scaling (StandardScaler)
        ↓
Trained Regression Model
        ↓
Glucose Level Prediction
        ↓
Result Display (Web Interface)

🖥️ Web Application

The trained model is deployed using Flask, providing a clean and minimal interface where users can enter patient health parameters and obtain an immediate glucose prediction.

Key Characteristics

Modular backend design

Clear separation between training and inference

Lightweight and easily deployable

Extensible for future clinical analytics features

🖼️ Screenshots & Demonstration

This section provides visual confirmation of the successful execution of the project and the integration between Machine Learning and web deployment.

1️⃣ Home Page – Input Interface

The home page allows users to input demographic, lifestyle, and clinical parameters required by the model.

<img width="1366" height="768" alt="Screenshot (50)" src="https://github.com/user-attachments/assets/a72ad146-5bcf-40cc-8084-bdd02edf6580" />

<img width="1366" height="768" alt="Screenshot (48)" src="https://github.com/user-attachments/assets/82ee22b2-1894-4d4f-9292-e2123842f56a" />

screenshots/home_page.png

2️⃣ Prediction Result Page

After submission, the system displays the predicted blood glucose level in mg/dL, ensuring clarity and interpretability.

<img width="1366" height="768" alt="Screenshot (49)" src="https://github.com/user-attachments/assets/0d0138ea-c7ac-41d5-86d2-575b5849d7e1" />

screenshots/prediction_result.png

📁 Project Structure
Glucose_Prediction_System/
│
├── dataset/
│   └── framingham.csv
│
├── model/
│   └── glucose_model.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── screenshots/
│   ├── home_page.png
│   ├── prediction_result.png
│   └── model_training_output.png
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

▶️ How to Run the Project
Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Train the model
python train_model.py

Step 3: Run the web application
python app.py

Step 4: Open in browser
http://127.0.0.1:5000/

📊 Output Description

Output Type: Continuous numerical value

Unit: mg/dL

Purpose: Estimation of blood glucose levels for analytical and educational use

🔬 Technology Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn

Machine Learning: Supervised Regression

Web Framework: Flask

🚀 Future Enhancements

Classification of glucose levels into clinical risk categories

Comparison with advanced regression models (Ridge, Lasso, Random Forest)

Feature importance analysis and visualization

Cloud deployment (Render / Railway)

Integration of explainable AI techniques

🎓 Academic & Professional Relevance

This project demonstrates:

Strong understanding of Machine Learning fundamentals

Practical handling of real-world medical datasets

Awareness of evaluation rigor and data leakage

Ability to build and deploy end-to-end ML systems

It is well-suited for:

DAAD scholarship applications

German public university Master’s programs

Research-oriented academic evaluation

Technical interviews and project defense

👤 Author

M V Karthikeya
Computer Science (AI & ML)
GitHub: https://github.com/Mvkarthikeya07

📜 License

This project is released under the MIT License and is intended strictly for academic and educational purposes.
