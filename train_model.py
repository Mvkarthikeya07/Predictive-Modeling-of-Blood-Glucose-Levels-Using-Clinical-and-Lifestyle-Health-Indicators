import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("dataset/framingham.csv")

# Drop missing values
data = data.dropna()

# Feature selection
features = [
    "male", "age", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate"
]

X = data[features]
y = data["glucose"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # ✅ FIXED

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# Save model
with open("model/glucose_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model trained and saved successfully")
