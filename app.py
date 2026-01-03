from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model/glucose_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        int(request.form["male"]),
        float(request.form["age"]),
        int(request.form["currentSmoker"]),
        float(request.form["cigsPerDay"]),
        int(request.form["BPMeds"]),
        int(request.form["prevalentHyp"]),
        int(request.form["diabetes"]),
        float(request.form["totChol"]),
        float(request.form["sysBP"]),
        float(request.form["diaBP"]),
        float(request.form["BMI"]),
        float(request.form["heartRate"])
    ]

    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]

    return render_template(
        "results.html",
        glucose=round(prediction, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
