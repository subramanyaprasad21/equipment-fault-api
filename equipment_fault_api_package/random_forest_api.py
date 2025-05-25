from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Load your dataset (make sure path is correct)
data = pd.read_csv("equipment_fault_api_package/equipment_anomaly_data.csv")

# Encode categorical columns and keep encoders
equipment_le = LabelEncoder()
location_le = LabelEncoder()
data['equipment'] = equipment_le.fit_transform(data['equipment'])
data['location'] = location_le.fit_transform(data['location'])

# Prepare features and target
X = data.drop(columns=["faulty"])
y = data["faulty"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert string to encoded if needed
        if isinstance(data.get("equipment"), str):
            data["equipment"] = int(equipment_le.transform([data["equipment"]])[0])
        if isinstance(data.get("location"), str):
            data["location"] = int(location_le.transform([data["location"]])[0])

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "confidence": {
                "not_faulty": round(proba[0], 3),
                "faulty": round(proba[1], 3)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
