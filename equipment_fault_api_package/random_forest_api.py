from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Load dataset
data = pd.read_csv("equipment_anomaly_data.csv")

# Encode categorical columns
data_clean = data.copy()
for col in ["equipment", "location"]:
    le = LabelEncoder()
    data_clean[col] = le.fit_transform(data_clean[col])

# Prepare features and target
X = data_clean.drop(columns=["faulty"])
y = data_clean["faulty"]

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Set up Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "confidence": {
                "not_faulty": round(probability[0], 3),
                "faulty": round(probability[1], 3),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
