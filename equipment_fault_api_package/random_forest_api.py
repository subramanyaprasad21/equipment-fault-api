from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

csv_path = os.path.join(os.path.dirname(__file__), "equipment_anomaly_data.csv")
data = pd.read_csv(csv_path)

equipment_le = LabelEncoder()
location_le = LabelEncoder()
data['equipment'] = equipment_le.fit_transform(data['equipment'])
data['location'] = location_le.fit_transform(data['location'])

X = data[['equipment', 'location', 'temperature', 'pressure', 'humidity']]
y = data['faulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Equipment Fault Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json()
    equipment = data_json['equipment']
    location = data_json['location']
    temperature = data_json['temperature']
    pressure = data_json['pressure']
    humidity = data_json['humidity']

    # Encode input
    equipment_enc = equipment_le.transform([equipment])[0]
    location_enc = location_le.transform([location])[0]
    X_pred = [[equipment_enc, location_enc, temperature, pressure, humidity]]

    prediction = int(model.predict(X_pred)[0])  # <-- Cast to int
    confidence = model.predict_proba(X_pred)[0][prediction]

    label = "Faulty" if prediction == 1 else "Not Faulty"

    result = {
        "faulty": prediction,
        "label": label,
        "confidence": round(confidence * 100, 2),
        "equipment": equipment,
        "location": location
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
