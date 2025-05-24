import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "temperature": 80.0,
    "pressure": 35.0,
    "vibration": 2.7,
    "humidity": 55.0,
    "equipment": 1,  # You can try 0, 1, or 2
    "location": 3,  # You can try 0 through 4
}

response = requests.post(url, json=data)
print("API Response:", response.json())
