import requests

from src.utils import load_dataset

X, y = load_dataset()
headers = {
    "Content-Type": "application/json",
}

data = {"input": list(X.iloc[0]), "model_type": "random_forest"}

response = requests.post("http://localhost:5000/predict", headers=headers, json=data)
prediction = response.json()["prediction"]
print(f"Predicted class: {prediction}")
