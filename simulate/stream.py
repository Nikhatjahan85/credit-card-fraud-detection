import requests
import time
import random

url = "http://127.0.0.1:8000/predict"

while True:
    sample = {
        "Time": random.randint(0, 100000),
        "Amount": random.uniform(1, 5000),
        # add remaining fields same as dataset
    }

    res = requests.post(url, json=sample)
    print(res.json())

    time.sleep(2)