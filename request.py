import numpy as np
from network import test_images, test_labels
import requests
import json

data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:10].tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/handrecog:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
for i in range(10):
    print('Prediction: {0}, Actual: {1}'.format(np.argmax(predictions[i]), test_labels[i]))