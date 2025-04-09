import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()


# Configuration
API_URL = os.getenv("CHATUI_API_URL")
model = 'llama3.2:1b'
#model = 'llama3:8b'

# Prepare the payload
payload = {
    "model": f"{model}",
    "prompt": "tell me about denmarks president in one sentence.",
    "stream": False,
    "options": {
        #"seed": 101,
        "temperature": 0.9
    }
}

# Send the POST request
response = requests.post(api_url, data=json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    response_data = response.json()
    print("Generated Response:", response_data.get('response'))
else:
    print("Error:", response.status_code, response.text)
