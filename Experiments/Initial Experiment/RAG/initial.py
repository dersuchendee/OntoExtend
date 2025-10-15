import requests
import json

# Define the URL and payload
url = "http://localhost:8000/run"
headers = {"Content-Type": "application/json"}

payload = {
    "cq": "Who owns a product?",
    "start_rag": 'true',
    "core": "OntoDESIDECoreOntology",
    "class_count": 15,
    "op_count": 4,
    "dp_count": 5,
    'prompt':'pipe',
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Parse the JSON response (similar to jq)
try:
    result = response.json()
except json.JSONDecodeError:
    print("Response is not valid JSON:")
    print(response.text)
    result = None

# Save in variable (and optionally print)
print(json.dumps(result, indent=2))

