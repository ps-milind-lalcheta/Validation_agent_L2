import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")

print(f"Checking models at {base_url}models")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

try:
    response = requests.get(f"{base_url}models", headers=headers)
    if response.status_code == 200:
        data = response.json()
        for model in data.get("data", []):
            if "flash" in model["id"].lower():
                print(f"ID: {model['id']}")
    else:
        print(f"Status: {response.status_code}, Body: {response.text}")
except Exception as e:
    print(f"Error: {e}")
