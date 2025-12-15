import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: No API key found in .env")
    exit()

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code != 200:
    print(f"Error connecting to Google: {response.status_code}")
    print(response.text)
    exit()

data = response.json()
print("\n--- YOUR AVAILABLE MODELS ---")
found_any = False
for model in data.get("models", []):
    if "generateContent" in model.get("supportedGenerationMethods", []):
        print(f"Name: {model['name']}")
        found_any = True

if not found_any:
    print("No chat models found! Check if 'Generative Language API' is enabled in Google Cloud Console.")