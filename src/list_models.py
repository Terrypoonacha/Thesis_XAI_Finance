import google.generativeai as genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("API Key not found")
else:
    genai.configure(api_key=api_key)
    try:
        print("Listing available models:")
        for m in genai.list_models():
            print(f"Model: {m.name}")
            print(f"Supported methods: {m.supported_generation_methods}")
    except Exception as e:
        print(f"Error listing models: {e}")
