import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY"
if not api_key:
    print("GOOGLE_API_KEY not found in .env file. Please set it.")
else:
    genai.configure(api_key=api_key)

    print("Available models supporting generateContent:")
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"An error occurred while listing models: {e}")