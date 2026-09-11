import os
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
print("Key loaded:", GROQ_API_KEY[:8] + "...")
if not GROQ_API_KEY:
    raise ValueError("GROQ API key is missing")