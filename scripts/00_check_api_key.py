import os
from dotenv import load_dotenv

load_dotenv()  # reads .env and loads its variables

key = os.environ.get("GROQ_API_KEY")

if key:
    print(f"Key loaded successfully. Starts with: {key[:8]}...")
else:
    print("No key found — check your .env file exists and is spelled correctly.")