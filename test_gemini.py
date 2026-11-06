import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

model = os.environ.get("GEMINI_MODEL")
key = os.environ.get("GEMINI_API_KEY")
print("model:", model)
print("key loaded:", bool(key), flush=True)

client = genai.Client(
    api_key=key,
    http_options=types.HttpOptions(timeout=20000),  # 20 s, in milliseconds
)

print("calling Gemini...", flush=True)
t = time.time()
try:
    r = client.models.generate_content(model=model, contents="Say hi in one word.")
    print("reply:", r.text, f"({time.time() - t:.1f}s)")
except Exception as e:
    print("ERROR after", f"{time.time() - t:.1f}s:", repr(e))