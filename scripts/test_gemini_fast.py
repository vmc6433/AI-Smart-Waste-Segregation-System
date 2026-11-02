from pathlib import Path
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE = Path(
    r"C:\Users\vchet\OneDrive\Desktop\plastic_black_BG.png.png"
)


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# CHECK IMAGE
# ============================================================

if not TEST_IMAGE.exists():
    raise FileNotFoundError(
        f"Test image not found:\n{TEST_IMAGE}"
    )


image_bytes = TEST_IMAGE.read_bytes()


# ============================================================
# FAST PROMPT
# ============================================================

prompt = """
Classify this waste image.

Choose exactly ONE:

E_WASTE
ORGANIC
PLASTIC
RECYCLABLE

Return ONLY:

Category: <category>
Confidence: <0-100>%
"""


# ============================================================
# TEST
# ============================================================

print("=" * 70)
print("FAST GEMINI SPEED TEST")
print("=" * 70)

print(
    "Model: gemini-3.5-flash-lite"
)

print(
    "Thinking: minimal"
)

print(
    f"Image: {TEST_IMAGE}"
)

print("-" * 70)

print(
    "Sending image..."
)

start_time = time.perf_counter()


response = client.models.generate_content(

    model="gemini-3.5-flash-lite",

    contents=[

        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        ),

        prompt
    ],

    config={
        "thinking_config": {
            "thinking_level": "minimal"
        },
        "max_output_tokens": 30,
    }
)


elapsed = (
    time.perf_counter()
    - start_time
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("GEMINI RESULT")
print("=" * 70)

print(
    response.text
)

print("-" * 70)

print(
    f"Response time: {elapsed:.2f} seconds"
)

print("=" * 70)
