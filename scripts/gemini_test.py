from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\vchet\OneDrive\Desktop\plastic_black_BG.png.png"
)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# ============================================================
# READ IMAGE
# ============================================================

image_bytes = IMAGE_PATH.read_bytes()


# ============================================================
# PROMPT
# ============================================================

prompt = """
You are an AI assistant for a waste segregation system.

Analyze the provided image carefully.

First identify what object or material is visible.

Then classify it into EXACTLY ONE of these four categories:

E_WASTE
ORGANIC
PLASTIC
RECYCLABLE

Definitions:

E_WASTE:
Electronic or electrical waste such as phones, chargers, cables,
batteries, keyboards, electronic components, etc.

ORGANIC:
Food waste, vegetable waste, fruit waste, leaves,
and other biodegradable organic material.

PLASTIC:
Waste primarily made of plastic such as plastic bottles,
plastic containers, plastic packaging, plastic bags, etc.

RECYCLABLE:
Common recyclable materials such as paper, cardboard,
glass, and metal that are not primarily plastic or electronic waste.

Return exactly:

Object: <identified object>
Category: <one category>
Confidence: <0-100>%
Reason: <short explanation>
"""


# ============================================================
# SEND IMAGE TO GEMINI
# ============================================================

print("=" * 70)
print("AI WASTE SEGREGATION SYSTEM")
print("GEMINI IMAGE TEST")
print("=" * 70)

print(f"Image: {IMAGE_PATH}")
print("Sending image to Gemini...")
print("-" * 70)


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        ),
        prompt
    ]
)


# ============================================================
# RESULT
# ============================================================

print("\nGEMINI RESULT")
print("=" * 70)
print(response.text)
print("=" * 70)