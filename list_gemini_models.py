#!/usr/bin/env python3
"""List available Gemini models"""

import os
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set")
    exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models:")
print("=" * 70)
models = genai.list_models()
for m in models:
    if "generateContent" in m.supported_generation_methods:
        print(f"  - {m.name}")
        print(f"    Supported methods: {m.supported_generation_methods}")
        print()
