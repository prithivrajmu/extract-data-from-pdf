#!/usr/bin/env python3
"""
Simple test script to verify Gemini 2.5 Flash API works
"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not installed.")
    exit(1)

# Configure API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ Error: GEMINI_API_KEY not set!")
    exit(1)

genai.configure(api_key=api_key)

print("=" * 70)
print("Testing Gemini 2.5 Flash API")
print("=" * 70)
print()

# Test 1: Create model
print("Test 1: Creating model instance...")
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    print(f"✅ Model created successfully")
    print(f"   Model object: {model}")
    print(f"   Internal model_name: {getattr(model, 'model_name', 'N/A')}")
    print()
except Exception as e:
    print(f"❌ Error creating model: {e}")
    exit(1)

# Test 2: Simple text generation
print("Test 2: Testing simple text generation...")
try:
    response = model.generate_content("Say 'Hello' in one word")
    print(f"✅ Text generation successful")
    print(f"   Response: {response.text}")
    print()
except Exception as e:
    print(f"❌ Error in text generation: {e}")
    exit(1)

# Test 3: JSON structured output
print("Test 3: Testing structured JSON output...")
try:
    response = model.generate_content(
        "Return a JSON object with one field 'greeting' set to 'Hello World'",
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
    )
    print(f"✅ JSON generation successful")
    print(f"   Response: {response.text}")
    print()
except Exception as e:
    print(f"❌ Error in JSON generation: {e}")
    print(f"   Error details: {str(e)}")
    exit(1)

# Test 4: Check if model supports file uploads
print("Test 4: Checking model capabilities...")
try:
    # List models to check capabilities
    models = genai.list_models()
    for m in models:
        if m.name == 'models/gemini-2.5-flash':
            print(f"✅ Found gemini-2.5-flash in available models")
            print(f"   Supported methods: {m.supported_generation_methods}")
            if 'generateContent' in m.supported_generation_methods:
                print(f"   ✅ Supports generateContent")
            break
    else:
        print(f"⚠️  Model gemini-2.5-flash not found in list")
    print()
except Exception as e:
    print(f"⚠️  Could not check model capabilities: {e}")
    print()

print("=" * 70)
print("✅ All tests passed! Gemini 2.5 Flash API is working correctly.")
print("=" * 70)

