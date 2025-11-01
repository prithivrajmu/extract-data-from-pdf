#!/usr/bin/env python3
"""
Test Gemini 2.5 Flash with file upload + JSON output
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
print("Testing Gemini 2.5 Flash with File Upload + JSON Output")
print("=" * 70)
print()

test_pdf = "ec/RG EC 103 4.pdf"
if not os.path.exists(test_pdf):
    print(f"❌ Test PDF not found: {test_pdf}")
    exit(1)

models_to_test = [
    ('gemini-2.5-flash', '2.5 Flash'),
    ('gemini-2.0-flash', '2.0 Flash'),
]

for model_name, display_name in models_to_test:
    print(f"\n{'='*70}")
    print(f"Testing: {display_name} ({model_name})")
    print(f"{'='*70}\n")
    
    pdf_file = None
    try:
        model = genai.GenerativeModel(model_name)
        print(f"✅ Model created")
        
        # Upload file
        print(f"📤 Uploading PDF...")
        pdf_file = genai.upload_file(path=test_pdf)
        print(f"✅ File uploaded")
        
        # Wait for processing
        import time
        while pdf_file.state.name == "PROCESSING":
            time.sleep(2)
            pdf_file = genai.get_file(pdf_file.name)
        
        # Test with JSON output
        print(f"🔍 Testing with response_mime_type='application/json'...")
        response = model.generate_content(
            ["Return a JSON object with one field 'test' set to 'success'", pdf_file],
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            }
        )
        print(f"✅ Success with JSON output!")
        print(f"   Response: {response.text[:200]}...")
        
        # Cleanup
        genai.delete_file(pdf_file.name)
        print(f"✅ {display_name} works with file upload + JSON!")
        break
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {e}")
        if "model name format" in error_msg.lower():
            print(f"   ⚠️  Model name format issue")
        if "400" in error_msg:
            print(f"   ⚠️  400 error - API issue")
        if pdf_file:
            try:
                genai.delete_file(pdf_file.name)
            except:
                pass
        continue

print("\n" + "=" * 70)

