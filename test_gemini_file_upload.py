#!/usr/bin/env python3
"""
Test Gemini 2.5 Flash with file upload
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
print("Testing Gemini 2.5 Flash with File Upload")
print("=" * 70)
print()

# Use one of the PDF files
test_pdf = "ec/RG EC 103 4.pdf"
if not os.path.exists(test_pdf):
    print(f"❌ Test PDF not found: {test_pdf}")
    exit(1)

# Test different model name formats
models_to_test = [
    'gemini-2.5-flash',
    'gemini-2.0-flash',  # Known to work
]

for model_name in models_to_test:
    print(f"\n{'='*70}")
    print(f"Testing with model: {model_name}")
    print(f"{'='*70}\n")
    
    try:
        # Create model
        model = genai.GenerativeModel(model_name)
        print(f"✅ Model created: {model_name}")
        
        # Upload file
        print(f"📤 Uploading PDF: {test_pdf}...")
        pdf_file = genai.upload_file(path=test_pdf)
        print(f"✅ File uploaded: {pdf_file.uri}")
        
        # Wait for processing
        import time
        while pdf_file.state.name == "PROCESSING":
            print("   Waiting for file processing...")
            time.sleep(2)
            pdf_file = genai.get_file(pdf_file.name)
        
        if pdf_file.state.name == "FAILED":
            print(f"❌ File upload failed")
            continue
        
        # Test with simple prompt
        print(f"🔍 Testing generate_content with file upload...")
        response = model.generate_content(
            ["What is this document about? Answer in one sentence.", pdf_file],
            generation_config={
                "temperature": 0.1,
            }
        )
        print(f"✅ Success! Response: {response.text}")
        
        # Cleanup
        genai.delete_file(pdf_file.name)
        print(f"🧹 Cleaned up uploaded file")
        
        print(f"\n✅ {model_name} works with file uploads!")
        break
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error with {model_name}: {e}")
        if "model name format" in error_msg.lower():
            print(f"   ⚠️  Model name format issue detected")
        if pdf_file:
            try:
                genai.delete_file(pdf_file.name)
            except:
                pass
        print()
        continue

print("\n" + "=" * 70)

