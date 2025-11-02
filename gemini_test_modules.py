#!/usr/bin/env python3
"""
Advanced Gemini API testing modules for Streamlit UI.
Wraps the test scripts for use in the web interface.

Test files should be placed in the test_file/ directory.
"""

import os
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()


def test_gemini_basic(api_key: str) -> Tuple[bool, Dict]:
    """
    Test Gemini 2.5 Flash basic functionality.
    Based on test_gemini_2.5_flash.py

    Returns:
        Tuple of (success: bool, results: dict)
    """
    results = {"success": False, "tests": [], "errors": []}

    try:
        import google.generativeai as genai
    except ImportError:
        results["errors"].append("google-generativeai package not installed")
        return False, results

    try:
        genai.configure(api_key=api_key)

        # Test 1: Create model
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            results["tests"].append(
                {
                    "name": "Model Creation",
                    "status": "success",
                    "message": "Model created successfully",
                }
            )
        except Exception as e:
            results["tests"].append(
                {
                    "name": "Model Creation",
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }
            )
            results["errors"].append(f"Model creation failed: {e}")
            return False, results

        # Test 2: Simple text generation
        try:
            response = model.generate_content("Say 'Hello' in one word")
            results["tests"].append(
                {
                    "name": "Text Generation",
                    "status": "success",
                    "message": f"Response: {response.text}",
                }
            )
        except Exception as e:
            results["tests"].append(
                {
                    "name": "Text Generation",
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }
            )
            results["errors"].append(f"Text generation failed: {e}")
            return False, results

        # Test 3: JSON structured output
        try:
            response = model.generate_content(
                "Return a JSON object with one field 'greeting' set to 'Hello World'",
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
            )
            results["tests"].append(
                {
                    "name": "JSON Output",
                    "status": "success",
                    "message": "JSON response received",
                }
            )
        except Exception as e:
            results["tests"].append(
                {
                    "name": "JSON Output",
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }
            )
            results["errors"].append(f"JSON generation failed: {e}")
            return False, results

        # Test 4: Check model capabilities
        try:
            models = genai.list_models()
            found = False
            for m in models:
                if "gemini-2.5-flash" in m.name or "models/gemini-2.5-flash" == m.name:
                    found = True
                    results["tests"].append(
                        {
                            "name": "Model Availability",
                            "status": "success",
                            "message": f"Model found with methods: {list(m.supported_generation_methods)}",
                        }
                    )
                    break
            if not found:
                results["tests"].append(
                    {
                        "name": "Model Availability",
                        "status": "warning",
                        "message": "Model not found in list (may still work)",
                    }
                )
        except Exception as e:
            results["tests"].append(
                {
                    "name": "Model Availability",
                    "status": "warning",
                    "message": f"Could not check: {str(e)}",
                }
            )

        results["success"] = True
        return True, results

    except Exception as e:
        results["errors"].append(f"Configuration error: {e}")
        return False, results


def test_gemini_file_upload(
    api_key: str, test_pdf_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Test Gemini with file upload.
    Based on test_gemini_file_upload.py

    Args:
        api_key: Gemini API key
        test_pdf_path: Optional path to test PDF file

    Returns:
        Tuple of (success: bool, results: dict)
    """
    results = {"success": False, "tests": [], "errors": [], "model_used": None}

    try:
        import google.generativeai as genai
    except ImportError:
        results["errors"].append("google-generativeai package not installed")
        return False, results

    # Find a test PDF file
    if not test_pdf_path:
        # Check test_file directory first (primary test directory)
        test_file_dir = "test_file"
        test_locations = []

        # Add files from test_file directory
        if os.path.exists(test_file_dir):
            for file in os.listdir(test_file_dir):
                if file.lower().endswith(".pdf"):
                    test_locations.append(os.path.join(test_file_dir, file))

        # Fallback to other locations
        test_locations.extend(
            [
                "ec/RG EC 103 4.pdf",
                "ec/RG EC 103 3.pdf",
                "ec2/EC 24.pdf",
            ]
        )

        test_pdf_path = None
        for loc in test_locations:
            if os.path.exists(loc):
                test_pdf_path = loc
                break

        if not test_pdf_path:
            results["errors"].append(
                "No test PDF file found. Please add a PDF file to the test_file/ directory."
            )
            return False, results

    if not os.path.exists(test_pdf_path):
        results["errors"].append(f"Test PDF not found: {test_pdf_path}")
        return False, results

    try:
        genai.configure(api_key=api_key)

        models_to_test = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

        pdf_file = None
        for model_name in models_to_test:
            try:
                # Test model creation
                model = genai.GenerativeModel(model_name)
                results["tests"].append(
                    {
                        "name": f"Model Creation ({model_name})",
                        "status": "success",
                        "message": "Model created",
                    }
                )

                # Test file upload
                try:
                    pdf_file = genai.upload_file(path=test_pdf_path)
                    results["tests"].append(
                        {
                            "name": "File Upload",
                            "status": "success",
                            "message": f"File uploaded: {os.path.basename(test_pdf_path)}",
                        }
                    )

                    # Wait for processing
                    import time

                    max_wait = 60
                    waited = 0
                    while pdf_file.state.name == "PROCESSING" and waited < max_wait:
                        time.sleep(2)
                        waited += 2
                        pdf_file = genai.get_file(pdf_file.name)

                    if pdf_file.state.name == "FAILED":
                        results["tests"].append(
                            {
                                "name": "File Processing",
                                "status": "error",
                                "message": "File processing failed",
                            }
                        )
                        if pdf_file:
                            try:
                                genai.delete_file(pdf_file.name)
                            except Exception:
                                pass
                        continue

                    # Test generate_content with file
                    try:
                        response = model.generate_content(
                            [
                                "What is this document about? Answer in one sentence.",
                                pdf_file,
                            ],
                            generation_config={
                                "temperature": 0.1,
                            },
                        )
                        results["tests"].append(
                            {
                                "name": "Content Generation with File",
                                "status": "success",
                                "message": f"Response received: {response.text[:100]}...",
                            }
                        )

                        results["model_used"] = model_name
                        results["success"] = True

                        # Cleanup
                        if pdf_file:
                            try:
                                genai.delete_file(pdf_file.name)
                            except Exception:
                                pass

                        return True, results

                    except Exception as e:
                        results["tests"].append(
                            {
                                "name": "Content Generation with File",
                                "status": "error",
                                "message": f"Error: {str(e)}",
                            }
                        )
                        if pdf_file:
                            try:
                                genai.delete_file(pdf_file.name)
                            except Exception:
                                pass
                        continue

                except Exception as e:
                    results["tests"].append(
                        {
                            "name": "File Upload",
                            "status": "error",
                            "message": f"Error: {str(e)}",
                        }
                    )
                    continue

            except Exception as e:
                results["tests"].append(
                    {
                        "name": f"Model Creation ({model_name})",
                        "status": "error",
                        "message": f"Error: {str(e)}",
                    }
                )
                continue

        results["errors"].append("All models failed")
        return False, results

    except Exception as e:
        results["errors"].append(f"Configuration error: {e}")
        return False, results


def test_gemini_json_upload(
    api_key: str, test_pdf_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Test Gemini with file upload + JSON output.
    Based on test_gemini_json_upload.py

    Args:
        api_key: Gemini API key
        test_pdf_path: Optional path to test PDF file

    Returns:
        Tuple of (success: bool, results: dict)
    """
    results = {"success": False, "tests": [], "errors": [], "model_used": None}

    try:
        import google.generativeai as genai
    except ImportError:
        results["errors"].append("google-generativeai package not installed")
        return False, results

    # Find a test PDF file
    if not test_pdf_path:
        # Check test_file directory first (primary test directory)
        test_file_dir = "test_file"
        test_locations = []

        # Add files from test_file directory
        if os.path.exists(test_file_dir):
            for file in os.listdir(test_file_dir):
                if file.lower().endswith(".pdf"):
                    test_locations.append(os.path.join(test_file_dir, file))

        # Fallback to other locations
        test_locations.extend(
            [
                "ec/RG EC 103 4.pdf",
                "ec/RG EC 103 3.pdf",
                "ec2/EC 24.pdf",
            ]
        )

        test_pdf_path = None
        for loc in test_locations:
            if os.path.exists(loc):
                test_pdf_path = loc
                break

        if not test_pdf_path:
            results["errors"].append(
                "No test PDF file found. Please add a PDF file to the test_file/ directory."
            )
            return False, results

    if not os.path.exists(test_pdf_path):
        results["errors"].append(f"Test PDF not found: {test_pdf_path}")
        return False, results

    try:
        genai.configure(api_key=api_key)

        models_to_test = [
            ("gemini-2.5-flash", "2.5 Flash"),
            ("gemini-2.0-flash", "2.0 Flash"),
        ]

        pdf_file = None
        for model_name, display_name in models_to_test:
            try:
                model = genai.GenerativeModel(model_name)
                results["tests"].append(
                    {
                        "name": f"Model Creation ({display_name})",
                        "status": "success",
                        "message": "Model created",
                    }
                )

                # Upload file
                try:
                    pdf_file = genai.upload_file(path=test_pdf_path)
                    results["tests"].append(
                        {
                            "name": "File Upload",
                            "status": "success",
                            "message": "File uploaded",
                        }
                    )

                    # Wait for processing
                    import time

                    max_wait = 60
                    waited = 0
                    while pdf_file.state.name == "PROCESSING" and waited < max_wait:
                        time.sleep(2)
                        waited += 2
                        pdf_file = genai.get_file(pdf_file.name)

                    if pdf_file.state.name == "FAILED":
                        results["tests"].append(
                            {
                                "name": "File Processing",
                                "status": "error",
                                "message": "File processing failed",
                            }
                        )
                        if pdf_file:
                            try:
                                genai.delete_file(pdf_file.name)
                            except Exception:
                                pass
                        continue

                    # Test with JSON output
                    try:
                        response = model.generate_content(
                            [
                                "Return a JSON object with one field 'test' set to 'success'",
                                pdf_file,
                            ],
                            generation_config={
                                "temperature": 0.1,
                                "response_mime_type": "application/json",
                            },
                        )
                        results["tests"].append(
                            {
                                "name": "JSON Output with File",
                                "status": "success",
                                "message": f"JSON response received: {response.text[:100]}...",
                            }
                        )

                        results["model_used"] = model_name
                        results["success"] = True

                        # Cleanup
                        if pdf_file:
                            try:
                                genai.delete_file(pdf_file.name)
                            except Exception:
                                pass

                        return True, results

                    except Exception as e:
                        results["tests"].append(
                            {
                                "name": "JSON Output with File",
                                "status": "error",
                                "message": f"Error: {str(e)}",
                            }
                        )
                        if pdf_file:
                            try:
                                genai.delete_file(pdf_file.name)
                            except Exception:
                                pass
                        continue

                except Exception as e:
                    results["tests"].append(
                        {
                            "name": "File Upload",
                            "status": "error",
                            "message": f"Error: {str(e)}",
                        }
                    )
                    continue

            except Exception as e:
                results["tests"].append(
                    {
                        "name": f"Model Creation ({display_name})",
                        "status": "error",
                        "message": f"Error: {str(e)}",
                    }
                )
                continue

        results["errors"].append("All models failed")
        return False, results

    except Exception as e:
        results["errors"].append(f"Configuration error: {e}")
        return False, results
