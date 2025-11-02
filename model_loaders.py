#!/usr/bin/env python3
"""
Generic model loaders for different OCR models.
Allows supporting multiple models via transformers library.
"""

import os
import subprocess
from pathlib import Path


def load_chandra_model(
    pdf_path: str, output_dir: str, use_cpu: bool = False
) -> tuple[str, dict | None]:
    """
    Load and run Chandra OCR model via CLI.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for results
        use_cpu: Force CPU mode

    Returns:
        Tuple of (extracted_text, structured_data)
    """
    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"] = "info"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HF_DEVICE_MAP"] = "cpu"

    # Run Chandra CLI
    process = subprocess.Popen(
        ["chandra", pdf_path, output_dir, "--method", "hf"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Chandra OCR failed: {stderr}")

    # Read output
    output_files = list(Path(output_dir).glob("*.md"))
    if output_files:
        text = output_files[0].read_text()
    else:
        text = stdout

    # Try to get structured data
    structured_data = None
    json_files = list(Path(output_dir).glob("*.json"))
    if json_files:
        import json

        structured_data = json.loads(json_files[0].read_text())

    return text, structured_data


def load_transformers_model(
    model_name: str, pdf_path: str, use_cpu: bool = False
) -> tuple[str, dict | None]:
    """
    Generic loader for models via transformers library.

    Args:
        model_name: HuggingFace model identifier
        pdf_path: Path to PDF file
        use_cpu: Force CPU mode

    Returns:
        Tuple of (extracted_text, structured_data)

    Note: This is a framework for future model support.
    Each model type may need custom preprocessing/postprocessing.
    """
    try:
        import torch
        from pdf2image import convert_from_path
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        model.to(device)
        model.eval()

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)

        # Process each page
        all_text_parts = []
        for image in images:
            # Process image (model-specific preprocessing)
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Generate text
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_length=512)

            # Decode text
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            all_text_parts.append(generated_text)

        full_text = "\n\n".join(all_text_parts)
        return full_text, None

    except ImportError as e:
        raise ImportError(
            f"Required packages not installed for {model_name}. "
            f"Install with: pip install transformers torch torchvision"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model {model_name}: {e}. "
            f"This model may require specific preprocessing or is not compatible with this loader."
        ) from e


def get_model_loader(model_name: str):
    """
    Get the appropriate loader function for a model.

    Args:
        model_name: Model identifier

    Returns:
        Loader function and its type
    """
    # Models that use Chandra CLI
    chandra_models = ["datalab-to/chandra", "chandra"]

    # Models that can use transformers
    transformers_models = [
        "microsoft/trocr-base-printed",
        "microsoft/trocr-base-handwritten",
        "microsoft/trocr-large-printed",
        "microsoft/trocr-large-handwritten",
    ]

    # Check if model uses Chandra CLI
    if any(chandra in model_name.lower() for chandra in chandra_models):
        return load_chandra_model, "chandra_cli"

    # Check if model can use transformers
    elif any(tf in model_name.lower() for tf in transformers_models):
        return load_transformers_model, "transformers"

    # Default: try transformers (may fail, but gives user feedback)
    else:
        return load_transformers_model, "transformers_attempt"


def is_model_supported(model_name: str) -> tuple[bool, str]:
    """
    Check if a model is supported and return status.

    Args:
        model_name: Model identifier

    Returns:
        Tuple of (is_supported, reason_message)
    """
    chandra_models = ["datalab-to/chandra", "chandra"]
    transformers_models = [
        "microsoft/trocr-base-printed",
        "microsoft/trocr-base-handwritten",
        "microsoft/trocr-large-printed",
        "microsoft/trocr-large-handwritten",
    ]

    if any(chandra in model_name.lower() for chandra in chandra_models):
        # Check if chandra CLI is available
        try:
            subprocess.run(
                ["chandra", "--version"], capture_output=True, check=True, timeout=5
            )
            return True, "Fully supported via Chandra CLI"
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return (
                False,
                "Chandra CLI not installed. Install with: pip install chandra-ocr",
            )

    elif any(tf in model_name.lower() for tf in transformers_models):
        return (
            True,
            "Supported via transformers library (may need additional dependencies)",
        )

    else:
        return (
            False,
            f"Model '{model_name}' needs custom implementation. We can add support - please file an issue!",
        )
