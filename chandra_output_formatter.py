#!/usr/bin/env python3
"""
Formatter for Chandra OCR output to make it human-readable and user-friendly.
Filters out technical details and shows only relevant progress information.
"""

import re
import sys
from typing import Optional


class ChandraOutputFormatter:
    """Formats Chandra OCR output for better readability."""

    def __init__(self):
        self.in_config_section = False
        self.config_lines = 0
        self.last_progress_message = ""

    def format_line(self, line: str) -> Optional[str]:
        """
        Format a single line of output.
        Returns None if line should be filtered out, otherwise returns formatted line.
        """
        line = line.rstrip()

        # Skip empty lines
        if not line.strip():
            return None

        # Show important status messages
        if "Starting OCR processing" in line:
            return "🚀 Starting OCR processing..."

        if "Loading model" in line:
            self.in_config_section = True
            self.config_lines = 0
            return "📥 Loading model (this may take a few minutes)..."

        # Skip model configuration details (Qwen3VLConfig block)
        if self.in_config_section:
            if "Model config" in line or line.startswith("  ") or line.startswith("}"):
                self.config_lines += 1
                # Skip all config lines, only show when done
                if line.strip() == "}":
                    self.in_config_section = False
                    return "✓ Model configuration loaded"
                return None

        # Show file loading progress with percentages if available
        if "loading" in line.lower() and "from cache" in line.lower():
            # Extract meaningful info
            if "configuration file" in line.lower():
                return "  → Loading configuration..."
            elif "weights file" in line.lower():
                # Check if there's a progress indicator
                match = re.search(r"(\d+)/(\d+)", line)
                if match:
                    current, total = match.groups()
                    percent = int((int(current) / int(total)) * 100)
                    return f"  → Loading model weights ({percent}%)"
                return "  → Loading model weights..."
            elif "model.safetensors" in line:
                return "  → Loading model files..."

        # Show progress indicators
        if (
            "Processing page" in line
            or "page" in line.lower()
            and any(x in line for x in ["1", "2", "3", "4", "5"])
        ):
            page_match = re.search(r"page\s+(\d+)", line, re.IGNORECASE)
            if page_match:
                page_num = page_match.group(1)
                return f"📄 Processing page {page_num}..."

        # Show percentage progress
        if "%" in line:
            percent_match = re.search(r"(\d+)%")
            if percent_match:
                percent = percent_match.group(1)
                return f"⏳ Progress: {percent}%"

        # Show download progress
        if "downloading" in line.lower() or "download" in line.lower():
            # Extract size info if available
            size_match = re.search(r"([\d.]+)\s*(MB|GB|KB)", line, re.IGNORECASE)
            if size_match:
                size, unit = size_match.groups()
                return f"⬇️  Downloading: {size} {unit.upper()}"
            return "⬇️  Downloading files..."

        # Show batch processing info
        if "batch" in line.lower():
            batch_match = re.search(r"batch\s+(\d+)", line, re.IGNORECASE)
            if batch_match:
                batch_num = batch_match.group(1)
                return f"📦 Processing batch {batch_num}..."

        # Show completion messages
        if "completed" in line.lower() or "finished" in line.lower():
            return f"✅ {line.strip()}"

        # Show error/warning messages (keep these)
        if (
            "error" in line.lower()
            or "warning" in line.lower()
            or "exception" in line.lower()
        ):
            if "error" in line.lower():
                return f"❌ {line.strip()}"
            return f"⚠️  {line.strip()}"

        # Skip very long technical lines (likely config or debug)
        if len(line) > 200:
            return None

        # Skip lines that are just JSON or config
        if line.strip().startswith("{") or line.strip().startswith("}"):
            return None

        # Skip lines with only brackets, commas, quotes
        if re.match(r'^[\s\[\]{}",:]+$', line):
            return None

        # Show informational messages that are short and meaningful
        if len(line) < 100 and not any(
            x in line
            for x in ["Qwen3VLConfig", "_attn_", "architectures", "base_model"]
        ):
            # Only show if it's a meaningful message
            if any(
                keyword in line.lower()
                for keyword in [
                    "processing",
                    "extracting",
                    "saving",
                    "output",
                    "input",
                    "method",
                ]
            ):
                return f"ℹ️  {line.strip()}"

        # Filter out most other technical details
        return None

    def format_output(self, lines: list) -> list:
        """
        Format a list of output lines.
        Returns list of formatted lines.
        """
        formatted = []
        last_output = None

        for line in lines:
            formatted_line = self.format_line(line)
            if formatted_line and formatted_line != last_output:
                formatted.append(formatted_line)
                last_output = formatted_line

        return formatted


def filter_chandra_output(input_stream, output_stream=sys.stdout):
    """
    Filter and format Chandra OCR output in real-time.

    Args:
        input_stream: Input stream (file-like object)
        output_stream: Output stream (default: stdout)
    """
    formatter = ChandraOutputFormatter()

    for line in input_stream:
        formatted = formatter.format_line(line)
        if formatted:
            output_stream.write(formatted + "\n")
            output_stream.flush()


if __name__ == "__main__":
    # Can be used as a standalone filter
    filter_chandra_output(sys.stdin, sys.stdout)
