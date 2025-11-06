#!/usr/bin/env python3
"""
Performance benchmarking utilities for PDF extraction methods.

This module provides tools to benchmark extraction performance across different
methods, comparing extraction time, memory usage, and other metrics.
"""

import os
import time
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

from logging_config import get_logger

logger = get_logger(__name__)


class ExtractionBenchmark:
    """Benchmark extraction performance metrics."""

    def __init__(self):
        """Initialize benchmark tracker."""
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.start_memory: float | None = None
        self.end_memory: float | None = None
        self.process = psutil.Process(os.getpid()) if psutil else None

    def start(self) -> None:
        """Start timing and memory tracking."""
        self.start_time = time.time()
        if self.process:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def stop(self) -> None:
        """Stop timing and memory tracking."""
        self.end_time = time.time()
        if self.process:
            self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Benchmark not started or stopped")
        return self.end_time - self.start_time

    def get_memory_delta(self) -> float | None:
        """Get memory delta in MB."""
        if self.start_memory is None or self.end_memory is None:
            return None
        return self.end_memory - self.start_memory

    def get_results(self) -> dict[str, Any]:
        """Get benchmark results as dictionary."""
        results = {
            "elapsed_time_seconds": self.get_elapsed_time(),
        }
        if self.start_memory is not None and self.end_memory is not None:
            results["memory_start_mb"] = self.start_memory
            results["memory_end_mb"] = self.end_memory
            results["memory_delta_mb"] = self.get_memory_delta()
        return results


def benchmark_extraction(
    extraction_func: callable,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Benchmark an extraction function.

    Args:
        extraction_func: Function to benchmark (should return list of dicts)
        *args: Positional arguments to pass to extraction function
        **kwargs: Keyword arguments to pass to extraction function

    Returns:
        Dictionary containing benchmark results:
        - elapsed_time_seconds: Time taken for extraction
        - memory_delta_mb: Memory usage change (if psutil available)
        - rows_extracted: Number of rows extracted
        - success: Whether extraction succeeded
    """
    benchmark = ExtractionBenchmark()
    rows_extracted = 0
    success = False
    error = None

    try:
        benchmark.start()
        result = extraction_func(*args, **kwargs)
        benchmark.stop()

        if isinstance(result, list):
            rows_extracted = len(result)
            success = True
        else:
            logger.warning("Extraction function did not return a list")
            success = False

    except Exception as e:
        benchmark.stop()
        success = False
        error = str(e)
        logger.exception("Extraction failed during benchmark")

    results = benchmark.get_results()
    results["rows_extracted"] = rows_extracted
    results["success"] = success
    if error:
        results["error"] = error

    return results


def compare_methods(
    pdf_path: str,
    methods: list[str],
    api_keys: dict[str, str],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """
    Compare performance of multiple extraction methods.

    Args:
        pdf_path: Path to PDF file to extract from
        methods: List of method names to compare
        api_keys: Dictionary of API keys
        **kwargs: Additional arguments to pass to extract_data

    Returns:
        Dictionary mapping method names to benchmark results
    """
    from extraction_service import extract_data

    results = {}
    for method in methods:
        logger.info(f"Benchmarking method: {method}")
        try:
            benchmark_result = benchmark_extraction(
                extract_data,
                pdf_path,
                method,
                api_keys,
                **kwargs,
            )
            results[method] = benchmark_result
        except Exception as e:
            logger.exception(f"Failed to benchmark method {method}")
            results[method] = {
                "success": False,
                "error": str(e),
            }

    return results


def print_benchmark_results(results: dict[str, dict[str, Any]]) -> None:
    """
    Print benchmark results in a formatted table.

    Args:
        results: Dictionary mapping method names to benchmark results
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    header = f"{'Method':<20} {'Time (s)':<12} {'Rows':<8} {'Memory Δ (MB)':<15} {'Status':<10}"
    print(header)
    print("-" * 80)

    # Results
    for method, result in results.items():
        time_str = f"{result.get('elapsed_time_seconds', 0):.2f}"
        rows_str = str(result.get("rows_extracted", 0))
        memory_str = (
            f"{result.get('memory_delta_mb', 0):.2f}"
            if result.get("memory_delta_mb") is not None
            else "N/A"
        )
        status = "✓ Success" if result.get("success") else "✗ Failed"
        if not result.get("success") and "error" in result:
            status += f": {result['error'][:30]}"

        row = f"{method:<20} {time_str:<12} {rows_str:<8} {memory_str:<15} {status:<10}"
        print(row)

    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark extraction methods")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["easyocr", "pytesseract"],
        help="Methods to benchmark",
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        help="Path to .env file with API keys",
    )

    args = parser.parse_args()

    # Load API keys if provided
    api_keys = {}
    if args.api_keys:
        from dotenv import load_dotenv

        load_dotenv(args.api_keys)
        from api_key_manager import load_api_key

        api_keys = {
            "datalab": load_api_key("datalab") or "",
            "gemini": load_api_key("gemini") or "",
            "deepseek": load_api_key("deepseek") or "",
            "huggingface": load_api_key("huggingface") or "",
        }

    # Run benchmarks
    results = compare_methods(args.pdf_path, args.methods, api_keys)
    print_benchmark_results(results)

