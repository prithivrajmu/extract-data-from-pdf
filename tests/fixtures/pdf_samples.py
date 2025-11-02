"""Utility helpers for creating sample PDF files for tests."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PyPDF2 import PdfWriter


@dataclass
class FakeUploadedFile:
    """Minimal stub mimicking Streamlit's UploadedFile for tests."""

    name: str
    _data: bytes

    def getvalue(self) -> bytes:  # pragma: no cover - simple accessor
        return self._data

    def seek(
        self, position: int, whence: int = 0
    ) -> None:  # pragma: no cover - delegate
        # Some downstream consumers expect seek to exist; nothing else uses the result.
        # We provide a no-op implementation because validate_pdf_file simply resets the pointer.
        # The file content is served from stored bytes so seek has no side-effects.
        _ = position
        _ = whence


def build_pdf_bytes(num_pages: int = 1) -> bytes:
    """Create a simple in-memory PDF document."""

    writer = PdfWriter()
    for _ in range(max(num_pages, 1)):
        writer.add_blank_page(width=72, height=72)

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_uploaded_pdf(name: str = "sample.pdf", num_pages: int = 1) -> FakeUploadedFile:
    """Return a FakeUploadedFile representing a valid PDF."""

    pdf_bytes = build_pdf_bytes(num_pages=num_pages)
    return FakeUploadedFile(name=name, _data=pdf_bytes)


def make_corrupt_pdf(
    name: str = "corrupt.pdf",
    *,
    missing_header: bool = False,
    missing_eof: bool = False,
) -> FakeUploadedFile:
    """Return a FakeUploadedFile with specific corruption applied."""

    pdf_bytes = build_pdf_bytes()

    if missing_header:
        pdf_bytes = pdf_bytes.replace(b"%PDF", b"%TXT", 1)
    if missing_eof:
        eof_index = pdf_bytes.rfind(b"%%EOF")
        if eof_index != -1:
            pdf_bytes = pdf_bytes[:eof_index]

    return FakeUploadedFile(name=name, _data=pdf_bytes)
