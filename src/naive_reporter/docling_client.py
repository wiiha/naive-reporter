"""Client for docling-serve PDF-to-text conversion."""

from pathlib import Path

import httpx

from naive_reporter.config import settings

DOCLING_CONVERT_URL = f"{settings.docling_url}/v1/convert/file"
TIMEOUT = 120.0


def convert_pdf(pdf_path: Path) -> str:
    """Send a PDF to docling-serve and return extracted text.

    Tries ``text_content`` first, falls back to ``md_content`` when that is
    ``None``.

    Parameters
    ----------
    pdf_path
        Path to the PDF file.

    Returns
    -------
    str
        Extracted document text.

    Raises
    ------
    RuntimeError
        If the response contains no usable text.
    """
    with httpx.Client(timeout=TIMEOUT) as client:
        with pdf_path.open("rb") as file:
            response = client.post(
                DOCLING_CONVERT_URL,
                files={"files": (pdf_path.name, file, "application/pdf")},
            )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            if not data:
                raise RuntimeError("Empty docling response list")
            data = data[0]

        document = data.get("document", {})

        text = document.get("text_content")
        if text:
            return str(text)

        text = document.get("md_content")
        if text:
            return str(text)

        raise RuntimeError(
            "docling response contains neither text_content nor md_content"
        )
