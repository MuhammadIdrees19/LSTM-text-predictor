from pathlib import Path

import pymupdf


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF.

    Returns the full text content of the document.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text
