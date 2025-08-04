import fitz
import os

def from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()