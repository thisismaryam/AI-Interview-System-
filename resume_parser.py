
from pypdf import PdfReader
from docx import Document


def extract_resume_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
