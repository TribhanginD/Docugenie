import os
import tempfile
import pytest
from pages.Chatbot import extract_text_from_pdf, create_chunks


@pytest.fixture
def sample_pdf(tmp_path):
    content = "Page1\n\nThis is a test.\n\nPage2\n\nAnother chunk."
    # write a minimal two-page PDF
    from reportlab.pdfgen import canvas
    path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(path))
    for line in content.split("\n"):
        c.drawString(10, 800, line)
        c.showPage()
    c.save()
    return str(path)

def test_extract_text_from_pdf(sample_pdf):
    text = extract_text_from_pdf(sample_pdf)
    assert "Page1" in text
    assert "Another chunk." in text

def test_create_chunks_small_text():
    text = "a " * 50
    chunks = create_chunks(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    assert all(len(c) <= 20 + 5 for c in chunks)
