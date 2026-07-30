"""PDF page rasterization via pypdfium2."""

import pypdfium2 as pdfium


def pdf_to_images(pdf_path, dpi=300):
    """Render each page of a PDF to a PIL RGB image."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        return [page.render(scale=dpi / 72).to_pil().convert("RGB") for page in pdf]
    finally:
        pdf.close()
