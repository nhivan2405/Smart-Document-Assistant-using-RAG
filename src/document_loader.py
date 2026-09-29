
"""Đọc file PDF và lấy text theo từng trang."""
from pathlib import Path # khai bao thu vien tung trang
from pypdf import PdfReader #khai bao thu vien pdf


def load_pdf(pdf_path):
    # Chuyển đường dẫn thành đối tượng Path
    pdf_path = Path(pdf_path)

    # Kiểm tra file PDF có tồn tại không
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

    # Mở và đọc file PDF
    reader = PdfReader(str(pdf_path))

    pages = []

    # Đọc lần lượt từng trang, bắt đầu đánh số từ 1
    for page_number, page in enumerate(reader.pages, start=1):

        # Lấy text của trang
        text = (page.extract_text() or "").strip()

        # Chỉ lưu trang có nội dung
        if text:
            pages.append({
                "filename": pdf_path.name,
                "page": page_number,
                "text": text
            })

    return pages