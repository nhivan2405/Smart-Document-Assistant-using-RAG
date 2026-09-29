"""Chia text của từng trang thành các chunk nhỏ."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages):
    # Cấu hình cách chia text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""] #separators là thứ tự ưu tiên vị trí để tách:
    )

    chunks = []

    # Lấy từng trang đã đọc từ document_loader
    for page in pages:

        # Chia text của trang thành nhiều chunk
        page_chunks = text_splitter.split_text(page["text"])

        # Lưu từng chunk cùng metadata
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "filename": page["filename"],
                "page": page["page"],
                "chunk_index": chunk_index,
                "text": chunk_text
            })

    return chunks