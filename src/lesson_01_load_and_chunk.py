"""Chạy thử quá trình: PDF → Text → Chunk."""
from config import DATA_DIR
from document_loader import load_pdf
from chunker import chunk_pages


# Đường dẫn đến file PDF cần test
PDF_PATH = DATA_DIR / "rag_test_document.pdf"


def main():
    print("===== LESSON 01: LOAD + CHUNK =====")

    # 1. Đọc PDF và lấy text theo từng trang
    pages = load_pdf(PDF_PATH)

    print(f"PDF: {PDF_PATH.name}")
    print(f"Số trang có text: {len(pages)}")

    # 2. Chia text của các trang thành chunks
    chunks = chunk_pages(pages)

    print(f"Số chunks: {len(chunks)}")

    # 3. In thử 5 chunks đầu tiên
    print("\n===== 5 CHUNKS ĐẦU TIÊN =====")

    for chunk in chunks[:5]:
        print("\n--- Chunk ---")
        print(f"File: {chunk['filename']}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Text: {chunk['text']}")


if __name__ == "__main__":
    main()