"""Test quá trình: PDF → Chunk → Embedding → Vector."""

from config import DATA_DIR
from document_loader import load_pdf
from chunker import chunk_pages
from embedder import embed_texts


# File PDF dùng để test
PDF_PATH = DATA_DIR / "rag_test_document.pdf"


def main():
    print("===== LESSON 02: EMBEDDING =====")

    # 1. Đọc PDF
    pages = load_pdf(PDF_PATH)

    # 2. Chia text thành chunks
    chunks = chunk_pages(pages)

    # 3. Lấy nội dung text của các chunk
    texts = [chunk["text"] for chunk in chunks]

    # 4. Chuyển các text thành vector
    vectors = embed_texts(texts)

    # 5. In kết quả để kiểm tra
    print(f"Số chunks: {len(chunks)}")
    print(f"Số vectors: {len(vectors)}")
    print(f"Số chiều vector: {len(vectors[0])}")

    # In thử chunk đầu tiên và vector của nó
    print("\n===== CHUNK ĐẦU TIÊN =====")
    print(chunks[0]["text"])

    print("\n===== VECTOR ĐẦU TIÊN =====")
    print(vectors[0])


if __name__ == "__main__":
    main()