"""Chạy quá trình: PDF → Chunk → Embedding → Qdrant."""

from config import DATA_DIR
from document_loader import load_pdf
from chunker import chunk_pages
from embedder import embed_texts
from vector_store import create_collection, store_vectors


PDF_PATH = DATA_DIR / "rag_test_document.pdf"


def main():
    print("===== LESSON 02: EMBEDDING + QDRANT =====")

    # 1. Đọc PDF
    pages = load_pdf(PDF_PATH)
    print(f"Số trang có text: {len(pages)}")

    # 2. Chia thành chunks
    chunks = chunk_pages(pages)
    print(f"Số chunks: {len(chunks)}")

    # 3. Lấy text từ các chunks
    texts = [chunk["text"] for chunk in chunks]

    # 4. Chuyển text thành vectors
    vectors = embed_texts(texts)

    print(f"Số vectors: {len(vectors)}")
    print(f"Số chiều vector: {len(vectors[0])}")

    # 5. Tạo collection trong Qdrant
    vector_size = len(vectors[0])
    create_collection(vector_size)

    # 6. Lưu vector + metadata vào Qdrant
    store_vectors(chunks, vectors)

    print("Đã lưu vectors vào Qdrant thành công!")


if __name__ == "__main__":
    main()