"""Chạy quá trình: PDF → Chunk → Embedding → Qdrant."""

from config import DATA_DIR
from document_loader import load_pdf
from chunker import chunk_pages
from embedder import embed_texts
from vector_store import store_vectors


PDF_PATH = DATA_DIR / "rag_test_document.pdf"


def main():
    print("===== LESSON 02: EMBEDDING + CHROMADB =====")

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

    # 5. Lưu vào ChromaDB
    collection = store_vectors(
        chunks,
        vectors
    )
     # 6. Kiểm tra số records
    print(f"Số records trong ChromaDB: {collection.count()}")

    print("\nĐã lưu vectors vào ChromaDB thành công!")


if __name__ == "__main__":
    main()