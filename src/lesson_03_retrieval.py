"""Lesson 03: Question → Embedding → ChromaDB → Top-K."""

from retriever import retrieve_chunks


def main():

    print("===== LESSON 03: RETRIEVAL =====")

    # 1. Người dùng nhập câu hỏi
    question = input("Nhập câu hỏi: ")

    # 2. Retrieval
    results = retrieve_chunks(question)

    # 3. Lấy dữ liệu từ kết quả ChromaDB
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 4. Hiển thị Top-K chunks
    print("\n===== TOP-K CHUNKS =====")

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        print(f"\n--- TOP {index} ---")
        print(f"Distance: {distance}")
        print(f"File: {metadata['filename']}")
        print(f"Page: {metadata['page']}")
        print(f"Chunk index: {metadata['chunk_index']}")
        print(f"Text: {document}")


if __name__ == "__main__":
    main()