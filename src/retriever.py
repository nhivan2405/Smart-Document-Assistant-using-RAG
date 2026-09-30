"""Tìm các chunks liên quan đến câu hỏi trong ChromaDB."""

import chromadb

from config import CHROMA_PATH, COLLECTION_NAME, TOP_K
from embedder import embed_texts


# Kết nối đến ChromaDB local
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

# Lấy collection đã được tạo ở Lesson 2
collection = client.get_collection(
    name=COLLECTION_NAME
)


def retrieve_chunks(question):
    """Tìm Top-K chunks liên quan nhất với câu hỏi."""

    # 1. Chuyển câu hỏi thành vector
    question_vector = embed_texts([question])[0]

    # 2. Tìm các chunks gần nhất trong ChromaDB
    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=TOP_K
    )

    return results