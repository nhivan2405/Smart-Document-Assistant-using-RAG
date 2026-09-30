"""Lưu vector, document và metadata vào ChromaDB."""

import chromadb

from config import CHROMA_PATH, COLLECTION_NAME


# Kết nối ChromaDB local
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


def create_collection():
    """Tạo collection mới để lưu document vectors."""

    # Chỉ dùng cho quá trình học/test:
    # nếu collection cũ tồn tại thì xóa để tạo lại sạch
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    return collection


def store_vectors(chunks, vectors):
    """Lưu chunks, vectors và metadata vào ChromaDB."""

    collection = create_collection()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)): #lưu dữ liệu

        ids.append(str(index))

        documents.append(
            chunk["text"]
        )

        embeddings.append(
            vector.tolist()
        )

        metadatas.append({
            "filename": chunk["filename"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"]
        })

    collection.add( #lưu tất cả vào chromadb
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return collection