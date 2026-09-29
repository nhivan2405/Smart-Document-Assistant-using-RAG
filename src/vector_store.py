"""Lưu vector và metadata vào Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_URL, COLLECTION_NAME


# Kết nối đến Qdrant
client = QdrantClient(url=QDRANT_URL)


def create_collection(vector_size):
    """Tạo collection để lưu vector."""

    # Nếu collection đã tồn tại thì xóa để test lại từ đầu
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # Tạo collection mới
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )


def store_vectors(chunks, vectors):
    """Lưu vectors cùng metadata của chunks vào Qdrant."""

    points = []

    # Ghép từng chunk với vector tương ứng
    for point_id, (chunk, vector) in enumerate(zip(chunks, vectors)):

        point = PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload={
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
        )

        points.append(point)

    # Lưu tất cả points vào Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )