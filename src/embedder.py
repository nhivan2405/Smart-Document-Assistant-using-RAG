"""Chuyển text thành vector bằng Embedding Model."""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts):
    """Chuyển danh sách text thành các vector."""

    vectors = model.encode(texts)

    return vectors