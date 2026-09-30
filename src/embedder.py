"""Chuyển text thành vector bằng Embedding Model."""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


# 1. Cấu hình / chọn model embedding
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 2. Hàm chuyển danh sách chữ thành vector
def embed_texts(texts):
    """Chuyển danh sách text thành các vector."""

    vectors = model.encode(texts)

    return vectors