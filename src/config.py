"""Cấu hình chung cho project RAG."""

from pathlib import Path

# Thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Thư mục chứa tài liệu PDF
DATA_DIR = PROJECT_ROOT / "data"

# Cấu hình chia văn bản thành các chunk
CHUNK_SIZE = 500       # Kích thước tối đa của mỗi chunk
CHUNK_OVERLAP = 50     # Phần nội dung chồng lấn giữa 2 chunk

# Model chuyển text thành vector
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Vector Database
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "course_documents"