# RAG_Project# Smart Document Assistant using RAG

## 1. Giới thiệu

**Smart Document Assistant using RAG** là một hệ thống trợ lý tài liệu thông minh sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)** để đọc tài liệu PDF, tìm kiếm thông tin liên quan và tạo câu trả lời dựa trên nội dung của tài liệu.

Thay vì để Large Language Model (LLM) trả lời hoàn toàn dựa trên kiến thức đã được huấn luyện trước đó, hệ thống sẽ tìm kiếm các đoạn nội dung liên quan trong tài liệu, đưa chúng vào ngữ cảnh và sau đó yêu cầu LLM tạo câu trả lời.

Project được xây dựng nhằm tìm hiểu và thực hành các thành phần chính của một hệ thống RAG:

```text
Document Processing
        ↓
Chunking
        ↓
Embedding
        ↓
Vector Database
        ↓
Retrieval
        ↓
LLM
        ↓
Answer + Citation
```

---

## 2. Bài toán

Các mô hình ngôn ngữ lớn như LLM có khả năng trả lời nhiều loại câu hỏi, tuy nhiên chúng có một số hạn chế:

- Không tự biết nội dung của tài liệu riêng do người dùng cung cấp.
- Có thể trả lời sai hoặc tạo ra thông tin không tồn tại trong tài liệu.
- Không thể luôn chỉ ra chính xác nguồn của câu trả lời.
- Không được cập nhật trực tiếp từ tài liệu mới nếu không có cơ chế bổ sung kiến thức.

Ví dụ, nếu người dùng có một tài liệu PDF môn học và hỏi:

```text
RAG là gì?
```

LLM thông thường có thể trả lời dựa trên kiến thức chung.

Trong project này, hệ thống sẽ:

```text
Câu hỏi
   ↓
Tìm nội dung liên quan trong PDF
   ↓
Lấy các đoạn phù hợp nhất
   ↓
Đưa vào LLM
   ↓
Sinh câu trả lời dựa trên tài liệu
```

Nhờ đó câu trả lời có thể được **grounded** vào nội dung thật của tài liệu.

---

## 3. Mục tiêu của project

Project hướng đến xây dựng một Smart Document Assistant có khả năng:

- Upload hoặc đọc tài liệu PDF.
- Trích xuất nội dung văn bản từ tài liệu.
- Chia nội dung thành các đoạn nhỏ (chunks).
- Tạo embedding cho từng chunk.
- Lưu embedding vào Vector Database.
- Chuyển câu hỏi của người dùng thành embedding.
- Tìm các chunks có nội dung liên quan nhất.
- Đưa các chunks đó vào LLM làm context.
- Sinh câu trả lời dựa trên nội dung tài liệu.
- Trả về nguồn tài liệu hoặc số trang của thông tin.
- Xử lý trường hợp tài liệu không chứa câu trả lời.

Trong các giai đoạn mở rộng, hệ thống có thể hỗ trợ thêm:

- Chat nhiều lượt.
- Upload nhiều tài liệu.
- Quiz generation.
- Tóm tắt tài liệu.
- FastAPI backend.
- Web UI.
- User authentication.
- Agent và Tools.
- Lưu lịch sử hội thoại.
- Automation với n8n.

---

# 4. RAG là gì?

**RAG - Retrieval-Augmented Generation** là kỹ thuật kết hợp giữa:

### Retrieval

Tìm kiếm những thông tin liên quan nhất từ một nguồn dữ liệu bên ngoài.

Ví dụ:

```text
PDF
↓
Vector Database
↓
Tìm Top-K chunks
```

### Generation

Sử dụng LLM để tạo câu trả lời dựa trên thông tin vừa được retrieval.

```text
Question + Retrieved Context
              ↓
             LLM
              ↓
            Answer
```

Do đó luồng đầy đủ là:

```text
User Question
      ↓
Question Embedding
      ↓
Vector Search
      ↓
Relevant Chunks
      ↓
Context + Question
      ↓
LLM
      ↓
Answer
```

---

# 5. RAG không phải là Train LLM

Trong project này, tài liệu **không được dùng để train hoặc fine-tune LLM**.

Quy trình đúng là:

```text
Document
   ↓
Extract Text
   ↓
Chunking
   ↓
Embedding
   ↓
Index vào Vector Database
   ↓
Retrieve khi có câu hỏi
```

LLM vẫn giữ nguyên.

Khi người dùng hỏi, hệ thống mới lấy thông tin liên quan từ Vector Database và cung cấp cho LLM.

---

# 6. Kiến trúc tổng thể

```text
                         USER
                           │
                           ▼
                   Smart Assistant
                           │
                           ▼
                  Question Processing
                           │
                           ▼
                   Embedding Model
                           │
                           ▼
                     Vector Search
                           │
                           ▼
                         Qdrant
                           │
                           ▼
                    Top-K Chunks
                           │
                           ▼
                 Context + Question
                           │
                           ▼
                          LLM
                           │
                           ▼
                 Answer + Citation
```

Phần xử lý tài liệu:

```text
PDF Document
      │
      ▼
Document Loader
      │
      ▼
Extract Text
      │
      ▼
Text Splitter
      │
      ▼
Chunks
      │
      ▼
Embedding Model
      │
      ▼
Vectors
      │
      ▼
Qdrant Vector Database
```

---

# 7. Luồng hoạt động của hệ thống

Hệ thống gồm hai quá trình chính.

## 7.1. Indexing Pipeline

Đây là quá trình xử lý tài liệu trước khi người dùng đặt câu hỏi.

```text
PDF
 ↓
Load Document
 ↓
Extract Text
 ↓
Chunking
 ↓
Embedding
 ↓
Store in Qdrant
```

### Bước 1 - Load Document

Hệ thống đọc file PDF.

Ví dụ:

```python
PdfReader(...)
```

### Bước 2 - Extract Text

Text được lấy ra từ từng trang.

```text
Page 1 → Text
Page 2 → Text
Page 3 → Text
```

### Bước 3 - Chunking

Văn bản dài được chia thành nhiều đoạn nhỏ.

Ví dụ:

```text
Document
   ↓
Chunk 1
Chunk 2
Chunk 3
...
```

### Bước 4 - Embedding

Mỗi chunk được chuyển thành một vector số.

```text
"Retrieval Augmented Generation"
            ↓
Embedding Model
            ↓
[0.18, -0.42, 0.91, ...]
```

### Bước 5 - Vector Storage

Các vector cùng metadata được lưu vào Qdrant.

Metadata có thể gồm:

```text
filename
page
chunk_index
text
```

---

## 7.2. Query Pipeline

Khi người dùng đặt câu hỏi:

```text
Question
   ↓
Embedding Model
   ↓
Question Vector
   ↓
Similarity Search
   ↓
Qdrant
   ↓
Top-K Chunks
   ↓
Context
   ↓
LLM
   ↓
Answer
```

Ví dụ:

```text
User:
RAG là gì?
```

Hệ thống tìm được:

```text
Chunk 12
Chunk 5
Chunk 8
```

Sau đó tạo prompt:

```text
Context:
[Chunk 12]
[Chunk 5]
[Chunk 8]

Question:
RAG là gì?
```

LLM sử dụng context này để tạo câu trả lời.

---

# 8. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| PDF Processing | pypdf |
| Text Splitting | LangChain Text Splitters |
| Embedding Model | Sentence Transformers |
| Vector Database | Qdrant |
| LLM | Llama 3.1 8B Instruct hoặc model tương đương |
| Local LLM Runtime | LM Studio |
| Backend | FastAPI - giai đoạn sau |
| Database | Supabase - giai đoạn sau |
| Automation | n8n - giai đoạn sau |

---

# 9. Giải thích công nghệ

## Python

Python được sử dụng làm ngôn ngữ chính để xây dựng RAG pipeline.

Lý do:

- phổ biến trong AI;
- nhiều thư viện machine learning;
- dễ tích hợp LLM;
- dễ làm backend với FastAPI.

---

## pypdf

`pypdf` được dùng để:

- đọc file PDF;
- lấy số trang;
- trích xuất text từ từng trang.

---

## LangChain Text Splitters

Dùng để chia văn bản thành các chunks nhỏ.

Project sử dụng:

```text
RecursiveCharacterTextSplitter
```

Các tham số quan trọng:

```text
chunk_size
chunk_overlap
```

Ví dụ:

```text
chunk_size = 500
chunk_overlap = 50
```

---

## Sentence Transformers

Sentence Transformers được dùng làm **Embedding Model**.

Nhiệm vụ:

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

Vector được dùng để so sánh mức độ tương đồng ngữ nghĩa giữa câu hỏi và nội dung tài liệu.

---

## Qdrant

Qdrant là Vector Database.

Qdrant lưu:

```text
Vector
+
Metadata
```

và hỗ trợ:

```text
Similarity Search
```

để tìm các chunks gần nhất với câu hỏi.

---

## LLM

LLM chịu trách nhiệm tạo câu trả lời cuối cùng.

Trong hệ thống:

```text
Retrieved Context
        +
     Question
        ↓
       LLM
        ↓
      Answer
```

LLM không phải Embedding Model.

Hai model có nhiệm vụ khác nhau:

```text
Embedding Model
→ chuyển text thành vector

LLM
→ hiểu context và tạo câu trả lời
```

---

# 10. Cấu trúc project dự kiến

```text
RAG_PROJECT/
│
├── data/
│   └── sample.pdf
│
├── src/
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── main.py
│
├── tests/
│
├── .venv/
│
├── .env
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 11. Ý nghĩa từng file

## `document_loader.py`

Phụ trách:

```text
PDF
 ↓
Extract Text
```

---

## `chunker.py`

Phụ trách:

```text
Text
 ↓
Chunks
```

---

## `embedding.py`

Phụ trách:

```text
Chunk
 ↓
Embedding Model
 ↓
Vector
```

---

## `vector_store.py`

Phụ trách kết nối và lưu dữ liệu vào Qdrant.

```text
Vector
 ↓
Qdrant
```

---

## `retriever.py`

Phụ trách tìm các chunks liên quan nhất.

```text
Question Vector
 ↓
Similarity Search
 ↓
Top-K Chunks
```

---

## `rag_pipeline.py`

Kết hợp toàn bộ quy trình:

```text
Question
 ↓
Retrieval
 ↓
Context
 ↓
LLM
 ↓
Answer
```

---

## `main.py`

Điểm chạy chính của ứng dụng.

---

# 12. Roadmap phát triển

Project được chia thành từng giai đoạn để dễ học, kiểm tra và giải thích.

## Giai đoạn 1 - Document Processing

```text
PDF
 ↓
Extract Text
 ↓
Chunking
```

Mục tiêu:

- đọc PDF;
- lấy text;
- chia chunks;
- giữ metadata.

---

## Giai đoạn 2 - Embedding

```text
Chunks
 ↓
Embedding Model
 ↓
Vectors
```

Mục tiêu:

- hiểu embedding;
- chuyển chunk thành vector;
- kiểm tra kích thước vector.

---

## Giai đoạn 3 - Vector Database

```text
Vectors
 ↓
Qdrant
```

Mục tiêu:

- tạo collection;
- lưu vector;
- lưu metadata.

---

## Giai đoạn 4 - Retrieval

```text
Question
 ↓
Question Embedding
 ↓
Qdrant Search
 ↓
Top-K Chunks
```

Mục tiêu:

- tìm các đoạn liên quan;
- kiểm tra retrieval có chính xác hay không.

---

## Giai đoạn 5 - RAG + LLM

```text
Top-K Chunks
      +
   Question
      ↓
     LLM
      ↓
    Answer
```

Mục tiêu:

- sinh câu trả lời dựa trên tài liệu.

---

## Giai đoạn 6 - Citation

Kết quả trả về:

```text
Answer
+
Filename
+
Page
+
Chunk
```

Ví dụ:

```text
RAG là kỹ thuật kết hợp retrieval và generation.

Nguồn:
sample.pdf - Trang 5
```

---

## Giai đoạn 7 - Chat Assistant

Phát triển giao diện:

```text
Upload PDF
   ↓
Ask Question
   ↓
Get Answer
   ↓
View Source
```

---

## Giai đoạn 8 - Advanced Features

Có thể mở rộng:

- Multiple documents.
- Chat history.
- Document summary.
- Quiz generation.
- FastAPI backend.
- Web frontend.
- User authentication.
- Agent.
- Tools.
- Supabase.
- n8n automation.

---

# 13. Cài đặt môi trường

## Clone repository

```bash
git clone https://github.com/nhivan2405/RAG_Project.git
cd RAG_Project
```

---

## Tạo virtual environment

Windows:

```powershell
python -m venv .venv
```

Kích hoạt:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Cài dependencies

```powershell
python -m pip install -r requirements.txt
```

---

# 14. requirements.txt ban đầu

Ở giai đoạn Document Processing:

```txt
pypdf
langchain-text-splitters
```

Các thư viện khác sẽ được bổ sung khi project phát triển.

---

# 15. Cách chạy

Ở giai đoạn đầu:

```powershell
python src\lesson_01_load_and_chunk.py
```

Kết quả mong đợi:

```text
PDF: sample.pdf
Pages with text: 10
Chunks created: 35
```

Sau đó hệ thống hiển thị thử một số chunks:

```text
Page: 1
Chunk: 0

Text:
...
```

---

# 16. Metadata

Mỗi chunk nên giữ thông tin:

```python
{
    "filename": "sample.pdf",
    "page": 1,
    "chunk_index": 0,
    "text": "..."
}
```

Metadata giúp:

- truy ngược nguồn;
- hiển thị citation;
- debug retrieval;
- kiểm tra chunk đến từ đâu.

---

# 17. Xử lý câu hỏi không có trong tài liệu

Một RAG Assistant không nên cố tạo câu trả lời khi tài liệu không chứa thông tin.

Ví dụ:

```text
User:
Ai là tổng thống của một quốc gia nào đó?
```

Nếu tài liệu không chứa thông tin này, hệ thống nên trả lời:

```text
Tôi không tìm thấy thông tin phù hợp trong tài liệu được cung cấp.
```

Thay vì tự tạo ra câu trả lời từ kiến thức bên ngoài.

---

# 18. Đánh giá hệ thống

RAG cần được kiểm tra bằng nhiều loại câu hỏi.

## Relevant Question

Câu hỏi có thông tin rõ ràng trong tài liệu.

Mục tiêu:

```text
Retrieve đúng chunk
+
Answer đúng
```

---

## Irrelevant Question

Câu hỏi không liên quan đến tài liệu.

Mục tiêu:

```text
Không retrieve sai nội dung
```

---

## Unanswerable Question

Câu hỏi có vẻ liên quan nhưng tài liệu không đủ thông tin để trả lời.

Mục tiêu:

```text
Assistant thừa nhận không đủ thông tin
```

---

# 19. Các câu hỏi cần hiểu khi bảo vệ project

Cần có khả năng giải thích:

```text
RAG là gì?

Tại sao cần RAG?

Chunk là gì?

Tại sao phải chia chunk?

chunk_size là gì?

chunk_overlap là gì?

Embedding là gì?

Embedding Model khác LLM như thế nào?

Vector Database là gì?

Tại sao sử dụng Qdrant?

Similarity Search là gì?

Top-K là gì?

RAG có train LLM không?

Tại sao phải lưu metadata?

Làm sao kiểm tra retrieval đúng?

Nếu không tìm được thông tin thì làm gì?

Nếu thay Embedding Model thì sửa ở đâu?

Nếu thay LLM thì sửa ở đâu?
```

---

# 20. Trạng thái project

## Đã hoàn thành

- [x] Tạo repository.
- [x] Tạo Python virtual environment.
- [x] Thiết lập Git và GitHub.
- [x] Tạo README.
- [x] Tạo requirements.txt.
- [x] Tạo .gitignore.

## Đang thực hiện

- [ ] PDF loading.
- [ ] Text extraction.
- [ ] Chunking.

## Chưa thực hiện

- [ ] Embedding.
- [ ] Qdrant.
- [ ] Retrieval.
- [ ] Top-K search.
- [ ] LLM integration.
- [ ] Citation.
- [ ] Chat Assistant.
- [ ] FastAPI.
- [ ] Agent.
- [ ] Tools.

---

# 21. Kết quả mong đợi cuối cùng

```text
User
 │
 │ Upload PDF
 ▼
Document Processing
 │
 ├── Extract Text
 ├── Chunk
 ├── Embedding
 └── Store in Qdrant
 │
 ▼
User asks a question
 │
 ▼
Question Embedding
 │
 ▼
Vector Search
 │
 ▼
Top-K Relevant Chunks
 │
 ▼
LLM
 │
 ▼
Answer + Citation
```

Hệ thống cuối cùng có thể hoạt động như một **trợ lý đọc tài liệu**, giúp người dùng tìm kiếm và hỏi đáp nhanh trên các tài liệu PDF.

---

# 22. Hướng mở rộng

Trong tương lai project có thể phát triển thành:

```text
RAG Core
   ↓
FastAPI Backend
   ↓
Supabase
   ↓
Web UI
   ↓
Agent + Tools
   ↓
Automation
```

Assistant khi đó không chỉ có khả năng **trả lời dựa trên kiến thức**, mà còn có thể thực hiện các hành động thông qua Tool hoặc API.

Có thể ghi nhớ:

```text
RAG   = Knowledge
Tool  = Action / Live Data
Agent = Decision
```

---

# 23. Tài liệu tham khảo

Project được tham khảo ý tưởng kiến trúc và cách tổ chức từ một số Smart Document Assistant sử dụng RAG:

- `vigkrishna/RAG-based-Smart-Document-Assistant`
  - PDF document processing.
  - Embedding.
  - Vector storage.
  - Context retrieval.
  - LLM generation.

- `bhavana1312/smart-doc-assistant`
  - PDF upload.
  - RAG Question Answering.
  - FastAPI backend.
  - Qdrant Vector Search.
  - Sentence Transformers.
  - Khả năng mở rộng thành quiz và learning assistant.

Project này không sao chép nguyên kiến trúc của các repository trên mà sử dụng chúng làm tài liệu tham khảo để xây dựng một pipeline RAG từ cơ bản đến hoàn chỉnh.

---

# 24. Tác giả

**Project:** Smart Document Assistant using RAG

**Môn học:** New Technologies in Software Engineering

**Năm:** 2026