# RAG Learning Course

This is a hands-on learning project for RAG (Retrieval-Augmented Generation). It includes course notes, examples, and sample data for building RAG applications step by step.

The course starts from the basic concepts of RAG, then moves into data loading, text chunking, embeddings, vector databases, multimodal embeddings, Milvus, and index optimization. Each chapter includes Markdown tutorials and Python examples so you can read and practice at the same time.

The learning path and chapter structure are inspired by [Datawhale all-in-rag](https://github.com/datawhalechina/all-in-rag/tree/main), which organizes RAG learning from fundamentals, data preparation, and index construction to retrieval optimization and system evaluation. This project rewrites the content with its own examples, data, and Gemini API setup.

## Course Goals

After completing this course, you should understand:

```text
What RAG is and why it is useful
How to load PDF, Markdown, TXT, image, and OCR data
How to split documents into retrieval-friendly chunks
How embedding models convert text or images into vectors
How vector databases store and search embeddings
How to build retrieval workflows with FAISS, LlamaIndex, and Milvus
How to improve retrieval quality with metadata, hybrid search, reranking, and sentence windows
```

## Learning Path

This project currently covers RAG fundamentals, data preprocessing, and index construction. Future chapters can extend the course into retrieval optimization, generation integration, and RAG evaluation.

```text
RAG fundamentals
-> Data loading and text chunking
-> Embeddings and multimodal embeddings
-> Vector databases and Milvus
-> Index optimization
-> Retrieval optimization and RAG system evaluation
```

Suggested learning order:

| Stage | Topic | Repo Section |
| --- | --- | --- |
| 1. RAG Fundamentals | Understand the RAG workflow and prepare the environment | Chapter 01 |
| 2. Data Preparation | Load data, apply OCR, and split text into chunks | Chapter 02 |
| 3. Index Construction | Embeddings, vector databases, and multimodal retrieval | Chapter 03 |
| 4. Index Optimization | Metadata, sentence windows, hybrid search, and reranking | Chapter 03, Section 05 |
| 5. Future Extensions | Retrieval optimization, generation integration, and evaluation | Planned |

## Project Structure

```text
RAG-Learning/
├── chapter/
│   ├── 01_what_is_RAG/
│   ├── 02_Data_Preprocessing/
│   └── 03_embedding/
├── data/
│   ├── C1/
│   ├── C2/
│   └── C3/
├── requirements.txt
├── README.md
└── README_en.md
```

## Chapter Guide

### Chapter 01: What Is RAG

| Chapter | Description |
| --- | --- |
| [Preparation](./chapter/01_what_is_RAG/01_preparation.md) | Set up the Python environment and Gemini API key |
| [What is RAG](./chapter/01_what_is_RAG/02_what_is_RAG.md) | RAG workflow with LangChain and LlamaIndex examples |

### Chapter 02: Data Preprocessing

| Chapter | Description |
| --- | --- |
| [Data Loading](./chapter/02_Data_Preprocessing/01_data_load.md) | Common document loading tools, PDF, Markdown, Unstructured, MarkItDown, and Chandra |
| [Text Chunking](./chapter/02_Data_Preprocessing/02_text_chuckling.md) | Character, recursive character, and semantic chunking |
| [OCR Data Loading](./chapter/02_Data_Preprocessing/03_data_load_ocr.md) | OCR tools and model overview |

### Chapter 03: Embeddings and Vector Databases

| Chapter | Description |
| --- | --- |
| [What Is Embedding](./chapter/03_embedding/01_what_is_embedding.md) | Vector spaces, embedding models, and the MTEB leaderboard |
| [Multimodal Embedding](./chapter/03_embedding/02_multimodal_embedding.md) | CLIP, Visualized-BGE, Gemini Embedding, and multimodal retrieval |
| [Vector Database](./chapter/03_embedding/03_vector_db.md) | Vector database concepts, FAISS, and LlamaIndex vector indexes |
| [Milvus](./chapter/03_embedding/04_Milvus.md) | Milvus concepts, collections, indexes, metadata filters, and multimodal image retrieval |
| [Index Optimization](./chapter/03_embedding/05_index_optimization.md) | Sentence windows, metadata, hierarchical indexes, hybrid search, and reranking |

## Environment Setup

This project recommends Python 3.12.

If `uv` is not installed, install it first:

```powershell
pip install uv
```

Create and activate the virtual environment:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

Check the environment:

```powershell
python --version
uv pip list
```

## Gemini API Key

This course mainly uses the Gemini API. Create an API key from Google AI Studio:

https://aistudio.google.com/app/apikey

Then create a `.env` file in the project root:

```powershell
New-Item -Path .env -ItemType File
```

Add:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

`.env` is already included in `.gitignore`. Do not upload your real API key to GitHub.

## Example Scripts

### Chapter 01

```powershell
python chapter/01_what_is_RAG/langchain_example.py
python chapter/01_what_is_RAG/llamaindex_example.py
```

### Chapter 02

```powershell
python chapter/02_Data_Preprocessing/unstructured_example.py
python chapter/02_Data_Preprocessing/character_splitter.py
python chapter/02_Data_Preprocessing/recursive_character_splitter.py
python chapter/02_Data_Preprocessing/semantic_chunker.py
```

### Chapter 03

```powershell
python chapter/03_embedding/02_multimodal_embedding.py
python chapter/03_embedding/03_langchain_faiss.py
python chapter/03_embedding/03_llamaindex_vector.py
python chapter/03_embedding/04_multi_milvus.py
```

`04_multi_milvus.py` requires Milvus to be running first.

## Technologies Used

This project currently covers:

```text
LangChain
LlamaIndex
Gemini API
FAISS
Milvus
Hugging Face Embeddings
Unstructured
MarkItDown / Chandra / OCR tools
```

## Reference Structure

The chapter planning of this course refers to:

- [Datawhale all-in-rag: Full-Stack RAG Technical Guide](https://github.com/datawhalechina/all-in-rag/tree/main)

The reference is mainly about the overall learning path, the way RAG modules are organized, and the progression from basic to advanced topics. The actual text, example code, images, and data layout are rewritten and adjusted for this project.

## License

This project is intended for learning purposes. If you cite or adapt it, please keep the source attribution and avoid committing any API keys or private data.
