# RAG Learning Course

[![English](https://img.shields.io/badge/Language-English-blue)](./README_en.md)

這是一份以實作為導向的 RAG（Retrieval-Augmented Generation，檢索增強生成）學習筆記與範例專案。

課程會從 RAG 的基本概念開始，逐步介紹資料載入、文字切分、Embedding、向量資料庫、多模態 Embedding、Milvus，以及索引優化等主題。每個章節都會搭配 Markdown 教學與 Python 範例，方便一邊閱讀一邊實作。

本教材的學習路徑與章節安排參考 [Datawhale all-in-rag](https://github.com/datawhalechina/all-in-rag/tree/main) 的整體架構：從 RAG 基礎、資料準備、索引建構一路延伸到檢索優化與系統評估。本專案會依照自己的教學節奏重新整理內容，並改用本 repo 的範例、資料與 Gemini API 設定。

## 課程目標

學完這份教材後，你應該能理解：

```text
RAG 是什麼，以及為什麼需要 RAG
如何載入 PDF、Markdown、TXT、圖片與 OCR 資料
如何把文件切成適合檢索的 chunks
Embedding model 如何把文字或圖片轉成向量
向量資料庫如何儲存與搜尋 embeddings
如何使用 FAISS、LlamaIndex、Milvus 建立檢索流程
如何透過 metadata、hybrid search、rerank、sentence window 改善檢索品質
```

## 學習路徑

本專案目前先完成 RAG 入門、資料前處理，以及索引建構三個部分，後續可以再往檢索優化、生成整合與評估延伸。

```text
RAG 基礎
-> 資料載入與文字切分
-> Embedding 與多模態 Embedding
-> 向量資料庫與 Milvus
-> 索引優化
-> 檢索優化與 RAG 系統評估
```

對應的學習順序：

| 階段 | 內容 | 本 repo 對應 |
| --- | --- | --- |
| 1. RAG 基礎入門 | 理解 RAG 流程、準備開發環境 | Chapter 01 |
| 2. 資料準備 | 載入資料、OCR、文字分塊 | Chapter 02 |
| 3. 索引建構 | Embedding、Vector DB、多模態檢索 | Chapter 03 |
| 4. 索引優化 | Metadata、Sentence Window、Hybrid Search、Rerank | Chapter 04 規劃中 |
| 5. 後續延伸 | 檢索優化、生成整合、系統評估 | 規劃中 |

## 專案結構

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
└── README.md
```

## 章節目錄

### Chapter 01：什麼是 RAG

| 章節 | 說明 |
| --- | --- |
| [課前準備](./chapter/01_what_is_RAG/01_preparation.md) | 建立 Python 環境、設定 Gemini API Key |
| [What is RAG](./chapter/01_what_is_RAG/02_what_is_RAG.md) | RAG 基本流程、LangChain 與 LlamaIndex 範例 |

### Chapter 02：資料前處理

| 章節 | 說明 |
| --- | --- |
| [資料載入](./chapter/02_Data_Preprocessing/01_data_load.md) | 常見文件載入工具、PDF、Markdown、Unstructured、MarkItDown、Chandra |
| [文字切分](./chapter/02_Data_Preprocessing/02_text_chuckling.md) | Character、Recursive Character、Semantic Chunking |
| [OCR 資料載入](./chapter/02_Data_Preprocessing/03_data_load_ocr.md) | OCR 工具與模型整理 |

### Chapter 03：Embedding 與向量資料庫

| 章節 | 說明 |
| --- | --- |
| [Embedding 是什麼](./chapter/03_embedding/01_what_is_embedding.md) | 向量空間、Embedding model、MTEB leaderboard |
| [多模態 Embedding](./chapter/03_embedding/02_multimodal_embedding.md) | CLIP、Visualized-BGE、Gemini Embedding、多模態檢索 |
| [向量資料庫](./chapter/03_embedding/03_vector_db.md) | Vector DB 概念、FAISS、LlamaIndex vector index |
| [Milvus](./chapter/03_embedding/04_Milvus.md) | Milvus 概念、collection、index、metadata filter、多模態圖片檢索 |
| [索引優化](./chapter/03_embedding/05_index_optimization.md) | Sentence Window、metadata、分層索引、hybrid search、rerank |

## 環境安裝

本專案建議使用 Python 3.12。

如果還沒有安裝 `uv`，可以先安裝：

```powershell
pip install uv
```

建立並啟動虛擬環境：

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
```

安裝套件：

```powershell
uv pip install -r requirements.txt
```

確認環境：

```powershell
python --version
uv pip list
```

## Gemini API Key

本課程主要使用 Gemini API。請先到 Google AI Studio 建立 API key：

https://aistudio.google.com/app/apikey

接著在專案根目錄建立 `.env`：

```powershell
New-Item -Path .env -ItemType File
```

加入：

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

`.env` 已經被加入 `.gitignore`，請不要把真實 API key 上傳到 GitHub。

## 範例程式

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

`04_multi_milvus.py` 需要先啟動 Milvus。

## 使用技術

本專案目前涵蓋：

```text
LangChain
LlamaIndex
Gemini API
FAISS
Milvus
Hugging Face Embeddings
Unstructured
MarkItDown / Chandra / OCR 工具整理
```

## 參考架構

本教材的章節規劃參考：

- [Datawhale all-in-rag：RAG 技術全棧指南](https://github.com/datawhalechina/all-in-rag/tree/main)

參考重點包含整體學習路徑、RAG 技術模組拆分方式，以及從基礎到進階的章節安排；實際文字、範例程式、圖片與資料配置會依照本專案重新撰寫與調整。

## License

本專案為學習用途教材。若要引用或改作，請保留來源說明並避免提交任何 API key 或私人資料。
