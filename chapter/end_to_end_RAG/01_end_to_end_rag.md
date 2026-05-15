# End-to-End RAG：使用 Chandra OCR 與 Milvus 建立課程知識庫

前面章節已經介紹過資料載入、OCR、chunking、embedding、向量資料庫、Milvus、Hybrid Search 與進階檢索技術。

這一節要把前面的概念串起來，建立一個完整的 end-to-end RAG 流程。

本範例的目標是：

```text
把 AI Agent 課程 PDF
-> 用 Chandra 轉成 Markdown
-> 切成 chunks
-> 產生 embeddings
-> 寫入 Milvus
-> 使用者提問時從 Milvus 找回相關內容
-> 交給 LLM 產生答案
```

資料來源：

```text
data/2025_AI_Agent_Course/
```

目前資料夾內包含：

```text
Agent Quality.pdf
Agent Tools & Interoperability with Model Context Protocol.pdf
Context Engineering_ Sessions & Memory.pdf
Introduction to Agents.pdf
Prototype to Production.pdf
```

這些資料都是 PDF，因此第一版流程會先以 **Chandra OCR / PDF parsing** 作為文件解析工具。

## 一、整體流程

End-to-end RAG 可以拆成兩條流程：

```text
離線建庫流程 ingestion pipeline
線上問答流程 query pipeline
```

離線建庫流程：

```text
PDF files
-> Chandra OCR
-> Markdown / metadata
-> clean text
-> chunking
-> embedding
-> Milvus collection
```

線上問答流程：

```text
user question
-> query embedding
-> Milvus similarity search
-> retrieve top-k chunks
-> build prompt
-> LLM answer
-> return answer with sources
```

兩條流程要分開設計，原因是：

| 流程 | 執行時機 | 目的 |
| --- | --- | --- |
| ingestion | 新增或更新文件時執行 | 建立知識庫 |
| query | 使用者提問時執行 | 從知識庫找資料並回答 |

不要每次提問都重新 OCR、重新切 chunk、重新寫入 Milvus。這些都應該在 ingestion 階段完成。

## 二、資料來源與輸出規劃

原始 PDF 放在：

```text
data/2025_AI_Agent_Course/
```

建議 Chandra 輸出放在：

```text
data/end_to_end_RAG/chandra_output/
```

Milvus 不會直接存原始 PDF。比較好的做法是：

```text
原始 PDF：保留在 data/2025_AI_Agent_Course/
Chandra Markdown：保留在 data/end_to_end_RAG/chandra_output/
Milvus：儲存 chunk text、embedding vector、metadata
```

建議資料夾結構：

```text
data/
├── 2025_AI_Agent_Course/
│   ├── Agent Quality.pdf
│   ├── Agent Tools & Interoperability with Model Context Protocol.pdf
│   ├── Context Engineering_ Sessions & Memory.pdf
│   ├── Introduction to Agents.pdf
│   └── Prototype to Production.pdf
└── end_to_end_RAG/
    ├── chandra_output/
    │   ├── Agent Quality/
    │   │   ├── output.md
    │   │   ├── output.html
    │   │   └── metadata.json
    │   └── ...
    └── processed/
        └── chunks.jsonl
```

其中 `chunks.jsonl` 可作為中間檔，方便 debug。

## 三、為什麼先用 Chandra

PDF 解析通常有兩種情況：

| PDF 類型 | 處理方式 |
| --- | --- |
| 文字型 PDF | 可以直接抽文字 |
| 掃描型 PDF / 圖片型 PDF | 需要 OCR |

課程 PDF 通常可能包含：

```text
標題
投影片式排版
表格
圖片
流程圖
頁碼
多欄或區塊式版面
```

如果只用一般 PDF text extractor，常見問題是：

```text
閱讀順序錯亂
表格結構消失
圖片中的文字沒有被讀出
頁首頁尾混入正文
標題層級不明確
```

Chandra 的價值是把 PDF 轉成較接近 RAG 可用的 Markdown / HTML / metadata，讓後面的 chunking 更好處理。

在這個流程中，Chandra 不是最後的知識庫，它只是 ingestion pipeline 的第一步。

## 四、Chandra OCR 階段

Chandra 的安裝與執行方式可能依照官方 repo 版本、GPU 環境與指令不同而改變，因此這份教材先定義輸入與輸出規格。

輸入：

```text
data/2025_AI_Agent_Course/*.pdf
```

輸出：

```text
data/end_to_end_RAG/chandra_output/<document_name>/
```

每份文件建議至少輸出：

| 檔案 | 用途 |
| --- | --- |
| `output.md` | 後續 chunking 的主要文字來源 |
| `output.html` | 檢查版面、表格與圖片位置 |
| `metadata.json` | 保存頁碼、圖片、區塊、解析資訊 |

如果 Chandra 實際輸出的檔名不同，可以在後續 loader 中做對應。

重點是要保留這三類資訊：

```text
文字內容
頁碼 / 區塊 metadata
原始來源檔案
```

## 五、資料清理

OCR 完成後，不建議直接把整份 Markdown 丟去 embedding。

通常需要先做清理：

```text
移除重複頁碼
移除頁首頁尾
合併錯誤斷行
保留 Markdown 標題
保留表格
移除空白過多的段落
保留圖片 caption 或 alt text
```

清理的目標不是把文字改寫得更漂亮，而是讓文字更適合：

```text
切 chunk
做 embedding
被 retrieval 找回
讓 LLM 引用來源
```

建議每個清理後的文字段落都保留 metadata：

```python
metadata = {
    "source": "data/2025_AI_Agent_Course/Introduction to Agents.pdf",
    "ocr_tool": "chandra",
    "document_name": "Introduction to Agents",
    "page": 3,
    "modality": "pdf",
}
```

## 六、Chunking 設計

第一版可以使用 `RecursiveCharacterTextSplitter`。

建議參數：

```python
chunk_size = 800
chunk_overlap = 120
```

原因是：

| 參數 | 說明 |
| --- | --- |
| `chunk_size=800` | 適合教材型文字，不會太短，也不會塞太多雜訊 |
| `chunk_overlap=120` | 保留上下文銜接，避免重要句子被切斷 |

如果 Markdown 標題結構很完整，也可以先用標題切，再 fallback 到字元切分：

```text
# 大標題
## 小標題
### 子主題
```

每個 chunk 建議 metadata：

```python
metadata = {
    "chunk_id": "introduction_to_agents_0001",
    "source": "data/2025_AI_Agent_Course/Introduction to Agents.pdf",
    "document_name": "Introduction to Agents",
    "page_start": 3,
    "page_end": 4,
    "section": "What is an Agent",
    "ocr_tool": "chandra",
}
```

Metadata 很重要，因為最後回答要能回傳來源。

## 七、Embedding Model

這個知識庫主要是英文 AI Agent 課程，但 repo 也有中英文混合內容。

建議第一版使用：

```text
BAAI/bge-m3
```

原因：

```text
支援多語言
適合 retrieval
可處理較長輸入
後續也能延伸到 hybrid search
```

如果要使用 Gemini embedding，也可以，但要注意：

```text
需要 API key
會有 API 成本
大量 ingestion 時要注意 rate limit
```

第一版建議先用本地 embedding model，讓流程比較容易 debug。

## 八、Milvus Collection 設計

Milvus collection 可以命名為：

```text
ai_agent_course_rag
```

建議 schema：

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `id` | `VARCHAR` | chunk id |
| `text` | `VARCHAR` | chunk 文字 |
| `vector` | `FLOAT_VECTOR` | embedding vector |
| `source` | `VARCHAR` | 原始 PDF 路徑 |
| `document_name` | `VARCHAR` | 文件名稱 |
| `page_start` | `INT64` | 起始頁 |
| `page_end` | `INT64` | 結束頁 |
| `section` | `VARCHAR` | Markdown 標題或章節 |
| `ocr_tool` | `VARCHAR` | 例如 `chandra` |

如果未來要做 hybrid search，可以再加：

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `sparse_vector` | `SPARSE_FLOAT_VECTOR` | sparse embedding 或 BM25 欄位 |

第一版先建立 dense vector RAG 即可。

## 九、Ingestion 程式流程

建議建立第一支程式：

```text
chapter/end_to_end_RAG/01_ingest_to_milvus.py
```

它負責：

```text
1. 掃描 data/2025_AI_Agent_Course/*.pdf
2. 檢查 Chandra output 是否存在
3. 讀取 Chandra 產生的 Markdown
4. 清理文字
5. 切 chunk
6. 產生 embedding
7. 建立 Milvus collection
8. 寫入 Milvus
```

建議不要在 ingestion 程式中自動重跑 Chandra。

比較穩定的做法是：

```text
Chandra OCR 單獨執行
ingestion 只讀 Chandra 的輸出
```

原因是 OCR 模型通常比較重，可能需要 GPU，也可能有不同 repo 的環境依賴。把 OCR 和 ingestion 拆開，debug 會簡單很多。

## 十、Retrieval 測試流程

建議建立第二支程式：

```text
chapter/end_to_end_RAG/02_query_milvus.py
```

它只做 retrieval，不接 LLM。

流程：

```text
使用者問題
-> query embedding
-> Milvus search top_k=5
-> 印出 chunk text、score、source、page
```

測試問題可以先用：

```text
What is an AI agent?
How should agent memory be designed?
What are common risks when moving agents from prototype to production?
How does MCP help agent tool interoperability?
How can we evaluate agent quality?
```

這一步很重要，因為如果 retrieval 找不到正確內容，接上 LLM 也不會變好。

## 十一、RAG Chat 流程

建議建立第三支程式：

```text
chapter/end_to_end_RAG/03_rag_chat.py
```

它負責：

```text
1. 接收使用者問題
2. 從 Milvus 找回 top-k chunks
3. 組 prompt
4. 呼叫 LLM
5. 回答並列出 sources
```

Prompt 可以先設計成：

```text
You are an AI Agent course assistant.
Answer the question using only the provided context.
If the context does not contain enough information, say that the course material does not provide enough information.
Always cite the source document and page when possible.
```

回答格式建議：

```text
Answer:
...

Sources:
- Introduction to Agents.pdf, page 3
- Context Engineering_ Sessions & Memory.pdf, page 8
```

## 十二、建議檔案結構

```text
chapter/end_to_end_RAG/
├── 01_end_to_end_rag.md
├── 01_ingest_to_milvus.py
├── 02_query_milvus.py
└── 03_rag_chat.py
```

資料輸出：

```text
data/end_to_end_RAG/
├── chandra_output/
├── processed/
│   └── chunks.jsonl
└── logs/
```

Milvus collection：

```text
ai_agent_course_rag
```

## 十三、開發順序

建議不要一次把所有功能寫完。

比較穩定的順序是：

```text
1. 先跑 Chandra，確認每份 PDF 都有 Markdown 輸出
2. 寫 Markdown loader，確認能讀出文字
3. 寫 chunking，輸出 chunks.jsonl 檢查切分結果
4. 寫 embedding，確認 vector 維度正確
5. 寫入 Milvus，確認 collection row count
6. 寫 query_milvus.py，確認 retrieval 找得到正確 chunk
7. 最後再接 LLM 生成答案
```

每一步都要能獨立測試，這樣比較容易定位問題。

## 十四、常見問題

### 14.1 為什麼不直接把 PDF 丟進 Milvus

Milvus 管理的是向量與 metadata，不是 PDF parser。

PDF 需要先被解析成文字，再切成 chunks，最後才轉成 embedding 存入 Milvus。

### 14.2 Chandra 輸出 Markdown 後還需要 chunking 嗎

需要。

Markdown 是完整文件，通常太長，不適合直接做單一 embedding。RAG 需要把文件切成可檢索的小單位。

### 14.3 Milvus 會存 metadata 嗎

會。

Milvus 的 scalar fields 可以存 `source`、`page_start`、`page_end`、`document_name`、`section` 等資訊。搜尋結果可以透過 `output_fields` 取回這些欄位。

### 14.4 要不要一開始就做 Hybrid Search

第一版先不用。

先完成 dense retrieval，確認資料解析、chunking、embedding、Milvus search 都正常。等 baseline 穩定後，再加入 hybrid search 或 reranker。

## 十五、本節重點

這個 end-to-end RAG 專案的核心流程是：

```text
Chandra 負責把 PDF 轉成 RAG 友善的 Markdown
Chunking 負責把長文件切成可檢索單位
Embedding 負責把 chunks 轉成向量
Milvus 負責儲存向量與 metadata
Retriever 負責找回相關 chunks
LLM 負責根據 context 生成答案
```

第一版要先追求「流程可跑、來源可追蹤、retrieval 找得到資料」。
等 baseline 穩定後，再逐步加入 hybrid search、rerank、context compression 或 Corrective RAG。
