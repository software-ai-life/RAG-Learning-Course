# 什麼是 RAG?

RAG 是 **Retrieval-Augmented Generation** 的縮寫，中文常翻成「檢索增強生成」。

一般的 LLM 只會根據模型本身已經學到的知識回答問題。RAG 則是在回答前，先從指定資料來源中找出相關內容，再把這些內容一起交給 LLM，讓模型根據「查到的資料」回答。

簡單來說，RAG 的流程是：

1. 載入文件
2. 把文件切成小段落
3. 將每個段落轉成 embedding 向量
4. 把向量存進 vector store
5. 使用者提出問題
6. 用問題去 vector store 搜尋相關段落
7. 把搜尋結果放進 prompt
8. 交給 LLM 生成答案

本章會根據 `langchain_example.py` 逐段說明一個最小可執行的 RAG 範例。


## 1. 初始設定

```python
import os
import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
```

`load_dotenv()` 會讀取專案中的 `.env` 檔案，並把裡面的內容載入到環境變數。

例如 `.env`：

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

載入後，就可以用：

```python
os.getenv("GEMINI_API_KEY")
```

取得 Gemini API key。


## 2. 準備 Markdown 資料及讀取

```python
markdown_path = "../../data/C1/markdown/what-is-rag.md"
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()
```

這段程式會把 Markdown 檔案讀進 LangChain 的 `Document` 格式。

### `UnstructuredMarkdownLoader(markdown_path)`

`UnstructuredMarkdownLoader` 會解析 Markdown 檔案，並轉成 LangChain 可處理的文件物件。

### `loader.load()`

```python
docs = loader.load()
```

`load()` 會真正讀取檔案，並回傳一個 `Document` list。

即使只有一個 Markdown 檔案，回傳值通常仍然是 list，因為 LangChain 的 loader 設計是可以一次回傳多份文件。

## 3. 將文件切成 chunks

```python
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)
```

LLM 和 embedding model 不適合一次處理太長的文件，所以 RAG 會先把原始文件切成較小的文字片段，這些片段稱為 chunk。

### `RecursiveCharacterTextSplitter()`

目前範例沒有傳入任何參數，因此會使用 LangChain 的預設設定。

這個 splitter 的特色是「遞迴切分」。它會先嘗試用較自然的分隔符切，例如段落、換行、句子，再退而求其次用字元長度切。

常見參數如下：

| 參數 | 說明 |
| --- | --- |
| `chunk_size` | 每個 chunk 的最大長度。chunk 太大會讓 retrieval 不精準；太小會讓上下文不完整 |
| `chunk_overlap` | 相鄰 chunk 之間重疊的字數。重疊可以避免重要句子剛好被切斷 |
| `separators` | 切分時使用的分隔符，例如 `\n\n`、`\n`、空白等 |
| `length_function` | 用來計算文字長度的函式，預設通常是 `len` |

如果要更明確控制切分方式，可以寫成：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### `split_documents(docs)`

```python
chunks = text_splitter.split_documents(docs)
```

回傳的 `chunks` 也是 `Document` list。每個 chunk 會保留原本的 metadata，因此之後可以追蹤答案來源。

## 4. 建立 embedding model

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

Embedding 是把文字轉成向量的過程。

向量可以讓電腦比較兩段文字的語意相似度。例如：

```text
"什麼是 RAG?"
"Retrieval-Augmented Generation 是什麼?"
```

這兩句文字字面不同，但語意相近，所以 embedding 向量距離會比較近。

### `model_name`

```python
model_name="BAAI/bge-m3"
```

`BAAI/bge-m3` 是一個多語言 embedding model，適合中英文檢索任務。因為本課程可能同時包含中文與英文資料，所以這個 model 比只支援英文的 embedding model 更適合。

### `encode_kwargs`

```python
encode_kwargs={'normalize_embeddings': True}
```

`encode_kwargs` 會影響文字轉向量時的設定。

| key | 目前值 | 說明 |
| --- | --- | --- |
| `normalize_embeddings` | `True` | 是否把 embedding 向量正規化 |

正規化後，向量長度會被縮放成一致，通常比較適合用 cosine similarity 做語意相似度搜尋。

在 RAG retrieval 中，建議開啟 `normalize_embeddings=True`，因為它能讓相似度比較更穩定。

## 5. 建立 vector store

```python
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)
```

Vector store 是用來保存 chunk 與 embedding 向量的地方。

在這個範例中，我們使用 `InMemoryVectorStore`，也就是把資料存在程式執行時的記憶體中。

### `InMemoryVectorStore(embeddings)`

| 參數 | 目前值 | 說明 |
| --- | --- | --- |
| `embedding` | `embeddings` | 指定 vector store 要用哪個 embedding model 產生向量 |

`InMemoryVectorStore` 的優點是簡單，不需要額外安裝資料庫，也不需要啟動服務。

缺點是資料不會永久保存。程式結束後，vector store 內容就會消失。正式專案通常會改用 Chroma、FAISS、Qdrant、Milvus 或 Pinecone。

### `add_documents(chunks)`

```python
vectorstore.add_documents(chunks)
```

| 參數 | 目前值 | 說明 |
| --- | --- | --- |
| `documents` | `chunks` | 要加入 vector store 的文件片段 |

執行這行時，LangChain 會做幾件事：

1. 讀取每個 chunk 的 `page_content`
2. 使用 `embeddings` 將文字轉成向量
3. 把原始 chunk、metadata、embedding 向量存進 vector store

## 6. 建立 prompt template

```python
prompt = ChatPromptTemplate.from_template("""
請根據以下內容回答問題。
如果內容中沒有答案，請說明無法從資料中找到答案。

內容：
{context}

問題：
{question}

回答：
""")
```

Prompt template 是給 LLM 的指令模板。

在 RAG 中，prompt 通常會包含兩個核心變數：

| 變數 | 說明 |
| --- | --- |
| `{context}` | retrieval 找到的相關文件內容 |
| `{question}` | 使用者提出的問題 |

LangChain 會把實際內容填入模板。

好的 RAG prompt 應該明確要求模型根據提供的 context 回答，而不是自由發揮。這可以降低 hallucination。

## 7. 建立 Gemini LLM client

```python
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    http_client=httpx.Client(verify=False)
)
```

雖然這裡使用的是 `ChatOpenAI`，但實際上呼叫的是 Gemini。

原因是 Gemini 提供 OpenAI-compatible endpoint，因此可以用 OpenAI 格式的 client 呼叫 Gemini 模型。

### `model`

```python
model="gemini-2.5-flash"
```

`gemini-2.5-flash` 是 Gemini 的快速模型，適合教學、互動式問答、RAG demo 等場景。

### `temperature`

```python
temperature=0.7
```

| 參數 | 說明 |
| --- | --- |
| `temperature` | 控制回答的隨機性 |

常見設定：

| 值 | 效果 |
| --- | --- |
| `0` | 最穩定、最保守，適合事實型問答 |
| `0.3` | 較穩定，但仍有一點彈性 |
| `0.7` | 比較自然、有變化，適合一般對話 |
| `1.0` 以上 | 更有創意，但也更容易偏離資料 |

RAG 通常希望模型忠於資料，因此正式問答系統可以考慮使用 `temperature=0` 到 `0.3`。本範例使用 `0.7`，回答會比較自然。

### `max_tokens`

```python
max_tokens=4096
```

| 參數 | 說明 |
| --- | --- |
| `max_tokens` | 限制模型最多輸出多少 token |

Token 可以粗略理解成文字單位。英文通常一個 token 約等於一小段單字或子字；中文則不一定等於一個字。

`max_tokens=4096` 代表模型最多可以輸出 4096 tokens。這可以避免模型回答過長，也能控制 API 成本。

### `base_url`

```python
base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
```

| 參數 | 說明 |
| --- | --- |
| `base_url` | 指定 API endpoint |

`ChatOpenAI` 預設會連到 OpenAI API。因為這裡要使用 Gemini，所以必須把 `base_url` 改成 Gemini 的 OpenAI-compatible endpoint。

如果沒有設定 `base_url`，程式會嘗試呼叫 OpenAI，而不是 Gemini。

### `http_client`

```python
http_client=httpx.Client(verify=False)
```

避免本機憑證問題導致 API 呼叫失敗。

正式環境不建議使用 `verify=False`。它會降低 HTTPS 連線的安全性。

## 8. 提出問題

```python
question = "什麼是 RAG?"
```

`question` 是使用者想問的問題。

在 RAG 流程中，這個問題會被使用兩次：

1. 用來搜尋相關文件
2. 放進 prompt 交給 LLM 回答


## 9. 檢索相關文件

```python
retrieved_docs = vectorstore.similarity_search(question, k=3)
```

這是 RAG 的 retrieval 步驟。

程式會把 `question` 轉成 embedding 向量，然後去 vector store 中搜尋最相似的 chunk。

### `similarity_search(question, k=3)`

| 參數 | 目前值 | 說明 |
| --- | --- | --- |
| `query` | `question` | 要搜尋的問題文字 |
| `k` | `3` | 回傳最相似的前 3 個 chunk |

`k` 是很重要的 retrieval 參數。

| `k` 值 | 效果 |
| --- | --- |
| 太小 | 可能找不到足夠資訊，回答不完整 |
| 適中 | 能提供足夠 context，又不會塞太多無關內容 |
| 太大 | 可能把不相關內容也放進 prompt，讓模型混淆 |

本範例使用 `k=3`，代表每次問答會找出最相關的 3 段文字。

## 10. 合併檢索到的 chunks

```python
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
```

這行會把 retrieval 找到的多個 chunk 合併成一段文字，準備放進 prompt 的 `{context}`。

### 參數與語法說明

| 寫法 | 說明 |
| --- | --- |
| `"\n\n"` | chunk 之間用兩個換行隔開，讓 context 比較容易閱讀 |
| `doc.page_content` | 取出每個 `Document` 的文字內容 |
| `for doc in retrieved_docs` | 逐一處理 retrieval 找到的文件 |
| `.join(...)` | 把多段文字合併成一個字串 |

合併後的 `docs_content` 會像這樣：

```text
第一段相關內容

第二段相關內容

第三段相關內容
```

## 11. 格式化 prompt 並呼叫 LLM

```python
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
```

這是最後的 generation 步驟。

### `prompt.format(...)`

```python
prompt.format(question=question, context=docs_content)
```

這行會產生最終 prompt


在 LangChain 中，`answer` 通常不是單純字串，而是 message object。若只想印出文字內容，可以使用：

```python
print(answer.content)
```

## 完整 RAG 流程

把整個程式流程串起來，可以理解成：

```text
Markdown file
    ↓
UnstructuredMarkdownLoader
    ↓
Document list
    ↓
RecursiveCharacterTextSplitter
    ↓
Chunks
    ↓
HuggingFaceEmbeddings
    ↓
Vectors
    ↓
InMemoryVectorStore
    ↓
similarity_search(question, k=3)
    ↓
Retrieved context
    ↓
ChatPromptTemplate
    ↓
Gemini model
    ↓
Answer
```

## 為什麼 RAG 很有用

RAG 的核心價值是讓 LLM 可以根據外部資料回答問題。

它解決了幾個常見問題：

| 問題 | RAG 的解法 |
| --- | --- |
| 模型不知道最新資料 | 把最新文件放進資料庫，回答前先搜尋 |
| 模型容易幻覺 | 要求模型根據 retrieval context 回答 |
| 企業資料沒有在模型訓練中 | 將內部文件轉成 vector store |
| 回答需要可追溯來源 | chunk metadata 可以保留檔案來源 |

## LlamaIndex 版本範例

除了 LangChain，本章也提供了另一個範例：

```text
chapter/01_what_is_RAG/llamaindex_example.py
```

這個範例使用 LlamaIndex 來完成同樣的 RAG 流程。

LangChain 和 LlamaIndex 都可以建立 RAG，但設計重點不同：

| 工具 | 特色 |
| --- | --- |
| LangChain | 流程拆得比較細，適合學習 loader、splitter、embedding、vector store、prompt、LLM 之間如何串接 |
| LlamaIndex | 對「文件索引」和「查詢引擎」包裝較完整，適合快速把資料建立成可查詢的 index |

LlamaIndex 範例的核心流程是：

```text
設定 Gemini LLM
    ↓
設定 embedding model
    ↓
讀取 Markdown 文件
    ↓
建立 VectorStoreIndex
    ↓
轉成 query engine
    ↓
提出問題並取得回答
```

### 1. 設定 Gemini LLM

```python
Settings.llm = OpenAILike(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    is_chat_model=True,
    http_client=httpx.Client(verify=False)
)
```

`Settings.llm` 是 LlamaIndex 的全域 LLM 設定。設定後，後續建立 index 或 query engine 時，LlamaIndex 會預設使用這個模型。

`OpenAILike` 的用途是讓 LlamaIndex 使用「類 OpenAI API 格式」的模型服務。Gemini 提供 OpenAI-compatible endpoint，所以可以用這種方式接上 Gemini。

正式環境不建議使用 `verify=False`，因為它會停用 HTTPS 憑證驗證。

### 2. 設定 embedding model

```python
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-m3")
```

`Settings.embed_model` 是 LlamaIndex 的全域 embedding 設定。它負責把文件與問題轉成向量。

### 3. 讀取文件

```python
docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/what-is-rag.md"]
).load_data()
```

`SimpleDirectoryReader` 是 LlamaIndex 用來讀取本機文件的工具。

`input_files` 使用 list，所以可以一次指定多個檔案：

```python
input_files=[
    "file1.md",
    "file2.pdf",
    "file3.txt"
]
```

`.load_data()` 會真正讀取檔案，並回傳 LlamaIndex 的 document list。

### 4. 建立 VectorStoreIndex

```python
index = VectorStoreIndex.from_documents(docs)
```

`VectorStoreIndex` 是 LlamaIndex 中常用的向量索引。

這行會做幾件事：

1. 讀取 `docs` 中的文件內容。
2. 使用 `Settings.embed_model` 將內容轉成 embedding。
3. 建立可以被查詢的 vector index。

和 LangChain 範例相比，LlamaIndex 把「切分文件、產生 embedding、建立 vector index」包裝得更簡潔。

### 5. 建立 query engine

```python
query_engine = index.as_query_engine()
```

`query_engine` 是 LlamaIndex 封裝好的查詢介面。

呼叫 `as_query_engine()` 後，就可以直接使用：

```python
query_engine.query("什麼是 RAG?")
```

常見重要參數包含：

| 參數 | 說明 |
| --- | --- |
| `similarity_top_k` | 每次查詢時要取回幾個最相關的 chunk，類似 LangChain 的 `k` |
| `response_mode` | 控制 LlamaIndex 如何整合多個 retrieved chunks 並生成答案 |

例如：

```python
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)
```

`similarity_top_k=3` 代表每次查詢會取回最相似的 3 段內容。

`response_mode="compact"` 代表 LlamaIndex 會盡量把檢索到的內容整理後，再交給 LLM 生成回答。

### 6. 查看 prompt

```python
print(query_engine.get_prompts())
```

這行會印出 query engine 內部使用的 prompt。

這對教學很有幫助，因為 LlamaIndex 幫我們包裝了許多細節。透過 `get_prompts()`，可以看到它實際上如何把 context 和 question 組成 prompt。


### LangChain 與 LlamaIndex 對照

| RAG 步驟 | LangChain 寫法 | LlamaIndex 寫法 |
| --- | --- | --- |
| 讀取文件 | `UnstructuredMarkdownLoader` | `SimpleDirectoryReader` |
| 切分文件 | `RecursiveCharacterTextSplitter` | 通常由 index 建立流程處理 |
| Embedding | `HuggingFaceEmbeddings` | `Settings.embed_model = HuggingFaceEmbedding(...)` |
| Vector store / index | `InMemoryVectorStore` | `VectorStoreIndex` |
| Retrieval | `similarity_search(question, k=3)` | `index.as_query_engine(similarity_top_k=3)` |
| Prompt | `ChatPromptTemplate` | Query engine 內建 prompt，可用 `get_prompts()` 查看 |
| LLM | `ChatOpenAI` | `OpenAILike` |
