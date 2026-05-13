# 第五節 索引優化

前面幾節已經介紹了 embedding、向量資料庫、FAISS 與 Milvus。到這裡為止，我們已經可以把文件切成 chunks、轉成向量，並用向量資料庫做相似度搜尋。

但是，能搜尋不代表搜尋品質就一定好。

在真實 RAG 系統裡，常見問題通常不是「完全找不到資料」，而是：

```text
找到了相關資料，但上下文不夠
找到了語意相近但其實不該回答的資料
大型知識庫搜尋範圍太大，結果被雜訊干擾
Top-K 回來的 chunks 太零散，LLM 很難組成完整答案
```

**索引優化（Index Optimization）** 的目的，就是在「資料如何被切分、儲存、標記、檢索與送進 LLM」這幾個環節中做設計，讓 retrieval 更精準，也讓最終回答更穩定。

## 一、索引優化在優化什麼

RAG 的索引不只是向量本身，而是一整套資料結構。

一個比較完整的索引通常包含：

| 組成 | 說明 |
| --- | --- |
| chunk text | 實際被 embedding 與檢索的文字 |
| embedding vector | 由 embedding model 產生的向量 |
| metadata | 來源、頁碼、章節、類型、日期、權限等資訊 |
| parent / child 關係 | 小 chunk 與大段落、章節、原始文件之間的關係 |
| index strategy | 使用純向量搜尋、metadata filter、hybrid search、分層檢索等策略 |

因此索引優化不是單純調整 `chunk_size`，而是要回答這幾個問題：

```text
要用多小的單位做精準檢索？
要用多大的上下文交給 LLM？
哪些 metadata 必須保存？
查詢時應該先過濾資料，還是直接全庫搜尋？
資料量變大後，如何避免不相關資料干擾？
```

## 二、為什麼需要上下文擴展

在第二章文字分塊時，我們提過 chunk 大小會影響 retrieval。

如果 chunk 太小，優點是檢索精準，缺點是上下文不足：

```text
使用者問：Milvus 適合什麼情境？

檢索到的 chunk：
「Milvus 適合大規模向量搜尋。」

問題：
這句話本身正確，但缺少更多背景，例如 metadata filter、服務化部署、多人共用等資訊。
```

如果 chunk 太大，優點是上下文完整，缺點是容易帶入雜訊：

```text
一整頁內容都被放進同一個 chunk
其中只有兩句和問題有關
其他內容會干擾向量相似度，也會佔用 LLM context window
```

所以實務上常見的做法是：

```text
用小單位做檢索，用較大的上下文做生成
```

這就是上下文擴展的核心想法。

## 三、句子窗口檢索

**句子窗口檢索（Sentence Window Retrieval）** 是一種常見的上下文擴展策略。

它的做法是：

```text
索引時：把文件切成一句一句的小節點
metadata 中：保存該句前後幾句的上下文窗口
檢索時：用單句做向量搜尋
送給 LLM 前：把單句替換成包含前後文的窗口內容
```

這樣可以同時兼顧兩件事：

| 目標 | 做法 |
| --- | --- |
| 檢索精準 | embedding 的內容是單一句子，比較容易對準問題 |
| 回答完整 | 最後送給 LLM 的不是孤立句子，而是前後文一起組成的窗口 |

### 3.1 概念流程

假設原文有這幾句：

```text
S1：RAG 會先把文件切成 chunks。
S2：每個 chunk 會被 embedding model 轉成向量。
S3：向量會存進 vector database。
S4：使用者提問時，系統會找回最相似的 chunks。
S5：LLM 會根據找回的內容生成答案。
```

如果 `window_size=1`，那麼 S3 這個節點會長這樣：

```python
{
    "text": "向量會存進 vector database。",
    "metadata": {
        "window": "每個 chunk 會被 embedding model 轉成向量。向量會存進 vector database。使用者提問時，系統會找回最相似的 chunks。",
        "original_text": "向量會存進 vector database。",
    },
}
```

檢索時，系統用 `text` 產生 embedding，因此檢索單位很小。  
但送進 LLM 前，系統會把 `text` 換成 `metadata["window"]`，因此回答時可以看到更多上下文。

### 3.2 LlamaIndex 範例

完整範例程式可以參考：[05_sentence_window_retrieval.py](./05_sentence_window_retrieval.py)

LlamaIndex 可以用 `SentenceWindowNodeParser` 建立句子窗口節點：

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

nodes = node_parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
```

重要參數：

| 參數 | 說明 |
| --- | --- |
| `window_size` | 每個句子前後要保留幾句作為上下文。`3` 代表前 3 句、自己、後 3 句 |
| `window_metadata_key` | 儲存上下文窗口的 metadata key，常用 `"window"` |
| `original_text_metadata_key` | 儲存原始單句的 metadata key，方便除錯與觀察檢索結果 |
| `similarity_top_k` | 檢索時取回幾個最相似節點 |
| `MetadataReplacementPostProcessor` | 在送入 LLM 前，把節點文字替換成指定 metadata 欄位 |

### 3.3 什麼時候適合用

句子窗口檢索適合：

1. 長文件問答，例如報告、論文、法規、技術文件。
2. 問題常常對應到某一句關鍵資訊，但回答需要前後文。
3. 不希望 chunk 太大，導致向量搜尋被雜訊干擾。

但它也有成本：

| 成本 | 說明 |
| --- | --- |
| 節點數變多 | 以句子為單位切分，向量數量會比一般 chunk 更多 |
| metadata 變大 | 每個節點都要保存上下文窗口 |
| context 仍需控制 | `window_size` 太大時，最後送入 LLM 的文字仍可能過長 |

## 四、結構化索引與 Metadata

另一個重要的索引優化方向，是把資料整理成可過濾、可路由的結構。

在 RAG 中，metadata 不只是補充資訊，而是檢索品質的重要控制條件。

常見 metadata 包含：

| Metadata | 用途 |
| --- | --- |
| `source` | 原始檔案路徑或文件名稱 |
| `page` | PDF 頁碼 |
| `chapter` | 章節名稱 |
| `section` | 小節標題 |
| `modality` | `text`、`image`、`table`、`audio` 等資料型態 |
| `created_at` | 文件建立時間 |
| `permission` | 權限或可見範圍 |
| `document_type` | FAQ、合約、財報、技術文件等 |

### 4.1 為什麼 metadata 很重要

假設知識庫裡有很多種文件：

```text
產品說明書
內部 SOP
合約
財報
教學文件
客服紀錄
```

如果使用者問：

```text
請整理 2025 年第二季財報中提到 AI 的部分
```

系統不應該直接對整個向量庫做 top-k 搜尋，而是應該先縮小範圍：

```python
metadata filter:
document_type == "financial_report"
year == 2025
quarter == "Q2"

vector search:
query = "AI 相關內容"
```

這種「先過濾，再做向量搜尋」的方式，可以減少雜訊，也能提升查詢速度。

### 4.2 Markdown 文件的結構化索引

對 Markdown 文件來說，標題本身就是很好的結構。

例如：

```markdown
# 第三章 Embedding
## 一、Embedding 是什麼
## 二、向量空間
## 三、常見 Embedding Model
```

切分時可以把標題存成 metadata：

```python
{
    "page_content": "Embedding 會把文字轉成向量...",
    "metadata": {
        "source": "01_what_is_embedding.md",
        "chapter": "第三章 Embedding",
        "section": "一、Embedding 是什麼",
    },
}
```

查詢時就可以指定：

```text
只搜尋 chapter == "第三章 Embedding"
```

這比單純依靠向量相似度更穩定。

## 五、分層索引與遞迴檢索

當資料量變大時，所有 chunks 放在同一個向量索引裡直接查詢，會遇到兩個問題：

```text
搜尋範圍太大，容易找回不相關 chunks
不同資料源的格式不同，例如 Markdown、PDF、表格、圖片，不能全部用同一種方式處理
```

這時可以使用 **分層索引（Hierarchical Index）** 或 **遞迴檢索（Recursive Retrieval）**。

核心概念是：

```text
第一層：先判斷問題應該去哪一個資料源
第二層：再進入該資料源內部做精準檢索
```

### 5.1 路由索引

假設有多個資料集：

```text
2023 財報
2024 財報
產品 FAQ
內部維運 SOP
RAG 課程教材
```

可以先為每個資料集建立一個摘要節點：

```python
route_docs = [
    {
        "text": "這份文件包含 2024 年公司財報與 AI 業務相關內容。",
        "metadata": {"target": "financial_report_2024"},
    },
    {
        "text": "這份文件包含 RAG 課程教材，介紹 chunking、embedding、vector database。",
        "metadata": {"target": "rag_course"},
    },
]
```

使用者查詢時，系統先在摘要節點中找到最可能的資料源，再到對應資料源裡搜尋。

```text
使用者問題
-> 搜尋摘要索引
-> 找到目標資料源
-> 對該資料源做 metadata filter 或子索引查詢
-> 回傳結果給 LLM
```

這種方式很適合大型知識庫。

### 5.2 遞迴檢索

遞迴檢索可以把一個節點當成「指向另一個查詢引擎的入口」。

概念上像這樣：

```text
問題：1994 年評分最高的電影是哪一部？

第一層索引：
找到「1994 年電影資料表」這個節點

第二層查詢：
進入 1994 年的表格查詢引擎

最後回答：
回傳該表格中的查詢結果
```

這種做法適合：

1. 多份文件或多個資料表。
2. 每個資料源需要不同查詢方式。
3. 不希望每次都對全資料庫搜尋。
4. 需要把「找資料源」和「在資料源內查答案」拆開處理。

### 5.3 關於表格查詢的安全提醒

有些工具可以把自然語言轉成 Pandas 程式碼，再用程式查表格。這種方式很方便，但要注意安全風險。

如果工具底層會執行模型產生的 Python code，例如透過 `eval()` 執行，就不適合直接用在正式環境。

比較安全的做法是：

```text
先用摘要索引判斷要查哪張表
再用 metadata filter 限制搜尋範圍
盡量避免讓 LLM 直接產生並執行任意 Python code
必要時使用 sandbox、權限隔離、查詢白名單
```

## 六、Hybrid Search

向量搜尋擅長語意相似，但不一定擅長精確關鍵字。

例如：

```text
錯誤碼：ERR-4291
產品型號：BGE-M3
合約條款：第 7.2 條
API 名稱：MetadataReplacementPostProcessor
```

這類查詢常常需要關鍵字搜尋。  
因此實務上常會把兩種方法結合：

| 方法 | 擅長 |
| --- | --- |
| Vector Search | 語意相似、同義詞、自然語言問題 |
| Keyword Search / BM25 | 精確詞、型號、錯誤碼、專有名詞 |
| Hybrid Search | 同時考慮語意與關鍵字 |

Hybrid Search 的概念是：

```text
同一個 query
-> 做 vector search
-> 做 keyword search
-> 合併與重新排序結果
-> 回傳最終 top-k
```

在技術文件、API 文件、錯誤排查、法規條文中，Hybrid Search 通常比純向量搜尋更穩定。

## 七、Rerank：重新排序檢索結果

向量資料庫回傳的 top-k，不一定就是最適合交給 LLM 的順序。

常見流程會加入 reranker：

```text
query
-> vector database 先取回 top-20
-> reranker 重新評分
-> 選出 top-5
-> 交給 LLM
```

Reranker 和 embedding model 的差異是：

| 模型 | 工作方式 |
| --- | --- |
| Embedding model | 先把 query 和 chunk 各自轉成向量，再計算相似度 |
| Reranker | 同時讀取 query 和 chunk，直接判斷這段內容是否能回答問題 |

Reranker 通常比較慢，但判斷更細。  
所以實務上常見做法是「先粗搜，再精排」。

## 八、索引優化的實務選擇

不同情境適合不同策略。

| 情境 | 建議策略 |
| --- | --- |
| 小型教學專案 | 一般 chunking + FAISS 即可 |
| 長文件問答 | 句子窗口檢索或 parent-child chunk |
| 多文件大型知識庫 | metadata filter + 分層索引 |
| API / 技術文件 | hybrid search + reranker |
| 表格資料 | 先路由資料源，再用安全方式查詢表格 |
| 多模態資料 | 保存 modality、image_path、page、caption 等 metadata |
| 權限敏感資料 | retrieval 前必須先做 permission filter |

可以把優化順序想成：

```text
先讓資料結構正確
再讓 metadata 完整
再調 chunk 與 index
最後加入 rerank / hybrid / hierarchical retrieval
```

## 九、評估索引效果

索引優化不能只靠感覺，需要用問題集測試。

可以準備一組 evaluation questions：

```text
這個問題應該從哪份文件找答案？
正確答案應該包含哪些關鍵資訊？
系統是否找回正確 chunk？
LLM 回答是否引用了正確來源？
```

常見觀察指標：

| 指標 | 說明 |
| --- | --- |
| Recall@K | 正確資料是否出現在 top-k 結果中 |
| Precision@K | top-k 裡有多少是真正相關資料 |
| MRR | 正確答案出現得越前面越好 |
| Answer quality | LLM 最後回答是否完整、正確、可追溯 |
| Latency | 查詢速度是否可接受 |
| Cost | embedding、rerank、LLM token 成本是否合理 |

如果 retrieval 沒有找回正確資料，後面的 LLM 再強也很難回答正確。  
因此在 RAG 系統中，索引與檢索品質通常比 prompt 技巧更關鍵。

## 十、本節重點整理

索引優化的核心不是把所有技巧都加上去，而是根據資料特性選擇合適的檢索結構。

```text
小 chunk 適合精準檢索，但上下文不足
大 chunk 上下文完整，但容易引入雜訊
句子窗口檢索可以先找小句子，再擴展上下文
metadata filter 可以縮小搜尋範圍
分層索引適合大型、多資料源知識庫
hybrid search 適合技術文件、錯誤碼、專有名詞
reranker 可以提升 top-k 結果排序品質
```

一個好的 RAG 索引設計，應該讓系統在回答問題前就先做到：

```text
找對資料源
找對段落
保留足夠上下文
排除不相關內容
控制查詢成本與延遲
```

## 參考資料

- [LlamaIndex：Building Performant RAG Applications for Production](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- [LlamaIndex：Metadata Replacement + Node Sentence Window](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/)
- [LlamaIndex：Recursive Retriever + Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/)
- [LlamaIndex：Pandas Query Engine](https://docs.llamaindex.ai/en/stable/api_reference/query_engine/pandas/)
