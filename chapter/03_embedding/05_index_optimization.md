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
| index strategy | 使用純向量搜尋、metadata filter、分層檢索等策略 |

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

### 5.1 分層索引 Hierarchical Index

完整範例程式可以參考：[05_hierarchical_index.py](./05_hierarchical_index.py)

這個範例使用四個資料源：

```text
financials：data/C3/excel/Financials.csv
product_faq：data/C3/hierarchical/product_faq.md
regional_strategy：data/C3/hierarchical/regional_strategy.md
risk_memo：data/C3/hierarchical/risk_memo.md
```

查詢問題是：

```python
QUERY = "What risks are associated with Mexico's discount strategy? Please answer based on the financial data and the risk memo."
```

這個問題同時需要兩類資料：

| 需要的資料 | 對應資料源 |
| --- | --- |
| Mexico 的 sales、discounts、profit 等數值摘要 | `financials` |
| 折扣策略可能造成什麼風險 | `risk_memo` |

如果把所有資料直接丟進同一個 index，系統可能會找回產品 FAQ 或區域策略中語意相近但不夠關鍵的內容。
分層索引會先判斷「應該查哪些資料源」，再進入那些資料源裡做細部檢索。

#### 第一層：建立資料源摘要索引

程式中的 `build_router_index()` 會為每個資料源建立一個摘要 `Document`：

```python
Document(
    text=(
        "Financials dataset. Use this source for quantitative questions "
        "about sales, units sold, discounts, COGS, profit, country, product, segment, month, and year."
    ),
    metadata={"source_id": "financials", "source_type": "csv"},
)
```

第一層不是拿來回答問題，而是拿來選資料源。
查詢時會先執行：

```python
router_retriever = router_index.as_retriever(similarity_top_k=TOP_SOURCE_K)
selected_sources = router_retriever.retrieve(query)
```

`TOP_SOURCE_K = 2` 代表第一層最多選出 2 個最相關資料源。
以範例問題來說，理想上會選到：

```text
financials
risk_memo
```

#### 第二層：進入被選中的資料源檢索

程式中的 `build_source_indexes()` 會為每個資料源建立自己的 `VectorStoreIndex`：

```python
source_documents = {
    "financials": build_financial_documents(),
    "product_faq": build_markdown_documents(...),
    "regional_strategy": build_markdown_documents(...),
    "risk_memo": build_markdown_documents(...),
}

source_indexes = {
    source_id: VectorStoreIndex.from_documents(documents)
    for source_id, documents in source_documents.items()
}
```

第二層會根據第一層選出的 `source_id`，進入對應的子索引：

```python
for source_node in selected_sources:
    source_id = source_node.node.metadata["source_id"]
    retriever = source_indexes[source_id].as_retriever(
        similarity_top_k=TOP_DOC_K
    )
    results = retriever.retrieve(query)
```

`TOP_DOC_K = 3` 代表每個被選中的資料源中，取回 3 筆最相關內容。

#### CSV 資料如何進入分層索引

`Financials.csv` 不是直接把每一列都變成一個 chunk。
範例程式會先依照不同維度建立摘要文件：

```text
country summary
product summary
segment summary
```

例如 `Mexico` 會被整理成一份 country summary，內容包含：

```text
Total units sold
Total sales
Total discounts
Total profit
Products
Segments
Years
```

這樣做的原因是：財務問題通常不只是問某一列，而是需要某個國家、產品或 segment 的彙總資訊。
先把 CSV 轉成可檢索摘要，可以讓 retrieval 更接近分析問題的粒度。

分層索引的完整流程可以整理成：

```text
使用者問題
-> 第一層 router index 選資料源
-> 例如選到 financials + risk_memo
-> 第二層進入 financials 子索引查 Mexico 財務摘要
-> 第二層進入 risk_memo 子索引查 discount risk
-> 回傳結果給後續 LLM 或人工分析
```

這種方式很適合大型知識庫，因為它會先縮小搜尋範圍，再做細部檢索。

### 5.2 遞迴檢索 Recursive Retrieval

遞迴檢索可以把第一層檢索到的節點，當成「指向另一個 retriever 的入口」。

完整範例程式可以參考：[05_recursive_retrieval.py](./05_recursive_retrieval.py)

這個範例使用：

```text
data/C3/excel/music.xlsx
```

這份 Excel 有多個年份 sheet：

```text
1950
2000
2010
```

每個 sheet 都是一個獨立資料源，欄位包含：

```text
artist_name
track_name
release_date
genre
lyrics
topic
```

程式用 `pd.ExcelFile()` 讀取整份 Excel，再把每個 sheet 轉成 list of dict

這樣 `1950`、`2000`、`2010` 會各自成為一個資料源，後面才能為每個年份建立自己的 retriever。

查詢問題是：

```python
QUERY = "Find 2010 pop songs related to violence and summarize what the lyrics are about."
```

這個問題其實包含兩層需求：

| 層級 | 任務 |
| --- | --- |
| 第一層 | 判斷應該進入哪個年份 sheet，例如 `2010` |
| 第二層 | 在該年份 sheet 裡找出 topic / lyrics 相關歌曲 |

#### 第一層：年份 sheet 索引

程式中的 `build_year_router_index()` 會先為每個年份 sheet 建立摘要節點。這裡使用的是 `IndexNode`，不是一般 `Document`：

```python
IndexNode(
    text=(
        f"Music dataset for year {year}. "
        f"This sheet contains {len(rows)} songs. "
        f"Top topics include {top_topics}. "
        f"Genres include {top_genres}. "
        "Use this source for year-specific music, artist, track, genre, topic, and lyrics questions."
    ),
    index_id=f"music_{year}",
    metadata={
        "source_id": f"music_{year}",
        "year": year,
        "source_type": "xlsx_sheet",
    },
)
```

`IndexNode` 的重點是 `index_id`。
當第一層選到 `index_id="music_2010"` 時，`RecursiveRetriever` 會知道下一步要進入 `retriever_dict["music_2010"]`。

這一層不是查歌曲，而是先判斷問題應該進入哪個年份 sheet。

```python
root_retriever = year_router_index.as_retriever(similarity_top_k=TOP_YEAR_K)
```

`TOP_YEAR_K = 1` 代表第一層只選出最相關的一個年份 sheet。
以範例問題來說，理想上會選到：

```text
music_2010
```

#### 第二層：歌曲索引

程式中的 `build_song_retrievers()` 會為每個年份 sheet 建立自己的歌曲索引，並轉成 retriever：

```python
song_index = VectorStoreIndex.from_documents(song_documents)
song_retrievers[f"music_{year}"] = song_index.as_retriever(
    similarity_top_k=TOP_SONG_K
)
```

每一首歌會被轉成一個 `Document`：

```python
text = (
    f"Year: {year}\n"
    f"Artist: {row.get('artist_name', '')}\n"
    f"Track: {row.get('track_name', '')}\n"
    f"Genre: {row.get('genre', '')}\n"
    f"Topic: {row.get('topic', '')}\n"
    f"Lyrics keywords: {row.get('lyrics', '')}"
)
```

`TOP_SONG_K = 5` 代表進入該年份後，取回 5 首最相關歌曲。

#### 用 `RecursiveRetriever` 串起兩層

```python
recursive_retriever = RecursiveRetriever(
    "root",
    retriever_dict={
        "root": root_retriever,
        **song_retrievers,
    },
    verbose=True,
)

results = recursive_retriever.retrieve(query)
```

`retriever_dict` 裡面有兩種 key：

| key | 作用 |
| --- | --- |
| `root` | 第一層年份 router retriever |
| `music_1950`、`music_2000`、`music_2010` | 第二層年份歌曲 retriever |

當 `root` retriever 找到 `IndexNode(index_id="music_2010")`，`RecursiveRetriever` 會自動用這個 `index_id` 找到對應的 `music_2010` retriever，接著在 2010 年的歌曲資料中繼續檢索。

#### 為什麼這是遞迴檢索

這個流程的重點是：第一層找到的不是最終答案，而是一個「通往下一層查詢的入口」。

```text
使用者問題
-> root retriever 查年份摘要索引
-> 找到 IndexNode(index_id="music_2010")
-> RecursiveRetriever 依照 index_id 進入 music_2010 retriever
-> 找到 topic / lyrics 最相關的歌曲
-> 回傳歌曲 metadata 與內容
```

如果未來有更多 sheet，例如：

```text
1980
1990
2020
```

就不需要把所有歌曲全部放進同一個索引裡直接搜尋，而是先選年份，再進入該年份查詢。這就是遞迴檢索的價值。

#### 執行方式

執行方式：

```powershell
python chapter/03_embedding/05_recursive_retrieval.py
```

因為 `2010` sheet 有兩萬多筆資料，教學範例預設每個 sheet 只取前 300 筆建立索引：

```python
DEFAULT_ROW_LIMIT = 300
```

如果想調整筆數：

```powershell
python chapter/03_embedding/05_recursive_retrieval.py --row-limit 1000
```

## 六、索引優化的實務選擇

不同情境適合不同策略。

| 情境 | 建議策略 |
| --- | --- |
| 小型教學專案 | 一般 chunking + FAISS 即可 |
| 長文件問答 | 句子窗口檢索或 parent-child chunk |
| 多文件大型知識庫 | metadata filter + 分層索引 |
| API / 技術文件 | 保留函式名稱、錯誤碼、版本號等 metadata，必要時先用 filter 縮小範圍 |
| 表格資料 | 先路由資料源，再用安全方式查詢表格 |
| 多模態資料 | 保存 modality、image_path、page、caption 等 metadata |
| 權限敏感資料 | retrieval 前必須先做 permission filter |

可以把優化順序想成：

```text
先讓資料結構正確
再讓 metadata 完整
再調 chunk 與 index
最後再依資料規模加入分層索引或遞迴檢索
```

## 七、評估索引效果

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
| Cost | embedding、LLM token 與查詢流程成本是否合理 |

如果 retrieval 沒有找回正確資料，後面的 LLM 再強也很難回答正確。  
因此在 RAG 系統中，索引與檢索品質通常比 prompt 技巧更關鍵。

## 八、本節重點整理

索引優化的核心不是把所有技巧都加上去，而是根據資料特性選擇合適的檢索結構。

```text
小 chunk 適合精準檢索，但上下文不足
大 chunk 上下文完整，但容易引入雜訊
句子窗口檢索可以先找小句子，再擴展上下文
metadata filter 可以縮小搜尋範圍
分層索引適合大型、多資料源知識庫
遞迴檢索可以讓上層節點指向下層 retriever
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
