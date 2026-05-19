# 第一節 混合檢索 Hybrid Search

前面章節已經介紹過 embedding、向量資料庫、Milvus，以及幾種索引優化方式。

到這裡我們已經可以用 dense embedding 做語意搜尋，例如：

```text
使用者問：「怎麼提升 RAG 檢索品質？」
系統可以找回「索引優化、chunking、rerank」相關內容
```

但是純向量搜尋並不總是夠用。真實知識庫裡常常會出現這些問題：

```text
使用者查產品型號、錯誤碼、函式名稱，但向量搜尋沒有精準命中
使用者用同義詞提問，關鍵字搜尋卻找不到
文件裡有專有名詞、版本號、法條、API 名稱，需要精準匹配
語意相近的 chunk 很多，但真正有關鍵字的內容反而排序較低
```

**混合檢索（Hybrid Search）** 的目的，就是把「關鍵字匹配」和「語意搜尋」結合起來，讓系統同時具備：

| 能力 | 對應方法 |
| --- | --- |
| 精準命中特定詞彙 | 稀疏向量 / BM25 |
| 理解語意、同義詞與上下文 | 密集向量 / Embedding |
| 合併兩種結果並重新排序 | RRF 或加權融合 |

簡單來說，Hybrid Search 不是要取代 dense vector search，而是補足它在精確詞彙上的弱點。

## 一、為什麼 RAG 需要混合檢索

RAG 的回答品質很大程度取決於 retrieval 是否找對資料。

如果 retrieval 階段漏掉關鍵 chunk，後面的 LLM 再強也很難回答正確。常見情境如下：

| 查詢類型 | 純向量搜尋可能的問題 | 關鍵字搜尋的價值 |
| --- | --- | --- |
| 錯誤碼 | `ERR-4291` 可能被當成不重要 token | 可以精準匹配錯誤碼 |
| API 名稱 | `MetadataReplacementPostProcessor` 太長且罕見 | 可以直接找出包含完整名稱的文件 |
| 產品型號 | `A17 Pro`、`RTX 4090` 容易和相似產品混在一起 | 可以保留型號差異 |
| 法條 / 條款 | 語意相近不代表條文相同 | 可以精準定位條號 |
| 中文同義詞 | 關鍵字搜尋找不到不同說法 | dense embedding 可以補足語意 |

因此實務上常見做法是：

```text
Dense retrieval 找語意相關內容
Sparse retrieval 找關鍵字精準命中內容
Hybrid search 合併兩邊結果
Reranker 或 fusion method 決定最後排序
```

## 二、稀疏向量與密集向量

理解 Hybrid Search 前，要先分清楚兩種向量。

### 2.1 稀疏向量 Sparse Vector

**稀疏向量**比較接近傳統資訊檢索。它會根據詞彙是否出現、出現頻率、詞彙重要性等因素建立向量。

它的特徵是：

| 特性 | 說明 |
| --- | --- |
| 維度很高 | 維度通常接近詞彙表大小 |
| 大多數值是 0 | 文件只會包含詞彙表中的少部分詞 |
| 可解釋性高 | 非零維度通常對應到具體詞彙 |
| 擅長精準匹配 | 適合錯誤碼、型號、專有名詞、關鍵字 |

例如一個簡化後的稀疏向量可以長這樣：

```json
{
  "88": 1.2,
  "666": 0.8,
  "999": 1.5
}
```

這代表只有第 `88`、`666`、`999` 這幾個詞彙維度有權重，其他大多數維度都是 0。

常見的稀疏檢索方法包含：

| 方法 | 說明 |
| --- | --- |
| TF-IDF | 根據詞頻與逆文件頻率計算詞彙重要性 |
| BM25 | 傳統搜尋引擎常用的排序方法 |
| SPLADE | 用神經網路產生稀疏表示 |
| BGE-M3 sparse vector | 同一個 embedding 模型同時輸出 sparse 與 dense 向量 |

### 2.2 BM25 的核心概念

BM25 是非常常見的 sparse retrieval 方法。它的直覺是：

```text
查詢詞出現在文件中，分數會提高
查詢詞越罕見，重要性越高
文件太長時，要做長度正規化，避免長文件天然佔優勢
同一個詞出現很多次，分數不會無限線性增加
```

BM25 常見公式如下：

$$
Score(Q, D) = \sum_{i=1}^{n} IDF(q_i) \cdot
\frac{f(q_i, D) \cdot (k_1 + 1)}
{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}
$$

其中：

| 參數 | 說明 |
| --- | --- |
| $Q$ | 使用者查詢 |
| $D$ | 被檢索的文件 |
| $q_i$ | 查詢中的第 i 個詞 |
| $IDF(q_i)$ | 詞彙的逆文件頻率，越少見通常越重要 |
| $f(q_i, D)$ | 該詞在文件中的出現次數 |
| $|D|$ | 文件長度 |
| $avgdl$ | 全部文件的平均長度 |
| $k_1$ | 控制詞頻飽和程度 |
| $b$ | 控制文件長度正規化程度 |

BM25 的優點是精準、可解釋、速度快。缺點是它不理解語意。

例如：

```text
文件：西紅柿炒蛋
查詢：番茄炒蛋
```

如果系統沒有同義詞擴展，BM25 可能會把「西紅柿」和「番茄」視為不同詞。

### 2.3 密集向量 Dense Vector

**密集向量**就是前面 embedding 章節介紹過的語意向量。它通常由 embedding model 產生，例如 BGE、OpenAI embedding、Gemini embedding 等。

它的特徵是：

| 特性 | 說明 |
| --- | --- |
| 維度固定 | 例如 768、1024、1536、3072 等 |
| 幾乎每個維度都有值 | 所以稱為 dense |
| 擅長語意相似 | 可以理解同義詞、上下文、概念接近 |
| 可解釋性較低 | 單一維度通常沒有明確人類語意 |

例如：

```json
[0.89, -0.12, 0.77, 0.03, -0.45]
```

實際向量會更長。這些數字本身不容易解讀，但在向量空間中，語意相近的內容距離會比較近。

Dense vector 適合：

```text
語意問答
同義詞搜尋
概念型問題
使用者沒有使用文件原文詞彙的情境
```

但它不一定適合：

```text
錯誤碼
產品型號
版本號
函式名稱
必須精準命中的條文
```

## 三、Hybrid Search 的基本流程

Hybrid Search 通常會同時做兩件事：

```text
1. 用 sparse vector / BM25 找關鍵字相關結果
2. 用 dense vector 找語意相關結果
3. 把兩組結果融合成一個排序
```

流程可以表示為：

```text
使用者問題
-> 產生 dense query vector
-> 產生 sparse query vector
-> dense search 找語意相近 chunks
-> sparse search 找關鍵字命中 chunks
-> fusion / rerank
-> 回傳 top-k 給 LLM
```

這裡的難點不是「同時查兩次」而已，而是如何把兩組分數合併。

因為 dense search 和 sparse search 的分數來源不同：

```text
dense score 可能是 cosine similarity 或 inner product
sparse score 可能是 BM25 或 sparse inner product
兩者分數尺度不一定一致
```

因此需要 fusion strategy。

## 四、常見融合方法

### 4.1 RRF：Reciprocal Rank Fusion 倒數排名融合法

**RRF（Reciprocal Rank Fusion）** 是很常見也很穩定的融合方法。

它不直接比較原始分數，而是看「排名」。

公式如下：

$$
RRF_{score}(d) = \sum_{i=1}^{k} \frac{1}{rank_i(d) + c}
$$

其中：

| 參數 | 說明 |
| --- | --- |
| $d$ | 某一筆候選文件 |
| $k$ | 檢索系統數量，例如 sparse + dense 就是 2 |
| $rank_i(d)$ | 文件在第 i 個檢索系統中的排名 |
| $c$ | 平滑常數，常見值是 60 |

RRF 的直覺是：

```text
如果一份文件在 dense search 排很前面，它會加分
如果它在 sparse search 也排很前面，它會再加分
如果它只在某一邊出現，也仍然有機會被保留
```

RRF 的優點是不用處理不同分數尺度，對多路檢索結果融合很方便。

### 4.2 加權分數融合

另一種方式是先把兩邊分數正規化，再做加權平均：

$$
Hybrid_{score} = \alpha \cdot Dense_{score} + (1 - \alpha) \cdot Sparse_{score}
$$

其中 `α` 控制 dense search 的權重：

| α 值 | 效果 |
| --- | --- |
| 接近 1 | 更偏重語意搜尋 |
| 接近 0 | 更偏重關鍵字搜尋 |
| 0.5 | sparse 和 dense 權重接近 |

例如：

```text
FAQ 問答：可以提高 dense 權重
錯誤碼搜尋：可以提高 sparse 權重
法律條文搜尋：通常不能忽略 sparse 權重
```

加權融合比較直覺，但要注意分數正規化。否則某一邊的分數範圍比較大，就會主導排序。

## 五、Hybrid Search 的優點與限制

| 面向 | 說明 |
| --- | --- |
| 優點 | 同時保留語意理解與關鍵字精準匹配 |
| 優點 | 對專有名詞、錯誤碼、型號、API 名稱更穩定 |
| 優點 | 可以提高召回率，降低漏掉關鍵 chunk 的機率 |
| 限制 | 需要維護 sparse 與 dense 兩種索引 |
| 限制 | 查詢成本比單一路徑高 |
| 限制 | RRF、權重、top-k 等參數需要實驗調整 |
| 限制 | 融合後的排序解釋性比單純 BM25 更複雜 |

如果資料與查詢符合以下情境，就很適合使用 Hybrid Search：

| 情境 | 原因 |
| --- | --- |
| 技術文件 | 函式名稱、類別名稱、錯誤碼需要精準命中 |
| 企業內部知識庫 | 文件標題、專案代號、人名、部門名很重要 |
| 法規文件 | 條號與原文措辭不能只靠語意相似 |
| 商品搜尋 | 型號、品牌、規格需要精準匹配 |
| 多語言資料 | dense vector 可補語意，sparse vector 可保留原文詞彙 |
| RAG 回答容易漏關鍵字 | sparse search 可以補回被 dense search 忽略的 chunk |

如果資料很小、查詢也很單純，純 dense vector search 可能已經足夠。

但只要系統開始面對真實使用者，通常很快會遇到：

```text
語意搜尋找得到相似內容，但找不到精確內容
關鍵字搜尋找得到精確詞，但找不到同義內容
```

這就是 Hybrid Search 的價值。

## 六、程式範例：BM25 + Gemini Embedding + RRF

本節範例程式放在：

[01_hybrid_search.py](./01_hybrid_search.py)

這支程式示範的是一個最小可執行的 Hybrid Search 流程：

```text
讀取 Markdown 文件
-> 切成 chunks
-> BM25Retriever 做 sparse retrieval
-> Gemini embedding 做 dense retrieval
-> EnsembleRetriever 用 RRF 合併排序
-> 比較 sparse、dense、hybrid 三組結果
```

範例資料放在：

```text
data/C4/hybrid_search/
```

這些文件刻意包含兩種查詢情境：

| 查詢情境 | 例子 | 目的 |
| --- | --- | --- |
| 語意問題 | `How does MCP help agent tool interoperability?` | 測試 dense retrieval 是否能找出真正回答問題的段落 |
| 精準詞彙 | `ERR-4291`、`MetadataReplacementPostProcessor` | 測試 BM25 是否能穩定命中錯誤碼、API 名稱或類別名稱 |

### 6.1 載入文件與切分 chunks

每個 chunk 會加上固定的 `chunk_id`：

```python
document.metadata["chunk_id"] = f"chunk_{index:04d}"
```

`chunk_id` 很重要，因為後面 RRF 融合時需要知道 sparse retrieval 和 dense retrieval 找到的是不是同一個 chunk。

### 6.2 BM25 sparse retrieval

使用 LangChain Community 的 `BM25Retriever`：

```python
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(
    chunks,
    preprocess_func=tokenize,
)
retriever.k = top_k
```

`preprocess_func=tokenize` 會把文字轉成 BM25 使用的 tokens：

```python
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())
```

這個 tokenizer 特別保留 `_`、`.`、`/`、`:`、`-`，是因為技術文件常有：

```text
ERR-4291
MetadataReplacementPostProcessor
api/v1/tools
model:gemini-embedding-2
```

如果 tokenizer 把這些符號全部切掉，BM25 對錯誤碼、API 路徑、類別名稱的精準匹配能力就會下降。

### 6.3 Gemini dense retrieval

Dense retrieval 使用 Gemini embedding：

```python
EMBEDDING_MODEL = "gemini-embedding-2"
OUTPUT_DIMENSIONALITY = 768
```

Dense retrieval 會用 cosine similarity 排序：

```python
cosine_similarity(query_vector, vector)
```

因此它比較擅長處理語意問題，例如：

```text
How does MCP help agent tool interoperability?
```

即使文件沒有完全出現相同句子，dense retrieval 仍可能找出「MCP 提供標準化工具通訊方式」這類真正回答問題的段落。

### 6.4 RRF hybrid fusion

使用 LangChain 的 `EnsembleRetriever`，根據Reciprocal Rank Fusion算法對結果進行重新排序：

```python
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.runnables import RunnableLambda

return EnsembleRetriever(
    retrievers=[bm25_retriever, RunnableLambda(dense_retriever.invoke)],
    weights=[0.5, 0.5],
    c=rrf_k,
    id_key="chunk_id",
)
```

幾個重要參數如下：

| 參數 | 說明 |
| --- | --- |
| `retrievers` | 放入多個 retriever，這裡是 BM25 和 Gemini dense retriever |
| `weights` | 控制每一路檢索的權重，`[0.5, 0.5]` 代表兩邊同等重要 |
| `c` | RRF 的平滑常數，預設使用 `60` |
| `id_key` | 用 metadata 裡的 `chunk_id` 判斷不同 retriever 是否命中同一筆資料 |

如果查詢偏向語意問題，可以提高 dense 權重：

```python
weights=[0.3, 0.7]
```

如果查詢偏向錯誤碼、API 名稱、產品型號，可以提高 BM25 權重：

```python
weights=[0.7, 0.3]
```

### 6.5 執行方式與結果解讀

執行查詢：

```bash
python chapter/04_hybrid_search/01_hybrid_search.py "How does MCP help agent tool interoperability?"
```

程式會輸出三組結果：

```text
Sparse / BM25Retriever results
Dense Gemini embedding results
Hybrid EnsembleRetriever RRF results
```

解讀時可以這樣看：

| 結果 | 觀察重點 |
| --- | --- |
| Sparse / BM25 | 是否命中錯誤碼、API 名稱、專有名詞 |
| Dense Gemini embedding | 是否找出真正回答問題的語意段落 |
| Hybrid RRF | 是否同時保留精準詞彙與語意相關內容 |

例如查詢 `How does MCP help agent tool interoperability?` 時，dense retrieval 通常會更容易找出真正解釋 MCP 的段落；BM25 可能會找出包含關鍵字的 retrieval notes。這代表 hybrid search 的權重需要依照資料與查詢型態調整。

## 七、Milvus + BGE-M3 Hybrid Search

Milvus 版本範例程式放在：

[01_hybrid_search_milvus.py](./01_hybrid_search_milvus.py)

這支程式和前面的 LangChain 版本使用同一份資料：

```text
data/C4/hybrid_search/
```

這裡不再使用 `Gemini dense embedding + BM25 sparse retrieval`，而是改成 `BGE-M3 dense vector + BGE-M3 sparse vector`。也就是 dense 和 sparse 兩種檢索表示都由 `BAAI/bge-m3` 產生，再交給 Milvus 儲存與查詢。

`BAAI/bge-m3` 可以這樣做，是因為它本身就是為 retrieval 任務設計的多功能 embedding model。一般 embedding model 通常只輸出一組 dense vector，用來做語意相似度搜尋；但 BGE-M3 在模型輸出中同時提供不同檢索表示：

| 表示 | 作用 |
| --- | --- |
| Dense vector | 把整段文字壓成語意向量，適合找語意相近內容 |
| Sparse vector | 保留詞彙層級的重要性，適合錯誤碼、專有名詞、API 名稱等精準匹配 |
| Multi-vector | 以更細的 token-level 表示做比對，本範例暫時不使用 |

所以這裡的 sparse vector 不是 BM25 算出來的，而是 BGE-M3 模型根據文字內容產生的 lexical weights。它和 BM25 一樣能補足 dense retrieval 對精準詞彙不穩的問題，但來源是神經模型，而不是傳統詞頻公式。

### 7.1 使用 BGE-M3 產生 dense 與 sparse vectors

程式使用 `pymilvus.model.hybrid` 提供的 `BGEM3EmbeddingFunction`：

```python
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

EMBEDDING_MODEL = "BAAI/bge-m3"

embedding_function = BGEM3EmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    device=device,
    normalize_embeddings=True,
    use_fp16=False,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
```

重要參數如下：

| 參數 | 說明 |
| --- | --- |
| `model_name` | 指定使用 `BAAI/bge-m3` |
| `device` | 指定執行裝置，例如 `cpu`、`cuda`、`cuda:0` |
| `normalize_embeddings` | 對 dense vector 做正規化，方便用 cosine 類型相似度 |
| `return_dense` | 是否回傳 dense embedding |
| `return_sparse` | 是否回傳 sparse embedding |
| `return_colbert_vecs` | 是否回傳 multi-vector，本範例先關閉 |

產生文件向量時：

```python
embeddings = embedding_function.encode_documents(
    [chunk.page_content for chunk in chunks]
)

dense_vectors = embeddings["dense"]
sparse_vectors = embeddings["sparse"]
```

查詢時也是同一個 embedding function：

```python
query_embeddings = embedding_function.encode_queries([query])

query_vector = query_embeddings["dense"][0]
sparse_query_vector = sparse_row_to_dict(query_embeddings["sparse"][0])
```

這樣做的好處是 sparse 和 dense 來自同一個 retrieval model，不需要另外維護 BM25 tokenizer。

### 7.2 建立 Milvus collection schema

Milvus collection 同時放入文字欄位、metadata、dense vector 和 sparse vector：

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(
        name="dense_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=dense_dim,
    ),
    FieldSchema(
        name="sparse_vector",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
    ),
]
```

這裡的關鍵是：

| 欄位 | 用途 |
| --- | --- |
| `dense_vector` | 存 BGE-M3 的 dense embedding，用於語意搜尋 |
| `sparse_vector` | 存 BGE-M3 的 sparse embedding，用於詞彙匹配 |
| `chunk_id` | 對應原始 chunk，方便追蹤資料來源 |
| `content` | 儲存原文內容，查詢後可以直接回傳給 LLM |

### 7.3 建立 dense 與 sparse index

Dense vector 使用 HNSW index：

```python
index_params.add_index(
    field_name="dense_vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,
        "efConstruction": 256,
    },
)
```

Sparse vector 使用 Milvus 的 sparse inverted index：

```python
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={
        "drop_ratio_build": 0.0,
    },
)
```

`drop_ratio_build` 會影響建立 sparse index 時是否丟棄低權重項目。本範例設為 `0.0`，代表教學時先完整保留 sparse vector 資訊。

### 7.4 寫入 chunks

寫入 Milvus 時，每一筆 chunk 會同時包含 dense vector 和 sparse vector：

```python
rows.append(
    {
        "id": index,
        "chunk_id": chunk.metadata["chunk_id"],
        "source": chunk.metadata["source"],
        "title": chunk.metadata["title"],
        "content": chunk.page_content,
        "dense_vector": [float(value) for value in dense_vector],
        "sparse_vector": sparse_row_to_dict(sparse_vector),
    }
)
```

`sparse_row_to_dict()` 的作用是把 BGE-M3 回傳的 sparse matrix 轉成 Milvus 可接受的 sparse vector 格式：

```python
{
    102: 0.8,
    981: 1.4,
}
```

也就是只保留非零維度，符合 sparse vector 的資料型態。

### 7.5 Milvus hybrid search

查詢時會建立兩個 `AnnSearchRequest`。

Dense request 查 `dense_vector`：

```python
dense_request = AnnSearchRequest(
    data=[query_vector],
    anns_field="dense_vector",
    param={
        "metric_type": "COSINE",
        "params": {"ef": 128},
    },
    limit=top_k,
)
```

Sparse request 查 `sparse_vector`：

```python
sparse_request = AnnSearchRequest(
    data=[sparse_query_vector],
    anns_field="sparse_vector",
    param={
        "metric_type": "IP",
        "params": {"drop_ratio_search": 0.0},
    },
    limit=top_k,
)
```

最後用 Milvus 的 `hybrid_search()` 和 `RRFRanker` 合併：

```python
results = client.hybrid_search(
    collection_name=COLLECTION_NAME,
    reqs=[sparse_request, dense_request],
    ranker=RRFRanker(k=rrf_k),
    limit=top_k,
    output_fields=["chunk_id", "source", "title", "content"],
)
```

這裡有幾個重點：

| 參數 | 說明 |
| --- | --- |
| `reqs` | 放入 sparse 和 dense 兩路檢索請求 |
| `RRFRanker(k=rrf_k)` | 使用 RRF 合併兩路檢索排名 |
| `limit` | 最終回傳多少筆 hybrid results |
| `output_fields` | 回傳 chunk metadata 和原文內容 |

### 7.6 執行方式

執行範例：

```bash
python chapter/04_hybrid_search/01_hybrid_search_milvus.py "ERR-4291"
```

輸出會分成三組：

```text
Sparse / Milvus BGE-M3 sparse vector results
Dense / Milvus BGE-M3 dense vector results
Hybrid / Milvus RRFRanker results
```

可以觀察 sparse、dense、hybrid 三種結果的差異。對錯誤碼、API 名稱這類查詢，sparse 往往會更穩；對概念型問題，dense 通常更容易找出真正回答問題的段落；hybrid 則用 RRF 把兩邊候選結果合併。

## 八、小結

本節介紹了 Hybrid Search 的核心概念：

```text
稀疏向量擅長精準匹配
密集向量擅長語意理解
Hybrid Search 把兩者結合
RRF 可以用排名融合不同檢索結果
Milvus 可以在同一個 collection 中同時儲存 sparse vector 與 dense vector
```

在 RAG 系統中，Hybrid Search 特別適合處理專有名詞多、查詢多變、需要兼顧精準與語意的資料集。
## 參考資料

- [Milvus Hybrid Search 文件](https://milvus.io/docs/multi-vector-search.md)
- [Milvus Sparse Vector 文件](https://milvus.io/docs/sparse_vector.md)
