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

### 4.1 RRF：Reciprocal Rank Fusion

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

Hybrid Search 很適合用在：

```text
企業知識庫
技術文件搜尋
客服 FAQ
法規 / 條文查詢
產品型錄搜尋
醫療、金融、製造等專有名詞很多的資料
```

## 六、Milvus 中的 Hybrid Search 概念

Milvus 支援在同一個 collection 中放入不同向量欄位，例如：

| 欄位 | 類型 | 用途 |
| --- | --- | --- |
| `dense_vector` | `FLOAT_VECTOR` | 儲存 dense embedding |
| `sparse_vector` | `SPARSE_FLOAT_VECTOR` | 儲存 sparse embedding |
| `title` / `content` | `VARCHAR` | 原始文字或摘要 |
| `source` / `category` | scalar fields | metadata filter |

概念上，一筆資料會長這樣：

```text
id: chunk_001
content: "Milvus 支援 dense vector、sparse vector 與 hybrid search..."
source: "04_Milvus.md"
dense_vector: [0.01, -0.03, ...]
sparse_vector: {102: 0.8, 981: 1.4, ...}
```

查詢時，也會同時產生兩種 query vector：

```text
query dense vector
query sparse vector
```

然後建立兩個 search request：

```python
dense_req = AnnSearchRequest(
    data=[dense_vec],
    anns_field="dense_vector",
    param=search_params,
    limit=top_k,
)

sparse_req = AnnSearchRequest(
    data=[sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
)
```

最後用 `hybrid_search()` 搭配 ranker 合併結果：

```python
rerank = RRFRanker(k=60)

results = collection.hybrid_search(
    [sparse_req, dense_req],
    rerank=rerank,
    limit=top_k,
    output_fields=["title", "content", "source"],
)
```

這裡幾個重要參數要理解：

| 參數 | 說明 |
| --- | --- |
| `anns_field` | 指定要查哪個向量欄位，例如 `dense_vector` 或 `sparse_vector` |
| `limit` | 每一路檢索取回多少候選結果 |
| `output_fields` | 查詢結果要回傳哪些 metadata 或文字欄位 |
| `RRFRanker(k=60)` | 使用 RRF 融合排名，`k` 越大排序越平滑 |
| `hybrid_search()` | 同時接收多個 `AnnSearchRequest`，再用 ranker 合併 |

## 七、BGE-M3 與 Hybrid Search

Hybrid Search 通常需要兩種模型或兩套方法：

```text
BM25 / sparse encoder
dense embedding model
```

但 `BAAI/bge-m3` 的特別之處是，它可以支援多種檢索表示，常見包含：

| 表示 | 用途 |
| --- | --- |
| dense embedding | 語意檢索 |
| sparse embedding | 關鍵字 / 詞彙匹配 |
| multi-vector | 更細粒度的 token-level matching |

因此在 Milvus 的 hybrid search 範例中，常會用 BGE-M3 同時產生 sparse 與 dense vector。

概念上會像這樣：

```python
embeddings = embedding_function(docs)

sparse_vectors = embeddings["sparse"]
dense_vectors = embeddings["dense"]
```

查詢時也是同樣流程：

```python
query_embeddings = embedding_function([query])

sparse_vec = query_embeddings["sparse"]
dense_vec = query_embeddings["dense"]
```

對 RAG 來說，這樣的好處是系統架構比較一致，不需要分別維護 BM25 tokenizer 和 dense embedding model。

## 八、什麼時候該使用 Hybrid Search

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

## 九、實作時的設計建議

在 RAG 系統中使用 Hybrid Search 時，可以從以下幾點開始調整：

| 設計點 | 建議 |
| --- | --- |
| Chunk 設計 | chunk 不要太大，否則 sparse 和 dense 都容易被雜訊影響 |
| Metadata | 保留 `source`、`chapter`、`page`、`category` 等欄位 |
| Top-K | sparse 和 dense 可以各取較大的候選數，再融合 |
| Fusion | 初期可先用 RRF，因為不需要手動正規化分數 |
| Filter | 先用 metadata filter 縮小範圍，再做 hybrid search |
| Rerank | 如果結果仍不穩，可在 hybrid search 後加 reranker |

一個比較完整的 RAG retrieval pipeline 可能會是：

```text
使用者問題
-> metadata filter
-> sparse retrieval
-> dense retrieval
-> RRF fusion
-> reranker
-> context packing
-> LLM generation
```

Hybrid Search 通常不是最後一步，而是 retrieval pipeline 中提升召回率的重要一層。

## 十、小結

本節介紹了 Hybrid Search 的核心概念：

```text
稀疏向量擅長精準匹配
密集向量擅長語意理解
Hybrid Search 把兩者結合
RRF 可以用排名融合不同檢索結果
Milvus 可以在同一個 collection 中同時儲存 sparse vector 與 dense vector
```

在 RAG 系統中，Hybrid Search 特別適合處理專有名詞多、查詢多變、需要兼顧精準與語意的資料集。

下一步可以實作一個簡單範例：

```text
建立 Milvus collection
為每個 chunk 產生 sparse vector 與 dense vector
分別查詢 sparse / dense 結果
用 RRF 合併排序
比較純 dense、純 sparse、hybrid search 的差異
```

## 參考資料

- [Datawhale all-in-rag：Hybrid Search](https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter4/11_hybrid_search.md)
- [Milvus Hybrid Search 文件](https://milvus.io/docs/multi-vector-search.md)
- [Milvus Sparse Vector 文件](https://milvus.io/docs/sparse_vector.md)
