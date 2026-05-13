# 第四節 Milvus

前一節介紹了向量資料庫的基本概念，也用 FAISS 示範如何在本地建立向量索引。

FAISS 很適合教學、本地實驗與 prototype；但如果資料量變大、需要多人共用、需要服務化部署、需要 metadata filter、需要長期維運，就會需要更完整的向量資料庫。

**Milvus** 就是常見的選擇之一。  

官網: https://milvus.io/  
GitHub: https://github.com/milvus-io/milvus

Milvus 是一個開源向量資料庫，主要用來處理大規模向量相似度搜尋。它適合用在：

```text
RAG 知識庫
語意搜尋
圖片搜尋
推薦系統
多模態檢索
大規模 embedding 管理
```

## 一、Milvus 是什麼

Milvus 是專門為向量資料設計的資料庫。和單純的本地向量索引工具不同，Milvus 更偏向正式服務。

它可以處理：

| 能力 | 說明 |
| --- | --- |
| 向量儲存 | 儲存文字、圖片、音訊、影片等資料產生的 embeddings |
| 相似度搜尋 | 根據 query vector 找出最接近的資料 |
| Metadata filter | 使用 scalar fields 過濾資料，例如章節、來源、日期、類型 |
| Index 管理 | 支援多種向量索引，提升查詢速度 |
| Collection 管理 | 用 collection 組織不同資料集 |
| 大規模部署 | 可從本地 standalone 擴展到分散式部署 |

在 RAG 中，Milvus 通常會位於這個位置：

```text
文件 / 圖片 / PDF
-> chunking / OCR / caption
-> embedding model
-> Milvus
-> retrieval
-> LLM
```

## 二、安裝與啟動 Milvus

Milvus 有多種部署方式，例如：

| 部署方式 | 適合情境 |
| --- | --- |
| Milvus Lite | 本地快速測試、Notebook、demo |
| Milvus Standalone | 單機服務、教學、開發環境 |
| Milvus Distributed | 大規模正式部署 |
| Zilliz Cloud | 不想自己維運 Milvus 的雲端託管版本 |

本節以 **Milvus Standalone** 作為概念說明。Standalone 版本通常會搭配 Docker 啟動，並對外提供 `19530` port 給 Python SDK 連線。

### 2.1 使用 Docker 啟動

官方文件提供 Docker / Docker Compose 等方式啟動 Milvus。概念上流程是：

```text
安裝 Docker
下載 Milvus 啟動設定
啟動 Milvus container
確認 19530 port 可連線
```

官方 Docker 安裝文件中，Milvus Standalone 啟動後會提供：

```text
Milvus service: localhost:19530
Milvus WebUI: http://127.0.0.1:9091/webui/
```

實際指令建議以官方文件為準，因為 Milvus 版本與部署腳本會更新。

## 三、Milvus 的核心概念

Milvus 的資料組織方式和傳統資料庫有些相似，但它的核心是向量。

可以先掌握幾個名詞：

| 名詞 | 說明 |
| --- | --- |
| Collection | 類似資料表，是向量與 metadata 的主要容器 |
| Schema | 定義 collection 有哪些欄位 |
| Field | 欄位，例如 id、text、embedding、source |
| Entity | 一筆資料，類似 table 裡的一 row |
| Partition | Collection 裡的邏輯分區 |
| Index | 加速向量搜尋的資料結構 |
| Metric Type | 相似度或距離計算方式，例如 COSINE、L2、IP |

## 四、Collection 與 Schema

在 Milvus 中，建立資料前要先建立 Collection。

Collection 需要 Schema。Schema 會定義每筆資料要有哪些欄位。

以 RAG 文字資料為例，一個 collection 可以設計成：

| 欄位 | 型態 | 用途 |
| --- | --- | --- |
| `id` | Primary Key | 唯一識別每個 chunk |
| `text` | VarChar | chunk 原文 |
| `embedding` | FloatVector | chunk 的 embedding vector |
| `source` | VarChar | 原始檔案來源 |
| `page` | Int64 | PDF 頁碼 |
| `chapter` | VarChar | 章節 |
| `modality` | VarChar | text、image、pdf_page 等資料型態 |

這和前面提到的 metadata 是同一個概念，只是在 Milvus 中 metadata 會以 scalar fields 的形式存在。

例如：

```text
text        -> 原始 chunk
embedding   -> 向量欄位
source      -> metadata
page        -> metadata
chapter     -> metadata
modality    -> metadata
```

設計 schema 時要注意：

1. 向量欄位的 dimension 必須和 embedding model 輸出的維度一致。
2. 常用來過濾的 metadata 應該設成 scalar fields。
3. 欄位不要亂加，schema 太複雜會增加管理成本。
4. 多模態資料可以用 `modality` 區分資料型態。

## 五、Index：讓搜尋變快

Milvus 支援多種向量索引。建立 index 的目的，是讓相似度搜尋更快。

常見 index 包括：

| Index | 說明 | 適合情境 |
| --- | --- | --- |
| FLAT | 暴力搜尋，結果最精確但速度慢 | 小資料、追求準確 |
| IVF_FLAT | 先分群再搜尋部分群集 | 通用大資料檢索 |
| IVF_SQ8 / IVF_PQ | 對向量做壓縮，降低記憶體成本 | 大資料、成本敏感 |
| HNSW | 圖結構索引，查詢快、召回高 | 低延遲搜尋 |
| DiskANN | 偏向磁碟的大規模索引 | 資料量超過記憶體 |

選 index 時要在幾件事之間取捨：

```text
查詢速度
召回率
記憶體使用量
建立索引時間
資料規模
```

如果只是學習或小資料測試，可以先用 `FLAT` 或 `IVF_FLAT` 理解流程。正式系統再依資料量與延遲需求測試不同 index。

## 六、Metric Type：如何計算相似度

Milvus 常見的 metric type 有：

| Metric | 說明 |
| --- | --- |
| `COSINE` | 比較向量方向，常用於文字 embedding |
| `IP` | Inner Product，常搭配 normalized embeddings |
| `L2` | 歐氏距離，距離越小越相似 |

選哪一種 metric，要和 embedding model 的建議一致。

例如本課程常用：

```python
encode_kwargs={"normalize_embeddings": True}
```

當向量已經 normalize 時，`COSINE` 或 `IP` 都常見；但實際選擇仍應以模型文件與向量資料庫設定為準。

## 七、多模態 Milvus 範例：圖片搜尋圖片

本章的 `04_multi_milvus.py` 示範的是 **多模態圖片檢索**。它不是一般文字 RAG，而是把圖片轉成向量後存進 Milvus，再用「查詢圖片 + 查詢文字」找回最相似的圖片。

這裡使用 **車子圖片檢索** 作為範例，因為車子類型明確、圖片容易準備，也很適合用文字條件描述。

相關素材已放入這個資料夾：

```text
data/C3/images/04_cars/
```

建議放入：

```text
query.jpg
car_01.jpg
car_02.jpg
car_03.jpg
car_04.jpg
```

圖片內容可以包含：

1. 紅色跑車。
2. 藍色轎車。
3. 黃色計程車。
4. 白色 SUV。
5. 黑色卡車。

`query.jpg` 建議放一張和查詢文字接近的圖片，例如紅色車。查詢文字可以設成：

```text
一台紅色車
```

這樣範例的目標就會很清楚：

```text
用 query.jpg + 「一台紅色車」
從車子圖片資料庫中找出最相似的圖片
```

## 八、程式流程

多模態 Milvus 範例可以拆成八個步驟：

```text
1. 載入 Gemini embedding client
2. 連線 Milvus
3. 讀取 data/C3/images/04_cars/*.jpg
4. 建立 Milvus collection
5. 把每張圖片轉成 image embedding 後插入 Milvus
6. 為 vector 欄位建立 HNSW index
7. 用 query image + query text 產生 query vector
8. 搜尋 Top-K 相似圖片並輸出結果
```

這個流程對應到 RAG / 多模態檢索中的幾個核心角色：

| 元件 | 在範例中的角色 |
| --- | --- |
| `gemini-embedding-2` | 把圖片、圖片加文字轉成向量 |
| Milvus | 儲存圖片向量，並做相似度搜尋 |
| Collection | 存放圖片資料集 |
| Schema | 定義 `id`、`vector`、`image_path` 欄位 |
| HNSW index | 加速圖片向量搜尋 |
| query image + query text | 多模態查詢條件 |

## 九、Schema 設計

這個範例的 schema 很簡單：

| 欄位 | 型態 | 說明 |
| --- | --- | --- |
| `id` | `INT64` | 主鍵，使用 `auto_id=True` 自動生成 |
| `vector` | `FLOAT_VECTOR` | 圖片 embedding |
| `image_path` | `VARCHAR` | 原始圖片路徑 |

概念上等於：

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]
```

這裡的 `dim` 不是手動亂填，而是先用模型對第一張圖片做 embedding，再取得向量長度：

```python
dim = len(encoder.encode_image(image_list[0]))
```

本課程的程式會把 `gemini-embedding-2` 的輸出維度設定為 `768`：

```python
config=types.EmbedContentConfig(
    output_dimensionality=768,
)
```

這點很重要。Milvus 的 vector field 維度必須和模型輸出的 embedding 維度一致，否則插入資料會失敗。

## 十、建立索引與搜尋

範例使用 HNSW index：

```python
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256},
)
```

這裡的重點是：

| 參數 | 說明 |
| --- | --- |
| `field_name="vector"` | 對圖片向量欄位建立索引 |
| `index_type="HNSW"` | 使用圖結構索引，加速近似最近鄰搜尋 |
| `metric_type="COSINE"` | 用 cosine similarity 比較向量方向 |
| `M` | 圖中每個節點連接的鄰居數，越大通常召回率越好但記憶體更高 |
| `efConstruction` | 建 index 時的搜尋範圍，越大通常 index 品質越好但建立更慢 |

查詢時，範例不是只用圖片，也不是只用文字，而是把兩者一起送進 `gemini-embedding-2`：

```python
query_vector = encoder.encode_query(
    image_path=query_image_path,
    text=query_text,
)
```

接著用這個 query vector 搜尋 Milvus：

```python
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["image_path"],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
)[0]
```

`output_fields=["image_path"]` 代表搜尋結果會回傳圖片路徑。這樣程式才能把找回來的圖片讀出來並視覺化。

## 十一、結果視覺化

範例最後會把查詢圖片和搜尋結果拼成一張圖。

概念上是：

```text
左側：Query 圖片
右側：Milvus 找回的 Top-K 圖片
```

這對多模態檢索很有幫助，因為你可以直接用肉眼檢查：

1. 找回來的圖片是否和 query image 相似。
2. 查詢文字是否影響結果。
3. Top-1 到 Top-5 的排序是否合理。

如果改用車子資料集，輸出圖可以命名為：

```text
data/C3/images/04_cars/search_result.png
```

## 十二、Metadata Filter

原始範例只存 `image_path`，所以它主要示範「圖片向量搜尋」。

如果要更接近正式多模態 RAG，可以增加 metadata 欄位，例如：

| 欄位 | 說明 |
| --- | --- |
| `category` | 車子類型，例如 sports_car、taxi、truck |
| `color` | 顏色，例如 red、blue、yellow |
| `source` | 圖片來源 |
| `modality` | 固定為 image |

這樣就可以做 filter：

```python
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    filter='color == "red"',
    output_fields=["image_path", "category", "color"],
    limit=5,
)
```

不過本節第一版先保留簡單 schema，讓重點集中在「圖片 embedding -> Milvus -> 圖文 query -> 相似圖片」。

## 十三、這個範例和 RAG 的關係

這個程式沒有直接呼叫 LLM，但它示範了多模態 RAG 的 retrieval 部分。

如果要接成完整多模態 RAG，可以把 Milvus 搜尋結果交給後續流程：

```text
query image + query text
-> gemini-embedding-2 query embedding
-> Milvus 找回相似圖片
-> 讀取圖片 metadata / caption / OCR
-> 交給 multimodal LLM 或文字 LLM 回答
```

因此這個範例的重點不是生成答案，而是學會如何用 Milvus 管理多模態向量資料。

## 十四、Milvus、FAISS、Chroma 怎麼選

| 情境 | 建議 |
| --- | --- |
| 只想理解向量搜尋流程 | FAISS |
| 本地小型 prototype | Chroma 或 FAISS |
| 教學展示持久化 index | FAISS / Chroma |
| 正式 RAG 服務 | Milvus / Qdrant / Weaviate / Pinecone |
| 資料量很大，需要服務化部署 | Milvus |
| metadata filter 很重要 | Milvus / Qdrant / Weaviate |
| 不想自己維運 | Zilliz Cloud / Pinecone |

本課程的順序是：

```text
InMemoryVectorStore -> FAISS -> LlamaIndex VectorStoreIndex -> Milvus
```

目標是先理解基本流程，再進入更接近正式系統的向量資料庫。

## 十五、常見問題

### 15.1 Milvus 會存 metadata 嗎

會。

Milvus 的 scalar fields 就是用來存 metadata 的地方，例如 `source`、`page`、`chapter`、`modality`。搜尋時可以透過 `output_fields` 取回，也可以透過 filter 條件限制搜尋範圍。

### 15.2 Collection 和 Partition 有什麼差別

Collection 是主要資料容器，類似資料表。

Partition 是 Collection 裡的邏輯分區，可以把同一個 collection 內的資料再分組。例如依照章節、資料類型、租戶或日期分區。

資料量不大時不一定需要 partition；等資料量變大、查詢範圍明確時再考慮。

### 15.3 Index 一定要建立嗎

資料量小時可以不急著建立複雜 index。

但在正式 RAG 中，資料量通常會增加。建立合適 index 可以大幅提升搜尋速度。

### 15.4 Milvus 可以取代傳統資料庫嗎

不建議。

Milvus 擅長向量搜尋，不是用來取代關聯式資料庫的交易、報表或複雜 Join。正式系統常見做法是：

```text
PostgreSQL / MySQL：管理結構化業務資料
Milvus：管理 embeddings 與向量搜尋
Object Storage：保存 PDF、圖片、音訊、影片原始檔
```

## 十六、本節重點整理

1. Milvus 是開源向量資料庫，適合大規模 RAG 與正式服務。
2. Collection 是 Milvus 管理向量資料的主要容器。
3. Schema 定義每筆資料有哪些欄位，包括 vector field 和 scalar fields。
4. Scalar fields 可以存 metadata，也可以用來做 filter。
5. Index 可以加速向量搜尋，但要在速度、召回率、記憶體與建立時間之間取捨。
6. 多模態圖片檢索可以把 image embeddings 存進 Milvus，再用 query image + query text 搜尋相似圖片。
7. `output_fields` 決定搜尋結果要回傳哪些原始內容與 metadata，例如 `image_path`。
8. 小型教學可以先用 FAISS；正式服務或大規模資料再考慮 Milvus。

## 十七、練習

1. 用 Docker 啟動 Milvus Standalone。
2. 使用 `pymilvus` 連線到 `localhost:19530`。
3. 準備 `data/C3/images/04_cars/` 圖片資料集。
4. 建立一個 `multimodal_car_demo` collection。
5. 設計 `id`、`vector`、`image_path` 欄位。
6. 將每張車子圖片轉成 embedding 後插入 Milvus。
7. 使用 `query.jpg` 和「一台紅色車」查詢 Top-5 相似圖片。
8. 將搜尋結果輸出成 `search_result.png`，觀察排序是否合理。

## 參考資料

- [Milvus Official Website](https://milvus.io/)
- [Milvus Docker Installation](https://milvus.io/docs/install_standalone-docker.md)
- [Milvus Quickstart](https://milvus.io/docs/quickstart.md)
- [Milvus GitHub](https://github.com/milvus-io/milvus)
- [pymilvus Documentation](https://milvus.io/api-reference/pymilvus/v2.6.x/About.md)
