# 第二節 進階檢索技術

前一節介紹了 Hybrid Search，重點是透過 sparse retrieval 與 dense retrieval 同時提高「精準匹配」和「語意召回」能力。

但在真實 RAG 系統裡，只把資料找回來還不夠。常見問題是：

```text
Top-K 裡有相關資料，但真正重要的 chunk 排在後面
檢索到的 chunk 太長，裡面混了很多無關內容
檢索結果看似相關，但其實不能回答問題
知識庫沒有答案，系統卻硬拿錯誤上下文生成回答
```

因此進階檢索通常會在「初步召回」後增加後處理流程，例如：

```text
retrieval
-> rerank
-> compression
-> correction / fallback
-> LLM generation
```

本節會介紹三類常見進階技術：

| 技術 | 目的 |
| --- | --- |
| Re-ranking | 重新排序檢索結果，讓最相關內容排前面 |
| Context Compression | 壓縮檢索內容，只保留和問題相關的部分 |
| Corrective RAG | 評估檢索結果是否可靠，不可靠時改走修正或外部搜尋 |

## 一、Re-ranking：重新排序檢索結果

向量資料庫回傳的 top-k 結果不一定就是最適合交給 LLM 的順序。

原因是第一階段 retrieval 通常追求速度與召回率，它會先找回一批「可能相關」的候選內容，但不一定能精準判斷哪一段最能回答問題。

常見流程如下：

```text
使用者問題
-> retriever 先取回 top-20 或 top-50
-> reranker 重新評分
-> 選出 top-3 或 top-5
-> 交給 LLM 生成答案
```

可以把 retrieval 和 rerank 分工想成：

| 階段 | 任務 | 特性 |
| --- | --- | --- |
| Retriever | 快速召回候選資料 | 速度快、召回多、排序可能不夠準 |
| Reranker | 精細判斷 query 與 chunk 的相關性 | 較慢、成本較高、排序更準 |

### 1.1 RRF：Reciprocal Rank Fusion

RRF 已經在 Hybrid Search 裡出現過。它是一種不依賴模型的重排方法，常用來融合多個 retriever 的結果。

它不看原始分數，而是看排名：

```text
某份文件在 dense search 排第 1
同一份文件在 sparse search 排第 3
代表它在不同檢索方式下都重要，因此應該提高最終排名
```

公式如下：

$$
RRF_{score}(d) = \sum_{i=1}^{k} \frac{1}{rank_i(d) + c}
$$

| 參數 | 說明 |
| --- | --- |
| $d$ | 某一份候選文件 |
| $k$ | 檢索器數量，例如 dense retriever 與 sparse retriever |
| $rank_i(d)$ | 文件在第 i 個檢索器中的排名 |
| $c$ | 平滑常數，常見值是 60 |

RRF 適合：

```text
Hybrid Search 結果融合
多個 retriever 結果融合
不想處理不同分數尺度時
```

限制是：RRF 只使用排名，不會理解 chunk 內容本身。

### 1.2 LLM-based Reranker

LLM-based reranker 的想法很直覺：

```text
既然最後是 LLM 要根據 context 回答，
那可以先讓 LLM 判斷哪些 context 最相關。
```

典型 prompt 會包含：

```text
使用者問題
候選文件 1
候選文件 2
候選文件 3
...
請按照相關性排序，並輸出分數或文件編號
```

例如：

```text
問題：Milvus 適合什麼情境？

文件 1：Milvus 是向量資料庫，適合大規模相似度搜尋...
文件 2：FAISS 是本地向量索引工具...
文件 3：OCR 可以把圖片文字轉成可檢索文字...

請輸出最相關文件順序。
```

LLM-based reranker 的優點是語意判斷能力強，能理解比較複雜的問題。缺點是：

```text
成本較高
延遲較高
輸出格式需要控制
候選文件太多時會佔用 context window
```

因此它比較適合高價值查詢，或資料量不大但答案品質要求高的場景。

### 1.3 Cross-Encoder Reranker

Cross-Encoder 是常見的 reranker 架構。

和 embedding model 不同，embedding model 通常會把 query 和 document 分別轉成向量，再計算相似度：

```text
query -> vector
document -> vector
similarity(query_vector, document_vector)
```

Cross-Encoder 則是把 query 和 document 放在一起輸入模型：

```text
[CLS] query [SEP] document [SEP]
-> model
-> relevance score
```

也就是模型會同時讀問題和候選文件，再直接輸出相關性分數。

這種方式通常比單純向量相似度更精準，因為它能看到 query 和 document 之間更細的互動關係。

流程如下：

```text
retriever 先召回 top-50
Cross-Encoder 逐一評分 query-document pair
依照 score 重新排序
取 top-5 給 LLM
```

優缺點如下：

| 面向 | 說明 |
| --- | --- |
| 優點 | 排序品質通常很好 |
| 優點 | 能直接判斷 query 和 chunk 是否匹配 |
| 限制 | 每個候選 chunk 都要跑一次模型 |
| 限制 | 候選數越多，延遲越高 |

Cross-Encoder 適合 top-k 精排，不適合拿來對整個資料庫做第一階段搜尋。

### 1.4 ColBERT：Late Interaction

ColBERT 是介於 bi-encoder 和 cross-encoder 之間的做法。

它的核心概念是 **Late Interaction**。

簡化流程如下：

```text
query 的每個 token 各自產生向量
document 的每個 token 各自產生向量
查詢時比較 query token 和 document token 的最大相似度
把每個 query token 的分數加總
得到 document relevance score
```

這種做法比一般 dense embedding 更細緻，因為它不是只比較整段文字的一個向量，而是保留 token-level 的互動。

和 Cross-Encoder 相比，ColBERT 的優勢是文件向量可以預先計算，因此查詢時通常比較有效率。

| 方法 | 互動方式 | 成本 | 適合情境 |
| --- | --- | --- | --- |
| Bi-Encoder | query 和 document 分別編碼，向量相似度比較 | 低 | 第一階段召回 |
| Cross-Encoder | query 和 document 一起輸入模型 | 高 | Top-K 精排 |
| ColBERT | 分別編碼，但保留 token-level late interaction | 中 | 需要更細粒度排序 |

### 1.5 重排方法比較

| 方法 | 核心機制 | 成本 | 適合場景 |
| --- | --- | --- | --- |
| RRF | 融合多個結果列表的排名 | 低 | Hybrid Search、多路召回融合 |
| LLM-based Reranker | 讓 LLM 判斷候選文件相關性 | 中到高 | 高價值查詢、複雜語意判斷 |
| Cross-Encoder | query-document pair 聯合編碼 | 高 | Top-K 精排 |
| ColBERT | token-level late interaction | 中 | 需要兼顧精度與效率的重排 |

實務上常見做法是：

```text
先用便宜快速的 retriever 召回較多候選
再用較貴但準確的 reranker 精排少量候選
```

## 二、Context Compression：上下文壓縮

Rerank 解決的是「排序」問題，但還有另一個常見問題：

```text
檢索到的 chunk 整體相關，
但 chunk 裡只有幾句真正能回答問題，
其他內容會佔用 context window。
```

這時可以使用 **Context Compression**。

上下文壓縮的目的不是壓縮檔案大小，而是壓縮「要交給 LLM 的上下文」。

它通常有兩種形式：

| 類型 | 說明 |
| --- | --- |
| 內容提取 | 從 chunk 中抽出與 query 相關的句子或段落 |
| 文件過濾 | 把雖然被召回、但其實不相關的 chunk 丟掉 |

流程如下：

```text
base retriever 取回 top-20
compressor 檢查每個 chunk
保留相關句子或過濾不相關 chunk
回傳壓縮後 context
```

### 2.1 LangChain 的 ContextualCompressionRetriever

LangChain 提供 `ContextualCompressionRetriever`，它本質上是一個包裝器：

```text
base_retriever 負責初步檢索
base_compressor 負責壓縮或過濾結果
ContextualCompressionRetriever 把兩者串起來
```

概念程式如下：

```python
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)

results = compression_retriever.invoke(query)
```

常見 compressor 包含：

| Compressor | 作用 |
| --- | --- |
| `LLMChainExtractor` | 用 LLM 從文件中抽出與 query 相關的內容 |
| `LLMChainFilter` | 用 LLM 判斷整份文件是否相關，不相關就丟掉 |
| `EmbeddingsFilter` | 用 embedding similarity 過濾低相關文件 |

三者差異可以這樣理解：

```text
LLMChainExtractor：保留文件中的相關句子
LLMChainFilter：保留或丟棄整份文件
EmbeddingsFilter：用相似度門檻快速過濾
```

### 2.2 DocumentCompressorPipeline

如果需要多個後處理步驟，可以使用 pipeline 概念。

例如：

```text
先 rerank
再 compression
最後回傳壓縮後結果
```

概念程式如下：

```python
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        reranker,
        compressor,
    ]
)

final_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=pipeline_compressor,
)
```

這樣呼叫 `final_retriever` 時，流程會變成：

```text
base retriever
-> reranker
-> compressor
-> final documents
```

這個架構的重點不是某個特定模型，而是把 retrieval 後處理拆成可組合的元件。

### 2.3 LlamaIndex 的檢索後處理

LlamaIndex 也有類似概念，通常稱為 **Node Postprocessor**。

例如前面句子窗口檢索提過的：

```text
先用小句子做 retrieval
再把 node 替換成較大的 window context
```

進階一點也可以做內容壓縮，例如只保留與 query embedding 最接近的句子。

這類方法的核心精神是：

```text
retrieval 找候選內容
postprocessor 決定哪些內容值得交給 LLM
```

## 三、Corrective RAG：校正檢索結果

傳統 RAG 有一個隱含假設：

```text
只要 retriever 找到資料，這些資料就足以回答問題。
```

但真實情況不一定如此。Retriever 可能會：

```text
找回語意相近但不相關的文件
找回過時文件
找回片段資訊但缺少關鍵答案
知識庫根本沒有答案
```

如果直接把這些 context 交給 LLM，模型可能會產生看似合理但其實錯誤的回答。

**Corrective RAG（C-RAG）** 的概念是：在生成答案前，先評估檢索結果品質，再決定下一步。

### 3.1 C-RAG 的基本流程

C-RAG 可以簡化成三個階段：

```text
Retrieve
-> Assess
-> Act
```

| 階段 | 說明 |
| --- | --- |
| Retrieve | 先從知識庫檢索候選文件 |
| Assess | 評估文件是否能回答問題 |
| Act | 根據評估結果決定生成、修正、重寫查詢或外部搜尋 |

評估結果常見可以分成：

| 評估結果 | 後續動作 |
| --- | --- |
| Correct | 文件足以回答問題，進入答案生成 |
| Incorrect | 文件不相關或錯誤，改用查詢重寫或外部搜尋 |
| Ambiguous | 文件可能有用但不完整，補充搜尋更多資料 |

### 3.2 為什麼需要 Corrective RAG

C-RAG 解決的是「不要盲目信任 retrieval」。

例如使用者問：

```text
Milvus hybrid search 如何同時使用 dense vector 和 sparse vector？
```

如果 retriever 找回的是：

```text
Milvus 的基本安裝方式
FAISS 本地向量索引
OCR 文件載入流程
```

這些內容都和 RAG 或向量資料庫有關，但不能回答 hybrid search 的問題。

此時系統應該先判斷「檢索結果不足」，而不是硬把這些內容塞給 LLM 生成答案。

### 3.3 Corrective RAG 的實作方式

概念上可以這樣設計：

```text
query
-> retrieve documents
-> evaluator 判斷每份文件相關性
-> 如果足夠：生成答案
-> 如果不足：rewrite query
-> 重新檢索或外部搜尋
-> 再生成答案
```

如果用 LangGraph 這類 graph workflow 工具，可以把每個步驟做成節點：

```text
retrieve_node
grade_documents_node
rewrite_query_node
web_search_node
generate_answer_node
```

再根據評估結果走不同分支：

```text
documents are relevant -> generate_answer
documents are not relevant -> rewrite_query -> retrieve again
knowledge base lacks answer -> web_search
```

C-RAG 的重點不是一定要做 Web search，而是要讓 RAG pipeline 有能力判斷：

```text
我找回來的資料是否真的能回答問題？
如果不能，下一步應該怎麼補救？
```

## 四、如何選擇進階檢索技術

不同技術解決的問題不同，不需要一次全部加上。

| 問題 | 優先考慮 |
| --- | --- |
| 找得到資料，但排序不準 | Reranker |
| Top-K 太長、雜訊太多 | Context Compression |
| 檢索結果經常不相關 | Corrective RAG |
| sparse / dense 結果需要融合 | RRF |
| 答案品質要求高但查詢量不大 | LLM-based reranker |
| 技術文件或專有名詞很多 | Hybrid Search + Rerank |
| 成本和延遲很敏感 | 先用 embedding filter 或 RRF，少用 LLM rerank |

一個較完整的進階 RAG pipeline 可能是：

```text
query
-> query rewrite
-> hybrid retrieval
-> RRF fusion
-> cross-encoder rerank
-> context compression
-> document grading
-> answer generation
```

但教學和 prototype 不需要一開始就做到這麼複雜。比較實際的順序是：

```text
先把 chunking、metadata、embedding 做好
再加入 hybrid search
再加入 rerank
最後才加入 compression / C-RAG
```

## 五、實務注意事項

進階檢索可以提升品質，但也會增加成本與複雜度。

設計時要注意：

| 面向 | 注意事項 |
| --- | --- |
| Latency | Cross-Encoder、LLM rerank、compression 都會增加延遲 |
| Cost | LLM-based 方法會增加 API 或 GPU 成本 |
| Observability | 要記錄每一步的輸入、輸出、分數與被丟棄的文件 |
| Evaluation | 不能只看單次回答，要用問題集評估 Recall@K、MRR、Answer Quality |
| Failure Handling | 當檢索不足時，要允許回答「資料不足」，不要硬答 |

尤其是 C-RAG 或 query rewrite 類流程，一定要保留追蹤資訊：

```text
原始 query 是什麼？
重寫後 query 是什麼？
哪些文件被判定相關？
哪些文件被過濾？
最後答案用了哪些來源？
```

這些資訊會直接影響 RAG 系統能不能 debug。

## 六、本節重點整理

本節介紹了三類進階檢索技術：

```text
Re-ranking：重新排序候選文件
Context Compression：壓縮或過濾檢索內容
Corrective RAG：評估檢索品質並決定是否修正
```

它們的共同目標是讓 LLM 拿到更可靠、更精準、更少雜訊的 context。

可以用一句話總結：

```text
retrieval 負責找候選資料，
rerank 負責排序，
compression 負責去雜訊，
correction 負責判斷資料是否可信。
```

## 參考資料

- [Datawhale all-in-rag：Advanced Retrieval Techniques](https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter4/15_advanced_retrieval_techniques.md)
- [LangChain：Contextual Compression](https://python.langchain.com/docs/how_to/contextual_compression/)
- [LlamaIndex：Node Postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)
- [Corrective Retrieval Augmented Generation, Jiang et al. 2024](https://arxiv.org/abs/2401.15884)
