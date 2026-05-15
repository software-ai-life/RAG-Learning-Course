# 第一節 RAG 系統評估

建立 RAG 系統後，下一個問題不是「它能不能回答」，而是：

```text
它回答得準不準？
它有沒有找對資料？
它的答案有沒有根據 context？
如果出錯，是 retrieval 出錯，還是 generation 出錯？
不同 RAG pipeline 之間要怎麼比較？
```

RAG 評估的目的，就是把「感覺好像可以」變成可以觀察、比較、迭代的 evaluation loop。

本節會從 **RAG 評估三元組（RAG Triad）** 開始，接著介紹 retrieval evaluation、response evaluation、Ragas，以及上線後的 monitoring / observability 工具選型。

## 一、RAG 評估三元組

RAG 系統的輸入與輸出可以拆成三個核心元素：

![RAG triad](./images/RAG%20triad.png)

source: trulen

```text
Query
Context
Answer
```

因此評估也可以拆成三個面向：

| 面向 | 評估對象 | 核心問題 |
| --- | --- | --- |
| Context Relevance | Retriever | 找回來的 context 是否和 query 相關？ |
| Faithfulness / Groundedness | Generator | 答案是否忠於 retrieved context？ |
| Answer Relevance | End-to-end system | 最終答案是否直接回答使用者問題？ |

這三個面向可以幫助我們定位問題。

例如：

```text
Context Relevance 低
-> retrieval 可能出錯，沒有找對資料

Faithfulness 低
-> LLM 可能沒有根據 context，產生 hallucination

Answer Relevance 低
-> 答案可能沒有切題，或只回答了一部分問題
```

### 1.1 Context Relevance

Context Relevance 評估的是 retrieval 階段。

它問的是：

```text
retriever 找回來的 chunks，是否真的和使用者問題相關？
```

這是 RAG 的第一關。
如果 retrieval 找回來的是錯的 context，後面的 LLM 再強也很難回答正確。

常見問題：

```text
找回語意相近但不能回答問題的 chunk
找回太多不相關內容
正確 chunk 排名太後面
metadata filter 沒有生效
```

### 1.2 Faithfulness / Groundedness

Faithfulness 評估的是答案是否忠於 context。

它問的是：

```text
答案中的每個主張，是否都能在 retrieved context 中找到依據？
```

這個指標主要用來衡量 hallucination。

一個答案可能看起來很流暢、也很像正確答案，但如果 context 裡沒有支持它的內容，對 RAG 來說就是不可靠。

### 1.3 Answer Relevance

Answer Relevance 評估的是最終答案是否真正回答了使用者問題。

它問的是：

```text
答案是否切題？
答案是否完整？
答案是否避免無關資訊？
```

Faithfulness 和 Answer Relevance 不完全一樣。

例如使用者問：

```text
What is an AI agent, and how does it use tools?
```

如果回答只說：

```text
An AI agent is a system that can reason and act.
```

這個答案可能是 faithful，因為它有根據 context；但它沒有回答工具使用，因此 answer relevance 不夠高。

## 二、評估工作流

RAG 評估可以拆成兩大類：

```text
retrieval evaluation
response evaluation
```

Retrieval evaluation 檢查「資料有沒有找對」。
Response evaluation 檢查「答案有沒有答好」。

建議流程：

```text
evaluation questions
-> run RAG pipeline
-> save retrieved contexts
-> save generated answers
-> calculate retrieval metrics
-> calculate response metrics
-> inspect failed cases
-> update chunking / embedding / retrieval / prompt
```

如果只看最終答案，會很難 debug。
所以評估資料應該至少保存：

```text
question
retrieved_contexts
answer
reference_answer
expected_sources
latency
token_usage
```

## 三、檢索評估 Retrieval Evaluation

Retrieval evaluation 主要評估 Context Relevance。

這類評估通常需要一組標註資料：

```json
{
  "question": "How does MCP help tool interoperability?",
  "relevant_sources": [
    "Agent Tools & Interoperability with Model Context Protocol (MCP).pdf"
  ],
  "relevant_chunk_ids": [
    "raw_000123",
    "raw_000124"
  ]
}
```

如果沒有標註 chunk，也可以先從文件層級開始標註：

```text
這題應該由哪份 PDF 回答？
```

### 3.1 Precision@K

Precision@K 衡量 top-k 結果中有多少是相關的。

$$
Precision@K = \frac{\text{top-k 中相關文件數}}{K}
$$

例如 top-5 中有 3 個相關 chunks：

```text
Precision@5 = 3 / 5 = 0.6
```

Precision 高代表 retrieval 噪音少。

### 3.2 Recall@K

Recall@K 衡量所有應該被找回的相關文件中，有多少出現在 top-k。

$$
Recall@K = \frac{\text{top-k 中找回的相關文件數}}{\text{所有相關文件數}}
$$

例如一題有 4 個正確 chunks，top-5 找回 3 個：

```text
Recall@5 = 3 / 4 = 0.75
```

Recall 高代表系統比較不容易漏掉關鍵資訊。

### 3.3 F1-score

F1-score 是 precision 和 recall 的調和平均。

$$
F_1 = 2 \cdot \frac{Precision \times Recall}{Precision + Recall}
$$

當你同時在意「找得準」和「找得全」時，可以看 F1。

### 3.4 MRR

MRR（Mean Reciprocal Rank）衡量第一個正確結果排得多前面。

$$
MRR = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{rank_q}
$$

其中：

| 符號 | 說明 |
| --- | --- |
| `|Q|` | 查詢總數 |
| `rank_q` | 第 q 個查詢中第一個相關結果的排名 |

如果正確結果常常排在第一名，MRR 會高。

### 3.5 MAP

MAP（Mean Average Precision）同時考慮多個相關結果的排名。

$$
MAP = \frac{1}{|Q|} \sum_{q=1}^{|Q|} AP(q)
$$

它適合用來評估「一題可能有多個正確 chunks」的 retrieval 任務。

## 四、回應評估 Response Evaluation

Response evaluation 主要評估：

```text
Faithfulness / Groundedness
Answer Relevance
```

也就是：

```text
答案是否根據 context？
答案是否回答了問題？
```

### 4.1 Faithfulness

Faithfulness 的評估方式通常是：

```text
1. 把答案拆成多個 claims
2. 檢查每個 claim 是否能被 context 支持
3. 計算被支持的 claims 比例
```

例如答案：

```text
MCP helps agents connect to tools through a standard protocol.
It also guarantees that every tool call is always safe.
```

如果 context 只支持第一句，不支持第二句，那第二句就是 unfaithful claim。

### 4.2 Answer Relevance

Answer Relevance 檢查答案是否對準使用者問題。

常見扣分情況：

```text
答非所問
只回答部分問題
加入太多無關背景
沒有針對 query 中的限制條件回答
```

### 4.3 Factual Correctness

Factual Correctness 通常需要 reference answer 或人工標準。

它問的是：

```text
答案中的事實是否正確？
```

這和 faithfulness 有關，但不完全相同。

Faithfulness 看的是「是否被 context 支持」。
Factual correctness 看的是「是否符合事實或標準答案」。

## 五、常見回應評估方法

回應評估通常有兩類方法：

```text
LLM-as-a-judge
classic lexical metrics
```

### 5.1 LLM-as-a-judge

LLM-as-a-judge 是目前常見的 RAG 評估方式。

它使用另一個 LLM 當評審，根據 question、context、answer、reference answer 進行評分。

適合評估：

```text
faithfulness
answer relevance
factual correctness
helpfulness
completeness
```

優點：

```text
能理解語意
能處理 paraphrase
能評估複雜回答
```

限制：

```text
成本較高
速度較慢
評審模型也可能有偏差
不同 evaluator prompt 會影響結果
```

因此 LLM-as-a-judge 最好搭配人工抽查，而不是完全盲信分數。

### 5.2 ROUGE

ROUGE 常用於摘要評估，偏向 recall。

它看的是：

```text
reference answer 中的內容，有多少被 generated answer 覆蓋？
```

ROUGE 適合檢查內容完整性，但不理解深層語意。

### 5.3 BLEU

BLEU 常用於機器翻譯，偏向 precision。

它看的是：

```text
generated answer 中有多少片段出現在 reference answer 中？
```

BLEU 也會考慮長度懲罰，避免過短答案拿到過高分。

但在 RAG 問答中，BLEU 不一定可靠，因為正確答案可能有很多不同表達方式。

### 5.4 METEOR

METEOR 同時考慮 precision 和 recall，也會嘗試處理詞形與同義詞。

相較 BLEU，METEOR 通常更接近人類判斷，但仍然屬於傳統文字相似度指標。

### 5.5 方法比較

| 方法 | 優點 | 限制 |
| --- | --- | --- |
| LLM-as-a-judge | 能理解語意，適合評估 RAG 回答品質 | 成本較高，可能有 evaluator bias |
| ROUGE | 快速、便宜、適合看內容覆蓋 | 不理解語意，容易誤判 paraphrase |
| BLEU | 快速、客觀、適合文字重疊比較 | 對開放式問答不一定合理 |
| METEOR | 比 BLEU 更平衡，考慮部分語意匹配 | 仍無法真正理解答案是否 grounded |

實務上可以組合使用：

```text
classic metrics 做快速初篩
LLM-as-a-judge 做語意評估
人工 review 檢查關鍵失敗案例
```

## 六、Ragas 簡介

Ragas 是一個用來評估 LLM application 的 library。官方文件把它定位成從「憑感覺檢查」走向「系統化 evaluation loop」的工具。

Ragas 提供：

```text
metrics
datasets
experiments
test data generation
framework integrations
observability integrations
```

Ragas 的 RAG metrics 包含：

| Metric | 對應評估面向 |
| --- | --- |
| Context Precision | retrieval precision / context relevance |
| Context Recall | retrieval recall |
| Context Entities Recall | entity-level retrieval coverage |
| Noise Sensitivity | 對無關 context 的敏感度 |
| Response Relevancy | answer relevance |
| Faithfulness | groundedness / hallucination |
| Multimodal Faithfulness | 多模態 groundedness |
| Multimodal Relevance | 多模態 relevance |

對本 repo 的 RAG 系統來說，第一版可以先使用：

```text
Context Precision
Context Recall
Response Relevancy
Faithfulness
Factual Correctness
```

## 七、Ragas 評估資料格式

Ragas 評估通常需要保存：

| 欄位 | 說明 |
| --- | --- |
| `user_input` / `question` | 使用者問題 |
| `retrieved_contexts` | RAG 找回的 context list |
| `response` / `answer` | RAG 產生的回答 |
| `reference` | 標準答案，部分 metrics 需要 |
| `reference_contexts` | 期望找回的 context，部分 retrieval metrics 需要 |

概念上：

```python
sample = {
    "user_input": "What is an AI agent?",
    "retrieved_contexts": [
        "An AI agent is a system that can reason, plan, and use tools..."
    ],
    "response": "An AI agent is a system that ...",
    "reference": "An AI agent is ...",
}
```

## 八、套用到本 repo 的評估流程

本 repo 目前有兩條 end-to-end RAG 路線：

```text
Traditional RAG: traditional_rag.py
LLM Wiki RAG: llm_wiki.py
```

建議新增：

```text
data/evaluation/
├── ai_agent_course_eval_questions.jsonl
├── traditional_rag_results.jsonl
├── llm_wiki_results.jsonl
└── ragas_scores.csv
```

流程：

```text
1. 準備 evaluation questions
2. 跑 traditional_rag.py，保存 answer + retrieved contexts
3. 跑 llm_wiki.py，保存 answer + retrieved contexts
4. 轉成 Ragas dataset
5. 跑 Ragas metrics
6. 比較 raw / wiki / wiki-grounded 模式
7. 人工 review 失敗案例
```

建議問題類型：

| 問題類型 | 範例 |
| --- | --- |
| 定義型 | What is an AI agent? |
| 原文細節 | What are the key components of an AI agent? |
| 跨文件比較 | Compare agent memory and context engineering. |
| 流程型 | What risks appear when moving agents from prototype to production? |
| 工具型 | How does MCP help tool interoperability? |
| 評估型 | How can we evaluate agent quality? |

## 九、Monitoring / Observability

Evaluation 通常是在開發或測試階段跑固定資料集。
Monitoring / Observability 則是上線後持續觀察真實使用情況。

RAG observability 應該記錄：

```text
user query
retrieved contexts
metadata / source
prompt
LLM response
latency
token usage
cost
errors
user feedback
evaluation scores
```

最重要的是 trace：

```text
query
-> retrieval
-> retrieved chunks
-> prompt
-> LLM call
-> answer
```

沒有 trace，就很難知道錯誤來自 retrieval、prompt、model，還是資料本身。

## 十、LangSmith / Langfuse / Arize Phoenix 怎麼選

| 工具 | 優點 | 限制 | 適合情境 |
| --- | --- | --- | --- |
| LangSmith | 和 LangChain / LangGraph 整合最好，datasets、experiments、evaluators 完整 | 偏 LangChain ecosystem，商業平台導向明顯 | 專案主要使用 LangChain / LangGraph |
| Langfuse | Open-source、可 self-host，包含 tracing、prompt management、evaluation、dashboard | 功能多，初期設定和概念較多 | 想做 production monitoring、prompt 管理、團隊協作 |
| Arize Phoenix | Open-source，基於 OpenTelemetry / OpenInference，RAG tracing、eval、experiments 適合 debug | 若要完整 enterprise monitoring，可再看 Arize AX | 教學、本機 debug、跨框架 RAG 分析 |

以這個 repo 來看，我會建議：

```text
第一選擇：Arize Phoenix
第二選擇：Langfuse
LangSmith：如果後續主要使用 LangChain / LangGraph，再選
```

原因：

```text
Phoenix 適合教學與本機 RAG debug
Langfuse 適合長期產品化與團隊監控
LangSmith 適合 LangChain-heavy 專案
```

## 十一、本課程建議採用的評估架構

建議本章採用：

```text
Ragas：離線自動評估
Phoenix：RAG trace 與 debug
人工 review：檢查 citation 與回答可用性
```

完整流程：

```text
evaluation questions
-> run traditional_rag.py / llm_wiki.py
-> save contexts + answers
-> Ragas scoring
-> Phoenix trace inspection
-> manual error analysis
-> update retrieval / prompt / index
```

最後可以輸出：

```text
data/evaluation/report.md
```

報告內容：

```text
整體分數
各題分數
失敗案例
retrieval 錯誤類型
generation 錯誤類型
下一步改善方向
```

## 十二、常見錯誤分析分類

| 錯誤類型 | 說明 | 改善方向 |
| --- | --- | --- |
| Retrieval miss | 沒找回正確 context | 調整 chunking、embedding、top-k、metadata filter |
| Context noise | 找回太多不相關 context | 加 rerank、compression、metadata filter |
| Grounding failure | 答案沒有根據 context | 加強 prompt、faithfulness eval、引用要求 |
| Incomplete answer | 有找對資料但回答不完整 | 增加 context window、改善 prompt |
| Citation error | 引用來源不支持答案 | 改 metadata、source_refs、citation validation |
| Latency / cost issue | 查詢太慢或成本太高 | 降低 top-k、cache、調整模型 |

## 十三、本節重點

RAG 評估不應只看最後答案。

完整評估要拆成：

```text
Context Relevance
Faithfulness / Groundedness
Answer Relevance
```

也就是：

```text
retrieval 是否找對資料
answer 是否根據 context
answer 是否真正回答問題
```

本 repo 的建議路線：

```text
先用 Ragas 建立離線評估
再用 Phoenix 看每次 RAG trace
最後搭配人工 review 建立錯誤分析與改善迴圈
```

## 參考資料

- [Datawhale all-in-rag：System Evaluation](https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter6/18_system_evaluation.md)
- [Ragas Documentation](https://docs.ragas.io/en/stable/)
- [Ragas Available Metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [LangSmith Evaluation Concepts](https://docs.langchain.com/langsmith/evaluation-concepts)
- [Langfuse Overview](https://langfuse.com/docs)
- [Arize Phoenix Documentation](https://arize.com/docs/phoenix)
