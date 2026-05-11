# 第二節 文字分塊

文字分塊（Text Chunking）是 RAG 資料前處理中的核心步驟。資料載入只能把 PDF、Markdown、文字檔轉成可處理的文字；但這些文字通常不能整份直接送進 embedding model 或 LLM。

RAG 需要先把長文件切成較小、語意相對完整的片段，這些片段稱為 **chunks**。後續 embedding、vector store、retrieval 都會以 chunk 為基本單位。

```text
原始文件
  ↓
資料載入
  ↓
文字分塊
  ↓
embedding
  ↓
vector store
  ↓
retrieval
```

本節會對照三個範例：

| 範例檔案 | 分塊方式 |
| --- | --- |
| `character_splitter.py` | 字元切分 |
| `recursive_character_splitter.py` | 遞迴字元切分 |
| `semantic_chunker.py` | 語意切分 |

## 一、為什麼需要文字分塊

RAG 系統中至少有兩個地方會受到長度限制：

1. **Embedding model 的輸入長度限制**
2. **LLM 的上下文視窗限制**

### 1. Embedding model 的 context window

Embedding model 會把文字轉成向量。這個向量之後會被放進 vector store，作為 retrieval 的比對基礎。

但 embedding model 本身也有輸入長度限制，通常稱為 **context window** 或 **max sequence length**。

以本課程使用的 `BAAI/bge-m3` 為例，它的最大輸入長度是 **8192 tokens**。這代表理論上，一個 chunk 最多可以長到 8192 tokens 以內，模型仍然能處理。

但是，這不代表 chunk 應該盡量切到 8192 tokens。

原因是 embedding 的本質是一種「語意壓縮」：

```text
一大段文字
  ↓
embedding model
  ↓
一個固定維度的向量
```

不管輸入是 100 tokens、1000 tokens，還是 8000 tokens，最後通常都會被壓成一個固定長度的向量。這個向量必須概括整個 chunk 的語意。

當 chunk 很短、主題很集中時，向量可以比較精準地代表內容。當 chunk 很長、包含多個主題時，向量就會變得模糊，因為它同時要代表很多不同語意。

所以，即使 `BAAI/bge-m3` 支援 8192 tokens，實務上仍然會把 chunk 控制在更小、更聚焦的範圍，讓每個 chunk 盡量對應一個清楚主題。

### 2. LLM 的 context window

LLM 在回答問題時，也有上下文視窗限制。

RAG 的 prompt 通常會包含：

```text
系統指令
使用者問題
retrieved chunk 1
retrieved chunk 2
retrieved chunk 3
回答格式要求
```

也就是說，LLM 的 context window 不是只放 retrieved chunks，還要放 prompt、問題、格式要求，以及可能的對話歷史。

如果每個 chunk 都很大，會造成兩個問題：

1. 能放進 prompt 的 chunk 數量變少。
2. LLM 需要在很長的上下文中尋找答案。

第二個問題會產生長上下文常見的 **lost in the middle** 現象，也就是模型比較容易注意到上下文開頭和結尾，卻忽略中間的關鍵資訊。

因此，文字分塊的目標不是「切得越小越好」，也不是「切得越大越好」，而是要讓 chunk 在「語意完整」與「檢索精準」之間取得平衡。

## 二、chunk 太大或太小的問題

### chunk 太大

chunk 太大時，常見問題包含：

| 問題 | 說明 |
| --- | --- |
| 檢索不精準 | 一個 chunk 可能包含多個主題，embedding 會變得模糊 |
| 關鍵資訊被稀釋 | 問題只和其中一小段相關，但整個 chunk 的語意太雜 |
| prompt 內容變少 | 每個 chunk 太大時，LLM 能接收的 chunks 數量會下降 |
| lost in the middle | 長上下文中間的資訊可能比較容易被模型忽略 |

可以從 embedding 的角度再理解一次。

Embedding model 會先把文字 token 化，接著計算每個 token 的語意表示，最後再透過 pooling 或其他方式，把整段文字壓縮成一個向量。這個向量會被拿去和 query 向量做相似度比較。

當 chunk 太大時，壓縮過程會把很多不同主題混在一起。即使模型的 context window 夠長，也不代表壓縮後的單一向量可以精準保留所有細節。

例如一份 RAG 教學文件，如果把「資料載入」、「文字清理」、「文字分塊」、「向量資料庫」全部放進同一個 chunk，當使用者只問「`chunk_overlap` 是什麼？」時，這個 chunk 的確包含答案，但它的向量也同時代表資料載入、OCR、embedding、vector store 等其他概念。這會讓它和問題的語意相似度被稀釋。

### chunk 太小

chunk 太小也會有問題：

| 問題 | 說明 |
| --- | --- |
| 上下文不足 | 單一 chunk 可能只剩半句話或片段資訊 |
| 回答不完整 | retrieval 找到的片段缺少前後文 |
| chunk 數量過多 | vector store 變大，檢索與管理成本增加 |
| 語意被切斷 | 標題、段落和說明可能被拆散 |

好的 chunk 應該盡量做到：

1. 大小適中。
2. 語意完整。
3. 主題集中。
4. 保留必要上下文。
5. 方便追蹤來源。

## 三、重要參數

### `chunk_size`

`chunk_size` 控制每個 chunk 的目標大小。

```python
chunk_size=200
```

在本章範例中，`chunk_size=200` 代表每個 chunk 大約控制在 200 個字元左右。真實專案可以依文件長度、embedding model、LLM context window 調整。

### `chunk_overlap`

`chunk_overlap` 控制相鄰 chunk 之間重疊多少內容。

```python
chunk_overlap=10
```

重疊的用途是避免重要資訊剛好落在切分邊界。若一句話被切在兩個 chunk 中間，overlap 可以保留一點前後文，降低語意斷裂。

### `separators`

`separators` 是遞迴字元切分常用的參數，用來指定切分優先順序。

```python
separators=["\n\n", "\n", "。", "，", " ", ""]
```

對中文文件來說，加入 `。`、`，` 這類中文標點很重要，因為中文不像英文一樣有穩定的空白作為詞邊界。

## 四、LangChain 分割策略

### 4.1 字元切分：`CharacterTextSplitter`

範例檔案：

```text
chapter/02_Data_Preprocessing/character_splitter.py
```

`CharacterTextSplitter` 是最直覺的分割器。它主要依照固定長度切分文字，適合用來理解 chunk 的基本概念。

```python
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
)
```

它的特色是：

| 面向 | 說明 |
| --- | --- |
| 切分依據 | 主要依照字元長度與分隔符 |
| 優點 | 簡單、快速、容易觀察 `chunk_size` 和 `chunk_overlap` 的效果 |
| 限制 | 不理解語意，可能把句子、段落或主題切開 |
| 適合 | 短文本、格式單純的文字、教學示範 |
| 不適合 | 長篇文章、Markdown 結構文件、語意邊界很重要的資料 |

在正式 RAG 專案中，`CharacterTextSplitter` 通常不是最佳選擇，但很適合當作第一個分割器範例，因為它能清楚展示「固定大小分塊」會帶來什麼效果。

### 4.2 遞迴字元切分：`RecursiveCharacterTextSplitter`

範例檔案：

```text
chapter/02_Data_Preprocessing/recursive_character_splitter.py
```

核心程式：

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],
    chunk_size=200,
    chunk_overlap=10,
)
```

遞迴字元切分比一般字元切分更適合真實文件。它會依照分隔符優先順序嘗試切分。

#### 切分流程

```text
先用段落切
  ↓ 如果還是太長
再用換行切
  ↓ 如果還是太長
再用句號切
  ↓ 如果還是太長
再用逗號、空白或單一字元切
```

這種方式能盡量保留較大的語意單位，例如段落和句子；只有在內容真的太長時，才逐步使用更細的切分方式。

#### 中文文件的 separators

中文文件建議加入中文標點：

```python
separators=["\n\n", "\n", "。", "，", " ", ""]
```

也可以依照文件型態加入更多符號：

```python
separators=[
    "\n\n",
    "\n",
    "。", "！", "？",
    "，", "、",
    " ",
    "",
]
```

如果文件是 Markdown，也可以優先保留標題結構：

```python
separators=[
    "\n# ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    "。",
    "，",
    " ",
    "",
]
```

#### 優點與限制

| 面向 | 說明 |
| --- | --- |
| 優點 | 比固定字元切分更能保留段落與句子結構 |
| 優點 | 適合中文、英文與一般教學文件 |
| 限制 | 仍然是根據符號切分，不是真正理解語意 |

實務上，如果不知道該先用哪種方法，`RecursiveCharacterTextSplitter` 通常是很好的預設選擇。

### 4.3 語意切分：`SemanticChunker`

`SemanticChunker` 和前面兩種 splitter 最大的差別是：它不是只看字元數，也不是只看標點符號，而是利用 embedding 判斷「語意在哪裡發生明顯轉折」。

範例檔案：

```text
chapter/02_Data_Preprocessing/semantic_chunker.py
```

核心程式：

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
)
```

#### 實現原理

`SemanticChunker` 的流程可以分成五步。

第一步是 **句子切分**。它會先把原始文字切成句子序列。這裡的基本單位不是 chunk，而是句子。

```text
原始文件
  ↓
句子 1
句子 2
句子 3
...
```

第二步是 **上下文感知 embedding**。語意切分不一定只對單句做 embedding，因為單句有時候太短，語意不足。`SemanticChunker` 會透過 `buffer_size` 把前後句子一起納入考量。

可以理解成：

```text
句子 i 的表示
= 前 buffer_size 個句子
  + 句子 i
  + 後 buffer_size 個句子
```

假設 `buffer_size=1`，當處理第 3 句時，實際拿去 embedding 的可能是：

```text
句子 2 + 句子 3 + 句子 4
```

這樣做的好處是，每個句子的向量不只代表單句，也包含附近上下文，比較能反映它在段落中的真正語意。

第三步是 **計算語意距離**。當每個句子都有 embedding 後，系統會計算相鄰句子向量之間的距離。

```text
句子 1 向量  vs  句子 2 向量
句子 2 向量  vs  句子 3 向量
句子 3 向量  vs  句子 4 向量
```

距離越小，代表兩句語意越接近；距離越大，代表語意跳躍越明顯。

第四步是 **找出斷點**。`SemanticChunker` 會觀察所有相鄰句子的語意距離，並用統計方法找出「特別大的距離」。這些位置就會被視為適合切分的 breakpoint。

第五步是 **合併成 chunks**。找到斷點後，系統會依照斷點把句子序列切開，並把每一段句子合併成最終 chunk。

```text
句子 1、句子 2、句子 3
  ↓ breakpoint
句子 4、句子 5
  ↓ breakpoint
句子 6、句子 7、句子 8
```

#### 重要參數

| 參數 | 說明 |
| --- | --- |
| `embeddings` | 用來計算句子語意向量的 embedding model |
| `breakpoint_threshold_type` | 判斷語意斷點的統計方法 |
| `breakpoint_threshold_amount` | 斷點門檻值，會依 `breakpoint_threshold_type` 有不同意義 |
| `buffer_size` | 每個句子在 embedding 時，要合併前後多少句作為上下文 |

#### `breakpoint_threshold_type`

這個參數決定「什麼樣的語意距離算是明顯跳躍」。

| 值 | 判斷方式 | 適合情境 |
| --- | --- | --- |
| `percentile` | 將所有語意距離排序，超過指定百分位的距離視為斷點 | 通用預設值，適合先嘗試 |
| `standard_deviation` | 距離大於平均值加上數倍標準差時視為斷點 | 距離分布接近常態時 |
| `interquartile` | 使用四分位距 IQR 找出異常大的語意距離 | 想降低極端值影響時 |
| `gradient` | 觀察距離變化率，找語意轉折點 | 長文件、主題轉折較平滑的文件 |

範例中使用：

```python
breakpoint_threshold_type="percentile"
```

這代表它會用百分位數方式找出語意距離特別大的位置。

#### `buffer_size`

`buffer_size` 控制 embedding 時要看多少鄰近句子。

| 設定 | 效果 |
| --- | --- |
| `buffer_size=0` | 每個句子只看自己，切分較敏感，但可能缺少上下文 |
| `buffer_size=1` | 每個句子會參考前後各一句，通常較穩定 |
| `buffer_size` 更大 | 語意表示更平滑，但斷點可能變少 |

如果文件句子很短，適合使用較大的 `buffer_size`。如果文件本身句子很長，`buffer_size` 可以小一點。

#### 優點與限制

| 面向 | 說明 |
| --- | --- |
| 優點 | 能根據語意變化切分，而不是只依靠長度或標點 |
| 優點 | 適合長文章、主題切換明顯的內容 |
| 優點 | chunk 內部語意通常更集中 |
| 限制 | 需要 embedding model，速度比字元切分慢 |
| 限制 | 結果會受到 embedding model 品質影響 |
| 限制 | 斷點數量不一定穩定，需要調參觀察 |

語意切分適合用在主題轉換明顯的內容，例如研究報告、長篇教學文章、會議逐字稿或多段落知識文件。若文件本身有清楚的 Markdown 標題結構，通常可以先用 Markdown 結構切分，再針對太長的段落使用語意切分。

### 4.4 以 Markdown 結構分塊為例

Markdown 文件通常自帶結構，例如：

```text
# 章節標題

## 小節標題

段落內容...

### 更小的小節

更多內容...
```

對 Markdown 來說，最理想的分塊方式通常不是單純每 500 字切一次，而是優先保留標題結構。

#### 為什麼要保留標題

標題在 RAG 中很重要，因為它可以提供上下文。

例如單看這句：

```text
它可以降低模型幻覺。
```

這句話本身不夠清楚。但如果它位於：

```text
## RAG 的優點
```

下面，模型就更容易知道「它」指的是 RAG。

#### Markdown 分塊策略

一個常見策略是：

1. 先按照 Markdown 標題切分。
2. 保留每個 chunk 所屬的標題路徑。
3. 如果某個章節仍然太長，再用遞迴字元切分。
4. 將標題作為 metadata 或補進 chunk 文字中。

流程如下：

```text
Markdown
  ↓
依 # / ## / ### 切出章節
  ↓
章節太長時再做 recursive chunking
  ↓
保留標題 metadata
  ↓
送進 embedding
```

#### 範例概念

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter


headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

docs = markdown_splitter.split_text(markdown_text)
```

`headers_to_split_on` 用來指定要根據哪些 Markdown 標題層級切分。

| 設定 | 說明 |
| --- | --- |
| `("#", "h1")` | 依一級標題切分，metadata 欄位名稱為 `h1` |
| `("##", "h2")` | 依二級標題切分，metadata 欄位名稱為 `h2` |
| `("###", "h3")` | 依三級標題切分，metadata 欄位名稱為 `h3` |

這種方式很適合教學文件、技術文件、README、API 文件等有明確標題階層的資料。

## 五、三種分塊方式比較

| 方法 | 優點 | 限制 | 適合情境 |
| --- | --- | --- | --- |
| `CharacterTextSplitter` | 簡單、快速 | 對語意邊界不敏感 | 基礎示範、簡單文字 |
| `RecursiveCharacterTextSplitter` | 保留段落與句子結構，實務常用 | 依然不是真正理解語意 | 一般文件、中文文章、Markdown 前處理 |
| `SemanticChunker` | 根據語意變化切分 | 需要 embedding model，速度較慢 | 長文章、主題切換明顯的內容 |
| Markdown 結構分塊 | 保留標題階層與 metadata | 只適合結構清楚的 Markdown | README、技術文件、課程講義 |

## 六、如何選擇 chunk 參數

沒有一組參數適合所有文件。可以先用下列方向作為起點：

| 文件類型 | 建議方式 |
| --- | --- |
| 短篇純文字 | `CharacterTextSplitter` 或 `RecursiveCharacterTextSplitter` |
| 中文教學文章 | `RecursiveCharacterTextSplitter`，加入中文標點 |
| Markdown 文件 | 先依標題切，再做 recursive chunking |
| 長篇報告 | `RecursiveCharacterTextSplitter` 或 `SemanticChunker` |
| 主題跳動明顯的文章 | `SemanticChunker` |
| 程式碼文件 | 使用語言感知的 splitter 或依函式 / class 切分 |

建議實驗流程：

1. 先用 `chunk_size=300~800`。
2. 設定 `chunk_overlap=30~100`。
3. 隨機抽幾個 chunk 檢查是否語意完整。
4. 用幾個問題測試 retrieval 結果。
5. 根據檢索品質調整參數。

在本章範例中，為了方便觀察，使用較小的：

```python
chunk_size=200
chunk_overlap=10
```

真實專案可以依文件長度和模型上下文調大。

## 七、練習

### 練習 1：調整 `chunk_size`

改動 `character_splitter.py` 中的：

```python
chunk_size=200
```

觀察 chunk 數量與內容完整度的變化。

### 練習 2：調整 `chunk_overlap`

改動：

```python
chunk_overlap=10
```

觀察相鄰 chunks 之間是否保留更多上下文。

### 練習 3：修改中文 separators

在 `recursive_character_splitter.py` 中嘗試：

```python
separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
```

觀察切分結果是否更符合中文句子邊界。

### 練習 4：比較語意切分

執行：

```powershell
python chapter/02_Data_Preprocessing/semantic_chunker.py
```

比較它和 recursive chunking 的差異：

1. chunk 數量是否不同？
2. 每個 chunk 是否更像同一個主題？
3. 執行速度是否變慢？

## 小結

文字分塊會直接影響 RAG 的檢索品質。太大的 chunk 會稀釋語意，太小的 chunk 會失去上下文。

實務上可以先從 `RecursiveCharacterTextSplitter` 開始，搭配合理的 `chunk_size`、`chunk_overlap` 和中文標點 separators。當文件主題轉換明顯或對語意一致性要求較高時，再考慮 `SemanticChunker`。

如果資料本身是 Markdown 或技術文件，則應優先利用標題階層做結構化分塊，這通常比單純依字數切分更適合 RAG。

## 參考資料

- [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [LangChain MarkdownHeaderTextSplitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)
