# 第一節 資料載入

[![English](https://img.shields.io/badge/Language-English-blue)](./01_data_load_en.md)

資料載入是 RAG pipeline 的第一步。這一步看起來只是「把檔案讀進來」，但實際上會直接影響後面的 chunk 切分、embedding、retrieval，以及最後 LLM 回答的品質。

如果資料載入階段抽到的文字是錯的、順序是亂的、表格被破壞，或是 metadata 不完整，後面再好的 embedding model 和 LLM 也很難補救。RAG 很符合一句資料工程常見的原則：

> Garbage in, garbage out.

也就是輸入資料品質不好，輸出結果通常也不會好。

本節會先介紹資料載入器的角色，再用 `Unstructured` 解析 PDF，最後說明重要參數與常見問題。

## 一、什麼是資料載入器

在 RAG 系統中，資料來源可能有很多種格式，例如：

| 類型 | 常見格式 |
| --- | --- |
| 文字文件 | `.txt`、`.md` |
| 辦公文件 | `.docx`、`.pptx`、`.xlsx` |
| 網頁資料 | `.html`、URL |
| PDF | 文字型 PDF、掃描型 PDF、論文、報告 |
| 結構化資料 | `.csv`、`.json` |

資料載入器的工作，是把這些不同格式的原始資料轉成程式可以處理的標準格式。

在 RAG pipeline 裡，資料載入器通常要完成三件事：

1. **讀取原始檔案**：例如讀取 PDF、Markdown、Word 或 HTML。
2. **抽取內容**：把文件中的標題、段落、表格、列表等內容取出來。
3. **保留 metadata**：例如檔名、頁碼、元素類型、來源路徑等資訊。

這些結果之後才會被送去做 chunk、embedding 和 vector database 建立。

## 二、常見文件載入工具

不同文件解析工具的設計重點不同。實務上不一定有單一最佳選擇，通常要依照文件格式、解析精度、速度和部署成本來決定。

| 工具 | 特色 | 適合情境 |
| --- | --- | --- |
| `TextLoader` | 輕量、單純讀純文字 | `.txt`、簡單文字資料 |
| `DirectoryLoader` | 批次讀取資料夾 | 多檔案文件庫 |
| `Unstructured` | 支援多種非結構化文件格式 | PDF、Word、HTML、Markdown、混合文件 |
| `MarkItDown` | 將多種文件轉成 Markdown，方便後續清理與切分 | Office 文件、PDF、HTML、圖片說明、需要 Markdown 輸出的流程 |
| `PyMuPDF` / `PyMuPDF4LLM` | PDF 文字抽取速度快 | 文字型 PDF、技術文件 |
| `LlamaParse` | PDF 結構解析能力強 | 論文、合約、表格多的文件 |
| `Docling` | 偏企業文件解析 | 報告、文件轉換、結構化輸出 |
| `Marker` | PDF 轉 Markdown | 書籍、論文、長篇 PDF |
| `MinerU` | 偏多模態文件解析 | 學術文件、表格、公式、版面複雜 PDF |
| `Chandra` | OCR / Document Intelligence 模型，可將圖片與 PDF 轉成 Markdown、HTML、JSON，並盡量保留版面資訊 | 掃描型 PDF、複雜表格、表單、手寫與版面複雜文件 |

本節使用 `Unstructured`，原因是它提供統一的 `partition(...)` 介面，可以用相似的方式處理不同格式文件，適合教學與 RAG 前處理入門。

## 三、Unstructured 簡介

`Unstructured` 是一套用於非結構化文件解析的工具。它的核心概念是 **partitioning**：將原始文件拆解成一個個 document elements。

例如 PDF 解析後，可能會得到：

| 元素類型 | 說明 |
| --- | --- |
| `Title` | 標題 |
| `NarrativeText` | 一般正文段落 |
| `ListItem` | 清單項目 |
| `Table` | 表格 |
| `Image` | 圖片相關資訊 |
| `FigureCaption` | 圖片說明 |
| `Header` | 頁首 |
| `Footer` | 頁尾 |
| `PageBreak` | 分頁 |
| `PageNumber` | 頁碼 |
| `UncategorizedText` | 尚未分類的文字 |
| `CompositeElement` | chunking 後形成的複合元素 |

每個 element 通常包含：

| 欄位 | 說明 |
| --- | --- |
| `category` | 元素類型，例如 `Title`、`NarrativeText` |
| `text` / `str(element)` | 元素文字內容 |
| `metadata` | 頁碼、檔案名稱、檔案路徑等補充資訊 |

對 RAG 來說，這些 element 是後續切 chunk 和建立索引的基礎。

## 四、程式範例

本節範例檔案：

```text
chapter/02_Data_Preprocessing/unstructured_example.py
```

目前範例使用 `Unstructured` 讀取 PDF：

```python
from collections import Counter

from unstructured.partition.auto import partition


pdf_path = "../../data/C2/pdf/Medium.pdf"

elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
)

total_characters = sum(len(str(element)) for element in elements)
print(f"總共解析出 {len(elements)} 個元素，共 {total_characters} 個字元")

types = Counter(element.category for element in elements)
print(f"元素類型統計: {dict(types)}")

print("\n解析出的元素:")
for i, element in enumerate(elements, 1):
    print(f"元素 {i} ({element.category}):")
    print(element)
    print("=" * 60)
```

這段程式做了四件事：

1. 指定 PDF 檔案路徑。
2. 用 `partition(...)` 解析 PDF。
3. 統計解析出的 element 數量與總字元數。
4. 印出每個 element 的類型與內容。

## 五、`partition(...)` 重要參數

`partition(...)` 是 Unstructured 最常用的入口。它會依照檔案類型，自動路由到對應的解析函式，例如 PDF 會交給 PDF 解析流程。

### `filename`

```python
filename=pdf_path
```

指定要解析的本機檔案路徑。

### `content_type`

```python
content_type="application/pdf"
```

指定檔案 MIME type。

如果有設定 `content_type`，Unstructured 就不需要完全依賴副檔名或自動偵測來判斷檔案類型。

常見值：

| 檔案 | `content_type` |
| --- | --- |
| PDF | `application/pdf` |
| HTML | `text/html` |
| Markdown | `text/markdown` |
| 純文字 | `text/plain` |

### `strategy`

```python
strategy="fast"
```

`strategy` 決定解析 PDF 或圖片時使用哪一種策略。

常見策略如下：

| 策略 | 說明 | 適合情境 |
| --- | --- | --- |
| `auto` | 讓 Unstructured 自動選擇策略 | 不確定文件型態時 |
| `fast` | 使用較快的文字抽取方式 | PDF 本身有可選取文字 |
| `ocr_only` | 使用 OCR 解析圖片中的文字 | 掃描型 PDF、圖片型文件 |
| `hi_res` | 使用版面偵測模型分析文件結構 | 表格、圖片、版面複雜的 PDF |

如果 PDF 是文字型 PDF，通常 `fast` 就可以解析，而且速度較快。

如果 PDF 是掃描圖片，`fast` 可能會抽不到文字，這時就需要 `ocr_only` 或 `hi_res`。

### `include_page_breaks`

```python
include_page_breaks=True
```

控制是否在結果中加入 `PageBreak` element。

如果你希望後續保留頁面邊界，例如要顯示「第幾頁的內容」，可以打開這個參數。

### `encoding`

```python
encoding="utf-8"
```

指定文字編碼。

大部分情況可以讓 Unstructured 自動判斷；如果處理純文字檔時出現亂碼，可以手動指定。

### `languages`

```python
languages=["eng", "chi_tra"]
```

指定 OCR 語言。

這個參數主要在 OCR 情境下重要。如果文件包含繁體中文，Tesseract 也需要安裝對應語言包，否則 OCR 效果會很差或無法執行。

## 八、範例輸出解讀

程式會先印出總數：

```text
總共解析出 42 個元素，共 12345 個字元
```

這代表 Unstructured 從 PDF 中切出 42 個 document elements。

接著會印出元素類型統計：

```text
元素類型統計: {'Title': 58, 'NarrativeText': 29, 'Header': 5, 'UncategorizedText': 3, 'Footer': 3, 'ListItem': 3}
```

這可以幫助你快速判斷解析結果是否合理。

如果全部都是 `UncategorizedText`，可能表示文件結構沒有被很好地辨識。

如果元素數量是 0，可能代表：

1. PDF 沒有可抽取文字層。
2. 需要 OCR。
3. Poppler 或 Tesseract 沒有安裝好。
4. 檔案路徑不正確。

## 九、在 RAG 中如何使用解析結果

`partition(...)` 回傳的是 elements。這些 elements 還不是最終的 vector database，它們只是資料載入與解析的結果。

後續通常會做：

```text
PDF
  ↓
Unstructured elements
  ↓
清理文字
  ↓
合併或切分 chunks
  ↓
embedding
  ↓
vector store
  ↓
retrieval
```

在實務上，你可以根據 element 類型決定保留或過濾內容。例如：

| 類型 | 處理方式 |
| --- | --- |
| `Title` | 通常保留，可作為章節資訊 |
| `NarrativeText` | 通常保留，是 RAG 的主要內容 |
| `ListItem` | 通常保留，尤其是步驟、條列、規格 |
| `Table` | 視需求轉成 Markdown table 或 HTML |
| `Header` / `Footer` | 常常過濾，避免頁首頁尾干擾 retrieval |
| `PageNumber` | 通常不放進正文，但可保留在 metadata |

## 十、練習

請嘗試修改 `unstructured_example.py`，觀察不同設定的解析結果。

### 練習 1：指定 `strategy="fast"`

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="fast",
)
```

觀察元素數量和文字內容是否正常。

### 練習 2：加入 page break

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    include_page_breaks=True,
)
```

觀察結果中是否出現 `PageBreak`。

### 練習 3：嘗試 OCR 策略

如果你已經安裝 Poppler 與 Tesseract，可以嘗試：

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="ocr_only",
    languages=["eng"],
)
```

如果文件包含繁體中文，請確認 OCR 工具已安裝繁中語言包，再嘗試：

```python
languages=["eng", "chi_tra"]
```

### 練習 4：比較不同 PDF

請分別用 `Medium.pdf` 和 `SAM3.pdf` 測試：

```python
pdf_path = "../../data/C2/pdf/Medium.pdf"
```

```python
pdf_path = "../../data/C2/pdf/SAM3.pdf"
```

比較兩者：

1. 哪一份解析出的元素較多？
2. 哪一份需要 OCR？
3. 元素類型是否合理？
4. 哪一份更適合直接做 RAG？

## 小結

資料載入不是 RAG 中可以隨便帶過的步驟。它決定了後續系統能看到什麼資料，也決定了 retrieval 的上限。

本節使用 `Unstructured` 示範如何解析 PDF，並觀察解析後的 elements。下一步通常會接續資料清理、chunk 切分與 embedding，讓這些文字真正進入 RAG 檢索流程。

## 參考資料

- [Unstructured Partitioning 官方文件](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- [Unstructured Document Elements 官方文件](https://docs.unstructured.io/ui/document-elements)
- [Unstructured Strategies 文件](https://unstructured.readthedocs.io/en/main/best_practices/strategies.html)
