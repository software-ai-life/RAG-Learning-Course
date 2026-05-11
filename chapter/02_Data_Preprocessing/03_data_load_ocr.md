# 第三節 OCR 與 Document AI 資料載入

前一節介紹的資料載入，主要處理「可以直接抽取文字」的文件。但實務上常見的 PDF 並不一定有文字層，例如掃描書籍、截圖型報告、合約影本、發票、表單、圖片型簡報等。

這類文件不能只靠一般文字抽取工具，通常需要 OCR 或 Document AI。

OCR 是 **Optical Character Recognition**，也就是光學文字辨識。它會從圖片或掃描頁面中辨識文字。Document AI 則更進一步，不只辨識文字，還會嘗試理解文件版面，例如標題、段落、表格、表單欄位、圖片說明、閱讀順序等。

## 一、什麼時候需要 OCR

可以先用下面方式判斷：

| 文件狀況 | 是否需要 OCR | 說明 |
| --- | --- | --- |
| 可以用滑鼠選取文字的 PDF | 通常不需要 | 可先用 `fast`、PyMuPDF、MarkItDown 等工具 |
| 每一頁看起來像圖片 | 通常需要 | 這是典型掃描型 PDF |
| 文字抽取結果是空的 | 通常需要 | 表示沒有可直接抽取的文字層 |
| 表格、圖片、版面很複雜 | 不一定，但常需要 Document AI | 只抽文字可能會破壞閱讀順序與表格結構 |
| 手寫、印章、簽名、表單欄位 | 通常需要進階 OCR / Document AI | 傳統 OCR 通常效果有限 |

如果你的 PDF 用 `Unstructured(strategy="fast")` 解析出 0 個元素，通常代表它可能是掃描型 PDF，或文字層無法直接抽取。

## 二、OCR 與一般文件載入的差異

一般文件載入工具通常做的是「文字抽取」：

```text
PDF 文字層
  ↓
抽出文字
  ↓
切 chunk
```

OCR 則是從影像辨識文字：

```text
PDF / image
  ↓
轉成頁面圖片
  ↓
OCR 辨識文字
  ↓
還原版面與閱讀順序
  ↓
輸出文字、Markdown、HTML 或 JSON
```

OCR 的成本通常比較高，速度較慢，也更容易受到解析度、掃描品質、語言包、版面複雜度影響。

## 三、常見強力 OCR / Document AI 工具

| 工具 | 類型 | 特色 | 適合情境 |
| --- | --- | --- | --- |
| `Tesseract OCR` | 開源 OCR engine | 老牌、免費、可本機執行，支援多語言；需要自行處理版面與後處理 | 單純圖片文字辨識、低成本本機 OCR |
| `PaddleOCR` / `PaddleOCR-VL` | 開源 OCR / 文件解析 | 支援多語言 OCR、版面分析、表格、公式等文件解析任務 | 中文文件、表格、版面複雜文件、本機部署 |
| `Surya` | 開源 OCR toolkit | 支援 OCR、layout analysis、reading order、table recognition，語言支援廣 | 需要本機 OCR、閱讀順序、表格辨識 |
| `Chandra` | 開源 OCR / Document Intelligence model | 可輸出 Markdown、HTML、JSON，強調複雜版面、表格、表單與手寫內容 | 想把掃描文件轉成 RAG 友善格式 |
| `dots.ocr` | 開源多語言文件解析 VLM | 將 layout detection 和 content recognition 放在同一個 vision-language model 中，支援 Markdown、JSON layout、視覺標註等輸出 | 多語言文件、網頁截圖、圖表、diagram、scene text |
| `olmOCR` | 開源 PDF / image OCR toolkit | 由 AI2 維護，可將 PDF、PNG、JPEG 轉成乾淨 Markdown，強調自然閱讀順序、表格、公式、手寫與複雜格式 | 大量 PDF 線性化、研究資料集、需要 Markdown 輸出的 RAG pipeline |
| `DeepSeek-OCR` | 開源 OCR / document understanding VLM | 可將文件轉成 Markdown，也支援 free OCR、figure parsing、visual grounding；核心概念是 visual-text compression | 文件 OCR、圖表解析、需要 self-host VLM OCR 的場景 |
| `Mistral OCR` | 商業 API | 可將 PDF / 文件抽成結構化內容，支援 Markdown / HTML 表格輸出設定 | 想快速取得 AI-ready Markdown，不想自行部署 OCR |
| `Azure AI Document Intelligence` | 商業 Document AI API | 可抽取文字、表格、selection marks、文件結構 | 企業文件、表單、表格、雲端工作流 |
| `Google Cloud Document AI` | 商業 Document AI API | 支援 OCR、layout parser、表格、表單與文件結構化 | 大量文件處理、GCP 生態、企業文件 |
| `Amazon Textract` | 商業 Document AI API | 可抽取文字、表格、表單、key-value pairs、簽名等 | AWS 生態、表單與半結構化文件 |
| `LlamaParse` | 商業文件解析 API | 偏向把 PDF 轉成 LLM / RAG 友善格式 | 論文、財報、表格與長文件 |
| `MinerU` | 開源 / 模型式文件解析 | 偏學術 PDF、公式、表格、圖片與多模態文件 | 論文、研究報告、版面複雜 PDF |

## 四、如何選擇 OCR 工具

| 文件狀況 | 建議優先嘗試 |
| --- | --- |
| 可以選取文字的 PDF | `Unstructured(strategy="fast")`、`PyMuPDF4LLM`、`MarkItDown` |
| 掃描型 PDF 或圖片 | `PaddleOCR`、`Surya`、`Chandra`、`dots.ocr`、`olmOCR`、`DeepSeek-OCR`、`Tesseract` |
| 表格很多 | `PaddleOCR-VL`、`Chandra`、`dots.ocr`、`olmOCR`、`Azure AI Document Intelligence`、`Amazon Textract` |
| 想直接轉 Markdown 給 RAG | `MarkItDown`、`Chandra`、`dots.ocr`、`olmOCR`、`DeepSeek-OCR`、`Mistral OCR`、`LlamaParse`、`Marker` |
| 企業雲端流程 | `Azure AI Document Intelligence`、`Google Cloud Document AI`、`Amazon Textract` |
| 想完全本機執行 | `Tesseract`、`PaddleOCR`、`Surya`、`Chandra`、`dots.ocr`、`olmOCR`、`DeepSeek-OCR`、`MinerU` |
| 不想處理環境安裝 | `Mistral OCR`、`Azure AI Document Intelligence`、`Google Cloud Document AI`、`Amazon Textract`、`LlamaParse` |

選擇工具時，通常要在四件事之間取捨：

1. **準確度**：是否能正確辨識文字、表格與閱讀順序。
2. **成本**：本機部署通常省 API 費，但需要硬體與維護；商業 API 則按量計費。
3. **速度**：OCR 通常比純文字抽取慢，hi-res layout parsing 又更慢。
4. **輸出格式**：RAG 常偏好 Markdown、JSON 或保留 metadata 的結構化結果。

## 五、Unstructured 中的 OCR 策略

使用 Unstructured 時，PDF 常見策略有：

| `strategy` | 說明 | 適合情境 |
| --- | --- | --- |
| `fast` | 直接抽取 PDF 中的文字層 | 文字型 PDF |
| `auto` | 讓 Unstructured 自動判斷策略 | 不確定文件型態 |
| `ocr_only` | 使用 OCR 辨識頁面圖片中的文字 | 掃描型 PDF |
| `hi_res` | 使用版面偵測模型，保留較多 layout 資訊 | 表格、圖片、版面複雜 PDF |

例如：

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="ocr_only",
    languages=["eng"],
)
```

如果文件包含繁體中文，可以嘗試：

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="ocr_only",
    languages=["eng", "chi_tra"],
)
```

但要注意，Tesseract 必須安裝對應語言包，否則 `chi_tra` 不會正常運作。

## 六、Windows 上的 Poppler 與 Tesseract

當 Unstructured 走到 OCR 或 PDF 轉圖片流程時，常會需要外部工具。

| 工具 | 用途 |
| --- | --- |
| Poppler | 將 PDF 頁面轉成圖片，提供 `pdfinfo`、`pdftoppm` 等工具 |
| Tesseract | OCR 文字辨識 |

Windows 可以用 `winget` 安裝：

```powershell
winget install oschwartz10612.Poppler
winget install UB-Mannheim.TesseractOCR
```

安裝後請重新開啟 PowerShell，讓新的 PATH 生效。

確認工具是否可用：

```powershell
pdfinfo -v
tesseract --version
```

如果出現：

```text
PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?
```

代表 Python 套件呼叫 `pdfinfo` 時找不到 Poppler。通常是尚未安裝，或安裝後 PATH 尚未生效。

## 七、OCR 輸出如何進入 RAG

OCR 的輸出不一定要直接丟進 vector database。比較好的流程是：

```text
PDF / image
  ↓
OCR / Document AI
  ↓
Markdown / HTML / JSON / elements
  ↓
清理頁首頁尾、頁碼、雜訊
  ↓
根據標題與段落切 chunk
  ↓
embedding
  ↓
vector store
```

對 RAG 來說，OCR 結果最好保留：

| 資訊 | 用途 |
| --- | --- |
| 頁碼 | 回答時可標示來源 |
| 區塊類型 | 可過濾頁首、頁尾、頁碼 |
| 表格結構 | 避免表格被攤平成難讀文字 |
| 閱讀順序 | 多欄文件或論文很重要 |
| 原始檔名 | 追蹤資料來源 |

如果 OCR 工具可以輸出 Markdown，通常會比單純純文字更適合 RAG，因為標題、列表、表格等結構比較容易保留。

## 八、練習

### 練習 1：比較 `fast` 與 `ocr_only`

用同一份 PDF 分別嘗試：

```python
strategy="fast"
```

和：

```python
strategy="ocr_only"
```

觀察：

1. 哪一個解析比較快？
2. 哪一個元素數量比較多？
3. 哪一個輸出的閱讀順序比較合理？

### 練習 2：測試文字型 PDF 與掃描型 PDF

分別使用：

```python
pdf_path = "../../data/C2/pdf/Medium.pdf"
```

和：

```python
pdf_path = "../../data/C2/pdf/SAM3.pdf"
```

觀察哪一份可以直接抽文字，哪一份需要 OCR。

### 練習 3：思考輸出格式

如果你要把 OCR 結果放進 RAG，請比較以下輸出格式：

| 格式 | 優點 | 缺點 |
| --- | --- | --- |
| 純文字 | 簡單、好處理 | 結構容易消失 |
| Markdown | 適合 RAG，保留標題與列表 | 表格很複雜時可能失真 |
| HTML | 能保留表格與版面 | 需要額外清理 |
| JSON | metadata 清楚，適合程式處理 | 不適合直接給 LLM，通常要轉換 |

## 小結

OCR 是資料載入中很重要但也很容易踩坑的一段。能直接抽文字的 PDF，優先用 `fast` 類工具；掃描型 PDF、表格很多或版面複雜的文件，才考慮 OCR 或 Document AI。

在 RAG 專案中，目標不是只把文字抽出來，而是要抽出「可檢索、可引用、結構合理」的內容。

## 參考資料

- [Unstructured Strategies](https://unstructured.readthedocs.io/en/main/best_practices/strategies.html)
- [Unstructured Document Elements](https://docs.unstructured.io/ui/document-elements)
- [Chandra GitHub](https://github.com/datalab-to/chandra)
- [dots.ocr GitHub](https://github.com/rednote-hilab/dots.ocr)
- [olmOCR GitHub](https://github.com/allenai/olmocr)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Surya GitHub](https://github.com/datalab-to/surya)
- [Mistral OCR Docs](https://docs.mistral.ai/capabilities/document_ai/basic_ocr)
- [Azure AI Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-layout)
- [Google Cloud Document AI](https://cloud.google.com/document-ai/docs/overview)
- [Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-analyzing.html)
