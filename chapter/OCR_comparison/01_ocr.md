# 第一節 OCR

前面章節已經介紹過一般資料載入、文字切分、Embedding 與向量資料庫。這些流程大多假設資料已經能被轉成乾淨文字。

但實務上常見的 PDF 不一定能直接抽出完整文字，例如掃描書籍、截圖型報告、合約影本、發票、表單、圖片型簡報、論文中的圖表與公式等。

這類文件不能只靠一般文字抽取工具，通常需要 OCR 或 VLM OCR。

OCR 是 **Optical Character Recognition**，也就是光學文字辨識。它會從圖片或掃描頁面中辨識文字。近年的 VLM OCR 則會結合視覺語言模型，嘗試保留標題、段落、表格、圖片說明與閱讀順序，讓輸出更適合後續 RAG 使用。

## 一、什麼時候需要 OCR

可以先用下面方式判斷：

| 文件狀況 | 是否需要 OCR | 說明 |
| --- | --- | --- |
| 可以用滑鼠選取文字的 PDF | 通常不需要 | 可先用 `fast`、PyMuPDF、MarkItDown 等工具 |
| 每一頁看起來像圖片 | 通常需要 | 這是典型掃描型 PDF |
| 文字抽取結果是空的 | 通常需要 | 表示沒有可直接抽取的文字層 |
| 表格、圖片、版面很複雜 | 不一定，但常需要進階 OCR / VLM OCR | 只抽文字可能會破壞閱讀順序與表格結構 |
| 手寫、印章、簽名、表單欄位 | 通常需要進階 OCR / VLM OCR | 傳統 OCR 通常效果有限 |

如果 PDF 可以直接抽出文字，通常可以先用文字層作為 baseline；如果抽出的文字順序混亂、表格破碎，或圖片 caption 遺失，就需要再考慮 OCR 或 VLM OCR。

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

## 三、常見 OCR 工具

| 工具 | 類型 | 特色 | 適合情境 |
| --- | --- | --- | --- |
| `Tesseract OCR` | 開源 OCR engine | 老牌、免費、可本機執行，支援多語言；需要自行處理版面與後處理 | 單純圖片文字辨識、低成本本機 OCR |
| `PaddleOCR` / `PaddleOCR-VL` | 開源 OCR / VLM OCR | 支援多語言 OCR、版面分析、表格、公式等文件任務 | 中文文件、表格、版面複雜文件、本機部署 |
| `Surya` | 開源 OCR toolkit | 支援 OCR、layout analysis、reading order、table recognition，語言支援廣 | 需要本機 OCR、閱讀順序、表格辨識 |
| `Chandra` | 開源 OCR model | 可輸出 Markdown、HTML、JSON，強調複雜版面、表格、表單與手寫內容 | 想把掃描文件轉成 RAG 友善格式 |
| `dots.ocr` | 開源多語言 OCR VLM | 將 layout detection 和 content recognition 放在同一個 vision-language model 中，支援 Markdown、JSON layout、視覺標註等輸出 | 多語言文件、網頁截圖、圖表、diagram、scene text |
| `olmOCR` | 開源 PDF / image OCR toolkit | 由 AI2 維護，可將 PDF、PNG、JPEG 轉成乾淨 Markdown，強調自然閱讀順序、表格、公式、手寫與複雜格式 | 大量 PDF 線性化、研究資料集、需要 Markdown 輸出的 RAG pipeline |
| `DeepSeek-OCR` | 開源 OCR VLM | 可將文件轉成 Markdown，也支援 free OCR、figure parsing、visual grounding；核心概念是 visual-text compression | 文件 OCR、圖表解析、需要 self-host VLM OCR 的場景 |
| `Mistral OCR` | 商業 API | 可將 PDF / 文件抽成結構化內容，支援 Markdown / HTML 表格輸出設定 | 想快速取得 AI-ready Markdown，不想自行部署 OCR |
| `LlamaParse` | 商業解析 API | 偏向把 PDF 轉成 LLM / RAG 友善格式 | 論文、財報、表格與長文件 |
| `MinerU` | 開源 / 模型式 OCR pipeline | 偏學術 PDF、公式、表格、圖片與多模態文件 | 論文、研究報告、版面複雜 PDF |

### 3.1 OCR Benchmark 參考

如果想知道哪些 OCR / VLM OCR 工具效果較好，可以參考公開 benchmark。不過 benchmark 分數只能當方向，實際專案仍要用自己的 PDF 測試。

| Benchmark | 重點 | 對本課程的啟發 |
| --- | --- | --- |
| [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) | PDF / document OCR，包含數學公式、頁首頁尾、小字、多欄、舊掃描文件與表格測試 | 評估論文 PDF 時，不能只看總字數，也要看公式、表格、caption、閱讀順序 |
| [OCRBench v2](https://99franklin.github.io/ocrbench_v2/) | 大型 text-centric OCR benchmark，涵蓋多種文字辨識、文件理解與場景文字任務 | 可以用更廣的角度觀察 OCR 工具在不同場景下是否穩定 |

這些 benchmark 可以幫助你選工具候選名單，但不能完全取代本地測試。對 RAG 來說，還是要看輸出 Markdown 是否乾淨、表格是否可讀、metadata 是否足夠、是否適合切 chunk。

## 四、如何選擇 OCR 工具

| 文件狀況 | 建議優先嘗試 |
| --- | --- |
| 可以選取文字的 PDF | `Unstructured(strategy="fast")`、`PyMuPDF4LLM`、`MarkItDown` |
| 掃描型 PDF 或圖片 | `PaddleOCR`、`Surya`、`Chandra`、`dots.ocr`、`olmOCR`、`DeepSeek-OCR`、`Tesseract` |
| 表格很多 | `PaddleOCR-VL`、`Chandra`、`dots.ocr`、`olmOCR`、`MinerU` |
| 想直接轉 Markdown 給 RAG | `MarkItDown`、`Chandra`、`dots.ocr`、`olmOCR`、`DeepSeek-OCR`、`Mistral OCR`、`LlamaParse`、`Marker` |
| 想完全本機執行 | `Tesseract`、`PaddleOCR`、`Surya`、`Chandra`、`dots.ocr`、`olmOCR`、`DeepSeek-OCR`、`MinerU` |
| 不想處理環境安裝 | `Mistral OCR`、`LlamaParse` |

選擇工具時，通常要在四件事之間取捨：

1. **準確度**：是否能正確辨識文字、表格與閱讀順序。
2. **成本**：本機部署通常省 API 費，但需要硬體與維護；商業 API 則按量計費。
3. **速度**：OCR 通常比純文字抽取慢，hi-res layout parsing 又更慢。
4. **輸出格式**：RAG 常偏好 Markdown、JSON 或保留 metadata 的結構化結果。

## 五、OCR 輸出如何進入 RAG

OCR 的輸出不一定要直接丟進 vector database。比較好的流程是：

```text
PDF / image
  ↓
OCR / VLM OCR
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

## 六、OCR 輸出格式

如果你要把 OCR 結果放進 RAG，請比較以下輸出格式：

| 格式 | 優點 | 缺點 |
| --- | --- | --- |
| 純文字 | 簡單、好處理 | 結構容易消失 |
| Markdown | 適合 RAG，保留標題與列表 | 表格很複雜時可能失真 |
| HTML | 能保留表格與版面 | 需要額外清理 |
| JSON | metadata 清楚，適合程式處理 | 不適合直接給 LLM，通常要轉換 |

## 七、OCR 評估可以看什麼

OCR 評估不只看字有沒有辨識出來，也要看輸出能不能進入後續 RAG 流程。如果是為了 RAG，建議觀察：

| 指標 | 說明 |
| --- | --- |
| Text Accuracy | 文字是否正確 |
| Reading Order | 多欄、圖文混排時順序是否合理 |
| Table Quality | 表格是否保留欄列關係 |
| Formula Quality | 公式是否可讀，是否嚴重遺失符號 |
| Header / Footer Handling | 頁首、頁尾、頁碼是否造成雜訊 |
| Tiny Text Quality | 小字、註解、圖中標籤是否能辨識 |
| Figure Caption Quality | 圖片 caption 是否保留並放在正確位置 |
| Markdown Usability | Markdown 是否乾淨，是否適合切 chunk |
| RAG Readiness | 是否能直接進入 embedding / retrieval 流程 |

## 小結

OCR 是資料載入中很重要但也很容易踩坑的一段。能直接抽文字的 PDF，可以先保留原生文字層作為 baseline；掃描型 PDF、表格很多、圖文混合或版面複雜的文件，才需要進一步使用 OCR 或 VLM OCR。

在 RAG 專案中，目標不是只把文字抽出來，而是要抽出「可檢索、可引用、結構合理」的內容。

## 參考資料

- [Chandra GitHub](https://github.com/datalab-to/chandra)
- [dots.ocr GitHub](https://github.com/rednote-hilab/dots.ocr)
- [olmOCR GitHub](https://github.com/allenai/olmocr)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Surya GitHub](https://github.com/datalab-to/surya)
- [Mistral OCR Docs](https://docs.mistral.ai/capabilities/document_ai/basic_ocr)
- [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
- [OCRBench v2](https://99franklin.github.io/ocrbench_v2/)
