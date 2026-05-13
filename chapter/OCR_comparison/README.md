# OCR 工具比較實驗

這個資料夾用來比較不同 OCR / VLM OCR 工具在同一份 PDF 上的解析效果。

預設資料來源：

```text
data/C2/pdf/SAM3.pdf
```

預設輸出位置：

```text
data/OCR/
```

比較工具：

```text
PaddleOCR
dots.ocr
Chandra
DeepSeek-OCR
```

## 為什麼不只看字數

`SAM3.pdf` 本身有文字層，也包含大量圖片、caption、表格與公式。  
因此比較 OCR 工具時，不應只看「抽出多少字」，而要看結果是否適合放進 RAG：

```text
閱讀順序是否合理
標題與段落是否保留
表格是否可讀
圖片 caption 是否保留
Markdown 是否容易切 chunk
metadata 是否足夠追蹤來源
```

## 參考 Benchmark

這個專案的比較方式可以參考兩個公開 benchmark：

| Benchmark | 重點 | 可以借用的評估角度 |
| --- | --- | --- |
| [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) | 針對 PDF / document OCR 的 benchmark，資料 split 包含 `arxiv_math`、`headers_footers`、`long_tiny_text`、`multi_column`、`old_scans`、`old_scans_math`、`table_tests` | 公式、頁首頁尾、小字、多欄閱讀順序、舊掃描文件、表格 |
| [OCRBench v2](https://99franklin.github.io/ocrbench_v2/) | 大型 text-centric OCR benchmark，涵蓋多種文字辨識、文件理解與場景文字任務 | 文字準確度、跨場景泛化能力、圖文問答、文件結構理解 |

本 repo 不直接下載或重跑這兩個 benchmark。  
這裡是借用它們的評估方向，設計適合 `SAM3.pdf` 的人工評分表與輸出檢查項目。

## 基本執行

在 Linux GPU server 上執行：

```bash
python chapter/OCR_comparison/ocr_benchmark.py \
  --pdf data/C2/pdf/SAM3.pdf \
  --pages 1-5 \
  --tools paddleocr chandra dots_ocr deepseek_ocr \
  --output-dir data/OCR
```

輸出結構：

```text
data/OCR/
└── SAM3/
    ├── pages/
    │   ├── page_001.png
    │   ├── page_002.png
    │   └── ...
    ├── paddleocr/
    │   ├── output.md
    │   ├── output.json
    │   └── raw/
    ├── chandra/
    ├── dots_ocr/
    ├── deepseek_ocr/
    ├── summary.csv
    └── manual_score.csv
```

## 先跑小範圍

建議先跑 1 到 2 頁，確認工具環境都可用：

```bash
python chapter/OCR_comparison/ocr_benchmark.py --pages 1-2
```

確認沒問題後再跑：

```bash
python chapter/OCR_comparison/ocr_benchmark.py --pages 1-5
python chapter/OCR_comparison/ocr_benchmark.py --pages 1,2,3,4,5,8
```

最後才跑完整 PDF：

```bash
python chapter/OCR_comparison/ocr_benchmark.py --pages all
```

## PDF 轉圖片依賴

主程式會先將 PDF 頁面轉成 PNG，讓四個 OCR 工具吃到相同輸入。

需要安裝：

```bash
pip install pymupdf
```

如果使用專案環境：

```bash
uv pip install -r requirements.txt
```

## PaddleOCR

`paddleocr` adapter 會直接呼叫 Python package。

Linux server 上可依照 PaddleOCR 官方文件安裝：

```bash
pip install paddlepaddle paddleocr
```

如果使用 GPU，請依照你的 CUDA 版本安裝對應的 PaddlePaddle。

執行單一工具：

```bash
python chapter/OCR_comparison/ocr_benchmark.py --tools paddleocr --pages 1-2
```

## Chandra

Chandra 的執行方式可能依安裝方式不同，因此這裡使用環境變數指定 command template。

你需要設定：

```bash
export CHANDRA_OCR_COMMAND='chandra {input} {output_dir}'
```

其中：

| Placeholder | 說明 |
| --- | --- |
| `{input}` | 單頁 PNG 路徑 |
| `{output_dir}` | 該頁 raw output 目錄 |
| `{output}` | 同 `{output_dir}` |

執行：

```bash
python chapter/OCR_comparison/ocr_benchmark.py --tools chandra --pages 1-2
```

如果你的 Chandra CLI 參數不同，只要修改 `CHANDRA_OCR_COMMAND`，不需要改 benchmark 主程式。

## dots.ocr

dots.ocr 通常需要照官方 repo 啟動模型或 vLLM server。  
這裡同樣使用 command template。

範例：

```bash
export DOTS_OCR_COMMAND='python /path/to/dots.ocr/demo/demo.py --input {input} --output {output_dir}'
```

執行：

```bash
python chapter/OCR_comparison/ocr_benchmark.py --tools dots_ocr --pages 1-2
```

## DeepSeek-OCR

DeepSeek-OCR 也建議依官方 repo 在 Linux GPU server 上安裝，再用 command template 串接。

範例：

```bash
export DEEPSEEK_OCR_COMMAND='python /path/to/DeepSeek-OCR/inference.py --image {input} --output {output_dir}'
```

執行：

```bash
python chapter/OCR_comparison/ocr_benchmark.py --tools deepseek_ocr --pages 1-2
```

## summary.csv

每次 benchmark 會產生：

```text
data/OCR/SAM3/summary.csv
```

欄位：

```text
tool
pdf
pages
success
runtime_seconds
output_md_path
output_json_path
total_characters
markdown_headings
markdown_tables
image_caption_count
empty_pages
error
```

這份表適合快速看：

1. 哪個工具有成功跑完。
2. 哪個工具速度比較快。
3. 哪個工具輸出比較多 Markdown 結構。
4. 哪些工具有錯誤或空白頁。

## manual_score.csv

`manual_score.csv` 是人工評分模板：

```text
tool
text_accuracy_1_to_5
reading_order_1_to_5
table_quality_1_to_5
formula_quality_1_to_5
header_footer_handling_1_to_5
tiny_text_quality_1_to_5
multi_column_order_1_to_5
figure_caption_quality_1_to_5
markdown_usability_1_to_5
rag_readiness_1_to_5
notes
```

建議評分時看這幾頁：

| 頁數 | 觀察重點 |
| --- | --- |
| Page 1 | 標題、作者、abstract |
| Page 2-6 | 圖片、caption、圖文順序 |
| Page 8 | 表格密集頁 |

這些欄位設計參考 `olmOCR-bench` 的文件 OCR split，例如數學公式、頁首頁尾、小字、多欄與表格；也參考 `OCRBench v2` 對文字辨識與文件理解的整體評估方向。

## Git 注意事項

`data/OCR/` 是 benchmark 輸出資料夾，已加入 `.gitignore`。  
除非要把實驗結果整理成教材，否則不建議提交 OCR 輸出結果。
