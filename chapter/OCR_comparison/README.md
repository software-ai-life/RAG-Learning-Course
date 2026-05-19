# OCR 工具比較實驗

這個資料夾用來比較不同 OCR / VLM OCR 工具在同一份 PDF 上的解析效果。

重點是：**每個 OCR 工具分開跑，跑完後再統一評比輸出結果**。

因為 PaddleOCR、Chandra、dots.ocr、DeepSeek-OCR 都是不同開源 repo，安裝方式、GPU 需求、執行指令和輸出格式都不同，所以本資料夾不負責統一執行所有 OCR。

## 流程

```text
1. 分別到各 OCR repo 跑模型
2. 把每個工具的輸出放到固定資料夾
3. 使用 evaluate_ocr_outputs.py 統一讀取結果
4. 產生 summary.csv 與 manual_score.csv
5. 依照人工評分欄位比較哪個輸出最適合 RAG
```

預設資料來源：

```text
data/C2/pdf/SAM3.pdf
```

預設輸出位置：

```text
chapter/OCR_comparison/outputs/
```

建議放成這個結構：

```text
chapter/OCR_comparison/outputs/
├── chandra/
│   └── SAM3/
│       ├── SAM3.md
│       ├── SAM3.html
│       └── SAM3_metadata.json
├── paddleocr/
│   └── SAM3/
├── dots_ocr/
│   └── SAM3/
└── deepseek_ocr/
    └── SAM3/
```

`outputs/` 是本機實驗結果資料夾，已加入 `.gitignore`。

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

## 如何放入已跑好的結果

例如你已經跑完 Chandra，輸出在：

```text
chapter/OCR_comparison/outputs/chandra/SAM3/
```

裡面可以包含：

```text
SAM3.md
SAM3.html
SAM3_metadata.json
*.png / *.jpg / *.webp
```

其他工具也照同樣結構放：

```text
chapter/OCR_comparison/outputs/paddleocr/SAM3/
chapter/OCR_comparison/outputs/dots_ocr/SAM3/
chapter/OCR_comparison/outputs/deepseek_ocr/SAM3/
```

檔名不一定要完全相同，評比程式會自動尋找：

```text
*.md
*.html
*metadata*.json
output.json
```

## 統一評比

執行：

```bash
python chapter/OCR_comparison/evaluate_ocr_outputs.py
```

預設會掃描：

```text
chapter/OCR_comparison/outputs/*/SAM3/
```

也可以指定工具：

```bash
python chapter/OCR_comparison/evaluate_ocr_outputs.py \
  --tools chandra paddleocr dots_ocr deepseek_ocr
```

或指定其他 dataset 名稱：

```bash
python chapter/OCR_comparison/evaluate_ocr_outputs.py --dataset SAM3
```

產生：

```text
chapter/OCR_comparison/outputs/SAM3_summary.csv
chapter/OCR_comparison/outputs/SAM3_manual_score.csv
```

## summary.csv

`summary.csv` 是自動統計結果，欄位包含：

```text
tool
dataset
output_dir
markdown_path
html_path
metadata_path
total_characters
markdown_headings
markdown_tables
html_tables
image_references
image_files
figure_caption_count
formula_marker_count
metadata_pages
metadata_chunks
metadata_images
notes
```

這份表適合快速看：

1. 哪個工具有輸出 Markdown / HTML / metadata。
2. 哪個工具輸出比較多結構資訊。
3. 哪個工具有圖片引用或圖片檔。
4. metadata 是否包含頁數、chunk 數、圖片數。

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

建議評分時看這幾頁或區段：

| 頁數 | 觀察重點 |
| --- | --- |
| Page 1 | 標題、作者、abstract |
| Page 2-6 | 圖片、caption、圖文順序 |
| Page 8 | 表格密集頁 |

這些欄位設計參考 `olmOCR-bench` 的文件 OCR split，例如數學公式、頁首頁尾、小字、多欄與表格；也參考 `OCRBench v2` 對文字辨識與文件理解的整體評估方向。

## 評比重點

對 RAG 來說，最終要看：

```text
Markdown 是否乾淨
表格是否可讀
圖片 caption 是否保留
閱讀順序是否合理
metadata 是否足夠追蹤來源
是否容易切 chunk
是否需要大量人工清理
```

所以 `evaluate_ocr_outputs.py` 只做自動統計，真正的品質判斷仍需要搭配 `manual_score.csv` 人工檢查。


### Chandra vLLM 啟動方式

```bash
sudo docker run --name chandra-vllm \
  --runtime nvidia --gpus device=0 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:v0.17.0 \
  --model datalab-to/chandra-ocr-2 \
  --no-enforce-eager \
  --max-num-seqs 64 \
  --dtype bfloat16 \
  --max-model-len 18000 \
  --max_num_batched_tokens 8192 \
  --gpu-memory-utilization .85 \
  --enable-prefix-caching \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}' \
  --served-model-name chandra
```

```bash
uv run chandra \
  data/2025_AI_Agent_Course \
  chapter/end_to_end_RAG/chandra_output \
  --method vllm \
  --max-workers 4
```
