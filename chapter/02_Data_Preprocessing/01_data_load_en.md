# Section 1: Data Loading

Data loading is the first step in a RAG pipeline. It may look like simply reading files, but it directly affects later chunking, embedding, retrieval, and final LLM answer quality.

If the extracted text is wrong, out of order, noisy, or missing metadata, later stages are hard to fix. In RAG, the common data engineering rule still applies:

> Garbage in, garbage out.

This section introduces the role of data loaders, then uses `Unstructured` to parse PDF files and explains important parameters and common issues.

## 1. What Is a Data Loader?

RAG systems may need to process many file formats:

| Type | Common formats |
| --- | --- |
| Text documents | `.txt`, `.md` |
| Office documents | `.docx`, `.pptx`, `.xlsx` |
| Web data | `.html`, URL |
| PDF | text-based PDF, scanned PDF, papers, reports |
| Structured data | `.csv`, `.json` |

A data loader converts these different formats into a standard structure that the program can process.

In a RAG pipeline, a data loader usually needs to:

1. Read the original file.
2. Extract content such as titles, paragraphs, tables, and lists.
3. Preserve metadata such as file name, page number, element type, and source path.

## 2. Common Document Loading Tools

Different tools focus on different file types and parsing quality.

| Tool | Feature | Best for |
| --- | --- | --- |
| `TextLoader` | Lightweight plain text loading | `.txt`, simple text data |
| `DirectoryLoader` | Batch loading from folders | Multi-file document collections |
| `Unstructured` | Supports many unstructured document formats | PDF, Word, HTML, Markdown, mixed documents |
| `MarkItDown` | Converts many formats into Markdown | Office files, PDF, HTML, Markdown-first workflows |
| `PyMuPDF` / `PyMuPDF4LLM` | Fast PDF text extraction | Text-based PDF, technical documents |
| `LlamaParse` | Strong PDF structure parsing | Papers, contracts, table-heavy files |
| `Docling` | Enterprise-oriented document parsing | Reports, document conversion, structured output |
| `Marker` | PDF to Markdown conversion | Books, papers, long PDFs |
| `MinerU` | Multimodal document parsing | Academic PDFs, tables, formulas, complex layouts |
| `Chandra` | OCR / Document Intelligence model that can output Markdown, HTML, and JSON | Scanned PDF, complex tables, forms, handwriting, complex layouts |

This section uses `Unstructured` because it provides a unified `partition(...)` interface and is convenient for learning RAG preprocessing.

## 3. Unstructured Overview

`Unstructured` is a toolkit for parsing unstructured documents. Its core idea is **partitioning**: splitting a file into document elements.

Common element types include:

| Element type | Description |
| --- | --- |
| `Title` | Title |
| `NarrativeText` | Normal paragraph text |
| `ListItem` | List item |
| `Table` | Table |
| `Image` | Image-related information |
| `FigureCaption` | Figure caption |
| `Header` | Page header |
| `Footer` | Page footer |
| `PageBreak` | Page break |
| `PageNumber` | Page number |
| `UncategorizedText` | Text that is not classified |
| `CompositeElement` | Combined element after chunking |

Each element usually contains:

| Field | Description |
| --- | --- |
| `category` | Element type, such as `Title` or `NarrativeText` |
| `text` / `str(element)` | Text content |
| `metadata` | Page number, file name, source path, and other information |

For RAG, these elements become the basis for later cleaning, chunking, and indexing.

## 4. Code Example

Example file:

```text
chapter/02_Data_Preprocessing/unstructured_example.py
```

Example:

```python
from collections import Counter

from unstructured.partition.auto import partition


pdf_path = "../../data/C2/pdf/Medium.pdf"

elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
)

total_characters = sum(len(str(element)) for element in elements)
print(f"Parsed {len(elements)} elements, {total_characters} characters in total")

types = Counter(element.category for element in elements)
print(f"Element type counts: {dict(types)}")
```

This code:

1. Sets the PDF file path.
2. Parses the PDF with `partition(...)`.
3. Counts elements and total characters.
4. Prints element categories.

## 5. Important `partition(...)` Parameters

`partition(...)` is the main entry point in Unstructured. It automatically routes files to the appropriate parser.

### `filename`

```python
filename=pdf_path
```

Specifies the local file path to parse.

### `content_type`

```python
content_type="application/pdf"
```

Specifies the MIME type. This helps Unstructured identify the file type.

| File | `content_type` |
| --- | --- |
| PDF | `application/pdf` |
| HTML | `text/html` |
| Markdown | `text/markdown` |
| Plain text | `text/plain` |

### `strategy`

```python
strategy="fast"
```

`strategy` controls how PDFs or images are parsed.

| Strategy | Description | Best for |
| --- | --- | --- |
| `auto` | Lets Unstructured choose automatically | Unknown document type |
| `fast` | Fast text extraction | PDFs with selectable text |
| `ocr_only` | OCR-based extraction | Scanned PDFs, image-only files |
| `hi_res` | Layout-aware parsing | PDFs with tables, images, complex layout |

### `include_page_breaks`

```python
include_page_breaks=True
```

Controls whether `PageBreak` elements are included.

### `encoding`

```python
encoding="utf-8"
```

Specifies text encoding. This is useful when plain text files show garbled characters.

### `languages`

```python
languages=["eng", "chi_tra"]
```

Specifies OCR languages. This matters when using OCR strategies.

## 6. Example Output

The script first prints summary statistics:

```text
Parsed 42 elements, 12345 characters in total
```

Then it prints element type counts:

```text
Element type counts: {'Title': 5, 'NarrativeText': 30, 'ListItem': 7}
```

This helps you quickly check whether parsing looks reasonable.

If the result has 0 elements, possible causes include:

1. The PDF has no extractable text layer.
2. OCR is required.
3. Poppler or Tesseract is not installed.
4. The file path is wrong.

## 7. Using Parsed Results in RAG

`partition(...)` returns elements. These are not yet the final vector database.

A typical flow is:

```text
PDF
  ↓
Unstructured elements
  ↓
text cleaning
  ↓
chunking
  ↓
embedding
  ↓
vector store
  ↓
retrieval
```

You can decide how to process content based on element type:

| Type | Suggested handling |
| --- | --- |
| `Title` | Usually keep; useful as section context |
| `NarrativeText` | Usually keep; main content for RAG |
| `ListItem` | Usually keep; useful for steps and specifications |
| `Table` | Convert to Markdown table or HTML if needed |
| `Header` / `Footer` | Often remove to avoid retrieval noise |
| `PageNumber` | Usually keep as metadata, not body text |

## 8. Exercises

### Exercise 1: Use `strategy="fast"`

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="fast",
)
```

Check whether the extracted text is normal.

### Exercise 2: Include page breaks

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    include_page_breaks=True,
)
```

Check whether `PageBreak` appears.

### Exercise 3: Try OCR

If Poppler and Tesseract are installed:

```python
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="ocr_only",
    languages=["eng"],
)
```

For Traditional Chinese documents, you may need:

```python
languages=["eng", "chi_tra"]
```

### Exercise 4: Compare Different PDFs

Try both:

```python
pdf_path = "../../data/C2/pdf/Medium.pdf"
```

and:

```python
pdf_path = "../../data/C2/pdf/SAM3.pdf"
```

Compare:

1. Which PDF returns more elements?
2. Which one needs OCR?
3. Are the element types reasonable?
4. Which one is more suitable for RAG?

## Summary

Data loading determines what the RAG system can see. Poor loading quality limits the quality of retrieval and generation.

This section used `Unstructured` to parse PDFs and inspect document elements. The next steps are usually cleaning, chunking, and embedding.

## References

- [Unstructured Partitioning](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- [Unstructured Document Elements](https://docs.unstructured.io/ui/document-elements)
- [Unstructured Strategies](https://unstructured.readthedocs.io/en/main/best_practices/strategies.html)
