from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any

from adapters import ADAPTERS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "data" / "C2" / "pdf" / "SAM3.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "OCR"
DEFAULT_TOOLS = ["paddleocr", "chandra", "dots_ocr", "deepseek_ocr"]
SUMMARY_COLUMNS = [
    "tool",
    "pdf",
    "pages",
    "success",
    "runtime_seconds",
    "output_md_path",
    "output_json_path",
    "total_characters",
    "markdown_headings",
    "markdown_tables",
    "image_caption_count",
    "empty_pages",
    "error",
]
MANUAL_SCORE_COLUMNS = [
    "tool",
    "text_accuracy_1_to_5",
    "reading_order_1_to_5",
    "table_quality_1_to_5",
    "formula_quality_1_to_5",
    "header_footer_handling_1_to_5",
    "tiny_text_quality_1_to_5",
    "multi_column_order_1_to_5",
    "figure_caption_quality_1_to_5",
    "markdown_usability_1_to_5",
    "rag_readiness_1_to_5",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OCR / VLM OCR tools on the same PDF pages."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF file to benchmark. Default: data/C2/pdf/SAM3.pdf",
    )
    parser.add_argument(
        "--pages",
        default="1-5",
        help="Pages to process, e.g. 1-5, 1,3,8, or all. Pages are 1-based.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=DEFAULT_TOOLS,
        choices=sorted(ADAPTERS.keys()),
        help="OCR tools to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Benchmark output root. Default: data/OCR",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI used when rendering PDF pages to PNG.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each page-level external OCR command.",
    )
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Re-render page images even if PNG files already exist.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_pdf_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to count PDF pages.") from exc

    return len(PdfReader(str(pdf_path)).pages)


def parse_pages(page_spec: str, total_pages: int) -> list[int]:
    spec = page_spec.strip().lower()
    if spec == "all":
        return list(range(1, total_pages + 1))

    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    invalid = [page for page in pages if page < 1 or page > total_pages]
    if invalid:
        raise ValueError(f"Page out of range: {invalid}; PDF has {total_pages} pages.")
    return sorted(pages)


def render_pdf_pages(
    pdf_path: Path,
    pages: list[int],
    pages_dir: Path,
    dpi: int,
    force_render: bool,
) -> list[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_images: list[Path] = []

    if importlib.util.find_spec("fitz") is None:
        raise RuntimeError(
            "PyMuPDF is required to render PDF pages. Install it with: pip install pymupdf"
        )

    import fitz

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as document:
        for page_number in pages:
            output_path = pages_dir / f"page_{page_number:03d}.png"
            if output_path.exists() and not force_render:
                page_images.append(output_path)
                continue

            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(output_path)
            page_images.append(output_path)

    return page_images


def count_markdown_tables(markdown: str) -> int:
    table_lines = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines += 1
    return table_lines


def count_image_captions(markdown: str) -> int:
    pattern = re.compile(r"\b(figure|fig\.|image|caption|圖|圖片)\b", re.IGNORECASE)
    return len(pattern.findall(markdown))


def count_empty_pages(payload: dict[str, Any]) -> int:
    pages = payload.get("pages", [])
    empty_pages = 0
    for page in pages:
        text = str(page.get("text", "")).strip()
        blocks = page.get("blocks", [])
        if not text and not blocks:
            empty_pages += 1
    return empty_pages


def make_summary_row(
    tool: str,
    pdf_path: Path,
    page_spec: str,
    runtime_seconds: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    markdown = result.get("markdown", "") or ""
    payload = result.get("json_payload", {}) or {}
    return {
        "tool": tool,
        "pdf": str(pdf_path),
        "pages": page_spec,
        "success": result.get("success", False),
        "runtime_seconds": f"{runtime_seconds:.2f}",
        "output_md_path": result.get("output_md_path", ""),
        "output_json_path": result.get("output_json_path", ""),
        "total_characters": len(markdown),
        "markdown_headings": sum(1 for line in markdown.splitlines() if line.startswith("#")),
        "markdown_tables": count_markdown_tables(markdown),
        "image_caption_count": count_image_captions(markdown),
        "empty_pages": count_empty_pages(payload),
        "error": result.get("error", ""),
    }


def write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_manual_score_template(path: Path, tools: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANUAL_SCORE_COLUMNS)
        writer.writeheader()
        for tool in tools:
            writer.writerow({column: "" for column in MANUAL_SCORE_COLUMNS} | {"tool": tool})


def main() -> int:
    args = parse_args()
    pdf_path = resolve_path(args.pdf)
    output_root = resolve_path(args.output_dir)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    total_pages = get_pdf_page_count(pdf_path)
    pages = parse_pages(args.pages, total_pages)
    benchmark_dir = output_root / pdf_path.stem
    pages_dir = benchmark_dir / "pages"

    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Benchmark pages: {pages}")
    print(f"Output: {benchmark_dir}")

    page_images = render_pdf_pages(
        pdf_path=pdf_path,
        pages=pages,
        pages_dir=pages_dir,
        dpi=args.dpi,
        force_render=args.force_render,
    )
    print(f"Rendered pages: {len(page_images)}")

    summary_rows: list[dict[str, Any]] = []
    for tool in args.tools:
        adapter = ADAPTERS[tool]
        tool_dir = benchmark_dir / tool
        print(f"\nRunning {tool}...")
        started_at = time.perf_counter()
        result = adapter(page_images=page_images, tool_dir=tool_dir, timeout=args.timeout)
        runtime_seconds = time.perf_counter() - started_at
        summary_rows.append(
            make_summary_row(
                tool=tool,
                pdf_path=pdf_path,
                page_spec=args.pages,
                runtime_seconds=runtime_seconds,
                result=result,
            )
        )
        status = "success" if result.get("success") else "failed"
        print(f"{tool}: {status} ({runtime_seconds:.2f}s)")

    write_summary(benchmark_dir / "summary.csv", summary_rows)
    write_manual_score_template(benchmark_dir / "manual_score.csv", args.tools)
    print(f"\nSummary: {benchmark_dir / 'summary.csv'}")
    print(f"Manual score template: {benchmark_dir / 'manual_score.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
