from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import collect_text_outputs, ensure_dirs, make_failure, run_command_template, write_outputs


def run_deepseek_ocr(
    page_images: list[Path],
    tool_dir: Path,
    timeout: int | None = None,
) -> dict[str, Any]:
    tool = "deepseek_ocr"
    raw_dir = ensure_dirs(tool_dir)
    template = os.getenv("DEEPSEEK_OCR_COMMAND")

    if not template:
        return make_failure(
            tool,
            tool_dir,
            "DEEPSEEK_OCR_COMMAND is not set. Example: DEEPSEEK_OCR_COMMAND='python /path/to/DeepSeek-OCR/inference.py --image {input} --output {output_dir}'",
        )

    pages: list[dict[str, Any]] = []
    markdown_parts = ["# DeepSeek-OCR Output"]
    errors: list[str] = []

    for page_index, image_path in enumerate(page_images, start=1):
        page_dir = raw_dir / f"page_{page_index:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_command_template(template, image_path, page_dir, timeout)
        except Exception as exc:
            errors.append(f"page {page_index}: {type(exc).__name__}: {exc}")
            continue

        page_text = collect_text_outputs(page_dir)
        if not page_text:
            page_text = result.stdout.strip()
        if result.stderr.strip():
            (page_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.stdout.strip():
            (page_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")

        if result.returncode != 0:
            errors.append(f"page {page_index}: returncode={result.returncode}; {result.stderr.strip()}")

        pages.append(
            {
                "page": page_index,
                "image_path": str(image_path),
                "returncode": result.returncode,
                "raw_dir": str(page_dir),
                "text": page_text,
            }
        )
        markdown_parts.append(f"\n\n## Page {page_index}\n\n{page_text}")

    success = bool(pages) and not errors
    markdown = "\n".join(markdown_parts).strip() + "\n"
    payload = {
        "tool": tool,
        "success": success,
        "pages": pages,
        "error": "\n".join(errors),
        "command_template": template,
    }
    paths = write_outputs(tool_dir, markdown, payload)
    return {
        "tool": tool,
        "success": success,
        "markdown": markdown,
        "json_payload": payload,
        **paths,
        "error": "\n".join(errors),
    }

