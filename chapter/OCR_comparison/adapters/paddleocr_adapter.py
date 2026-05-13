from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .utils import ensure_dirs, make_failure, write_outputs


def _normalize_old_result(result: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not result:
        return blocks

    for item in result:
        if not item:
            continue
        lines = item if isinstance(item, list) else [item]
        for line in lines:
            try:
                bbox, text_info = line
                text, score = text_info
            except Exception:
                continue
            blocks.append(
                {
                    "text": str(text),
                    "score": float(score) if score is not None else None,
                    "bbox": bbox,
                }
            )
    return blocks


def _normalize_new_result(result: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    items = result if isinstance(result, list) else [result]
    for item in items:
        if isinstance(item, dict):
            texts = item.get("rec_texts") or item.get("texts") or []
            scores = item.get("rec_scores") or []
            boxes = item.get("rec_polys") or item.get("dt_polys") or []
            for index, text in enumerate(texts):
                blocks.append(
                    {
                        "text": str(text),
                        "score": scores[index] if index < len(scores) else None,
                        "bbox": boxes[index].tolist()
                        if index < len(boxes) and hasattr(boxes[index], "tolist")
                        else boxes[index]
                        if index < len(boxes)
                        else None,
                    }
                )
    return blocks


def run_paddleocr(
    page_images: list[Path],
    tool_dir: Path,
    timeout: int | None = None,
) -> dict[str, Any]:
    tool = "paddleocr"
    if importlib.util.find_spec("paddleocr") is None:
        return make_failure(
            tool,
            tool_dir,
            "paddleocr is not installed. Install it on the Linux GPU server before running this adapter.",
        )

    ensure_dirs(tool_dir)

    try:
        from paddleocr import PaddleOCR

        try:
            ocr = PaddleOCR(use_angle_cls=True, lang="en")
        except TypeError:
            ocr = PaddleOCR(lang="en")
        pages: list[dict[str, Any]] = []
        markdown_parts = ["# PaddleOCR Output"]

        for page_index, image_path in enumerate(page_images, start=1):
            try:
                if hasattr(ocr, "predict"):
                    raw_result = ocr.predict(str(image_path))
                    blocks = _normalize_new_result(raw_result)
                else:
                    raw_result = ocr.ocr(str(image_path), cls=True)
                    blocks = _normalize_old_result(raw_result)
            except TypeError:
                raw_result = ocr.ocr(str(image_path))
                blocks = _normalize_old_result(raw_result)

            page_text = "\n".join(block["text"] for block in blocks if block.get("text"))
            pages.append(
                {
                    "page": page_index,
                    "image_path": str(image_path),
                    "blocks": blocks,
                    "text": page_text,
                }
            )
            markdown_parts.append(f"\n\n## Page {page_index}\n\n{page_text}")

        markdown = "\n".join(markdown_parts).strip() + "\n"
        payload = {
            "tool": tool,
            "success": True,
            "pages": pages,
            "error": "",
        }
        paths = write_outputs(tool_dir, markdown, payload)
        return {
            "tool": tool,
            "success": True,
            "markdown": markdown,
            "json_payload": payload,
            **paths,
            "error": "",
        }
    except Exception as exc:
        return make_failure(tool, tool_dir, f"{type(exc).__name__}: {exc}")
