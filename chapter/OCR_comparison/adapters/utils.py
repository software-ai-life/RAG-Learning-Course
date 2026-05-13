import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def ensure_dirs(tool_dir: Path) -> Path:
    raw_dir = tool_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def write_outputs(tool_dir: Path, markdown: str, payload: dict[str, Any]) -> dict[str, str]:
    tool_dir.mkdir(parents=True, exist_ok=True)
    md_path = tool_dir / "output.md"
    json_path = tool_dir / "output.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output_md_path": str(md_path),
        "output_json_path": str(json_path),
    }


def make_failure(tool: str, tool_dir: Path, error: str) -> dict[str, Any]:
    paths = write_outputs(
        tool_dir,
        f"# {tool}\n\nFailed to run OCR.\n\n```text\n{error}\n```\n",
        {
            "tool": tool,
            "success": False,
            "pages": [],
            "error": error,
        },
    )
    return {
        "tool": tool,
        "success": False,
        "markdown": "",
        "json_payload": {"tool": tool, "success": False, "error": error},
        **paths,
        "error": error,
    }


def command_exists(command: str) -> bool:
    from shutil import which

    return which(command) is not None


def run_command_template(
    template: str,
    input_path: Path,
    output_dir: Path,
    timeout: int | None,
) -> subprocess.CompletedProcess[str]:
    command = template.format(
        input=str(input_path),
        output=str(output_dir),
        output_dir=str(output_dir),
    )
    return subprocess.run(
        shlex.split(command, posix=os.name != "nt"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def collect_text_outputs(raw_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".html", ".json"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if content:
            rel_path = path.relative_to(raw_dir)
            parts.append(f"## {rel_path}\n\n{content}")
    return "\n\n".join(parts)

