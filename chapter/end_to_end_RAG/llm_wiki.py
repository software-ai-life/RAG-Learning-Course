import argparse
import json
import os
import re
import ssl
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests

    requests.packages.urllib3.disable_warnings()
except ImportError:
    pass


# 教學環境若遇到公司代理或本機憑證問題，可先關閉 SSL 驗證。
# 正式專案建議改成設定可信任 CA。
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "2025_AI_Agent_Course"
LLM_WIKI_DIR = PROJECT_ROOT / "data" / "end_to_end_RAG" / "llm_wiki"
WIKI_DIR = LLM_WIKI_DIR / "wiki"
WIKI_PROCESSED_DIR = LLM_WIKI_DIR / "processed"

COLLECTION_NAME = "ai_agent_course_wiki"
MILVUS_URI = "http://localhost:19530"
EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
BATCH_SIZE = 64
TOP_K = 5
MAX_SOURCE_CHARS = 18000


def slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "untitled"


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def ensure_wiki_dirs() -> None:
    for path in [
        WIKI_DIR,
        WIKI_DIR / "sources",
        WIKI_DIR / "concepts",
        WIKI_DIR / "comparisons",
        WIKI_DIR / "questions",
        WIKI_PROCESSED_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 GEMINI_API_KEY。請在專案根目錄建立 .env，並加入：\n"
            "GEMINI_API_KEY=your_gemini_api_key_here"
        )

    return api_key


def get_gemini_client() -> Any:
    from google import genai

    return genai.Client(api_key=get_api_key())


def call_llm(prompt: str, model: str) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    text = response.text or ""
    if not text.strip():
        raise RuntimeError("Gemini 沒有回傳文字內容。")

    return strip_markdown_fence(text.strip())


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown") :].strip()
    elif text.startswith("```"):
        text = text[len("```") :].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def load_course_documents(source_mode: str) -> list[Any]:
    import traditional_rag

    return traditional_rag.load_documents(source_mode=source_mode)


def group_documents_by_source(documents: list[Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for document in documents:
        metadata = document.metadata
        document_name = metadata.get("document_name", "Untitled")
        source = metadata.get("source", "")
        entry = grouped.setdefault(
            document_name,
            {
                "document_name": document_name,
                "source": source,
                "parser": metadata.get("parser", ""),
                "pages": [],
                "texts": [],
            },
        )
        entry["pages"].append(metadata.get("page_start", 0))
        entry["texts"].append(document.page_content)

    return grouped


def make_source_prompt(source: dict[str, Any]) -> str:
    document_name = source["document_name"]
    source_path = source["source"]
    parser = source["parser"]
    text = "\n\n".join(source["texts"])[:MAX_SOURCE_CHARS]

    return f"""You are maintaining an LLM Wiki for an AI Agent course.

Create a source wiki page in Markdown for the course document below.

Requirements:
- Write in Traditional Chinese.
- Preserve the original English technical terms when useful.
- Do not invent facts that are not supported by the document.
- Include source references in the "Source Notes" section.
- Output only Markdown.

Use this exact structure:

---
title: {document_name}
type: source
sources:
  - {source_path}
tags:
  - ai-agent
  - course
updated: {today()}
---

# {document_name}

## Summary

## Key Ideas

## Important Terms

## Related Concepts

## Source Notes

Document metadata:
- source: {source_path}
- parser: {parser}

Document text:

{text}
"""


def make_concept_prompt(source_pages: list[Path]) -> str:
    source_texts = []
    for path in source_pages:
        source_texts.append(f"## {path.name}\n\n{path.read_text(encoding='utf-8')}")

    joined_text = "\n\n".join(source_texts)[:MAX_SOURCE_CHARS]

    return f"""You are maintaining an LLM Wiki for an AI Agent course.

Based on the source wiki pages below, create 5 to 8 concept wiki pages.

Output format:
- Return a JSON array.
- Each item must have: path, title, tags, source_refs, markdown.
- path must start with "concepts/" and end with ".md".
- markdown must be a complete Markdown page with YAML frontmatter.
- Write markdown content in Traditional Chinese.
- Preserve important English technical terms.
- Do not invent facts beyond the provided source pages.

Each concept page should use this structure:

---
title: ...
type: concept
sources:
  - ...
tags:
  - ...
updated: {today()}
---

# ...

## Summary

## Key Ideas

## Why It Matters

## Related Concepts

## Source Notes

Source wiki pages:

{joined_text}
"""


def parse_json_array(text: str) -> list[dict[str, Any]]:
    text = strip_markdown_fence(text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("LLM 回傳內容不是 JSON array。")

    return json.loads(text[start : end + 1])


def write_source_pages(model: str, source_mode: str, limit: int | None) -> list[Path]:
    documents = load_course_documents(source_mode=source_mode)
    grouped_sources = list(group_documents_by_source(documents).values())
    if limit is not None:
        grouped_sources = grouped_sources[:limit]

    output_paths: list[Path] = []
    for source in grouped_sources:
        slug = slugify(source["document_name"])
        output_path = WIKI_DIR / "sources" / f"{slug}.md"
        prompt = make_source_prompt(source)
        markdown = call_llm(prompt, model=model)
        output_path.write_text(markdown + "\n", encoding="utf-8")
        output_paths.append(output_path)
        print(f"已建立 source page：{output_path.relative_to(PROJECT_ROOT)}")

    return output_paths


def write_concept_pages(model: str, source_pages: list[Path]) -> list[Path]:
    prompt = make_concept_prompt(source_pages)
    response = call_llm(prompt, model=model)
    concept_items = parse_json_array(response)

    output_paths: list[Path] = []
    for item in concept_items:
        relative_path = Path(item["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"不安全的 concept path：{relative_path}")

        output_path = WIKI_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        markdown = item.get("markdown", "").strip()
        if not markdown:
            continue

        output_path.write_text(markdown + "\n", encoding="utf-8")
        output_paths.append(output_path)
        print(f"已建立 concept page：{output_path.relative_to(PROJECT_ROOT)}")

    return output_paths


def extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def first_summary_line(markdown: str) -> str:
    lines = markdown.splitlines()
    in_summary = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Summary":
            in_summary = True
            continue
        if in_summary and stripped.startswith("## "):
            break
        if in_summary and stripped and not stripped.startswith("-"):
            return stripped
    return ""


def write_index(source_pages: list[Path], concept_pages: list[Path]) -> None:
    lines = [
        "# AI Agent Course Wiki Index",
        "",
        "## Sources",
        "",
    ]

    for path in sorted(source_pages):
        markdown = path.read_text(encoding="utf-8")
        title = extract_title(markdown, path.stem)
        summary = first_summary_line(markdown)
        rel_path = path.relative_to(WIKI_DIR).as_posix()
        lines.append(f"- [{title}](./{rel_path})：{summary}")

    lines.extend(["", "## Concepts", ""])

    for path in sorted(concept_pages):
        markdown = path.read_text(encoding="utf-8")
        title = extract_title(markdown, path.stem)
        summary = first_summary_line(markdown)
        rel_path = path.relative_to(WIKI_DIR).as_posix()
        lines.append(f"- [{title}](./{rel_path})：{summary}")

    (WIKI_DIR / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已更新：{(WIKI_DIR / 'index.md').relative_to(PROJECT_ROOT)}")


def write_overview(model: str, source_pages: list[Path], concept_pages: list[Path]) -> None:
    page_text = []
    for path in source_pages + concept_pages:
        page_text.append(path.read_text(encoding="utf-8"))

    prompt = f"""Create a course overview page for an AI Agent course wiki.

Write in Traditional Chinese. Preserve important English technical terms.
Use this Markdown structure:

---
title: AI Agent Course Overview
type: overview
sources:
  - data/2025_AI_Agent_Course
tags:
  - ai-agent
  - course
updated: {today()}
---

# AI Agent Course Overview

## Summary

## Learning Map

## Core Concepts

## How the Source Materials Connect

## Source Notes

Wiki pages:

{chr(10).join(page_text)[:MAX_SOURCE_CHARS]}
"""

    markdown = call_llm(prompt, model=model)
    output_path = WIKI_DIR / "overview.md"
    output_path.write_text(markdown + "\n", encoding="utf-8")
    print(f"已更新：{output_path.relative_to(PROJECT_ROOT)}")


def append_log(source_pages: list[Path], concept_pages: list[Path]) -> None:
    log_path = WIKI_DIR / "log.md"
    if not log_path.exists():
        log_path.write_text("# Wiki Log\n", encoding="utf-8")

    lines = [
        "",
        f"## [{today()}] build-wiki | AI Agent Course",
        "",
        "### Source pages",
        "",
    ]
    lines.extend(
        f"- {path.relative_to(WIKI_DIR).as_posix()}"
        for path in source_pages
    )
    lines.extend(["", "### Concept pages", ""])
    lines.extend(
        f"- {path.relative_to(WIKI_DIR).as_posix()}"
        for path in concept_pages
    )

    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"已更新：{log_path.relative_to(PROJECT_ROOT)}")


def build_wiki(model: str, source_mode: str, limit: int | None) -> None:
    ensure_wiki_dirs()
    source_pages = write_source_pages(model=model, source_mode=source_mode, limit=limit)
    concept_pages = write_concept_pages(model=model, source_pages=source_pages)
    write_overview(model=model, source_pages=source_pages, concept_pages=concept_pages)
    write_index(source_pages=source_pages, concept_pages=concept_pages)
    append_log(source_pages=source_pages, concept_pages=concept_pages)

    print("\nLLM Wiki 建立完成")
    print(f"Wiki 位置：{WIKI_DIR.relative_to(PROJECT_ROOT)}")


def get_embedder() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    if not markdown.startswith("---"):
        return {}, markdown

    end = markdown.find("\n---", 3)
    if end == -1:
        return {}, markdown

    raw_frontmatter = markdown[3:end].strip()
    body = markdown[end + 4 :].strip()
    metadata: dict[str, Any] = {}
    current_key: str | None = None

    for line in raw_frontmatter.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            metadata.setdefault(current_key, []).append(line[4:].strip())
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            if value:
                metadata[key] = value
            else:
                metadata[key] = []

    return metadata, body


def load_wiki_documents() -> list[Any]:
    from langchain_core.documents import Document

    if not WIKI_DIR.exists():
        raise FileNotFoundError(
            f"找不到 wiki 目錄：{WIKI_DIR}\n"
            "請先執行：python chapter/end_to_end_RAG/llm_wiki.py build-wiki"
        )

    documents: list[Any] = []
    for path in sorted(WIKI_DIR.rglob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(markdown)
        wiki_path = path.relative_to(WIKI_DIR).as_posix()
        page_type = str(frontmatter.get("type", "wiki"))
        title = str(frontmatter.get("title") or extract_title(body, path.stem))
        sources = frontmatter.get("sources", [])
        tags = frontmatter.get("tags", [])

        documents.append(
            Document(
                page_content=body,
                metadata={
                    "wiki_path": wiki_path,
                    "page_type": page_type,
                    "title": title,
                    "source_refs": json.dumps(sources, ensure_ascii=False),
                    "tags": json.dumps(tags, ensure_ascii=False),
                    "updated": str(frontmatter.get("updated", "")),
                    "pipeline": "llm_wiki",
                },
            )
        )

    if not documents:
        raise RuntimeError(f"{WIKI_DIR} 沒有任何 Markdown wiki pages。")

    return documents


def split_wiki_documents(documents: list[Any]) -> list[Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            "。",
            "，",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"wiki_{index:06d}"

    return chunks


def create_collection(client: Any, dim: int, recreate: bool) -> None:
    from pymilvus import CollectionSchema, DataType, FieldSchema

    if client.has_collection(COLLECTION_NAME):
        if not recreate:
            print(f"Collection 已存在，沿用：{COLLECTION_NAME}")
            return

        client.drop_collection(COLLECTION_NAME)
        print(f"已刪除既有 Collection：{COLLECTION_NAME}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="wiki_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="source_refs", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="updated", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="pipeline", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(
        fields,
        description="LLM Wiki collection for the 2025 AI Agent course",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )
    print(f"已建立 Collection：{COLLECTION_NAME}")


def create_index(client: Any) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )

    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )
    client.load_collection(collection_name=COLLECTION_NAME)
    print("已建立 HNSW index，並載入 Collection")


def chunk_to_record(chunk: Any, vector: list[float]) -> dict[str, Any]:
    metadata = chunk.metadata
    return {
        "id": metadata["chunk_id"],
        "vector": vector,
        "text": chunk.page_content.strip()[:65535],
        "wiki_path": metadata.get("wiki_path", ""),
        "page_type": metadata.get("page_type", ""),
        "title": metadata.get("title", ""),
        "source_refs": metadata.get("source_refs", "[]")[:4096],
        "tags": metadata.get("tags", "[]")[:2048],
        "updated": metadata.get("updated", ""),
        "pipeline": metadata.get("pipeline", "llm_wiki"),
    }


def insert_chunks(client: Any, embedder: Any, chunks: list[Any]) -> None:
    from tqdm import tqdm

    total_inserted = 0

    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="寫入 Milvus"):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]
        vectors = embedder.embed_documents(texts)
        records = [
            chunk_to_record(chunk, vector)
            for chunk, vector in zip(batch, vectors)
        ]
        result = client.insert(collection_name=COLLECTION_NAME, data=records)
        total_inserted += result["insert_count"]

    print(f"已寫入 {total_inserted} 個 wiki chunks")


def save_chunks_debug_file(chunks: list[Any]) -> None:
    WIKI_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WIKI_PROCESSED_DIR / "wiki_chunks.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wiki chunks debug 檔已輸出：{output_path.relative_to(PROJECT_ROOT)}")


def index_wiki(recreate: bool) -> None:
    from pymilvus import MilvusClient

    documents = load_wiki_documents()
    chunks = split_wiki_documents(documents)
    embedder = get_embedder()
    sample_vector = embedder.embed_query("dimension check")

    client = MilvusClient(uri=MILVUS_URI)
    create_collection(client, dim=len(sample_vector), recreate=recreate)
    insert_chunks(client, embedder, chunks)
    create_index(client)
    save_chunks_debug_file(chunks)

    print("\nLLM Wiki indexing 完成")
    print(f"Wiki pages：{len(documents)}")
    print(f"Wiki chunks：{len(chunks)}")
    print(f"Milvus collection：{COLLECTION_NAME}")


def query_wiki(query: str, top_k: int) -> None:
    from pymilvus import MilvusClient

    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(COLLECTION_NAME):
        raise RuntimeError(
            f"找不到 Collection：{COLLECTION_NAME}\n"
            "請先執行：python chapter/end_to_end_RAG/llm_wiki.py index"
        )

    client.load_collection(collection_name=COLLECTION_NAME)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=[
            "text",
            "wiki_path",
            "page_type",
            "title",
            "source_refs",
            "tags",
            "updated",
        ],
    )

    print("\n查詢問題：")
    print(query)
    print("\nLLM Wiki 檢索結果：")

    for rank, hit in enumerate(results[0], start=1):
        entity = hit["entity"]
        print(f"\n--- Result {rank} ---")
        print(f"score: {hit['distance']}")
        print(f"title: {entity.get('title')}")
        print(f"wiki_path: {entity.get('wiki_path')}")
        print(f"page_type: {entity.get('page_type')}")
        print(f"source_refs: {entity.get('source_refs')}")
        print("text:")
        print(entity.get("text", "")[:1200])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Wiki pipeline for the 2025 AI Agent Course."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-wiki", help="Generate Markdown wiki.")
    build_parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help="Gemini model used to generate wiki pages.",
    )
    build_parser.add_argument(
        "--source-mode",
        choices=["auto", "chandra", "pdf"],
        default="auto",
        help=(
            "auto: prefer Chandra Markdown and fallback to PDF text; "
            "chandra: require Chandra output; pdf: use pypdf directly."
        ),
    )
    build_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N source documents for testing.",
    )

    index_parser = subparsers.add_parser("index", help="Index wiki pages into Milvus.")
    index_parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Do not drop the existing Milvus collection before indexing.",
    )

    query_parser = subparsers.add_parser("query", help="Search wiki Milvus collection.")
    query_parser.add_argument(
        "query",
        nargs="?",
        default="How is agent memory related to context engineering?",
        help="Question to search for.",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of wiki chunks to return.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build-wiki":
        build_wiki(
            model=args.model,
            source_mode=args.source_mode,
            limit=args.limit,
        )
    elif args.command == "index":
        index_wiki(recreate=not args.no_recreate)
    elif args.command == "query":
        query_wiki(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
