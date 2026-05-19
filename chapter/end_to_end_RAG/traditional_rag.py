import argparse
import os
import ssl
from pathlib import Path
from typing import Any


# 教學環境若遇到公司代理或本機憑證問題，可先關閉 SSL 驗證。
# 正式專案建議改成設定可信任 CA。
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import requests

    requests.packages.urllib3.disable_warnings()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "2025_AI_Agent_Course"
CHANDRA_OUTPUT_DIR = PROJECT_ROOT / "chapter" / "end_to_end_RAG" / "chandra_output"
PROCESSED_DIR = PROJECT_ROOT / "chapter" / "end_to_end_RAG" / "processed"

COLLECTION_NAME = "ai_agent_course_raw"
MILVUS_URI = "http://localhost:19530"
EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 64
TOP_K = 5


def normalize_document_name(path: Path) -> str:
    return path.stem.strip()


def find_chandra_markdown(pdf_path: Path) -> Path | None:
    document_name = normalize_document_name(pdf_path)
    candidate_dir = CHANDRA_OUTPUT_DIR / document_name

    candidates = [
        candidate_dir / "output.md",
        candidate_dir / f"{document_name}.md",
        candidate_dir / "result.md",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if candidate_dir.exists():
        markdown_files = sorted(candidate_dir.glob("*.md"))
        if markdown_files:
            return markdown_files[0]

    return None


def load_from_chandra_markdown(pdf_path: Path) -> Any | None:
    from langchain_core.documents import Document

    markdown_path = find_chandra_markdown(pdf_path)
    if markdown_path is None:
        return None

    text = markdown_path.read_text(encoding="utf-8")
    return Document(
        page_content=text,
        metadata={
            "source": str(pdf_path.relative_to(PROJECT_ROOT)),
            "document_name": normalize_document_name(pdf_path),
            "page_start": 0,
            "page_end": 0,
            "modality": "pdf",
            "parser": "chandra",
            "chandra_markdown": str(markdown_path.relative_to(PROJECT_ROOT)),
        },
    )


def load_from_pdf(pdf_path: Path) -> list[Any]:
    from langchain_core.documents import Document
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    documents: list[Any] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path.relative_to(PROJECT_ROOT)),
                    "document_name": normalize_document_name(pdf_path),
                    "page_start": page_index,
                    "page_end": page_index,
                    "modality": "pdf",
                    "parser": "pypdf",
                    "chandra_markdown": "",
                },
            )
        )

    return documents


def load_documents(source_mode: str) -> list[Any]:
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"找不到 PDF：{DATA_DIR}")

    documents: list[Any] = []
    for pdf_path in pdf_paths:
        chandra_doc = load_from_chandra_markdown(pdf_path)

        if source_mode == "chandra":
            if chandra_doc is None:
                raise FileNotFoundError(
                    "找不到 Chandra Markdown：\n"
                    f"{pdf_path}\n\n"
                    "請先將 Chandra 輸出放到：\n"
                    f"{CHANDRA_OUTPUT_DIR / normalize_document_name(pdf_path)}"
                )
            documents.append(chandra_doc)
            continue

        if source_mode == "auto" and chandra_doc is not None:
            documents.append(chandra_doc)
            continue

        documents.extend(load_from_pdf(pdf_path))

    if not documents:
        raise RuntimeError("沒有載入任何可用文字。請確認 PDF 或 Chandra 輸出。")

    return documents


def split_documents(documents: list[Any]) -> list[Any]:
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
        chunk.metadata["chunk_id"] = f"raw_{index:06d}"
        chunk.metadata["pipeline"] = "traditional_rag"

    return chunks


def get_embedder() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_start", dtype=DataType.INT64),
        FieldSchema(name="page_end", dtype=DataType.INT64),
        FieldSchema(name="parser", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="pipeline", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(
        fields,
        description="Traditional RAG collection for the 2025 AI Agent course",
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
    text = chunk.page_content.strip()

    return {
        "id": metadata["chunk_id"],
        "vector": vector,
        "text": text[:65535],
        "source": metadata.get("source", ""),
        "document_name": metadata.get("document_name", ""),
        "page_start": int(metadata.get("page_start", 0) or 0),
        "page_end": int(metadata.get("page_end", 0) or 0),
        "parser": metadata.get("parser", ""),
        "pipeline": metadata.get("pipeline", "traditional_rag"),
    }


def insert_chunks(
    client: Any,
    embedder: Any,
    chunks: list[Any],
) -> None:
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

    print(f"已寫入 {total_inserted} 個 chunks")


def save_chunks_debug_file(chunks: list[Any]) -> None:
    import json

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "traditional_rag_chunks.jsonl"

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

    print(f"chunks debug 檔已輸出：{output_path.relative_to(PROJECT_ROOT)}")


def ingest(source_mode: str, recreate: bool) -> None:
    from pymilvus import MilvusClient

    documents = load_documents(source_mode=source_mode)
    chunks = split_documents(documents)
    embedder = get_embedder()
    sample_vector = embedder.embed_query("dimension check")

    client = MilvusClient(uri=MILVUS_URI)
    create_collection(client, dim=len(sample_vector), recreate=recreate)
    insert_chunks(client, embedder, chunks)
    create_index(client)
    save_chunks_debug_file(chunks)

    print("\nIngestion 完成")
    print(f"資料來源：{DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"載入文件數：{len(documents)}")
    print(f"chunks 數：{len(chunks)}")
    print(f"Milvus collection：{COLLECTION_NAME}")


def query_milvus(query: str, top_k: int) -> None:
    from pymilvus import MilvusClient

    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(COLLECTION_NAME):
        raise RuntimeError(
            f"找不到 Collection：{COLLECTION_NAME}\n"
            "請先執行 ingest：\n"
            "python chapter/end_to_end_RAG/traditional_rag.py ingest"
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
            "source",
            "document_name",
            "page_start",
            "page_end",
            "parser",
        ],
    )

    print("\n查詢問題：")
    print(query)
    print("\nMilvus 檢索結果：")

    for rank, hit in enumerate(results[0], start=1):
        entity = hit["entity"]
        print(f"\n--- Result {rank} ---")
        print(f"score: {hit['distance']}")
        print(f"source: {entity.get('source')}")
        print(f"document: {entity.get('document_name')}")
        print(f"page: {entity.get('page_start')} - {entity.get('page_end')}")
        print(f"parser: {entity.get('parser')}")
        print("text:")
        print(entity.get("text", "")[:1200])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional RAG baseline for the 2025 AI Agent Course."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build Milvus index.")
    ingest_parser.add_argument(
        "--source-mode",
        choices=["auto", "chandra", "pdf"],
        default="auto",
        help=(
            "auto: prefer Chandra Markdown and fallback to PDF text; "
            "chandra: require Chandra output; pdf: use pypdf directly."
        ),
    )
    ingest_parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Do not drop the existing Milvus collection before ingesting.",
    )

    query_parser = subparsers.add_parser("query", help="Search Milvus index.")
    query_parser.add_argument(
        "query",
        nargs="?",
        default="What is an AI agent?",
        help="Question to search for.",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of chunks to return.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "ingest":
        ingest(
            source_mode=args.source_mode,
            recreate=not args.no_recreate,
        )
    elif args.command == "query":
        query_milvus(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
