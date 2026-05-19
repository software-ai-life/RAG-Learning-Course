import argparse
import os
import ssl
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


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
DATA_DIR = PROJECT_ROOT / "data" / "C4" / "hybrid_search"

COLLECTION_NAME = "hybrid_search_course_demo"
MILVUS_URI = "http://localhost:19530"

EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 3
DEFAULT_RRF_K = 60

DEFAULT_QUERIES = [
    "How does MCP help agent tool interoperability?",
    "ERR-4291",
    "MetadataReplacementPostProcessor",
    "How is agent memory different from context engineering?",
    "Why does dense retrieval miss exact error codes?",
]


def load_markdown_documents() -> list[Document]:
    markdown_paths = sorted(DATA_DIR.glob("*.md"))
    if not markdown_paths:
        raise FileNotFoundError(
            f"找不到 Hybrid Search 範例資料：{DATA_DIR}\n"
            "請先建立 data/C4/hybrid_search/*.md"
        )

    documents: list[Document] = []
    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        title = markdown_path.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(markdown_path.relative_to(PROJECT_ROOT)),
                    "title": title,
                },
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
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
            " ",
            "",
        ],
    )
    split_docs = splitter.split_documents(documents)

    chunks: list[Document] = []
    for index, document in enumerate(split_docs):
        document.page_content = document.page_content.strip()
        document.metadata["chunk_id"] = f"chunk_{index:04d}"
        chunks.append(document)

    return chunks


def sparse_row_to_dict(row: Any) -> dict[int, float]:
    coo = row.tocoo()
    return {
        int(index): float(value)
        for index, value in zip(coo.col, coo.data)
        if float(value) != 0.0
    }


def build_embedding_function(device: str | None) -> BGEM3EmbeddingFunction:
    return BGEM3EmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=device,
        normalize_embeddings=True,
        use_fp16=False,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )


def create_collection(
    client: MilvusClient,
    dense_dim: int,
    drop_existing: bool,
) -> bool:
    if client.has_collection(COLLECTION_NAME):
        if not drop_existing:
            print(f"Collection 已存在，將重用：{COLLECTION_NAME}")
            client.load_collection(collection_name=COLLECTION_NAME)
            return False

        client.drop_collection(COLLECTION_NAME)
        print(f"已刪除既有 Collection：{COLLECTION_NAME}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=64,
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=4096,
        ),
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=dense_dim,
        ),
        FieldSchema(
            name="sparse_vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Hybrid Search demo with BGE-M3 dense and sparse vectors",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )
    print(f"已建立 Collection：{COLLECTION_NAME}")
    return True


def create_indexes(client: MilvusClient) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": 16,
            "efConstruction": 256,
        },
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={
            "drop_ratio_build": 0.0,
        },
    )

    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )
    client.load_collection(collection_name=COLLECTION_NAME)
    print("已建立 dense / sparse indexes，並將 Collection 載入記憶體")


def insert_chunks(
    client: MilvusClient,
    chunks: list[Document],
    dense_vectors: list[list[float]],
    sparse_vectors: Any,
) -> None:
    rows = []
    for index, (chunk, dense_vector, sparse_vector) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        rows.append(
            {
                "id": index,
                "chunk_id": chunk.metadata["chunk_id"],
                "source": chunk.metadata["source"],
                "title": chunk.metadata["title"],
                "content": chunk.page_content,
                "dense_vector": [float(value) for value in dense_vector],
                "sparse_vector": sparse_row_to_dict(sparse_vector),
            }
        )

    result = client.insert(
        collection_name=COLLECTION_NAME,
        data=rows,
    )
    print(f"已插入 {result['insert_count']} 筆 chunks")


def build_milvus_index(
    milvus_uri: str,
    drop_existing: bool,
    device: str | None,
) -> tuple[MilvusClient, BGEM3EmbeddingFunction, list[Document]]:
    documents = load_markdown_documents()
    chunks = split_documents(documents)

    print(f"已載入 {len(documents)} 份 Markdown 文件")
    print(f"已切成 {len(chunks)} 個 chunks")

    print(f"初始化 BGE-M3 embedding function：{EMBEDDING_MODEL}")
    embedding_function = build_embedding_function(device=device)

    print("使用 BGE-M3 產生 dense / sparse vectors")
    embeddings = embedding_function.encode_documents(
        [chunk.page_content for chunk in chunks]
    )
    dense_vectors = embeddings["dense"]
    sparse_vectors = embeddings["sparse"]

    client = MilvusClient(uri=milvus_uri)
    created_collection = create_collection(
        client=client,
        dense_dim=embedding_function.dim["dense"],
        drop_existing=drop_existing,
    )
    if created_collection:
        insert_chunks(client, chunks, dense_vectors, sparse_vectors)
        create_indexes(client)

    return client, embedding_function, chunks


def search_dense(
    client: MilvusClient,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    return client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="dense_vector",
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": 128},
        },
        limit=top_k,
        output_fields=["chunk_id", "source", "title", "content"],
    )[0]


def search_sparse(
    client: MilvusClient,
    sparse_query_vector: dict[int, float],
    top_k: int,
) -> list[dict[str, Any]]:
    return client.search(
        collection_name=COLLECTION_NAME,
        data=[sparse_query_vector],
        anns_field="sparse_vector",
        search_params={
            "metric_type": "IP",
            "params": {"drop_ratio_search": 0.0},
        },
        limit=top_k,
        output_fields=["chunk_id", "source", "title", "content"],
    )[0]


def search_hybrid(
    client: MilvusClient,
    query_vector: list[float],
    sparse_query_vector: dict[int, float],
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    dense_request = AnnSearchRequest(
        data=[query_vector],
        anns_field="dense_vector",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 128},
        },
        limit=top_k,
    )
    sparse_request = AnnSearchRequest(
        data=[sparse_query_vector],
        anns_field="sparse_vector",
        param={
            "metric_type": "IP",
            "params": {"drop_ratio_search": 0.0},
        },
        limit=top_k,
    )

    return client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[sparse_request, dense_request],
        ranker=RRFRanker(k=rrf_k),
        limit=top_k,
        output_fields=["chunk_id", "source", "title", "content"],
    )[0]


def print_results(label: str, results: list[dict[str, Any]]) -> None:
    print(f"\n## {label}")
    if not results:
        print("沒有找到結果。")
        return

    for rank, hit in enumerate(results, start=1):
        entity = hit.get("entity", {})
        content = entity.get("content", "")
        preview = " ".join(content.split())
        if len(preview) > 280:
            preview = preview[:280] + "..."

        print(f"\n[{rank}]")
        print(f"score: {hit.get('distance')}")
        print(f"source: {entity.get('source')}")
        print(f"title: {entity.get('title')}")
        print(f"chunk_id: {entity.get('chunk_id')}")
        print(f"text: {preview}")


def run_demo(
    query: str,
    top_k: int,
    rrf_k: int,
    milvus_uri: str,
    drop_existing: bool,
    keep_collection: bool,
    device: str | None,
) -> None:
    client, embedding_function, _ = build_milvus_index(
        milvus_uri=milvus_uri,
        drop_existing=drop_existing,
        device=device,
    )

    print(f"查詢：{query}")

    query_embeddings = embedding_function.encode_queries([query])
    query_vector = query_embeddings["dense"][0]
    sparse_query_vector = sparse_row_to_dict(query_embeddings["sparse"][0])

    sparse_results = search_sparse(client, sparse_query_vector, top_k=top_k)
    dense_results = search_dense(client, query_vector, top_k=top_k)
    hybrid_results = search_hybrid(
        client=client,
        query_vector=query_vector,
        sparse_query_vector=sparse_query_vector,
        top_k=top_k,
        rrf_k=rrf_k,
    )

    print_results("Sparse / Milvus BGE-M3 sparse vector results", sparse_results)
    print_results("Dense / Milvus BGE-M3 dense vector results", dense_results)
    print_results("Hybrid / Milvus RRFRanker results", hybrid_results)

    client.release_collection(collection_name=COLLECTION_NAME)
    if not keep_collection:
        client.drop_collection(collection_name=COLLECTION_NAME)
        print(f"\n已釋放並刪除 Collection：{COLLECTION_NAME}")
    else:
        print(f"\n已釋放 Collection，資料仍保留在 Milvus：{COLLECTION_NAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milvus BGE-M3 dense + sparse vector hybrid search demo."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERIES[0],
        help="Query to search. If omitted, a default MCP query is used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to show for each search method.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF smoothing constant used by Milvus RRFRanker.",
    )
    parser.add_argument(
        "--milvus-uri",
        default=MILVUS_URI,
        help="Milvus URI, for example http://localhost:19530.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for BGE-M3, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep the demo collection after search.",
    )
    parser.add_argument(
        "--reuse-collection",
        action="store_true",
        help="Reuse an existing collection instead of dropping it first.",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print suggested demo queries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_queries:
        print("Suggested demo queries:")
        for query in DEFAULT_QUERIES:
            print(f"- {query}")
        return

    run_demo(
        query=args.query,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        milvus_uri=args.milvus_uri,
        drop_existing=not args.reuse_collection,
        keep_collection=args.keep_collection,
        device=args.device,
    )


if __name__ == "__main__":
    main()
