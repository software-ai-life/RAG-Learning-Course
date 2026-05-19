import argparse
import math
import os
import re
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
DATA_DIR = PROJECT_ROOT / "data" / "C4" / "hybrid_search"
EMBEDDING_MODEL = "gemini-embedding-2"
OUTPUT_DIMENSIONALITY = 768

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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())


def load_markdown_documents() -> list[Any]:
    from langchain_core.documents import Document

    markdown_paths = sorted(DATA_DIR.glob("*.md"))
    if not markdown_paths:
        raise FileNotFoundError(
            f"找不到 Hybrid Search 範例資料：{DATA_DIR}\n"
            "請先建立 data/C4/hybrid_search/*.md"
        )

    documents: list[Any] = []
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
            " ",
            "",
        ],
    )
    split_docs = splitter.split_documents(documents)

    chunks: list[Any] = []
    for index, document in enumerate(split_docs):
        document.page_content = document.page_content.strip()
        document.metadata["chunk_id"] = f"chunk_{index:04d}"
        chunks.append(document)

    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class GeminiEmbedder:
    def __init__(self) -> None:
        import httpx
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "找不到 GEMINI_API_KEY。請在專案根目錄建立 .env，並加入：\n"
                "GEMINI_API_KEY=your_gemini_api_key_here"
            )

        self.types = types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                # 教學環境若遇到公司代理或本機自簽憑證，httpx 會拋出
                # CERTIFICATE_VERIFY_FAILED。正式專案建議改成指定可信任 CA。
                httpxClient=httpx.Client(verify=False, follow_redirects=True),
            ),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                self.types.Content(
                    parts=[
                        self.types.Part.from_text(
                            text=f"task: retrieval document | content: {text}"
                        )
                    ]
                )
                for text in texts
            ],
            config=self.types.EmbedContentConfig(
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )
        return [embedding.values for embedding in result.embeddings]

    def embed_query(self, query: str) -> list[float]:
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                self.types.Content(
                    parts=[
                        self.types.Part.from_text(
                            text=f"task: search result | query: {query}"
                        )
                    ]
                )
            ],
            config=self.types.EmbedContentConfig(
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )
        return result.embeddings[0].values


def build_bm25_retriever(chunks: list[Any], top_k: int) -> Any:
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=tokenize,
    )
    retriever.k = top_k
    return retriever


class GeminiDenseRetriever:
    def __init__(self, chunks: list[Any], top_k: int) -> None:
        self.chunks = chunks
        self.top_k = top_k
        self.embedder = GeminiEmbedder()
        self.vectors = self.embedder.embed_documents(
            [chunk.page_content for chunk in chunks]
        )

    def invoke(self, query: str, config: Any | None = None) -> list[Any]:
        from langchain_core.documents import Document

        query_vector = self.embedder.embed_query(query)
        scored_results = [
            (
                cosine_similarity(query_vector, vector),
                chunk,
            )
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored_results.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, chunk in scored_results[: self.top_k]:
            results.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "dense_score": round(score, 6),
                    },
                )
            )

        return results


def build_hybrid_retriever(
    bm25_retriever: Any,
    dense_retriever: Any,
    rrf_k: int,
) -> Any:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever

    from langchain_core.runnables import RunnableLambda

    return EnsembleRetriever(
        retrievers=[bm25_retriever, RunnableLambda(dense_retriever.invoke)],
        weights=[0.5, 0.5],
        c=rrf_k,
        id_key="chunk_id",
    )


def print_results(label: str, results: list[Any]) -> None:
    print(f"\n## {label}")
    if not results:
        print("沒有找到結果。")
        return

    for rank, document in enumerate(results, start=1):
        preview = " ".join(document.page_content.split())
        if len(preview) > 280:
            preview = preview[:280] + "..."

        print(f"\n[{rank}]")
        if "dense_score" in document.metadata:
            print(f"dense_score: {document.metadata['dense_score']}")
        print(f"source: {document.metadata.get('source')}")
        print(f"title: {document.metadata.get('title')}")
        print(f"chunk_id: {document.metadata.get('chunk_id')}")
        print(f"text: {preview}")


def run_demo(query: str, top_k: int, rrf_k: int) -> None:
    documents = load_markdown_documents()
    chunks = split_documents(documents)

    print(f"已載入 {len(documents)} 份 Markdown 文件")
    print(f"已切成 {len(chunks)} 個 chunks")
    print(f"查詢：{query}")

    bm25_retriever = build_bm25_retriever(chunks, top_k=top_k)
    dense_retriever = GeminiDenseRetriever(chunks, top_k=top_k)
    hybrid_retriever = build_hybrid_retriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        rrf_k=rrf_k,
    )

    sparse_results = bm25_retriever.invoke(query)
    dense_results = dense_retriever.invoke(query)
    hybrid_results = hybrid_retriever.invoke(query)[:top_k]

    print_results("Sparse / BM25Retriever results", sparse_results)
    print_results("Dense Gemini embedding results", dense_results)
    print_results("Hybrid EnsembleRetriever RRF results", hybrid_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25 + dense embedding + RRF hybrid search demo."
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
        help="RRF smoothing constant. Larger values make ranking smoother.",
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

    run_demo(query=args.query, top_k=args.top_k, rrf_k=args.rrf_k)


if __name__ == "__main__":
    main()
