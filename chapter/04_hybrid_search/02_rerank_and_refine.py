import argparse
import os
import ssl
from pathlib import Path
from typing import Any


# 教學環境若遇到公司代理或本機憑證問題，可先關閉 SSL 驗證。
# 正式專案建議改成設定可信任 CA。
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import requests

    requests.packages.urllib3.disable_warnings()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "C4" / "hybrid_search"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-base"
GEMINI_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DEFAULT_CANDIDATE_K = 12
DEFAULT_TOP_N = 5

DEFAULT_QUERIES = [
    "How does MCP help agent tool interoperability?",
    "Why does dense retrieval miss exact error codes?",
    "What should I check when ERR-4291 happens?",
    "How is agent memory different from context engineering?",
]


def configure_huggingface_ssl() -> None:
    try:
        import requests
        from huggingface_hub import configure_http_backend

        def backend_factory() -> requests.Session:
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=backend_factory)
    except ImportError:
        pass


def load_markdown_documents() -> list[Any]:
    from langchain_core.documents import Document

    markdown_paths = sorted(DATA_DIR.glob("*.md"))
    if not markdown_paths:
        raise FileNotFoundError(
            f"找不到範例資料：{DATA_DIR}\n"
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


def build_base_retriever(
    chunks: list[Any],
    candidate_k: int,
    embedding_model: str,
    device: str | None,
) -> Any:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    configure_huggingface_ssl()

    model_kwargs = {"device": device} if device else {}
    embeddings = HuggingFaceEmbeddings(
        model=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": candidate_k})


def build_rerank_retriever(
    base_retriever: Any,
    top_n: int,
    reranker_model: str,
    device: str | None,
) -> tuple[Any, Any]:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import (
        CrossEncoderReranker,
    )
    from langchain_community.cross_encoders.huggingface import (
        HuggingFaceCrossEncoder,
    )

    configure_huggingface_ssl()

    model_kwargs = {"device": device} if device else {}
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=reranker_model,
        model_kwargs=model_kwargs,
    )
    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=top_n,
    )
    rerank_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker,
    )
    return rerank_retriever, cross_encoder


def add_rerank_scores(
    query: str,
    documents: list[Any],
    cross_encoder: Any,
) -> list[Any]:
    if not documents:
        return []

    scores = cross_encoder.score(
        [(query, document.page_content) for document in documents]
    )
    scored_documents = []
    for document, score in zip(documents, scores):
        document.metadata["rerank_score"] = round(float(score), 6)
        scored_documents.append(document)

    return scored_documents


def get_gemini_client() -> Any:
    import httpx
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            # 教學環境若遇到公司代理或本機自簽憑證，httpx 會拋出
            # CERTIFICATE_VERIFY_FAILED。正式專案建議改成指定可信任 CA。
            httpxClient=httpx.Client(verify=False, follow_redirects=True),
        ),
    )


def refine_context_with_gemini(
    query: str,
    documents: list[Any],
    model: str,
) -> str | None:
    client = get_gemini_client()
    if client is None:
        return None

    context_blocks = []
    for index, document in enumerate(documents, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[Document {index}]",
                    f"source: {document.metadata.get('source')}",
                    f"title: {document.metadata.get('title')}",
                    f"chunk_id: {document.metadata.get('chunk_id')}",
                    document.page_content,
                ]
            )
        )

    prompt = f"""You are refining retrieved context for a RAG system.

Question:
{query}

Retrieved context:
{chr(10).join(context_blocks)}

Task:
Extract only the sentences or bullet points that are directly useful for answering the question.
Keep source references in square brackets, for example [Document 1].
Do not answer the question. Only return the refined context.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    text = response.text or ""
    return text.strip() or None


def print_documents(label: str, documents: list[Any], max_chars: int = 260) -> None:
    print(f"\n## {label}")
    if not documents:
        print("沒有找到結果。")
        return

    for rank, document in enumerate(documents, start=1):
        preview = " ".join(document.page_content.split())
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."

        print(f"\n[{rank}]")
        if "rerank_score" in document.metadata:
            print(f"rerank_score: {document.metadata['rerank_score']}")
        print(f"source: {document.metadata.get('source')}")
        print(f"title: {document.metadata.get('title')}")
        print(f"chunk_id: {document.metadata.get('chunk_id')}")
        print(f"text: {preview}")


def run_demo(
    query: str,
    candidate_k: int,
    top_n: int,
    embedding_model: str,
    reranker_model: str,
    gemini_model: str,
    device: str | None,
    skip_refine: bool,
) -> None:
    documents = load_markdown_documents()
    chunks = split_documents(documents)

    print(f"已載入 {len(documents)} 份 Markdown 文件")
    print(f"已切成 {len(chunks)} 個 chunks")
    print(f"查詢：{query}")
    print(f"第一階段召回數量：{candidate_k}")
    print(f"Rerank 後保留數量：{top_n}")

    base_retriever = build_base_retriever(
        chunks=chunks,
        candidate_k=candidate_k,
        embedding_model=embedding_model,
        device=device,
    )
    rerank_retriever, cross_encoder = build_rerank_retriever(
        base_retriever=base_retriever,
        top_n=top_n,
        reranker_model=reranker_model,
        device=device,
    )

    base_results = base_retriever.invoke(query)
    reranked_results = rerank_retriever.invoke(query)
    reranked_results = add_rerank_scores(
        query=query,
        documents=reranked_results,
        cross_encoder=cross_encoder,
    )

    print_documents("Base retrieval results", base_results)
    print_documents("Reranked results", reranked_results)

    if skip_refine:
        return

    refined_context = refine_context_with_gemini(
        query=query,
        documents=reranked_results,
        model=gemini_model,
    )
    print("\n## Refined context")
    if refined_context is None:
        print("未執行 Gemini refine。請確認 .env 是否有 GEMINI_API_KEY，或使用 --skip-refine。")
    else:
        print(refined_context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced retrieval demo: retrieve, rerank, and refine context."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERIES[0],
        help="Query to search.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of documents retrieved before reranking.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of documents kept after reranking.",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help="Embedding model used by the first-stage FAISS retriever.",
    )
    parser.add_argument(
        "--reranker-model",
        default=RERANKER_MODEL,
        help="Cross-encoder reranker model.",
    )
    parser.add_argument(
        "--gemini-model",
        default=GEMINI_MODEL,
        help="Gemini model used for optional context refinement.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for local Hugging Face models, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--skip-refine",
        action="store_true",
        help="Only run retrieval and rerank. Skip Gemini context refinement.",
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
        candidate_k=args.candidate_k,
        top_n=args.top_n,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        gemini_model=args.gemini_model,
        device=args.device,
        skip_refine=args.skip_refine,
    )


if __name__ == "__main__":
    main()
