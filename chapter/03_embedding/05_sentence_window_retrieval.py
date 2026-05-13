import os
import ssl
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 教學環境若遇到本機憑證或公司代理問題，可先關閉 SSL 驗證。
# 正式專案建議改成設定可信任 CA。
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import requests

    requests.packages.urllib3.disable_warnings()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "C3"
    / "pdf"
    / "Why you Should Choose Multimodal RAG and Vision Language Models Over OCR + LLM _ LinkedIn.pdf"
)

WINDOW_SIZE = 3
SIMILARITY_TOP_K = 2
QUERY = "為什麼文章建議使用 Multimodal RAG 和 Vision Language Models，而不是 OCR 加 LLM？"


def load_documents():
    documents = SimpleDirectoryReader(
        input_files=[str(PDF_PATH)],
        filename_as_id=True,
    ).load_data()

    for document in documents:
        source_path = Path(document.metadata.get("file_path", document.id_))
        document.metadata.update(
            {
                "source": str(source_path.relative_to(PROJECT_ROOT)),
                "chapter": "03_embedding",
                "modality": "pdf",
            }
        )

    return documents


def build_sentence_window_index(documents):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=WINDOW_SIZE,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    return index, nodes


def print_node_result(index, source_node):
    node = source_node.node
    original_text = node.metadata.get("original_text", "")
    window_text = node.metadata.get("window", "")

    print(f"\n--- Result {index} ---")
    print("score:")
    print(source_node.score)

    print("\nmetadata:")
    print(
        {
            "source": node.metadata.get("source"),
            "chapter": node.metadata.get("chapter"),
            "modality": node.metadata.get("modality"),
        }
    )

    print("\n原始檢索句子 original_text:")
    print(original_text or node.get_content())

    print("\n送給 LLM 的上下文 window:")
    print(window_text or node.get_content())


def main():
    Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-m3")
    Settings.llm = None

    documents = load_documents()
    index, nodes = build_sentence_window_index(documents)

    query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        response_mode="no_text",
        llm=None,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    response = query_engine.query(QUERY)

    print(f"已載入 {len(documents)} 份 PDF 文件")
    print(f"資料來源: {PDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"已建立 {len(nodes)} 個 sentence window nodes")
    print(f"window_size: {WINDOW_SIZE}")
    print(f"similarity_top_k: {SIMILARITY_TOP_K}")

    print("\n查詢問題:")
    print(QUERY)

    print("\n查詢結果:")
    for result_index, source_node in enumerate(response.source_nodes, start=1):
        print_node_result(result_index, source_node)


if __name__ == "__main__":
    main()
