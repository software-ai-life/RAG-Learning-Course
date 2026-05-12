import os
import ssl
from pathlib import Path

import requests
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAPTER_DIR = PROJECT_ROOT / "chapter" / "03_embedding"
PERSIST_DIR = PROJECT_ROOT / "data" / "C3" / "llamaindex_index"

markdown_paths = [
    CHAPTER_DIR / "01_what_is_embedding.md",
    CHAPTER_DIR / "02_multimodal_embedding.md",
    CHAPTER_DIR / "03_vector_db.md",
]

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-m3")
Settings.llm = None
Settings.node_parser = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n",
)

docs = SimpleDirectoryReader(
    input_files=[str(path) for path in markdown_paths],
    filename_as_id=True,
).load_data()

for doc in docs:
    source_path = Path(doc.metadata.get("file_path", doc.id_))
    doc.metadata.update(
        {
            "source": str(source_path.relative_to(PROJECT_ROOT)),
            "chapter": "03_embedding",
            "modality": "text",
        }
    )

index = VectorStoreIndex.from_documents(docs)
index.storage_context.persist(persist_dir=str(PERSIST_DIR))

print(f"已載入 {len(docs)} 份 Markdown 文件")
print(f"LlamaIndex index 已儲存到：{PERSIST_DIR}")

storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
loaded_index = load_index_from_storage(storage_context)

query_engine = loaded_index.as_query_engine(
    similarity_top_k=2,
    llm=None,
    response_mode="no_text",
)

query = "RAG 為什麼需要向量資料庫？"
response = query_engine.query(query)

print("\n查詢問題：")
print(query)

print("\n查詢結果：")
for index, source_node in enumerate(response.source_nodes, start=1):
    node = source_node.node
    print(f"\n--- Result {index} ---")
    print(node.get_content())
    print("score:")
    print(source_node.score)
    print("metadata:")
    print(node.metadata)
