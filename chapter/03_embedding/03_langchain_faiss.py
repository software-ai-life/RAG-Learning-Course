from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import ssl
import requests
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAPTER_DIR = PROJECT_ROOT / "chapter" / "03_embedding"
FAISS_PATH = PROJECT_ROOT / "data" / "C3" / "faiss_index"

markdown_paths = [
    CHAPTER_DIR / "01_what_is_embedding.md",
    CHAPTER_DIR / "02_multimodal_embedding.md",
    CHAPTER_DIR / "03_vector_db.md",
]

docs = []
for markdown_path in markdown_paths:
    loader = TextLoader(str(markdown_path), encoding="utf-8")
    loaded_docs = loader.load()

    for doc in loaded_docs:
        doc.metadata.update(
            {
                "source": str(markdown_path.relative_to(PROJECT_ROOT)),
                "chapter": "03_embedding",
                "modality": "text",
            }
        )

    docs.extend(loaded_docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        "。",
        "，",
        " ",
        "",
    ],
)

chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(str(FAISS_PATH))

print(f"已載入 {len(docs)} 份 Markdown 文件")
print(f"已切成 {len(chunks)} 個 chunks")
print(f"FAISS index 已儲存到：{FAISS_PATH}")

loaded_vectorstore = FAISS.load_local(
    str(FAISS_PATH),
    embeddings,
    allow_dangerous_deserialization=True,
)

query = "RAG 為什麼需要向量資料庫？"
results = loaded_vectorstore.similarity_search(query, k=2)

print("\n查詢問題：")
print(query)

print("\n查詢結果：")
for index, doc in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(doc.page_content)
    print("metadata:")
    print(doc.metadata)
