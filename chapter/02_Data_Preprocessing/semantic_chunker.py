from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


# 1. 建立 embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 2. 建立語意切分器
text_splitter = SemanticChunker(
    embeddings,
    # 也可以改成 "standard_deviation"、"interquartile" 或 "gradient"。
    breakpoint_threshold_type="percentile",
)

# 3. 載入文字檔
project_root = Path(__file__).resolve().parents[2]
txt_path = project_root / "data" / "C2" / "txt" / "rag_data.txt"

loader = TextLoader(str(txt_path), encoding="utf-8")
documents = loader.load()

# 4. 根據語意變化將文件切成 chunks
docs = text_splitter.split_documents(documents)

# 5. 顯示切分結果
print(f"文件總共被切成 {len(docs)} 個 chunks\n")
print("--- 前 2 個 chunks 預覽 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f"第 {i + 1} 個 chunk（長度: {len(chunk.page_content)}）:")
    print(chunk.page_content)
