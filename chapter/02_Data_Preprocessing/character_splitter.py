from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


# 1. 載入文字檔
project_root = Path(__file__).resolve().parents[2]
txt_path = project_root / "data" / "C2" / "txt" / "rag_data.txt"

loader = TextLoader(str(txt_path), encoding="utf-8")
docs = loader.load()

# 2. 建立字元切分器
text_splitter = CharacterTextSplitter(
    chunk_size=200,  # 每個 chunk 的最大字元數
    chunk_overlap=10,  # 相鄰 chunk 之間重疊的字元數
)

# 3. 將文件切成 chunks
chunks = text_splitter.split_documents(docs)

# 4. 顯示切分結果
print(f"文件總共被切成 {len(chunks)} 個 chunks\n")
print("--- 前 5 個 chunks 預覽 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f"第 {i + 1} 個 chunk（長度: {len(chunk.page_content)}）:")
    print(chunk.page_content)
