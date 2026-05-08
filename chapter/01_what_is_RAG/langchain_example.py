import os
import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import ssl
import requests
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()

load_dotenv()

markdown_path = "../../data/C1/markdown/what-is-rag.md"

# 載入本地 markdown 檔案
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分塊
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 建立向量儲存
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示詞模板
prompt = ChatPromptTemplate.from_template("""請根據下面提供的上下文資訊來回答問題。
請確保你的回答完全基於這些上下文。
如果上下文中沒有足夠的資訊來回答問題，請直接告知：「抱歉，我無法根據提供的上下文找到相關資訊來回答此問題。」

上下文：
{context}

問題：{question}

回答："""
                                          )

# 設定大型語言模型

# 使用 Gemini 模型
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    http_client=httpx.Client(verify=False)
)


# 使用者查詢
question = "文中舉了哪些例子？"

# 在向量儲存中查詢相關文件
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
