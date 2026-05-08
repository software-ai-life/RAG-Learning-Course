import os
from dotenv import load_dotenv
import httpx
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import ssl
import requests
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()

load_dotenv()

# 使用 Gemini 模型
Settings.llm = OpenAILike(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    is_chat_model=True,
    http_client=httpx.Client(verify=False)
)

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-m3")

docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/what-is-rag.md"]).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中舉了哪些例子？"))