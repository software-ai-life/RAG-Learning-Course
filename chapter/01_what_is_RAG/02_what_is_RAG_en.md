# What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

A normal LLM answers from knowledge learned during training. RAG adds a retrieval step before generation: it first searches your documents for relevant content, then sends that content to the LLM together with the user question.

In simple terms, a RAG flow is:

1. Load documents.
2. Split documents into smaller chunks.
3. Convert each chunk into an embedding vector.
4. Store the vectors in a vector store.
5. Receive a user question.
6. Search the vector store for relevant chunks.
7. Put the retrieved chunks into a prompt.
8. Send the prompt to the LLM to generate an answer.

This chapter explains a basic RAG example based on `langchain_example.py`.

## 1. Initial Setup

```python
import os
import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
```

`load_dotenv()` loads variables from the `.env` file into environment variables.

Example `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

After loading the `.env` file, the Gemini API key can be read with:

```python
os.getenv("GEMINI_API_KEY")
```

## 2. Prepare and Load Markdown Data

```python
markdown_path = "../../data/C1/markdown/what-is-rag.md"
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()
```

This step reads the Markdown file and converts it into LangChain `Document` objects.

### `UnstructuredMarkdownLoader(markdown_path)`

`UnstructuredMarkdownLoader` parses the Markdown file and converts it into documents that LangChain can process.

### `loader.load()`

```python
docs = loader.load()
```

`load()` actually reads the file and returns a `Document` list.

Even when there is only one Markdown file, the result is still usually a list because LangChain loaders are designed to support multiple documents.

## 3. Split Documents into Chunks

```python
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)
```

LLMs and embedding models are not ideal for processing very long documents at once. RAG usually splits the original document into smaller text pieces called chunks.

### `RecursiveCharacterTextSplitter()`

The example does not pass any parameters, so it uses LangChain's default settings.

This splitter is recursive. It first tries natural separators such as paragraphs and new lines, then falls back to smaller separators when needed.

Common parameters:

| Parameter | Description |
| --- | --- |
| `chunk_size` | Maximum length of each chunk. If it is too large, retrieval may be less precise. If it is too small, context may be incomplete |
| `chunk_overlap` | Overlap between neighboring chunks. This helps avoid cutting important sentences in half |
| `separators` | Separators used for splitting, such as `\n\n`, `\n`, and spaces |
| `length_function` | Function used to calculate text length. The default is usually `len` |

If you want more control over splitting, you can write:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### `split_documents(docs)`

```python
chunks = text_splitter.split_documents(docs)
```

The returned `chunks` are also a `Document` list. Each chunk keeps the original metadata, so sources can still be traced later.

## 4. Create an Embedding Model

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

Embedding means converting text into vectors.

Vectors allow the computer to compare semantic similarity between text. For example:

```text
"What is RAG?"
"What does Retrieval-Augmented Generation mean?"
```

These two questions use different words, but their meanings are similar, so their embedding vectors should be close.

### `model_name`

```python
model_name="BAAI/bge-m3"
```

`BAAI/bge-m3` is a multilingual embedding model. It is useful for Chinese and English retrieval tasks.

### `encode_kwargs`

```python
encode_kwargs={'normalize_embeddings': True}
```

`encode_kwargs` controls how text is encoded into vectors.

| Key | Current value | Description |
| --- | --- | --- |
| `normalize_embeddings` | `True` | Whether to normalize the embedding vectors |

After normalization, vector lengths are scaled consistently. This is usually helpful when using cosine similarity for semantic search.

For RAG retrieval, `normalize_embeddings=True` is recommended because it makes similarity comparison more stable.

## 5. Create a Vector Store

```python
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)
```

A vector store saves chunks and their embedding vectors.

In this example, `InMemoryVectorStore` stores everything in memory while the program is running.

### `InMemoryVectorStore(embeddings)`

| Parameter | Current value | Description |
| --- | --- | --- |
| `embedding` | `embeddings` | Specifies which embedding model the vector store should use |

`InMemoryVectorStore` is simple and does not require an external database or service.

The limitation is that the data is not persisted. When the program ends, the vector store is gone. Production projects usually use Chroma, FAISS, Qdrant, Milvus, or Pinecone.

### `add_documents(chunks)`

```python
vectorstore.add_documents(chunks)
```

| Parameter | Current value | Description |
| --- | --- | --- |
| `documents` | `chunks` | Document chunks to add into the vector store |

When this line runs, LangChain:

1. Reads each chunk's `page_content`.
2. Uses `embeddings` to convert the text into vectors.
3. Stores the original chunk, metadata, and embedding vector in the vector store.

## 6. Create a Prompt Template

```python
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context.
If the answer cannot be found in the context, say that it cannot be found.

Context:
{context}

Question:
{question}

Answer:
""")
```

A prompt template is the instruction template given to the LLM.

In RAG, the prompt usually contains two core variables:

| Variable | Description |
| --- | --- |
| `{context}` | Relevant document content retrieved from the vector store |
| `{question}` | The user's question |

LangChain fills these variables into the prompt.

A good RAG prompt should clearly ask the model to answer based on the provided context. This helps reduce hallucination.

## 7. Create a Gemini LLM Client

```python
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    http_client=httpx.Client(verify=False)
)
```

Although the class is `ChatOpenAI`, this example is actually calling Gemini.

This works because Gemini provides an OpenAI-compatible endpoint, so an OpenAI-style client can call Gemini.

### `model`

```python
model="gemini-2.5-flash"
```

`gemini-2.5-flash` is a fast Gemini model. It is suitable for teaching examples, interactive Q&A, and RAG demos.

### `temperature`

```python
temperature=0.7
```

| Parameter | Description |
| --- | --- |
| `temperature` | Controls randomness in the answer |

Common settings:

| Value | Effect |
| --- | --- |
| `0` | Most stable and conservative, good for factual QA |
| `0.3` | Stable with a little flexibility |
| `0.7` | More natural and varied, good for general conversation |
| `1.0` or higher | More creative, but more likely to drift away from the data |

RAG usually wants the model to stay faithful to the retrieved data, so production QA systems often use `temperature=0` or `0.3`. This example uses `0.7`, so the answer may sound more natural.

### `max_tokens`

```python
max_tokens=4096
```

| Parameter | Description |
| --- | --- |
| `max_tokens` | Limits the maximum number of output tokens |

Tokens are text units used by language models. `max_tokens=4096` means the model can output up to 4096 tokens. This prevents overly long answers and helps control API cost.

### `base_url`

```python
base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
```

| Parameter | Description |
| --- | --- |
| `base_url` | Specifies the API endpoint |

`ChatOpenAI` normally calls the OpenAI API. Because this example uses Gemini, `base_url` must be changed to Gemini's OpenAI-compatible endpoint.

If `base_url` is not set, the code will try to call OpenAI instead of Gemini.

### `http_client`

```python
http_client=httpx.Client(verify=False)
```

This custom HTTP client is used when calling the API.

`verify=False` disables SSL certificate verification. This can help with local certificate issues, but it is not recommended for production because it weakens HTTPS security.

## 8. Ask a Question

```python
question = "What is RAG?"
```

`question` is the user's question.

In the RAG flow, it is used twice:

1. To search for relevant documents.
2. To fill the prompt before calling the LLM.

## 9. Retrieve Relevant Documents

```python
retrieved_docs = vectorstore.similarity_search(question, k=3)
```

This is the retrieval step in RAG.

The code converts `question` into an embedding vector, then searches the vector store for the most similar chunks.

### `similarity_search(question, k=3)`

| Parameter | Current value | Description |
| --- | --- | --- |
| `query` | `question` | The question text used for search |
| `k` | `3` | Returns the top 3 most similar chunks |

`k` is an important retrieval parameter.

| `k` value | Effect |
| --- | --- |
| Too small | May not retrieve enough information, making the answer incomplete |
| Balanced | Provides enough context without too much irrelevant content |
| Too large | May include irrelevant content and confuse the model |

This example uses `k=3`, meaning each question retrieves the 3 most relevant chunks.

## 10. Combine Retrieved Chunks

```python
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
```

This line combines the retrieved chunks into one text block, which will be inserted into the `{context}` variable in the prompt.

### Parameters and Syntax

| Syntax | Description |
| --- | --- |
| `"\n\n"` | Separates chunks with two new lines to make the context easier to read |
| `doc.page_content` | Gets the text content from each `Document` |
| `for doc in retrieved_docs` | Iterates through the retrieved documents |
| `.join(...)` | Combines multiple strings into one string |

The combined `docs_content` looks like:

```text
First retrieved chunk

Second retrieved chunk

Third retrieved chunk
```

## 11. Format the Prompt and Call the LLM

```python
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
```

This is the final generation step.

### `prompt.format(...)`

```python
prompt.format(question=question, context=docs_content)
```

This fills the prompt template with the actual question and retrieved context.

In LangChain, `answer` is usually not just a plain string. It is often a message object. If you only want to print the answer text, use:

```python
print(answer.content)
```

## Full RAG Flow

The whole LangChain RAG flow can be summarized as:

```text
Markdown file
    ↓
UnstructuredMarkdownLoader
    ↓
Document list
    ↓
RecursiveCharacterTextSplitter
    ↓
Chunks
    ↓
HuggingFaceEmbeddings
    ↓
Vectors
    ↓
InMemoryVectorStore
    ↓
similarity_search(question, k=3)
    ↓
Retrieved context
    ↓
ChatPromptTemplate
    ↓
Gemini model
    ↓
Answer
```

## Why RAG is Useful

The core value of RAG is that it lets an LLM answer based on external data.

It helps with several common problems:

| Problem | RAG solution |
| --- | --- |
| The model does not know the latest data | Put the latest documents into the data source and retrieve before answering |
| The model may hallucinate | Ask the model to answer based on retrieval context |
| Company data was not in model training | Convert internal documents into a vector store |
| Answers need traceable sources | Chunk metadata can preserve document source information |

## LlamaIndex Example

Besides LangChain, this chapter also includes another example:

```text
chapter/01_what_is_RAG/llamaindex_example.py
```

This example uses LlamaIndex to complete the same RAG flow.

LangChain and LlamaIndex can both build RAG systems, but their design focus is different:

| Tool | Feature |
| --- | --- |
| LangChain | The pipeline is more explicit, so it is easier to learn how loader, splitter, embedding, vector store, prompt, and LLM connect |
| LlamaIndex | Indexing and query engine abstractions are more complete, so it is convenient for quickly turning data into a searchable index |

The core LlamaIndex flow is:

```text
Set Gemini LLM
    ↓
Set embedding model
    ↓
Load Markdown document
    ↓
Create VectorStoreIndex
    ↓
Convert to query engine
    ↓
Ask a question and get an answer
```

### 1. Set Gemini LLM

```python
Settings.llm = OpenAILike(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    is_chat_model=True,
    http_client=httpx.Client(verify=False)
)
```

`Settings.llm` is the global LLM setting in LlamaIndex. After setting it, later index and query engine operations use this model by default.

`OpenAILike` lets LlamaIndex use a model service that follows an OpenAI-like API format. Since Gemini provides an OpenAI-compatible endpoint, this can be used to connect LlamaIndex to Gemini.

Do not use `verify=False` in production because it disables HTTPS certificate verification.

### 2. Set the Embedding Model

```python
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-m3")
```

`Settings.embed_model` is the global embedding setting in LlamaIndex. It converts documents and questions into vectors.

### 3. Read Documents

```python
docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/what-is-rag.md"]
).load_data()
```

`SimpleDirectoryReader` is a LlamaIndex tool for reading local files.

`input_files` uses a list, so multiple files can be specified:

```python
input_files=[
    "file1.md",
    "file2.pdf",
    "file3.txt"
]
```

`.load_data()` reads the file and returns a LlamaIndex document list.

### 4. Create VectorStoreIndex

```python
index = VectorStoreIndex.from_documents(docs)
```

`VectorStoreIndex` is a commonly used vector index in LlamaIndex.

This line does several things:

1. Reads the content from `docs`.
2. Uses `Settings.embed_model` to convert the content into embeddings.
3. Creates a vector index that can be queried.

Compared with the LangChain example, LlamaIndex wraps document splitting, embedding generation, and vector index creation into a simpler workflow.

### 5. Create a Query Engine

```python
query_engine = index.as_query_engine()
```

`query_engine` is LlamaIndex's query interface.

After calling `as_query_engine()`, you can ask questions like:

```python
query_engine.query("What is RAG?")
```

Common important parameters:

| Parameter | Description |
| --- | --- |
| `similarity_top_k` | Number of relevant chunks to retrieve for each query, similar to LangChain's `k` |
| `response_mode` | Controls how LlamaIndex combines retrieved chunks into an answer |

Example:

```python
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)
```

`similarity_top_k=3` means each query retrieves the top 3 most similar chunks.

`response_mode="compact"` means LlamaIndex will compact the retrieved content before sending it to the LLM for answer generation.

### 6. View Prompts

```python
print(query_engine.get_prompts())
```

This prints the prompts used internally by the query engine.

This is useful for learning because LlamaIndex wraps many details. With `get_prompts()`, you can see how it combines context and question into a prompt.

### LangChain and LlamaIndex Comparison

| RAG step | LangChain approach | LlamaIndex approach |
| --- | --- | --- |
| Read documents | `UnstructuredMarkdownLoader` | `SimpleDirectoryReader` |
| Split documents | `RecursiveCharacterTextSplitter` | Usually handled during index creation |
| Embedding | `HuggingFaceEmbeddings` | `Settings.embed_model = HuggingFaceEmbedding(...)` |
| Vector store / index | `InMemoryVectorStore` | `VectorStoreIndex` |
| Retrieval | `similarity_search(question, k=3)` | `index.as_query_engine(similarity_top_k=3)` |
| Prompt | `ChatPromptTemplate` | Built into query engine, can be viewed with `get_prompts()` |
| LLM | `ChatOpenAI` | `OpenAILike` |
