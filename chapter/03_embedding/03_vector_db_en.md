# Section 3: Vector Databases

The previous sections introduced embeddings and multimodal embeddings. An embedding model converts text, images, or documents into vectors, but vectors alone are not enough.

A RAG system also needs a place to store those vectors and search them efficiently when a user asks a question.

That place is a **vector database**.

In RAG, a vector database usually handles:

```text
storing chunks / images / document pages
storing embedding vectors
storing metadata
performing similarity search with a query vector
returning relevant content to the LLM
```

You can think of it as the semantic search engine of a RAG system.

## 1. Why Do We Need a Vector Database?

If you only have 10 chunks, you can calculate similarity one by one.

But real RAG projects may contain:

```text
thousands of documents
millions of chunks
multiple file types
metadata filters
many users
production deployment requirements
```

In those cases, a vector database provides:

| Capability | Description |
| --- | --- |
| Persistence | Store vectors and metadata permanently. |
| Fast search | Use indexes to accelerate similarity search. |
| Metadata filtering | Filter by source, page, category, date, permission, etc. |
| Scalability | Support larger datasets and concurrent users. |
| Operations | Manage collections, indexes, and updates. |

## 2. Basic Vector Database Workflow

A typical RAG workflow looks like this:

```text
documents
-> split into chunks
-> generate embeddings
-> insert vectors and metadata into vector database
```

At query time:

```text
user question
-> query embedding
-> vector search
-> top-k chunks
-> LLM answer
```

The vector database does not generate answers. It retrieves relevant context for the LLM.

## 3. Core Concepts

### 3.1 Collection

A collection is similar to a table in a traditional database.

It stores a group of vectors and their metadata.

For example:

```text
collection: rag_course_chunks
```

### 3.2 Vector Field

The vector field stores embedding vectors.

Example:

```text
vector: [0.12, -0.03, 0.44, ...]
```

The vector dimension must match the output dimension of the embedding model.

### 3.3 Scalar Fields / Metadata

Scalar fields store metadata:

```text
source
document_name
page
chapter
modality
created_at
```

Metadata is important because RAG systems should return not only the answer, but also the source.

### 3.4 Index

An index accelerates vector search.

Without an index, the database may need to compare the query vector with every stored vector. With an index, it can search much faster.

Common index types include:

```text
Flat
IVF
HNSW
DiskANN
```

### 3.5 Metric Type

Metric type defines how vector similarity is calculated.

| Metric | Description |
| --- | --- |
| COSINE | Measures angle similarity. Common for normalized text embeddings. |
| IP | Inner product. Often used for normalized embeddings. |
| L2 | Euclidean distance. |

## 4. Metadata and Source Tracking

A good RAG system must be traceable.

Each chunk should store metadata such as:

```python
metadata = {
    "source": "chapter/03_embedding/01_what_is_embedding.md",
    "chapter": "03_embedding",
    "modality": "text",
    "page": 3,
}
```

This allows the system to:

```text
show citations
debug bad retrieval results
filter by source or chapter
support permissions
separate text, image, and table chunks
```

## 5. LangChain + FAISS Example

This course includes a FAISS example:

```text
03_langchain_faiss.py
```

The example reads Markdown files from this repository, splits them into chunks, generates embeddings with `BAAI/bge-m3`, and stores them in a FAISS index.

FAISS is useful for local experiments and teaching because it is simple and fast.

However, FAISS is usually not enough for production systems that need:

```text
shared service
metadata filtering
persistence management
large-scale deployment
multi-user access
```

## 6. LlamaIndex Vector Example

This course also includes a LlamaIndex example:

```text
03_llamaindex_vector.py
```

LlamaIndex provides a higher-level abstraction for building indexes and query engines.

Compared with a low-level vector store example, it wraps more retrieval logic and makes it easier to connect indexing, retrieval, and query workflows.

## 7. Mainstream Vector Databases and Tools

![Vector databases](./images/vector_database.png)

Common options include:

| Tool | Description |
| --- | --- |
| FAISS | Local vector search library, good for experiments. |
| Chroma | Lightweight vector database for prototypes. |
| Milvus | Open-source vector database for large-scale and production use. |
| Qdrant | Vector database with strong filtering support. |
| Weaviate | Vector database with schema and hybrid search support. |
| Pinecone | Managed vector database service. |

## 8. How to Choose

| Scenario | Suggested Tool |
| --- | --- |
| Small teaching demo | FAISS |
| Local prototype | FAISS or Chroma |
| Production service | Milvus, Qdrant, Weaviate, or Pinecone |
| Large-scale open-source deployment | Milvus or Qdrant |
| Strong metadata filtering | Qdrant, Milvus, Weaviate |
| Multimodal retrieval | Milvus, Weaviate, Qdrant |

For this course:

```text
FAISS is used for local teaching examples.
Milvus is used when we need a more realistic vector database.
```

## 9. Key Takeaways

1. A vector database stores embeddings and metadata.
2. It performs similarity search at query time.
3. Metadata is essential for filtering, citations, and debugging.
4. FAISS is good for local experiments.
5. Milvus and similar systems are better suited for production-scale RAG.
