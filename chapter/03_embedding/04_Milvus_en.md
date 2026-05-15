# Section 4: Milvus

The previous section introduced vector databases and showed how FAISS can be used to build a local vector index.

FAISS is useful for teaching, local experiments, and prototypes. But when the data becomes larger, or when a system needs shared access, service deployment, metadata filtering, and long-term maintenance, a more complete vector database is needed.

**Milvus** is one common choice.

Official website: https://milvus.io/
GitHub: https://github.com/milvus-io/milvus

Milvus is an open-source vector database for large-scale vector similarity search.

It is suitable for:

```text
RAG knowledge bases
semantic search
image search
recommendation systems
multimodal retrieval
large-scale embedding management
```

## 1. What Is Milvus?

Milvus is a database designed specifically for vector data.

Compared with local vector index tools, Milvus is closer to a production service.

It can handle:

| Capability | Description |
| --- | --- |
| Vector storage | Store embeddings for text, images, audio, video, and other data. |
| Similarity search | Find vectors closest to a query vector. |
| Metadata filter | Filter by scalar fields such as source, date, category, or modality. |
| Index management | Build vector indexes to improve search speed. |
| Collection management | Organize different datasets with collections. |
| Scalable deployment | Run locally or deploy as a larger service. |

In a RAG system, Milvus usually appears here:

```text
documents / images / PDFs
-> chunking / OCR / captioning
-> embedding model
-> Milvus
-> retrieval
-> LLM
```

## 2. Milvus Deployment Options

Milvus can be deployed in different ways:

| Deployment | Suitable For |
| --- | --- |
| Milvus Lite | Local testing, notebooks, demos. |
| Milvus Standalone | Single-machine development and teaching. |
| Milvus Distributed | Large-scale production deployment. |
| Zilliz Cloud | Managed cloud service for teams that do not want to operate Milvus. |

For teaching, Milvus Standalone or Milvus Lite is usually enough.

## 3. Core Concepts

### 3.1 Collection

A collection is the main container for vector data.

It is similar to a table in a relational database.

For example:

```text
collection: ai_agent_course_raw
```

### 3.2 Schema

Before inserting data, you need to define a schema.

For a text RAG collection, fields may include:

| Field | Type | Description |
| --- | --- | --- |
| `id` | `VARCHAR` or `INT64` | Primary key. |
| `vector` | `FLOAT_VECTOR` | Embedding vector. |
| `text` | `VARCHAR` | Chunk text. |
| `source` | `VARCHAR` | Source file path. |
| `page` | `INT64` | Page number. |
| `modality` | `VARCHAR` | text, image, table, etc. |

### 3.3 Vector Field

The vector field stores embedding vectors.

The dimension must match the embedding model output.

If the embedding model outputs 768-dimensional vectors, the Milvus vector field must also use:

```text
dim = 768
```

### 3.4 Scalar Fields

Scalar fields store metadata.

Examples:

```text
source
document_name
page
chapter
modality
image_path
caption
```

Milvus can return these fields in search results through `output_fields`, and can also use them for filtering.

### 3.5 Index and Metric Type

Milvus supports multiple vector index types.

Common choices include:

| Index | Description |
| --- | --- |
| FLAT | Exact search. Accurate but slower for large datasets. |
| IVF_FLAT | Partitions vectors into clusters before search. |
| HNSW | Graph-based approximate nearest neighbor search. |

Common metric types include:

| Metric | Description |
| --- | --- |
| COSINE | Common for normalized text embeddings. |
| IP | Inner product. |
| L2 | Euclidean distance. |

## 4. Basic Milvus RAG Workflow

A basic Milvus text RAG workflow:

```text
1. Load documents
2. Split into chunks
3. Generate embeddings
4. Create Milvus collection
5. Insert text, vectors, and metadata
6. Build index
7. Search with query vector
8. Return top-k chunks to the LLM
```

At query time:

```text
user question
-> query embedding
-> Milvus search
-> top-k chunks
-> LLM answer
```

## 5. Multimodal Milvus Example

This course includes:

```text
04_multi_milvus.py
```

This example demonstrates multimodal image retrieval.

It stores image embeddings in Milvus and searches for similar images using an image or text-image query.

The goal is not to generate an answer, but to understand:

```text
image embedding
-> Milvus
-> multimodal vector search
-> retrieved images
```

## 6. Metadata in Multimodal Retrieval

For multimodal RAG, metadata is especially important.

For example:

```python
metadata = {
    "image_path": "data/C3/images/04_cars/car_01.jpg",
    "caption": "white car front view",
    "modality": "image",
    "source": "car_dataset",
}
```

Without metadata, the system may retrieve a vector but fail to explain where the result came from.

## 7. Milvus vs FAISS

| Feature | FAISS | Milvus |
| --- | --- | --- |
| Local experiment | Strong | Possible |
| Production service | Limited | Strong |
| Metadata filtering | Limited | Strong |
| Multi-user access | Not the main focus | Supported |
| Deployment | Library | Database service |
| Persistence management | Manual | Built-in |

FAISS is a good starting point. Milvus is better when the system needs to become a real service.

## 8. Common Questions

### Does Milvus store metadata?

Yes. Milvus scalar fields can store metadata such as `source`, `page`, `chapter`, `modality`, and `image_path`.

### Can Milvus replace a traditional database?

Not fully. Milvus is designed for vector search. Traditional databases are still better for transactions, reports, joins, and strict relational queries.

A common production architecture is:

```text
Milvus: embeddings and vector search
Object storage: original files
Relational database: business data and permissions
LLM service: answer generation
```

### Should every RAG project use Milvus?

No.

For small teaching projects, FAISS may be enough. Use Milvus when the project needs service deployment, metadata filtering, larger scale, or long-term management.

## 9. Key Takeaways

1. Milvus is an open-source vector database for large-scale vector search.
2. A collection stores vectors and metadata.
3. Metadata fields are essential for RAG source tracking and filtering.
4. Milvus is more suitable than FAISS for production-style systems.
5. Milvus can support text retrieval and multimodal retrieval.
