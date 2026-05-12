# Section 2: Text Chunking

Text chunking is a core step in RAG preprocessing. Data loading turns files into text, but long documents usually cannot be sent directly into an embedding model or an LLM.

RAG first splits long documents into smaller, semantically coherent pieces called **chunks**. Later embedding, vector storage, and retrieval all use chunks as the basic unit.

```text
raw document
  ↓
data loading
  ↓
text chunking
  ↓
embedding
  ↓
vector store
  ↓
retrieval
```

This section follows three examples:

| Example file | Strategy |
| --- | --- |
| `character_splitter.py` | character-based splitting |
| `recursive_character_splitter.py` | recursive character splitting |
| `semantic_chunker.py` | semantic chunking |

## 1. Why Text Chunking Is Needed

At least two components in a RAG system have length limits:

1. the embedding model input window
2. the LLM context window

### 1. Embedding Model Context Window

An embedding model converts text into vectors. These vectors are stored in a vector store and used for retrieval.

Embedding models also have input length limits, often called **context window** or **max sequence length**.

For example, the course uses `BAAI/bge-m3`, whose max input length is **8192 tokens**. This means a chunk can theoretically be up to 8192 tokens.

However, this does not mean chunks should be as large as possible.

Embedding is a kind of semantic compression:

```text
long text
  ↓
embedding model
  ↓
one fixed-size vector
```

Whether the input has 100 tokens or 8000 tokens, it is usually compressed into one fixed-size vector. If the chunk contains many unrelated topics, this vector becomes blurry and less useful for precise retrieval.

### 2. LLM Context Window

The LLM also has a context window limit.

A RAG prompt usually contains:

```text
system instruction
user question
retrieved chunk 1
retrieved chunk 2
retrieved chunk 3
answer format
```

If each chunk is too large, fewer chunks can fit into the prompt. The LLM also needs to search through a longer context, which can lead to the **lost in the middle** problem.

The goal of chunking is therefore not to make chunks as small or as large as possible, but to balance semantic completeness and retrieval precision.

## 2. Problems with Chunks That Are Too Large or Too Small

### Chunks Too Large

| Problem | Description |
| --- | --- |
| Poor retrieval precision | One chunk may contain multiple topics |
| Important details get diluted | The query may match only a small part of the chunk |
| Fewer chunks fit into the prompt | Large chunks reduce context diversity |
| Lost in the middle | Key information in the middle may be ignored |

From the embedding perspective, large chunks mix many topics into one vector. Even if the model can accept a long input, one vector may not preserve all details precisely.

### Chunks Too Small

| Problem | Description |
| --- | --- |
| Not enough context | A chunk may contain only a fragment |
| Incomplete answers | Retrieved text may miss surrounding information |
| Too many chunks | Vector store size and retrieval cost increase |
| Broken meaning | Titles, paragraphs, or explanations may be separated |

Good chunks should be:

1. reasonably sized
2. semantically complete
3. focused on one topic
4. able to preserve necessary context
5. traceable to source

## 3. Important Parameters

### `chunk_size`

`chunk_size` controls the target size of each chunk.

```python
chunk_size=200
```

The examples use a small value to make chunking behavior easy to observe.

### `chunk_overlap`

`chunk_overlap` controls how much neighboring chunks overlap.

```python
chunk_overlap=10
```

Overlap helps avoid losing context at chunk boundaries.

### `separators`

`separators` defines the priority order for recursive splitting.

```python
separators=["\n\n", "\n", "。", "，", " ", ""]
```

For Chinese text, punctuation such as `。` and `，` is important because Chinese does not use spaces as word boundaries.

## 4. LangChain Splitting Strategies

### 4.1 Character Splitting: `CharacterTextSplitter`

Example file:

```text
chapter/02_Data_Preprocessing/character_splitter.py
```

`CharacterTextSplitter` is the most straightforward splitter. It is useful for understanding the basic idea of fixed-size chunks.

```python
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
)
```

| Aspect | Description |
| --- | --- |
| Basis | Character length and separators |
| Strength | Simple, fast, easy to observe |
| Limitation | Does not understand semantic boundaries |
| Good for | short text, simple text, teaching examples |
| Not good for | long articles, Markdown documents, semantic-sensitive data |

### 4.2 Recursive Character Splitting: `RecursiveCharacterTextSplitter`

Example file:

```text
chapter/02_Data_Preprocessing/recursive_character_splitter.py
```

Core code:

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],
    chunk_size=200,
    chunk_overlap=10,
)
```

Recursive splitting tries separators in priority order.

#### Splitting Flow

```text
split by paragraph
  ↓ if still too long
split by line break
  ↓ if still too long
split by sentence punctuation
  ↓ if still too long
split by comma, space, or character
```

This preserves larger semantic units when possible and only uses smaller separators when needed.

#### Separators for Chinese Text

For Chinese documents:

```python
separators=["\n\n", "\n", "。", "，", " ", ""]
```

You can add more punctuation:

```python
separators=[
    "\n\n",
    "\n",
    "。", "！", "？",
    "，", "、",
    " ",
    "",
]
```

#### Strengths and Limitations

| Aspect | Description |
| --- | --- |
| Strength | Preserves paragraphs and sentences better than fixed character splitting |
| Strength | Works well for Chinese, English, and general documents |
| Limitation | Still separator-based, not true semantic understanding |

### 4.3 Semantic Chunking: `SemanticChunker`

`SemanticChunker` does not split only by length or punctuation. It uses embeddings to detect where the meaning changes significantly.

Example file:

```text
chapter/02_Data_Preprocessing/semantic_chunker.py
```

Core code:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
)
```

#### Implementation Idea

`SemanticChunker` works in five steps.

First, it splits the text into sentences.

Second, it creates context-aware embeddings. A single sentence may be too short, so `buffer_size` can include neighboring sentences when creating the embedding.

```text
representation of sentence i
= previous buffer_size sentences
  + sentence i
  + next buffer_size sentences
```

If `buffer_size=1`, the embedding for sentence 3 may be based on:

```text
sentence 2 + sentence 3 + sentence 4
```

Third, it calculates semantic distance between neighboring sentence embeddings.

Fourth, it detects breakpoints. Large semantic distances are treated as possible split points.

Fifth, it merges sentences between breakpoints into final chunks.

#### Important Parameters

| Parameter | Description |
| --- | --- |
| `embeddings` | Embedding model used to compute semantic vectors |
| `breakpoint_threshold_type` | Statistical method for detecting breakpoints |
| `breakpoint_threshold_amount` | Threshold value; meaning depends on threshold type |
| `buffer_size` | Number of neighboring sentences included during embedding |

#### `breakpoint_threshold_type`

| Value | Logic | Best for |
| --- | --- | --- |
| `percentile` | Treat distances above a percentile as breakpoints | Good default |
| `standard_deviation` | Use mean plus N standard deviations | Distance distribution close to normal |
| `interquartile` | Use IQR to detect unusually large distances | Reducing outlier influence |
| `gradient` | Use distance change rate to find turns | Long documents with smooth topic shifts |

#### `buffer_size`

| Setting | Effect |
| --- | --- |
| `buffer_size=0` | Sentence only; more sensitive but less context |
| `buffer_size=1` | Includes one sentence before and after; usually more stable |
| larger `buffer_size` | Smoother semantic representation, possibly fewer breakpoints |

Semantic chunking is useful for long documents, topic shifts, meeting transcripts, and knowledge documents. It is slower than character splitting because it needs an embedding model.

### 4.4 Markdown Structure-Based Chunking

Markdown documents already have structure:

```text
[H1] Chapter Title

[H2] Section Title

Paragraph content...

[H3] Subsection

More content...
```

For Markdown, it is usually better to preserve heading structure instead of splitting only by character count.

#### Why Preserve Headings?

Headings provide context. For example:

```text
It can reduce hallucination.
```

This sentence is unclear alone. Under this heading:

```text
[H2] Benefits of RAG
```

the meaning becomes clearer.

#### Markdown Chunking Strategy

Common approach:

1. Split by Markdown headings.
2. Preserve heading path as metadata.
3. If a section is still too long, apply recursive chunking.
4. Add heading metadata to chunks.

#### Example

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter


headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

docs = markdown_splitter.split_text(markdown_text)
```

## 5. Comparison

| Method | Strength | Limitation | Best for |
| --- | --- | --- | --- |
| `CharacterTextSplitter` | Simple and fast | Not semantic-aware | simple text, teaching |
| `RecursiveCharacterTextSplitter` | Preserves paragraphs and sentences | Still separator-based | general documents, Chinese text |
| `SemanticChunker` | Splits by semantic shifts | Slower, needs embeddings | long documents, topic changes |
| Markdown structure chunking | Preserves heading metadata | Requires structured Markdown | README, technical docs |

## 6. How to Choose Chunk Parameters

Suggested starting points:

| Document type | Suggested method |
| --- | --- |
| short plain text | `CharacterTextSplitter` or `RecursiveCharacterTextSplitter` |
| Chinese tutorial article | `RecursiveCharacterTextSplitter` with Chinese punctuation |
| Markdown document | split by headings first, then recursive chunking |
| long report | `RecursiveCharacterTextSplitter` or `SemanticChunker` |
| topic-shifting article | `SemanticChunker` |
| code document | split by function / class or use language-aware splitters |

Experiment flow:

1. Start with `chunk_size=300~800`.
2. Set `chunk_overlap=30~100`.
3. Inspect random chunks.
4. Test retrieval with real questions.
5. Adjust based on retrieval quality.

## 7. Exercises

### Exercise 1: Adjust `chunk_size`

Try:

```python
chunk_size=100
```

and:

```python
chunk_size=500
```

Compare chunk count and completeness.

### Exercise 2: Adjust `chunk_overlap`

Try:

```python
chunk_overlap=50
```

Check whether neighboring chunks preserve more context.

### Exercise 3: Modify Chinese Separators

Try:

```python
separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
```

### Exercise 4: Compare Semantic Chunking

Run:

```powershell
python chapter/02_Data_Preprocessing/semantic_chunker.py
```

Compare it with recursive chunking.

## Summary

Text chunking directly affects retrieval quality. Large chunks dilute meaning; tiny chunks lose context.

In practice, `RecursiveCharacterTextSplitter` is often a good default. Use `SemanticChunker` when semantic boundaries matter, and use Markdown structure when the source document already has clear headings.

## References

- [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [LangChain MarkdownHeaderTextSplitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)
- [Datawhale all-in-rag text chunking chapter](https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter2/05_text_chunking.md)
