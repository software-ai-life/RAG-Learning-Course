# Section 2: Multimodal Embedding

The previous section focused on text embeddings: converting text into vectors and retrieving semantically similar content.

However, real-world data is not text-only. Enterprise documents, course materials, product manuals, academic papers, slides, and web pages often contain:

```text
text
images
tables
charts
screenshots
audio
video
```

If a RAG system can only process text, it may fail to answer questions such as:

```text
What does this architecture diagram explain?
Find images similar to this product photo.
Which page in the PDF contains a flowchart?
Which document section is most related to the uploaded image?
```

Multimodal embeddings extend retrieval from text-only data to multiple data types.

## 1. What Is Multimodal Embedding?

A multimodal embedding model converts different modalities into vectors.

For example:

```text
text -> vector
image -> vector
text + image -> vector
```

The key idea is to make different modalities searchable in a shared or compatible vector space.

For example:

```text
Text: "a cat sitting on a sofa"
Image: a photo of a cat on a sofa
```

If the text vector and image vector are close, the system can retrieve the image with a text query or retrieve related text with an image query.

![Multimodal embedding shared space](./images/multimodal_embedding.png)

## 2. Why Multimodal Embedding Matters for RAG

Many documents contain important information in visual form.

Examples:

| Data Type | Why It Matters |
| --- | --- |
| Architecture diagrams | Relationships may be shown through arrows and blocks. |
| Charts | Trends may be easier to understand visually. |
| Screenshots | UI state and layout may not appear in raw text. |
| Tables | Structure matters as much as text. |
| Product images | Visual similarity may be the main retrieval target. |

For many PDF-based RAG systems, OCR is the first step. But OCR only extracts text. If the image itself contains important semantic meaning, multimodal embedding or image captioning may be needed.

## 3. CLIP: A Representative Image-Text Embedding Model

CLIP is a representative model for image-text embedding.

Its core idea is simple:

```text
matching image-text pairs should be close
non-matching image-text pairs should be far apart
```

CLIP uses two encoders:

| Encoder | Input | Output |
| --- | --- | --- |
| Image encoder | Image | Image vector |
| Text encoder | Text | Text vector |

Because image and text vectors are aligned, CLIP can support:

```text
text-to-image search
image-to-text search
image similarity search
zero-shot image classification
```

![CLIP](./images/CLIP.png)

## 4. Common Multimodal Retrieval Patterns

### 4.1 Text-to-Image Retrieval

The user enters a text query:

```text
"a diagram about agent memory"
```

The system embeds the text and retrieves the most similar images.

### 4.2 Image-to-Image Retrieval

The user uploads an image. The system embeds the query image and finds visually or semantically similar images.

### 4.3 Image-to-Text Retrieval

The user uploads an image, and the system retrieves related document sections, captions, or descriptions.

### 4.4 Text + Image Query

The query may contain both image and text:

```text
uploaded image + "find slides that explain this architecture"
```

This is useful for visual question answering and multimodal RAG.

## 5. Common Multimodal Embedding Models

### 5.1 Visualized-BGE / bge-visualized-m3

`bge-visualized-m3` belongs to the Visualized-BGE family from BAAI.

It extends the BGE text embedding framework with visual capability, allowing the model to work with text, images, and image-text data.

It inherits important ideas from `BAAI/bge-m3`:

| Feature | Description |
| --- | --- |
| Multilinguality | Useful for multilingual or mixed-language data. |
| Multi-functionality | Can support multiple retrieval needs. |
| Multi-granularity | Can work with short text and longer documents. |

For this course, it is useful for understanding how text embedding can be extended to image-text embedding.

### 5.2 Gemini Embedding

Gemini embedding can be used through an API-based workflow.

It is useful when you want a hosted model and do not want to manage local multimodal embedding infrastructure.

In the course example, Gemini embedding can be used to embed image or image-text inputs and then store the vectors in a vector database such as Milvus.

## 6. Multimodal Leaderboards

For evaluating multimodal embedding models, leaderboards can provide a useful reference.

The MMEB leaderboard is one such reference:

https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard

![MMEB leaderboard](./images/MMEB_leaderboard.png)

Leaderboards should not be the only selection criterion, but they help compare model performance across different image-text retrieval tasks.

## 7. Metadata Design for Multimodal RAG

Multimodal RAG requires careful metadata design.

For an image extracted from a PDF, useful metadata may include:

```python
metadata = {
    "source": "agent_course.pdf",
    "page": 8,
    "modality": "image",
    "image_path": "images/page_008_fig_01.png",
    "caption": "Agent memory architecture diagram",
}
```

For text:

```python
metadata = {
    "source": "agent_course.pdf",
    "page": 8,
    "modality": "text",
    "section": "Memory and Context Engineering",
}
```

Metadata can be stored in the vector database as scalar fields. It helps with:

```text
source citation
metadata filtering
debugging retrieval
separating text and image results
reconstructing the original document context
```

## 8. OCR or Multimodal Embedding?

For PDFs with images, the choice depends on what kind of information matters.

| Situation | Recommended Approach |
| --- | --- |
| The image mainly contains text | OCR first. |
| The image is a chart, diagram, or screenshot | OCR + captioning or multimodal embedding. |
| The task is text Q&A | OCR-based text RAG is usually enough for the first version. |
| The task needs visual understanding | Add multimodal embedding or VLM-based processing. |

For course PDFs, a practical path is:

```text
Version 1: OCR -> Markdown -> Text RAG
Version 2: generate captions for important figures
Version 3: add image embeddings and multimodal retrieval
```

## 9. Key Takeaways

1. Multimodal embedding allows text, images, and image-text data to be searched together.
2. CLIP is a representative image-text embedding model.
3. Multimodal RAG is useful when visual information matters.
4. OCR is still important for extracting text from PDFs and images.
5. Metadata design is critical for multimodal retrieval and source tracking.
