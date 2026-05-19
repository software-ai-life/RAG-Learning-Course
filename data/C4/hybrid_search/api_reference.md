# Agent Framework API Reference Notes

This document contains exact API and class names used in agent and RAG systems.

Keyword retrieval is important because these names may not be semantically obvious.

## MetadataReplacementPostProcessor

`MetadataReplacementPostProcessor` is commonly used in sentence window retrieval.

It replaces the retrieved node text with a larger context stored in metadata.

For example, the retriever may find a single sentence, but the postprocessor can replace it with a surrounding sentence window before the answer is generated.

Useful when:

- small chunks are needed for precise retrieval
- larger context is needed for generation
- sentence-level retrieval is used

## RecursiveRetriever

`RecursiveRetriever` can follow an `IndexNode` to another retriever.

It is useful when the system first retrieves a high-level summary node, then enters a more detailed child retriever.

Example flow:

```text
root retriever
-> IndexNode(index_id="music_2010")
-> child retriever for music_2010
-> detailed results
```

## AnnSearchRequest

`AnnSearchRequest` is used in Milvus hybrid search to define one vector search request.

A hybrid search may create:

- one request for dense vectors
- one request for sparse vectors

Then a ranker such as `RRFRanker` merges the results.

## RRFRanker

`RRFRanker` applies Reciprocal Rank Fusion.

It combines multiple ranked result lists without requiring the raw scores to have the same scale.

## Retrieval Notes

This document is useful for exact API queries:

- MetadataReplacementPostProcessor
- RecursiveRetriever
- IndexNode
- AnnSearchRequest
- RRFRanker

It is also useful for semantic questions about sentence window retrieval, recursive retrieval, and Milvus hybrid search.

