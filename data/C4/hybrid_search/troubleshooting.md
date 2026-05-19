# AI Agent Troubleshooting Guide

This guide explains common troubleshooting steps for AI agent systems.

Production agents can fail for many reasons: tool errors, retrieval misses, memory pollution, prompt drift, or model output format problems.

## Agent Does Not Use the Right Tool

Symptoms:

- the agent answers from memory instead of calling a tool
- the agent calls an unrelated tool
- the agent refuses to call a tool even when the task requires it

Possible causes:

- tool description is too vague
- tool schema is missing required examples
- retrieval did not include tool documentation
- system prompt does not define tool-use policy

Fixes:

- improve tool descriptions
- add examples for when to use each tool
- add metadata filter for tool documentation
- evaluate tool-call accuracy

## Agent Hallucinates After Retrieval

Symptoms:

- retrieved context is correct
- final answer adds unsupported claims
- answer cites a source that does not support the claim

Possible causes:

- prompt does not require grounded answers
- context contains irrelevant noise
- model overgeneralizes from partial evidence

Fixes:

- require citation for every major claim
- add faithfulness evaluation
- use context compression
- add a reranker

## Agent Retrieves the Wrong Document

Symptoms:

- answer is fluent but based on the wrong source
- correct source exists but is not in top-k
- similar documents confuse the retriever

Possible causes:

- chunking is too coarse
- metadata is missing
- query contains exact identifiers that dense retrieval ignores

Fixes:

- add metadata fields
- use hybrid search
- increase top-k before reranking
- add BM25 for exact terms

## Retrieval Notes

This document is useful for troubleshooting questions such as:

- Why did the agent use the wrong tool?
- How do I fix hallucination after retrieval?
- Why does dense retrieval miss an exact error code?
- When should I add hybrid search?

