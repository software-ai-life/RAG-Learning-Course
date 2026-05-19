# Agent Memory and Context Engineering

Agent memory is the mechanism that allows an AI agent to preserve useful information across turns, sessions, or tasks.

Context engineering is broader. It is the practice of deciding what information should be placed into the model context window at the right time.

Memory and context engineering are related, but they are not the same thing.

## Short-Term Memory

Short-term memory usually refers to information from the current conversation or current task. It may include:

- recent user messages
- intermediate reasoning notes
- tool results
- temporary task state

Short-term memory is often stored in the prompt or session state.

## Long-Term Memory

Long-term memory stores information that should persist beyond a single conversation. It may include:

- user preferences
- project facts
- previous decisions
- summaries of past interactions
- reusable knowledge

Long-term memory often requires retrieval, filtering, and update policies.

## Context Engineering

Context engineering decides which information should be sent to the LLM. Too little context can make the model miss important facts. Too much context can introduce noise and increase cost.

Good context engineering requires:

- selecting relevant memory
- filtering irrelevant history
- preserving tool outputs
- compressing long conversations
- keeping source references

## Common Failure

If an agent keeps all previous messages in context, the prompt can become noisy and expensive.

If an agent stores memory without source tracking, the system may reuse outdated or incorrect facts.

## Retrieval Notes

This document is useful for semantic questions such as:

- What is agent memory?
- How is memory different from context engineering?
- Why does an agent need long-term memory?

Important exact terms:

- session memory
- long-term memory
- context engineering
- memory retrieval
- context window

