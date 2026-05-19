# Production Agent Error Codes

This document lists common error codes that may appear in production AI agent systems.

These codes are intentionally written as exact identifiers because keyword search should retrieve them reliably.

## ERR-4291: Tool Rate Limit Exceeded

`ERR-4291` means the agent called an external tool too frequently.

Common causes:

- the planner retries too aggressively
- `max_retries` is too high
- multiple agents share the same API quota
- the tool provider returns HTTP 429

Recommended fixes:

- add exponential backoff
- reduce parallel tool calls
- cache repeated tool results
- lower `max_retries`
- monitor token and API usage

## ERR-5007: Tool Schema Validation Failed

`ERR-5007` means the tool call arguments do not match the tool schema.

Common causes:

- missing required parameter
- wrong parameter type
- outdated schema version
- model generated extra fields

Recommended fixes:

- validate arguments before execution
- return clear schema errors to the model
- update the tool description
- add few-shot examples for tool usage

## ERR-7132: Memory Write Conflict

`ERR-7132` means two memory updates conflict with each other.

Common causes:

- concurrent sessions update the same memory key
- memory store does not support versioning
- stale context writes outdated facts

Recommended fixes:

- use optimistic locking
- attach timestamps and source references
- ask the model to summarize conflicting memories
- keep an audit log

## Retrieval Notes

This document is useful for exact error-code queries:

- ERR-4291
- ERR-5007
- ERR-7132

Dense retrieval may understand the meaning of rate limits or schema validation, but sparse retrieval is better at exact error-code lookup.

