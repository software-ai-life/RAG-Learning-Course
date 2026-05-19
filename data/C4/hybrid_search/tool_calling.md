# Tool Calling and JSON-RPC Requests

Tool calling allows an AI agent to interact with external functions, APIs, databases, or services.

The LLM decides that a tool is needed, fills in the required arguments, and sends a structured request. The tool executes the operation and returns a result.

## Tool Schema

A tool schema describes:

- tool name
- tool description
- required parameters
- optional parameters
- parameter types
- return format

For example, a weather tool may require:

```json
{
  "location": "Taipei",
  "unit": "celsius"
}
```

If the schema is unclear, the model may call the tool with missing or invalid arguments.

## JSON-RPC

Some tool systems use **JSON-RPC** as the message format.

JSON-RPC requests usually contain:

- `jsonrpc`
- `method`
- `params`
- `id`

Example:

```json
{
  "jsonrpc": "2.0",
  "method": "tools.call",
  "params": {
    "name": "search_documents",
    "arguments": {
      "query": "agent memory"
    }
  },
  "id": "call-001"
}
```

## Common Parameters

Important fields in tool-calling systems include:

- `tool_name`
- `arguments`
- `timeout_seconds`
- `max_retries`
- `schema_version`
- `request_id`

## Retrieval Notes

This document is useful for exact keyword queries such as:

- JSON-RPC
- tools.call
- schema_version
- request_id
- timeout_seconds

It is also useful for semantic questions such as:

- How do agents call tools?
- Why are tool schemas important?
- What happens when tool arguments are invalid?

