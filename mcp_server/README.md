# RegAgent MCP server

Plug RegAgent into any MCP-compatible host — Claude Desktop, an IDE assistant, or
the customer's own AI agent stack — as two tools:

- `ask_regulation(question, lang)` — a grounded, cited answer over the EU AI Act,
  DORA, GDPR and NIS2.
- `analyze_compliance(question, lang)` — decompose a "does our system comply?"
  question across regulations and synthesise one cited answer.

It's a **thin client**: it calls the hosted RegAgent API over HTTPS. Nothing —
no model, no corpus, no data — runs in your infrastructure. That's the point:
integrate the agent into your environment without deploying it.

## Setup

```bash
pip install -r requirements.txt
```

## Configure your MCP host

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "regagent": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/regagent_mcp.py"],
      "env": {
        "REGAGENT_URL": "https://Lagra21-regagent.hf.space",
        "REGAGENT_API_KEY": "your-tenant-key-if-any"
      }
    }
  }
}
```

`REGAGENT_API_KEY` is optional — set it when calling a tenant deployment that
requires a key (the public demo runs open). Then ask your assistant a compliance
question and it will call RegAgent for a grounded, cited answer.
