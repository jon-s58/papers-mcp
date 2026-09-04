# Papers MCP (`papers-mcp`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Spec](https://img.shields.io/badge/MCP-1.28+-green.svg)](https://modelcontextprotocol.io/)

A local, high-accuracy academic paper retrieval and **Model Context Protocol (MCP)** server designed for AI coding agents and LLM research workflows.

Turn any folder of research PDFs into an indexed, queryable academic corpus with state-of-the-art hybrid search, LaTeX-aware chunking, cross-encoder reranking, and full MCP tool integration.

---

## Key Features

* **Advanced Hybrid Retrieval Pipeline**:
  * **Lexical Search**: SQLite FTS5 BM25 with query normalization and phrase matching.
  * **Dense Semantic Search**: `Qwen/Qwen3-Embedding-4B` (2048-token context, bfloat16, MPS/CUDA accelerated).
  * **Reciprocal Rank Fusion (RRF)**: Combines dense and lexical candidate pools ($k=60$).
  * **Cross-Encoder Reranking**: `Qwen/Qwen3-Reranker-4B` reranks top candidates with instruction-guided relevance scoring.
* **Math & LaTeX-Aware Chunking**:
  * Extracts structural document trees via `pymupdf4llm`.
  * Preserves atomic LaTeX environments (`equation`, `align`, `cases`, `Theorem`, `Proof`) without splitting mathematical proofs across chunk boundaries.
* **Full Model Context Protocol (MCP) Support**:
  * Exposes 7 tools allowing LLMs to search, outline, inspect sections, and find exact citations.
* **100% Local & Private**:
  * Runs entirely on your local machine (Apple Silicon MPS or NVIDIA CUDA). No external API calls required.
* **Robust Crash Isolation**:
  * PDF extraction runs in isolated worker processes with memory limits and timeouts to prevent segfaults on malformed PDFs.

---

## MCP Tools Exposed

When connected to an MCP host (Claude Code, Claude Desktop, Cursor, Hermes Agent), the following tools are available:

| Tool | Description |
|---|---|
| `search_papers` | Hybrid BM25 + dense search over paper chunks with reciprocal rank fusion. |
| `research_search` | End-to-end multi-stage search with cross-encoder reranking and snippet extraction. |
| `paper_outline` | Returns the hierarchical table of contents and section structure of a paper. |
| `read_section` | Reads a specific section from a paper using exact token-budgeted pagination. |
| `expand_context` | Expands preceding and succeeding chunks around a search hit. |
| `find_in_paper` | Exact regex/string search within a specific paper. |
| `related_papers` | Discovers conceptually related papers via embedding nearest neighbors. |

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jon-s58/papers-mcp.git
cd papers-mcp

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# (Optional) Install PyTorch and model dependencies for dense retrieval & reranking
pip install -e ".[models]"
```

### 2. Configuration

Copy the example configuration:

```bash
cp config.example.toml config.toml
```

Edit `config.toml` to point `pdf_roots` to your local folder of PDF papers:

```toml
[paths]
pdf_roots = ["./papers"]     # Path to your research papers folder
database = "./data/papers.db"
```

### 3. Ingestion

Ingest and index your papers:

```bash
# Fast extraction & lexical index (without waiting for GPU embeddings)
papers-mcp ingest --no-embeddings

# Or full extraction + GPU dense embeddings
papers-mcp ingest
```

### 4. Search via CLI

```bash
# Test lexical/hybrid search from terminal
papers-mcp search "transformer attention mechanism"
```

### 5. Run the MCP Server

```bash
papers-mcp serve
```

---

## MCP Client Configuration

### Claude Code (`claude.json` / `.mcp.json`)

```json
{
  "mcpServers": {
    "papers-research": {
      "command": "/path/to/papers-mcp/.venv/bin/papers-mcp",
      "args": ["serve", "--config", "/path/to/papers-mcp/config.toml"]
    }
  }
}
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "papers-research": {
      "command": "/path/to/papers-mcp/.venv/bin/papers-mcp",
      "args": ["serve", "--config", "/path/to/papers-mcp/config.toml"]
    }
  }
}
```

---

## Architecture Overview

```
PDFs / Markdown
       │
       ▼
[ PyMuPDF4LLM ] ────────► Structural AST / Markdown
       │
       ▼
[ LaTeX-Aware Chunker ] ──► Target 850 Tokens, Preserves Equations
       │
       ├──────────────────────────┐
       ▼                          ▼
[ SQLite FTS5 BM25 ]     [ Qwen3-Embedding-4B ]
       │                          │
       └───────────┬──────────────┘
                   ▼
         [ Reciprocal Rank Fusion ]
                   │
                   ▼
        [ Qwen3-Reranker-4B ]
                   │
                   ▼
          MCP Tools / Agent Output
```

---

## License

This project is licensed under the [MIT License](LICENSE).
