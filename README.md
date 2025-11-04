# logseq-chat (local RAG over Logseq)

A fully local RAG pipeline using LlamaIndex + Ollama + Chroma to query your Logseq notes.

## Prereqs
- Python 3.13+
- Ollama running (https://ollama.com)
- Pull a chat and embedding model:
  ```bash
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ```

  or (lighter weight):

  ```bash
  ollama pull llama3.1
  ollama pull all-minilm
  ```

## Setup
```bash
cd logseq-chat
make install
```

Edit `config.yaml` and at a minimum set `logseq_root` to your Logseq graph directory.

## Build index
```bash
make ingest
```

## Chat
```bash
make chat
```

## Tests
```bash
make test
```

### Example questions
- Summarize tasks tagged #home in October 2025.
- Find notes referencing [[Team Topologies]] and list my pros/cons.

## Notes
- Skips `assets/` by default. Enable OCR later if needed.
- Uses Markdown-aware chunking; tags from `#tag` and `tags::` stored in metadata.
- For faster machines, try bigger models; for CPU-only, consider `llama3.2` or `qwen2.5:7b` and smaller chunks.
