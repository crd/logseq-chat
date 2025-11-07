# logseq-chat (local RAG over Logseq)

A fully local RAG pipeline using LlamaIndex + Ollama + Chroma to query your [Logseq](https://logseq.com/) notes.

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

Copy `config.yaml.sample` to `config.yaml` and customise the values. The sample
lists tuned defaults plus alternative values (chunk sizes, retrieval depth,
synonym lists, etc.) that you can toggle as you experiment. At minimum set
`logseq_root` to your Logseq graph directory.

## Configuration cheat sheet
- **Chunk size / overlap** – controls how much context each embedding sees.
  Smaller chunks with slightly larger overlaps (`chunk_size: 650`,
  `chunk_overlap: 160`) improve recall; larger chunks (`chunk_size: 1200`) speed
  things up on slower machines.
- **Retrieval depth** – adjust `retrieval.top_k` and `retrieval.mmr.enabled` to
  trade recall for latency.
- **Query expansion** – populate `retrieval.query_expansion.synonyms` with
  domain-specific vocabulary. Asking “What did I write about sailing?” will also
  search for “sloop” and “schooner” with the default config.
- **Model temperature** – lower values keep answers grounded; increase towards
  `0.3` for more conversational replies.

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

## Evaluate presets
```bash
make evaluate
```

The evaluation harness ingests your graph for each preset listed in
`evaluations/configurations.yaml`, runs the labelled queries from
`evaluations/datasets/baseline.yaml`, and prints a leaderboard ranked by the
weighted scoring formula defined in `config.yaml`. The bundled presets are:

| Name        | Purpose                                           |
| ----------- | ------------------------------------------------- |
| balanced    | Default profile – accuracy, coverage, and speed.  |
| high_recall | Smaller chunks, deeper retrieval, more overlap.   |
| fast_local  | Larger chunks, shallow retrieval for quick tests. |

After the run, the best-scoring configuration is reported and summarised in
`evaluations/results/latest.yaml`. Use that preset as a starting point for new
experiments or promote it to your day-to-day `config.yaml`.

### Example questions
- Summarize tasks tagged #home in October 2025.
- Find notes referencing [[Team Topologies]] and list my pros/cons.

## Notes
- Skips `assets/` by default. Enable OCR later if needed.
- Uses Markdown-aware chunking; tags from `#tag` and `tags::` stored in metadata.
- The default configuration enables targeted synonym expansion to improve recall
  for concept-driven queries (e.g. “sailing” → “sloop”, “schooner”).
- For faster machines, try bigger models; for CPU-only, consider `llama3.2` or
  `qwen2.5:7b` and larger chunk sizes to reduce request volume.
