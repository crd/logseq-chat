"""Utilities for turning a Logseq graph into a retriever-friendly index.

This script walks through each stage of a Retrieval Augmented Generation (RAG)
workflow. The ``main`` function mirrors the typical steps:

1. Collect source files from Logseq.
2. Clean and tag the raw markdown.
3. Split the content into small, retrievable "nodes".
4. Persist the resulting embeddings into a vector database (Chroma).

Running ``python ingest.py`` demonstrates how raw notes are transformed into
something a chatbot can search. The helper functions below keep the individual
tasks digestible and ready for experimentation.
"""

import glob
import os
import pathlib
import re
from typing import List

import chromadb
import yaml
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

PAGE_LINK = re.compile(r"\[\[([^\]]+)\]\]")                 # [[Page]]
BLOCK_REF = re.compile(r"\(\(([a-zA-Z0-9_-]{6,})\)\)")       # ((block-id))
TAG_HASH  = re.compile(r"(?<!\w)#([A-Za-z0-9/_-]+)")            # #tag
TAG_PROP  = re.compile(r"^\s*tags::\s*(.+)$", re.MULTILINE)     # tags:: a, b

def normalize_logseq_links(text: str) -> str:
    """Replace Logseq-specific link syntax with plain text.

    Special markup can confuse language models. This helper demonstrates how to
    pre-process text so the downstream embedding model sees natural language
    instead of wiki-style tokens.
    """

    text = PAGE_LINK.sub(lambda m: m.group(1), text)
    text = BLOCK_REF.sub(lambda m: f"[ref:{m.group(1)}]", text)
    return text

def parse_tags(text: str) -> List[str]:
    """Extract Logseq tags from both inline ``#hashtags`` and ``tags::`` fields.

    The returned list is sorted to keep results predictable when you display or
    filter by tags later in the workflow.
    """

    tags = set()
    for m in TAG_HASH.finditer(text):
        tags.add(m.group(1))
    for m in TAG_PROP.finditer(text):
        raw = [t.strip(" ,#") for t in m.group(1).split(",")]
        for t in raw:
            if t:
                tags.add(t)
    return sorted(tags)

def page_title_from_path(path: str) -> str:
    """Convert a file path into a human-friendly Logseq page title."""

    name = pathlib.Path(path).stem
    return name.replace("_", "-")

def collect_files(root: str, include_dirs: List[str], file_exts: List[str], exclude_globs: List[str]) -> List[str]:
    """Locate Logseq files to ingest based on the config settings.

    This function connects the configuration knobs to the actual file system. It
    assembles the "raw corpus" that feeds the remaining steps of the pipeline.
    """

    files = []
    for rel in include_dirs:
        base = os.path.join(root, rel)
        for ext in file_exts:
            files.extend(glob.glob(os.path.join(base, f"**/*{ext}"), recursive=True))
    excluded = set()
    for pat in exclude_globs:
        excluded.update(glob.glob(os.path.join(root, pat), recursive=True))
    return [f for f in files if f not in excluded and os.path.isfile(f)]

def load_documents(paths: List[str]) -> List[Document]:
    """Read markdown files and create ``Document`` objects with helpful metadata.

    Each ``Document`` becomes a single unit of knowledge for LlamaIndex. We add
    metadata like the title, directory, and tags so the chat interface can show
    meaningful references instead of opaque file names.
    """

    docs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            continue

        clean = normalize_logseq_links(txt)

        # compute tags here so tags_csv is in scope
        tags_list = parse_tags(txt)
        tags_csv = ", ".join(tags_list) if tags_list else None

        title = page_title_from_path(p)
        meta = {
            "source": p,
            "title": title,
            "tags": tags_csv,  # scalar (str/None), not a list
            "basename": os.path.basename(p),
            "dir": os.path.basename(os.path.dirname(p)),
        }
        docs.append(Document(text=clean, metadata=meta))
    return docs

def main():
    """Run the full ingestion workflow using settings from ``config.yaml``.

    Running this function end-to-end shows how data collection, cleaning,
    chunking, and indexing fit together in a practical RAG pipeline.
    """

    root = CONFIG["logseq_root"]
    include_dirs = CONFIG["include_dirs"]
    file_exts = CONFIG["file_exts"]
    exclude = CONFIG["exclude_globs"]

    if not os.path.isdir(root):
        raise SystemExit(f"Logseq root does not exist: {root}\nEdit config.yaml to set logseq_root.")

    paths = collect_files(root, include_dirs, file_exts, exclude)
    print(f"Found {len(paths)} markdown files.")

    docs = load_documents(paths)
    print(f"Loaded {len(docs)} documents.")

    Settings.llm = Ollama(model=CONFIG["models"]["llm"], request_timeout=180)
    Settings.embed_model = OllamaEmbedding(model_name=CONFIG["models"]["embedding"])

    parser = SimpleNodeParser.from_defaults(
        include_metadata=True,
        chunk_size=CONFIG["chunk"]["chunk_size"],
        chunk_overlap=CONFIG["chunk"]["chunk_overlap"]
    )
    nodes = parser.get_nodes_from_documents(docs)
    print(f"Parsed into {len(nodes)} nodes.")

    chroma_path = CONFIG["storage"]["chroma_path"]
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("logseq_rag")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    _ = VectorStoreIndex(nodes, storage_context=storage_ctx)
    print("Index built and persisted to Chroma.")

if __name__ == "__main__":
    main()
