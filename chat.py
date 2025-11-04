"""Simple command-line chat client for exploring the indexed Logseq graph.

This module keeps the runtime experience intentionally transparent: it shows how
to rebuild a query engine from the stored embeddings and how to send natural
language questions to it. The print statements highlight how answers relate to
the original notes.
"""

import chromadb
import yaml
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

def build_query_engine():
    """Create a ``QueryEngine`` that can answer questions over the Logseq index.

    The steps here mirror the high-level components of a RAG system: choose an
    LLM, choose an embedding model, open the vector store, then ask LlamaIndex
    for a query interface. Reading through the code reinforces the mental model
    introduced in ``ingest.py``.

    Returns
    -------
    BaseQueryEngine
        The object that exposes ``query("...")`` for the interactive loop.
    """

    # Models (local via Ollama)
    Settings.llm = Ollama(
        model=CONFIG["models"]["llm"],
        request_timeout=180,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=CONFIG["models"]["embedding"],
    )

    # Vector store
    client = chromadb.PersistentClient(path=CONFIG["storage"]["chroma_path"])
    collection = client.get_or_create_collection("logseq_rag")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Index from existing Chroma collection
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Let LlamaIndex create the retriever internally; pass our knobs only
    query_engine = index.as_query_engine(
        similarity_top_k=CONFIG["retrieval"]["top_k"],
        use_mmr=CONFIG["retrieval"]["mmr"],
    )
    return query_engine

def main():
    """Start an interactive chat loop backed by the previously ingested notes.

    Type questions in plain English to see how the retriever surfaces relevant
    pages. Use ``:q`` to exit when you are done experimenting.
    """

    print("Loading query engine...")
    qe = build_query_engine()
    print("Ready. Type your question (or :q to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q == ":q":
            break

        resp = qe.query(q)

        print("\n--- Answer ---")
        print(resp.response)

        print("\n--- Top refs ---")
        for s in resp.source_nodes[:5]:
            meta = s.node.metadata or {}
            title = meta.get("title", "(untitled)")
            d = meta.get("dir")
            src = meta.get("source")
            tags_csv = meta.get("tags")  # CSV string or None
            if tags_csv:
                print(f"{title} [{d}] tags: {tags_csv} -> {src}")
            else:
                print(f"{title} [{d}] -> {src}")
        print()

if __name__ == "__main__":
    main()
