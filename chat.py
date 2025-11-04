import os
import yaml
import chromadb

from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

CONFIG = yaml.safe_load(open("config.yaml", "r"))

def build_query_engine():
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
