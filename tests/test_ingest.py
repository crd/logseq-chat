import importlib
import sys
from pathlib import Path
import textwrap

import importlib
import sys
import types
from pathlib import Path
import textwrap

import pytest


def _install_dependency_stubs():
    if "chromadb" not in sys.modules:
        chromadb = types.ModuleType("chromadb")

        class _Collection:
            pass

        class _Client:
            def __init__(self, *_, **__):
                self._collection = _Collection()

            def get_or_create_collection(self, *_args, **_kwargs):
                return self._collection

            def delete_collection(self, *_args, **_kwargs):
                return None

        chromadb.PersistentClient = _Client
        sys.modules["chromadb"] = chromadb

    if "llama_index" not in sys.modules:
        root = types.ModuleType("llama_index")
        sys.modules["llama_index"] = root

    if "llama_index.core" not in sys.modules:
        core = types.ModuleType("llama_index.core")

        class _DummyDocument:
            def __init__(self, text: str, metadata: dict):
                self.text = text
                self.metadata = metadata

        class _DummySettings:
            llm = None
            embed_model = None

        class _DummyStorageContext:
            @classmethod
            def from_defaults(cls, **_kwargs):
                return cls()

        class _DummyVectorStoreIndex:
            def __init__(self, *_, **__):
                pass

        core.Document = _DummyDocument
        core.Settings = _DummySettings
        core.StorageContext = _DummyStorageContext
        core.VectorStoreIndex = _DummyVectorStoreIndex
        sys.modules["llama_index.core"] = core

    if "llama_index.core.node_parser" not in sys.modules:
        node_parser = types.ModuleType("llama_index.core.node_parser")

        class _Parser:
            @classmethod
            def from_defaults(cls, **_kwargs):
                return cls()

            def get_nodes_from_documents(self, documents):
                return documents

        node_parser.SimpleNodeParser = _Parser
        sys.modules["llama_index.core.node_parser"] = node_parser

    if "llama_index.embeddings.ollama" not in sys.modules:
        embeddings = types.ModuleType("llama_index.embeddings.ollama")

        class _DummyEmbedding:
            def __init__(self, *_, **__):
                pass

        embeddings.OllamaEmbedding = _DummyEmbedding
        sys.modules["llama_index.embeddings.ollama"] = embeddings

    if "llama_index.llms.ollama" not in sys.modules:
        llms = types.ModuleType("llama_index.llms.ollama")

        class _DummyLLM:
            def __init__(self, *_, **__):
                pass

        llms.Ollama = _DummyLLM
        sys.modules["llama_index.llms.ollama"] = llms

    if "llama_index.vector_stores.chroma" not in sys.modules:
        vector_store = types.ModuleType("llama_index.vector_stores.chroma")

        class _DummyVectorStore:
            def __init__(self, *_, **__):
                pass

        vector_store.ChromaVectorStore = _DummyVectorStore
        sys.modules["llama_index.vector_stores.chroma"] = vector_store

    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")

        def _safe_load(data):
            return {}

        def _safe_dump(_data, _fh, **_kwargs):
            return None

        yaml_stub.safe_load = _safe_load
        yaml_stub.safe_dump = _safe_dump
        sys.modules["yaml"] = yaml_stub


@pytest.fixture(scope="session")
def ingest_module():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    created = False

    _install_dependency_stubs()

    if not config_path.exists():
        config_path.write_text(
            textwrap.dedent(
                """
                logseq_root: /tmp
                include_dirs: []
                file_exts: []
                exclude_globs: []
                runtime:
                  request_timeout: 30
                models:
                  llm:
                    name: llama3.1
                    temperature: 0.0
                  embedding:
                    name: nomic-embed-text
                storage:
                  chroma_path: /tmp/chroma
                  collection_name: test_collection
                  clear_before_ingest: true
                retrieval:
                  top_k: 5
                  mmr:
                    enabled: false
                  query_expansion:
                    enabled: false
                chunk:
                  chunk_size: 512
                  chunk_overlap: 50
                evaluation:
                  dataset: evaluations/datasets/baseline.yaml
                  configurations_file: evaluations/configurations.yaml
                  max_queries: null
                  scoring:
                    accuracy_weight: 0.35
                    coverage_weight: 0.2
                    relevance_weight: 0.2
                    hallucination_weight: 0.15
                    speed_weight: 0.1
                """
            ).strip()
        )
        created = True

    added_to_path = False
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        added_to_path = True

    try:
        if "ingest" in sys.modules:
            module = sys.modules["ingest"]
        else:
            module = importlib.import_module("ingest")
        yield module
    finally:
        if added_to_path and str(project_root) in sys.path:
            sys.path.remove(str(project_root))
        if created and config_path.exists():
            config_path.unlink()


def test_normalize_logseq_links(ingest_module):
    text = "Follow [[Page Name]] then see ((abc123))."
    result = ingest_module.normalize_logseq_links(text)
    assert result == "Follow Page Name then see [ref:abc123]."


def test_parse_tags_combines_sources(ingest_module):
    text = """
    #alpha introduces the topic
    Another line with #beta and #alpha
    tags:: gamma, beta , delta
    """
    result = ingest_module.parse_tags(text)
    assert result == ["alpha", "beta", "delta", "gamma"]


def test_page_title_from_path(ingest_module):
    path = "/tmp/logseq/pages/project_notes.md"
    assert ingest_module.page_title_from_path(path) == "project-notes"


def test_collect_files_respects_ext_and_excludes(tmp_path, ingest_module):
    pages = tmp_path / "pages"
    journals = tmp_path / "journals"
    archive = pages / "archive"
    pages.mkdir()
    journals.mkdir()
    archive.mkdir()

    keep_pages = pages / "alpha.md"
    keep_journal = journals / "2025-01-01.md"
    ignore_ext = pages / "ignore.txt"
    excluded = archive / "old.md"

    keep_pages.write_text("alpha")
    keep_journal.write_text("journal")
    ignore_ext.write_text("nope")
    excluded.write_text("archive")

    found = ingest_module.collect_files(
        str(tmp_path),
        ["pages", "journals"],
        [".md"],
        ["pages/archive/*"],
    )

    assert set(found) == {str(keep_pages), str(keep_journal)}


def test_load_documents_applies_metadata(monkeypatch, tmp_path, ingest_module):
    docs_dir = tmp_path / "pages"
    docs_dir.mkdir()
    doc_path = docs_dir / "demo_page.md"
    doc_path.write_text(
        """
        #alpha tag at the top
        tags:: beta, alpha
        Content referencing [[Other Page]] and ((xyz789)).
        """
    )

    class DummyDocument:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    monkeypatch.setattr(ingest_module, "Document", DummyDocument)

    docs = ingest_module.load_documents([str(doc_path)])

    assert len(docs) == 1
    doc = docs[0]
    assert doc.text.strip().startswith("#alpha tag at the top")
    assert "[[" not in doc.text and "((" not in doc.text
    assert doc.metadata["source"] == str(doc_path)
    assert doc.metadata["title"] == "demo-page"
    assert doc.metadata["tags"] == "alpha, beta"
    assert doc.metadata["basename"] == "demo_page.md"
    assert doc.metadata["dir"] == "pages"
