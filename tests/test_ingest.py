import importlib
import sys
from pathlib import Path
import textwrap

import pytest


@pytest.fixture(scope="session")
def ingest_module():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    created = False

    if not config_path.exists():
        config_path.write_text(
            textwrap.dedent(
                """
                logseq_root: /tmp
                include_dirs: []
                file_exts: []
                exclude_globs: []
                models:
                  llm: llama3.1
                  embedding: nomic-embed-text
                storage:
                  chroma_path: /tmp/chroma
                retrieval:
                  top_k: 5
                  mmr: false
                chunk:
                  chunk_size: 512
                  chunk_overlap: 50
                """
            ).strip()
        )
        created = True

    try:
        if "ingest" in sys.modules:
            module = sys.modules["ingest"]
        else:
            module = importlib.import_module("ingest")
        yield module
    finally:
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
