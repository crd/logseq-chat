.PHONY: venv install ingest chat evaluate lock clean

# Create/refresh a local .venv and install deps from pyproject/uv.lock
install:
	uv sync --extra dev

# Optional: create venv explicitly (uv sync will also create one if missing)
venv:
	uv venv -q

# Run scripts using the project env without manual activation
ingest:
	uv run ingest.py

chat:
        uv run chat.py

evaluate:
        uv run evaluation/runner.py

# Run the automated test suite
test:
        uv run --extra dev pytest

# Create/update a lockfile explicitly (optional; uv sync also updates it)
lock:
	uv lock

clean:
	rm -rf .rag
