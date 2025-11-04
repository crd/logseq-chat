.PHONY: venv install ingest chat lock clean

# Create/refresh a local .venv and install deps from pyproject/uv.lock
install:
	uv sync

# Optional: create venv explicitly (uv sync will also create one if missing)
venv:
	uv venv -q

# Run scripts using the project env without manual activation
ingest:
	uv run ingest.py

chat:
	uv run chat.py

# Create/update a lockfile explicitly (optional; uv sync also updates it)
lock:
	uv lock

clean:
	rm -rf .rag
