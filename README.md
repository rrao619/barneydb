# Candidate A – Notion upload & transcript tooling

Python tools for OCR/transcription and uploading frozen documents and meeting transcripts to Notion.

## Setup with uv

[uv](https://docs.astral.sh/uv/) is used for dependency management.

1. **Install uv** (if needed):

   ```bash
   pip install uv
   # or: curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create a virtualenv and install dependencies**:

   ```bash
   uv venv
   uv pip sync
   ```

   Or with a lockfile (after `uv lock`):

   ```bash
   uv sync
   ```

3. **Environment / secrets** (not committed):

   - `OPENAI_API_KEY` – for OCR and metadata extraction in `main.py` and `upload_frozen_to_notion.py`.
   - `NOTION_API_KEY` or `NOTION_TOKEN` – for Notion API (or use a local `keys` file; see `notion.py`).

## Scripts

- **`main.py`** – Batch OCR/transcription of PDFs (uses OpenAI vision + optional glossary).
- **`notion.py`** – Notion API helpers (databases, pages, blocks).
- **`upload_frozen_to_notion.py`** – Upload frozen PDFs to a Notion database (with optional OCR fallback).
- **`list_notion_files.py`** – List files/documents in Notion.
- **`analyze_hashes.py`** – SHA256 analysis of PDFs in a `frozen/` folder.
- **`count_pdfs.py`** – Count PDFs in `frozen/` and compare with Notion document count.

Data (PDFs, CSVs, zips, etc.) and the `keys` file are gitignored; add them locally as needed.
