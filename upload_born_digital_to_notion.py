"""
Upload Born Digital documents from _BORN_DIGITAL_MASTER_METADATA_FOR_NOTION.v3.csv
to a Notion database (Singletons-style table).

- Maps CSV rows to local paths under LOCAL_UPLOAD_STAGING (e.g. C:\\Users\\rohit\\_UPLOAD_STAGING).
- Skips: .md, Excel (.xls, .xlsx, .xlsm, .xlsb), PowerPoint (.ppt, .pptx).
- Extracts text with PyMuPDF (PDF), mammoth (DOCX), LibreOffice headless (.doc), and direct read (.txt).
- Uses LLM to extract summary, key participants, topics, document date, date confidence.
- Creates Notion pages with the same property set as the Singletons table.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil import parser as dateutil_parser
from openai import OpenAI
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    import mammoth
except ImportError:
    mammoth = None
try:
    import olefile
except ImportError:
    olefile = None

# ----------------------------
# Config
# ----------------------------
NOTION_VERSION = "2022-06-28"
DEFAULT_CSV = "_BORN_DIGITAL_MASTER_METADATA_FOR_NOTION.v3.csv"
LOCAL_UPLOAD_STAGING = os.environ.get("UPLOAD_STAGING_PATH", r"C:\Users\rohit\_UPLOAD_STAGING")
# Notion duplicate table (Singletons format)
NOTION_DATABASE_ID_BORN_DIGITAL = "3020ea6befa2801a811df40a011c1f9b"

# Extensions to skip (do not upload)
SKIP_EXTENSIONS = {
    ".md",
    ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".ppt", ".pptx",
}

# Max text chars to send to LLM
MAX_TEXT_FOR_LLM = 14_000

# If page body has less than this many chars, consider "missing full text" and re-extract/update
MIN_BODY_TEXT_CHARS = 1500


# ----------------------------
# Notion helpers (minimal set)
# ----------------------------
def read_notion_token(keys_path: str = "keys") -> Optional[str]:
    try:
        p = Path(keys_path)
        if not p.exists():
            return os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN")
        raw = p.read_text(encoding="utf-8").strip()
        if "=" in raw:
            return raw.split("=", 1)[1].strip()
        return None
    except Exception:
        return os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN")


def read_openai_key(main_py_path: str = "main.py") -> Optional[str]:
    try:
        p = Path(main_py_path)
        if not p.exists():
            return os.environ.get("OPENAI_API_KEY")
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'api_key\s*=\s*"(?P<key>sk-[A-Za-z0-9_-]+)"', text)
        if m:
            return m.group("key").strip()
        return os.environ.get("OPENAI_API_KEY")
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


def notion_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def notion_get_database(database_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    url = f"https://api.notion.com/v1/databases/{database_id}"
    res = requests.get(url, headers=headers)
    if res.status_code == 404:
        raise RuntimeError(
            "Notion 404: wrong database ID or integration not connected to this database."
        )
    if res.status_code in (401, 403):
        raise RuntimeError(f"Notion auth error: {res.status_code}. Check your token and database connections.")
    res.raise_for_status()
    return res.json()


NOTION_MAX_BLOCKS_PER_REQUEST = 100


def notion_create_page(
    database_id: str,
    headers: Dict[str, str],
    properties: Dict[str, Any],
    children: Optional[List[Dict[str, Any]]] = None,
) -> str:
    payload = {"parent": {"database_id": database_id}, "properties": properties}
    if children:
        payload["children"] = children[:NOTION_MAX_BLOCKS_PER_REQUEST]
    res = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
    if res.status_code not in (200, 201):
        try:
            err_body = res.json()
        except Exception:
            err_body = res.text
        raise RuntimeError(f"Notion create failed ({res.status_code}): {err_body}")
    return res.json()["id"]


def notion_append_children(page_id: str, headers: Dict[str, str], children: List[Dict[str, Any]]) -> None:
    """Append blocks to a page in batches of 100."""
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    for i in range(0, len(children), NOTION_MAX_BLOCKS_PER_REQUEST):
        batch = children[i : i + NOTION_MAX_BLOCKS_PER_REQUEST]
        res = requests.patch(url, headers=headers, json={"children": batch})
        if res.status_code not in (200, 201):
            try:
                err_body = res.json()
            except Exception:
                err_body = res.text
            raise RuntimeError(f"Notion append blocks failed ({res.status_code}): {err_body}")
        time.sleep(0.3)


def notion_get_block_children(block_id: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetch all block children (paginated)."""
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    results: List[Dict[str, Any]] = []
    cursor = None
    while True:
        params: Dict[str, Any] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        results.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        time.sleep(0.2)
    return results


def notion_delete_block(block_id: str, headers: Dict[str, str]) -> None:
    """Archive (delete) a block."""
    res = requests.delete(f"https://api.notion.com/v1/blocks/{block_id}", headers=headers)
    res.raise_for_status()


def block_children_text_length(blocks: List[Dict[str, Any]]) -> int:
    """Total character count of plain text in paragraph/heading blocks."""
    total = 0
    for b in blocks:
        t = b.get("type")
        if t in ("paragraph", "heading_1", "heading_2", "heading_3"):
            rich = (b.get(t) or {}).get("rich_text") or []
            for r in rich:
                total += len((r.get("plain_text") or r.get("text", {}).get("content") or ""))
    return total


def notion_find_page_by_file_hash(
    database_id: str, headers: Dict[str, str], file_hash: str
) -> Optional[str]:
    payload = {
        "page_size": 1,
        "filter": {"property": "File Hash", "rich_text": {"equals": file_hash}},
    }
    try:
        res = requests.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers=headers,
            json=payload,
        )
        res.raise_for_status()
        results = res.json().get("results", [])
        if not results:
            return None
        return results[0]["id"]
    except Exception:
        return None


def notion_build_file_hash_to_page_id(
    database_id: str, headers: Dict[str, str]
) -> Dict[str, str]:
    """Fetch all pages and build file_hash -> page_id map. Much faster than one query per file."""
    hash_map, _ = notion_build_existing_pages_maps(database_id, headers)
    return hash_map


def notion_build_existing_pages_maps(
    database_id: str, headers: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Fetch all pages; return (file_hash -> page_id, source_path -> page_id). One pass for dedup by hash or path."""
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    hash_to_page: Dict[str, str] = {}
    source_path_to_page: Dict[str, str] = {}
    cursor = None
    while True:
        payload: Dict[str, Any] = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        for page in data.get("results", []):
            pid = page.get("id")
            if not pid:
                continue
            props = page.get("properties") or {}
            # File Hash
            prop = props.get("File Hash")
            if prop and prop.get("type") == "rich_text":
                parts = prop.get("rich_text") or []
                h = "".join(p.get("plain_text", "") for p in parts).strip()
                if h:
                    hash_to_page[h] = pid
            # Source Path (for dedup when same path, different hash)
            prop = props.get("Source Path")
            if prop and prop.get("type") == "rich_text":
                parts = prop.get("rich_text") or []
                path_str = "".join(p.get("plain_text", "") for p in parts).strip()
                if path_str:
                    source_path_to_page[path_str] = pid
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        time.sleep(0.3)
    return hash_to_page, source_path_to_page


# ----------------------------
# Path and CSV
# ----------------------------
def relpath_to_local(relpath: str, base: str) -> Path:
    """Convert CSV relpath to local path under base."""
    r = (relpath or "").strip().replace("/", os.sep).lstrip(os.sep)
    return Path(base) / r


def source_abs_path_to_local(source_abs_path: str, base: str) -> Optional[Path]:
    """Derive local path from CSV source_abs_path (Google Drive path). Uses part after _UPLOAD_STAGING."""
    if not source_abs_path or not base:
        return None
    # e.g. /Users/.../My Drive/.../_UPLOAD_STAGING/ subdir/file.docx (use last occurrence for nested staging)
    marker = "_UPLOAD_STAGING"
    i = source_abs_path.rfind(marker)
    if i == -1:
        return None
    suffix = source_abs_path[i + len(marker) :].strip().strip("/\\").replace("/", os.sep)
    if not suffix:
        return None
    return Path(base) / suffix


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def normalize_title(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    if s.startswith("\ufeff"):
        s = s[1:].strip()
    return s[:2000]


def safe_parse_iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    try:
        dt = dateutil_parser.parse(s, fuzzy=True).date()
        return dt.isoformat()
    except Exception:
        return None


# ----------------------------
# Text extraction (fast)
# ----------------------------
def extract_text_pdf(path: Path, max_chars: int = 50_000) -> str:
    if not fitz:
        return ""
    try:
        doc = fitz.open(str(path))
        texts = []
        total = 0
        try:
            for i in range(len(doc)):
                if total >= max_chars:
                    break
                t = doc[i].get_text("text") or ""
                if t.strip():
                    texts.append(t)
                    total += len(t)
        finally:
            doc.close()
        return "\n\n".join(texts).strip()[:max_chars]
    except Exception:
        return ""


def extract_text_docx(path: Path, max_chars: int = 50_000) -> str:
    if not mammoth:
        return ""
    try:
        with path.open("rb") as f:
            result = mammoth.extract_raw_text(f)
            return (result.value or "").strip()[:max_chars]
    except Exception:
        return ""


def extract_text_txt(path: Path, max_chars: int = 100_000) -> str:
    """Read plain text file; try UTF-8 then fallback encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding).strip()[:max_chars]
        except (UnicodeDecodeError, OSError):
            continue
    return ""


LIBREOFFICE_SOFFICE_WIN = r"C:\Program Files\LibreOffice\program\soffice.exe"


def extract_text_doc_libreoffice(path: Path, max_chars: int = 50_000) -> str:
    """Extract text from .doc (Word 97-2003) via LibreOffice headless. Returns '' if not available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        try:
            exe = "soffice"
            if os.name == "nt":
                # Explicit path (common Windows install)
                if Path(LIBREOFFICE_SOFFICE_WIN).exists():
                    exe = LIBREOFFICE_SOFFICE_WIN
                else:
                    for name in ("soffice.exe", "libreoffice.exe"):
                        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
                            if not base:
                                continue
                            candidate = Path(base) / "LibreOffice" / "program" / name
                            if candidate.exists():
                                exe = str(candidate)
                                break
                        if exe != "soffice":
                            break
            subprocess.run(
                [exe, "--headless", "--convert-to", "txt", "--outdir", str(outdir), str(path)],
                capture_output=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            txt_name = path.stem + ".txt"
            txt_path = outdir / txt_name
            if txt_path.exists():
                return txt_path.read_text(encoding="utf-8", errors="replace").strip()[:max_chars]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    return ""


def _decode_worddocument_stream(data: bytes, skip_header: int) -> str:
    """Decode WordDocument stream slice; return cleaned text. skip_header must be even."""
    if len(data) <= skip_header:
        return ""
    data = data[skip_header:]
    if len(data) % 2 != 0:
        data = data[:-1]
    decoded = data.decode("utf-16-le", errors="replace")
    cleaned: List[str] = []
    for c in decoded:
        if c == "\uFFFD":
            continue
        if c in "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f":
            continue
        if c.isspace():
            if cleaned and cleaned[-1] != " ":
                cleaned.append(" ")
            continue
        if c.isprintable():
            cleaned.append(c)
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def _latin_ratio(s: str) -> float:
    """Ratio of chars that are ASCII/Latin (U+0020–U+007F, U+00A0–U+024F). Used to prefer English over garbage/CJK."""
    if not s:
        return 0.0
    latin = sum(1 for c in s if "\u0020" <= c <= "\u007f" or "\u00a0" <= c <= "\u024f")
    return latin / len(s)


def extract_text_doc_olefile(path: Path, max_chars: int = 50_000) -> str:
    """Pure Python: extract text from .doc using olefile (WordDocument stream). Best-effort, no LibreOffice needed."""
    if not olefile:
        return ""
    try:
        ole = olefile.OleFileIO(str(path))
        if not ole.exists("WordDocument"):
            return ""
        stream = ole.openstream("WordDocument")
        data = stream.read()
        ole.close()
    except Exception:
        return ""
    if len(data) < 2:
        return ""
    # WordDocument stream starts with FIB (File Information Block) — decoding from 0 gives binary as "text" (often looks like CJK). Skip header.
    best_text = ""
    best_ratio = 0.0
    for skip in (512, 256, 768, 1024, 0):  # try common header sizes first
        if skip >= len(data):
            continue
        text_le = _decode_worddocument_stream(data, skip)
        if not text_le:
            continue
        ratio = _latin_ratio(text_le)
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = text_le
        # Also try UTF-16-BE in case of big-endian
        if skip < len(data) - 2:
            slice_be = data[skip:]
            if len(slice_be) % 2 != 0:
                slice_be = slice_be[:-1]
            try:
                dec_be = slice_be.decode("utf-16-be", errors="replace")
                cleaned_be = re.sub(r"\s+", " ", "".join(c for c in dec_be if c.isprintable() and c != "\uFFFD")).strip()
                if cleaned_be and _latin_ratio(cleaned_be) > best_ratio:
                    best_ratio = _latin_ratio(cleaned_be)
                    best_text = cleaned_be
            except Exception:
                pass
    # If the best decode is mostly non-Latin (garbage/CJK/mojibake), return nothing so caller uses LibreOffice or CSV abstract.
    if best_text and best_ratio < 0.35:
        return ""
    return best_text[:max_chars] if best_text else ""


def extract_text_for_file(path: Path, csv_abstract: str = "") -> str:
    """Extract text using fast local tools; fallback to csv_abstract on failure."""
    suf = path.suffix.lower()
    text = ""
    if suf == ".pdf":
        text = extract_text_pdf(path)
    elif suf == ".docx":
        text = extract_text_docx(path)
    elif suf == ".doc":
        # Prefer LibreOffice for .doc (reliable); then mammoth; olefile last (often produces mojibake).
        text = extract_text_doc_libreoffice(path)
        if not text:
            text = extract_text_docx(path)  # mammoth sometimes works on .doc
        if not text:
            text = extract_text_doc_olefile(path)  # pure Python fallback (skipped if result looks like garbage)
    elif suf == ".txt":
        text = extract_text_txt(path)
    if not text and csv_abstract:
        text = (csv_abstract or "").strip()[:max(MAX_TEXT_FOR_LLM, 5000)]
    return text


# ----------------------------
# LLM extraction (Singletons-style: summary, participants, topics, date)
# ----------------------------
def extract_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except Exception:
        return None


def extract_summary_participants_with_llm(
    client: OpenAI,
    model: str,
    title: str,
    filename: str,
    date_hint: Optional[str],
    text: str,
) -> Optional[Dict[str, Any]]:
    """Returns dict with: summary, participants, topics, document_date, date_confidence, triage_* (optional)."""
    sample = (text or "")[:MAX_TEXT_FOR_LLM]
    prompt = f"""
You are extracting metadata for a document to store in a Notion database (Singletons style).

Return ONLY valid JSON with these keys:
- summary: string, 2-4 sentences summarizing the document. Do NOT start with "This document", "This memorandum", "This memo", "This paper", or similar — go straight to the substance (e.g. "Outlines U.S. policy toward..." or "Recommends regionalizing...").
- participants: string, comma-separated list of key people/organizations (authors, recipients, main actors)
- topics: array of 5-12 short topic tags (2-5 words each)
- document_date: ISO date "YYYY-MM-DD" or null
- date_confidence: one of "high", "medium", "low"
- triage_legibility: number 0-3 (optional)
- triage_participant_identifiability: number 0-3 (optional)
- triage_subject_clarity: number 0-3 (optional)
- triage_total: number 0-9 (optional)

Context:
- title: {title}
- filename: {filename}
- filename_date_hint: {date_hint or "null"}

Document text (may be partial):
{sample}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        data = extract_first_json_object(content)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


# ----------------------------
# Notion property builders (Singletons / build_frozen_document_properties style)
# ----------------------------
def build_singletons_properties(
    title: str,
    source_filename: str,
    source_path: str,
    file_hash: str,
    prepared: Dict[str, Any],
    db_property_names: Optional[set] = None,
) -> Dict[str, Any]:
    """Build Notion properties matching Singletons table: Title, Source Filename, Summary, Participants, Topics, Date Confidence, Document Date, Triage*."""
    topics = prepared.get("topics") or []
    if isinstance(topics, list):
        topics = [str(t).strip() for t in topics if str(t).strip()]
    else:
        topics = [str(topics).strip()] if str(topics).strip() else []
    topics_str = ", ".join(topics)[:2000]

    db_props = db_property_names or set()
    props = {
        "Title": {"title": [{"text": {"content": normalize_title(title or prepared.get("title") or "Untitled")[:2000]}}]},
        "Source Filename": {"rich_text": [{"text": {"content": (source_filename or "")[:2000]}}]},
        "Summary": {"rich_text": [{"text": {"content": (str(prepared.get("summary") or "")[:2000])}}]},  # LLM 2–4 sentence summary (fallback: CSV abstract)
        "Participants": {"rich_text": [{"text": {"content": (str(prepared.get("participants") or "")[:2000])}}]},
        "Topics": {"rich_text": [{"text": {"content": topics_str}}]},
        "Date Confidence": {"rich_text": [{"text": {"content": (str(prepared.get("date_confidence") or "low")[:2000])}}]},
    }
    if prepared.get("document_date"):
        props["Document Date"] = {"date": {"start": prepared["document_date"]}}
    else:
        props["Document Date"] = {"date": None}

    # Triage fields only if DB has them
    if "Triage Legibility" in db_props:
        props["Triage Legibility"] = {"number": prepared.get("triage_legibility")}
    if "Triage Participant Identifiability" in db_props:
        props["Triage Participant Identifiability"] = {"number": prepared.get("triage_participant_identifiability")}
    if "Triage Subject Clarity" in db_props:
        props["Triage Subject Clarity"] = {"number": prepared.get("triage_subject_clarity")}
    if "Triage Total" in db_props:
        props["Triage Total"] = {"number": prepared.get("triage_total")}

    if "Source Path" in db_props:
        props["Source Path"] = {"rich_text": [{"text": {"content": (source_path or "")[:2000]}}]}
    if "File Hash" in db_props:
        props["File Hash"] = {"rich_text": [{"text": {"content": (file_hash or "")[:2000]}}]}

    # Notion rejects unknown properties; only return props that exist in the DB
    return {k: v for k, v in props.items() if k in db_props}


def build_page_blocks_abstract(abstract: str, max_chunk: int = 1800) -> List[Dict[str, Any]]:
    """Simple body: abstract as paragraphs."""
    blocks = []
    text = (abstract or "").strip()
    if not text:
        return blocks
    for i in range(0, len(text), max_chunk):
        chunk = text[i : i + max_chunk]
        blocks.append({"type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": chunk}}]}})
    return blocks


def build_page_blocks_with_full_text(
    summary_str: str, full_text: str, max_chunk: int = 1800, source_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Page body: Source path (Barney), Summary, then Full text (chunked). Full text is the extracted file content."""
    blocks: List[Dict[str, Any]] = []
    summary_str = (summary_str or "").strip()
    full_text = (full_text or "").strip()

    if source_path and source_path.strip():
        blocks.append({"type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Source path (Barney)"}}]}})
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": source_path.strip()[:2000]}}]},
        })
    if summary_str:
        blocks.append({"type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}})
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": summary_str[:2000]}}]},
        })
    if full_text:
        blocks.append({"type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Full text"}}]}})
        for i in range(0, len(full_text), max_chunk):
            chunk = full_text[i : i + max_chunk]
            blocks.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": chunk}}]},
            })
    if not blocks:
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": "(No content extracted.)"}}]},
        })
    return blocks


# ----------------------------
# Update missing full text
# ----------------------------
def _run_update_missing_full_text(
    to_process: List[Tuple[Dict[str, Any], Path]],
    headers: Dict[str, str],
    database_id: str,
    db_property_names: set,
    report_only: bool,
    errors_output: str,
    only_doc_mode: bool = False,
) -> None:
    """Re-extract from file and replace page body. If only_doc_mode: only .doc files (no body length check). Else: only pages with little body text."""
    from datetime import datetime

    errors: List[Tuple[str, str, str]] = []  # (path, reason, detail) - real errors only
    updated_paths: List[str] = []  # paths we would update or did update
    updated = 0
    skipped_ok = 0
    script_dir = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = errors_output.strip()
    if not out_path:
        out_path = str(script_dir / f"update_errors_{ts}.csv")
    missing_path = str(script_dir / f"missing_full_text_{ts}.txt")

    print("Building File Hash -> page ID map from Notion...")
    hash_to_page = notion_build_file_hash_to_page_id(database_id, headers)
    print(f"Found {len(hash_to_page)} pages in database.")
    if only_doc_mode:
        print("Mode: only .doc files — re-extract full text and replace body (no body length check).")

    for row, local_path in tqdm(to_process, desc="Update full text"):
        path_str = str(local_path)
        try:
            file_hash = (row.get("sha256") or "").strip() or sha256_file(local_path)
            page_id = hash_to_page.get(file_hash)
            if not page_id:
                err_detail = "No page with this File Hash in database"
                errors.append((path_str, "not_in_notion", err_detail))
                tqdm.write(f"  [ERROR] {local_path.name}: {err_detail}")
                continue

            blocks = notion_get_block_children(page_id, headers)
            if not only_doc_mode:
                body_len = block_children_text_length(blocks)
                if body_len >= MIN_BODY_TEXT_CHARS:
                    skipped_ok += 1
                    continue

            updated_paths.append(path_str)
            if report_only:
                body_len = block_children_text_length(blocks) if not only_doc_mode else 0
                tqdm.write(f"  [would update] {local_path.name}" + (f" (body {body_len} chars)" if not only_doc_mode else " (.doc)"))
            source_path = (row.get("source_abs_path") or row.get("source_path") or "").strip() or path_str
            csv_abstract = (row.get("abstract") or "").strip()
            text = extract_text_for_file(local_path, csv_abstract=csv_abstract)
            full_text_for_body = text or csv_abstract or ""

            if not full_text_for_body.strip():
                err_detail = "No text extracted and no CSV abstract"
                errors.append((path_str, "extraction_failed", err_detail))
                tqdm.write(f"  [ERROR] {local_path.name}: {err_detail}")
                continue

            if report_only:
                continue

            summary_str = csv_abstract[:2000] if csv_abstract else "(No summary)"
            new_children = build_page_blocks_with_full_text(
                summary_str, full_text_for_body, source_path=source_path
            )
            for blk in blocks:
                bid = blk.get("id")
                if bid:
                    notion_delete_block(bid, headers)
                    time.sleep(0.15)
            for i in range(0, len(new_children), NOTION_MAX_BLOCKS_PER_REQUEST):
                batch = new_children[i : i + NOTION_MAX_BLOCKS_PER_REQUEST]
                notion_append_children(page_id, headers, batch)
                time.sleep(0.3)
            updated += 1
            tqdm.write(f"  [updated] {local_path.name}")
        except Exception as e:
            errors.append((path_str, "error", str(e)))
            tqdm.write(f"  [ERROR] {local_path.name}: {e}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason", "detail"])
        for path_str, reason, detail in errors:
            w.writerow([path_str, reason, detail])

    with open(missing_path, "w", encoding="utf-8") as f:
        for p in updated_paths:
            f.write(p + "\n")

    print(f"Update full text: updated={updated}, skipped={skipped_ok}, processed={len(updated_paths)}, errors={len(errors)}")
    print(f"Errors list (path, reason, detail): {out_path}")
    print(f"All files with missing/short full text: {missing_path}")


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Upload Born Digital CSV rows to Notion (Singletons-style DB)")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to _BORN_DIGITAL_MASTER_METADATA_FOR_NOTION.v3.csv")
    parser.add_argument("--staging", default=LOCAL_UPLOAD_STAGING, help="Local _UPLOAD_STAGING root (default: env UPLOAD_STAGING_PATH or C:\\Users\\rohit\\_UPLOAD_STAGING)")
    parser.add_argument("--database-id", default=NOTION_DATABASE_ID_BORN_DIGITAL, help="Notion database ID (Singletons duplicate)")
    parser.add_argument("--dry-run", action="store_true", help="Only list what would be uploaded")
    parser.add_argument("--limit", type=int, default=0, help="Max number of files to process (0 = no limit)")
    parser.add_argument("--ext", type=str, default="", help="Only process this extension, e.g. .doc (optional)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rows whose File Hash already exists in Notion")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for summary/participants extraction")
    parser.add_argument(
        "--update-missing-full-text",
        action="store_true",
        help="Find pages with little body text, re-extract from file, and replace page body (no new uploads).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="With --update-missing-full-text: only scan and write error/missing lists, do not update pages.",
    )
    parser.add_argument(
        "--errors-output",
        default="",
        help="Path for errors CSV (default: update_errors_YYYYMMDD_HHMM.csv in script dir).",
    )
    parser.add_argument(
        "--only-doc",
        action="store_true",
        help="With --update-missing-full-text: only process .doc files and re-run full text for them (ignore body length).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    staging_base = Path(args.staging)
    if not staging_base.exists():
        print(f"Staging folder not found: {staging_base}")
        return

    notion_token = read_notion_token()
    if not notion_token:
        print("Set NOTION_API_KEY or add a 'keys' file with 'internal integration = ntn_...'")
        return

    # Load DB schema to know optional properties
    headers = notion_headers(notion_token)
    try:
        db = notion_get_database(args.database_id, headers)
    except Exception as e:
        print(f"Notion database error: {e}")
        return
    db_property_names = set((db.get("properties") or {}).keys())
    print("Notion database ID:", args.database_id)
    print("Notion database properties:", sorted(db_property_names))
    if "Title" not in db_property_names:
        print("WARNING: Database has no 'Title' property; page creation may fail.")

    rows = load_csv_rows(str(csv_path))
    print(f"Loaded {len(rows)} rows from {csv_path.name}")

    # Filter: build local path, skip excluded extensions, require file exists
    to_process: List[Tuple[Dict[str, Any], Path]] = []
    for row in rows:
        relpath = row.get("relpath") or row.get("source_path") or ""
        local_path = None
        if relpath.strip():
            local_path = relpath_to_local(relpath, str(staging_base))
        if not local_path or not local_path.exists():
            local_path = source_abs_path_to_local(row.get("source_abs_path") or "", str(staging_base))
        if not local_path or not local_path.exists():
            continue
        ext = local_path.suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue
        if args.ext:
            ext_filter = args.ext.lower() if args.ext.startswith(".") else f".{args.ext.lower()}"
            if ext != ext_filter:
                continue
        to_process.append((row, local_path))

    print(f"After filtering (exist, not md/xl/ppt): {len(to_process)} files")

    if args.limit > 0:
        to_process = to_process[: args.limit]
        print(f"Limited to first {args.limit}")

    if args.dry_run:
        for row, path in to_process[:20]:
            print(f"  {path}")
        if len(to_process) > 20:
            print(f"  ... and {len(to_process) - 20} more")
        return

    # ---------- Update missing full text ----------
    if args.update_missing_full_text:
        if args.only_doc:
            to_process = [(r, p) for r, p in to_process if p.suffix.lower() == ".doc"]
            print(f"Only .doc files: {len(to_process)} to process.")
        _run_update_missing_full_text(
            to_process=to_process,
            headers=headers,
            database_id=args.database_id,
            db_property_names=db_property_names,
            report_only=args.report_only,
            errors_output=args.errors_output,
            only_doc_mode=args.only_doc,
        )
        return

    openai_key = read_openai_key()
    if not openai_key:
        print("Set OPENAI_API_KEY or add key in main.py for LLM extraction.")
        return
    client = OpenAI(api_key=openai_key)

    if not fitz:
        print("Warning: PyMuPDF (fitz) not installed; PDF text extraction will be empty.")
    if not mammoth:
        print("Warning: mammoth not installed; DOCX text extraction will be empty. Install: pip install mammoth")

    uploaded = 0
    skipped_hash = 0
    skipped_path = 0
    errors = 0
    hash_to_page: Dict[str, str] = {}
    source_path_to_page: Dict[str, str] = {}
    if args.skip_existing:
        print("Building existing pages map (hash + Source Path) for dedup...")
        hash_to_page, source_path_to_page = notion_build_existing_pages_maps(args.database_id, headers)
        print(f"  {len(hash_to_page)} by hash, {len(source_path_to_page)} by Source Path")

    for row, local_path in tqdm(to_process, desc="Upload"):
        try:
            tqdm.write(f"  {local_path.name}")
            file_hash = (row.get("sha256") or "").strip() or sha256_file(local_path)
            # Use path as in CSV (Barney's path) for Source Path and for path-based dedup
            source_path = (row.get("source_abs_path") or row.get("source_path") or "").strip() or str(local_path)
            if args.skip_existing:
                existing_id = hash_to_page.get(file_hash) or source_path_to_page.get(source_path)
                if existing_id:
                    if hash_to_page.get(file_hash):
                        skipped_hash += 1
                        tqdm.write(f"    -> skipped (existing hash)")
                    else:
                        skipped_path += 1
                        tqdm.write(f"    -> skipped (existing Source Path)")
                    continue

            title = normalize_title(row.get("title_guess") or row.get("title") or local_path.stem)
            source_filename = local_path.name
            csv_abstract = (row.get("abstract") or "").strip()

            text = extract_text_for_file(local_path, csv_abstract=csv_abstract)
            if not text and csv_abstract:
                text = csv_abstract[:MAX_TEXT_FOR_LLM]

            prepared = {
                "title": title,
                "summary": "",
                "participants": "",
                "topics": [],
                "document_date": safe_parse_iso_date(row.get("date_inferred") or row.get("document_date")),
                "date_confidence": "low",
            }
            if text:
                llm_out = extract_summary_participants_with_llm(
                    client, args.model, title, source_filename,
                    row.get("date_inferred"), text,
                )
                if llm_out:
                    prepared["summary"] = llm_out.get("summary") or ""
                    prepared["participants"] = llm_out.get("participants") or ""
                    prepared["topics"] = llm_out.get("topics") or []
                    if llm_out.get("document_date"):
                        prepared["document_date"] = safe_parse_iso_date(llm_out["document_date"])
                    prepared["date_confidence"] = str(llm_out.get("date_confidence") or "low").lower()
                    if prepared["date_confidence"] not in ("high", "medium", "low"):
                        prepared["date_confidence"] = "low"
                    prepared["triage_legibility"] = llm_out.get("triage_legibility")
                    prepared["triage_participant_identifiability"] = llm_out.get("triage_participant_identifiability")
                    prepared["triage_subject_clarity"] = llm_out.get("triage_subject_clarity")
                    prepared["triage_total"] = llm_out.get("triage_total")
            if not prepared["document_date"] and row.get("date_inferred"):
                prepared["document_date"] = safe_parse_iso_date(row["date_inferred"])
            if not prepared["summary"] and csv_abstract:
                prepared["summary"] = csv_abstract[:2000]

            props = build_singletons_properties(
                title, source_filename, source_path, file_hash, prepared, db_property_names
            )
            # Page body: Barney's path from CSV, summary, then full extracted file text (searchable in Notion)
            full_text_for_body = text or csv_abstract or ""
            children = build_page_blocks_with_full_text(
                prepared.get("summary") or csv_abstract, full_text_for_body, source_path=source_path
            )

            page_id = notion_create_page(args.database_id, headers, props, children[:NOTION_MAX_BLOCKS_PER_REQUEST])
            if len(children) > NOTION_MAX_BLOCKS_PER_REQUEST:
                notion_append_children(page_id, headers, children[NOTION_MAX_BLOCKS_PER_REQUEST:])
            uploaded += 1
            tqdm.write(f"    -> uploaded")
            time.sleep(0.35)
        except Exception as e:
            errors += 1
            tqdm.write(f"Error {local_path}: {e}")

    print(f"Done. Uploaded: {uploaded}, skipped (existing hash): {skipped_hash}, skipped (existing path): {skipped_path}, errors: {errors}")


if __name__ == "__main__":
    main()
