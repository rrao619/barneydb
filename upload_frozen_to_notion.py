import argparse
import base64
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests
from dateutil import parser as dateutil_parser
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from notion import prepare_meeting_transcript_for_frozen_format, build_frozen_document_properties_from_meeting


NOTION_VERSION = "2022-06-28"
NOTION_DATABASE_ID = "2ff0ea6befa2807d8eb1d42c3caa3065"
DEFAULT_SOURCE_FOLDER = r"C:\Users\rohit\Desktop\Barneydb\Candidate_A\ATOMIC_MEETINGS_CANONICAL_462\ATOMIC_MEETINGS_CANONICAL_462"
SOURCE_PATH_PREFIX = "/Users/barneyrubin/Library/CloudStorage/GoogleDrive-barnettrrubin@gmail.com/My Drive/Afgh Peace Process Memoir/_UPLOAD_STAGING/ATOMIC_DEDUP/ATOMIC_MEETINGS_CANONICAL_462"


# ----------------------------
# Notion helpers
# ----------------------------
def read_notion_token_from_keys_file(keys_path: str = "keys") -> Optional[str]:
    """
    Best-effort read of a Notion integration token from the repo `keys` file.
    Expected format (example): 'internal integration = ntn_...'
    """
    try:
        p = Path(keys_path)
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8").strip()
        if "=" in raw:
            return raw.split("=", 1)[1].strip()
        return None
    except Exception:
        return None


def read_openai_key_from_main_py(main_py_path: str = "main.py") -> Optional[str]:
    """
    Best-effort extraction of a hard-coded OpenAI key from `main.py`.
    This is intentionally a fallback for this repo's current setup.
    """
    try:
        p = Path(main_py_path)
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'api_key\s*=\s*"(?P<key>sk-[A-Za-z0-9_-]+)"', text)
        if not m:
            return None
        return m.group("key").strip()
    except Exception:
        return None


def notion_headers(notion_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def notion_get_database(database_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    url = f"https://api.notion.com/v1/databases/{database_id}"
    res = requests.get(url, headers=headers)
    if res.status_code == 404:
        raise RuntimeError(
            "Notion returned 404 for the database. This usually means either:\n"
            "- the database ID is wrong, OR\n"
            "- your integration token does not have access to this database.\n\n"
            "Fix: open the database in Notion → ⋯ → Connections → Add connections → select your integration."
        )
    if res.status_code == 401:
        raise RuntimeError("Notion returned 401 (Unauthorized). Your NOTION_TOKEN/NOTION_API_KEY is invalid or expired.")
    if res.status_code == 403:
        raise RuntimeError(
            "Notion returned 403 (Forbidden). Your integration exists, but it is not permitted to access this database.\n"
            "Fix: database → ⋯ → Connections → Add connections → select your integration."
        )
    res.raise_for_status()
    return res.json()


def notion_query_database(
    database_id: str, headers: Dict[str, str], payload: Dict[str, Any]
) -> Dict[str, Any]:
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()


def notion_update_page(
    page_id: str, headers: Dict[str, str], properties: Dict[str, Any]
) -> Dict[str, Any]:
    url = f"https://api.notion.com/v1/pages/{page_id}"
    res = requests.patch(url, headers=headers, json={"properties": properties})
    if res.status_code != 200:
        print(f"    [ERROR] Notion API error: {res.status_code}")
        try:
            error_data = res.json()
            print(f"    Error details: {error_data}")
        except:
            print(f"    Raw response: {res.text}")
    res.raise_for_status()
    return res.json()


def notion_get_page(page_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    url = f"https://api.notion.com/v1/pages/{page_id}"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()


def notion_create_page(
    database_id: str,
    headers: Dict[str, str],
    properties: Dict[str, Any],
    children: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Creates a page with up to 100 initial children. Returns (page_id, remaining_children).
    """
    MAX_INITIAL_CHILDREN = 100
    initial = children[:MAX_INITIAL_CHILDREN]
    remaining = children[MAX_INITIAL_CHILDREN:]

    payload = {"parent": {"database_id": database_id}, "properties": properties, "children": initial}
    res = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
    if res.status_code not in (200, 201):
        # better diagnostics
        try:
            raise RuntimeError(f"Notion create failed ({res.status_code}): {res.json()}")
        except Exception:
            raise RuntimeError(f"Notion create failed ({res.status_code}): {res.text}")
    page = res.json()
    return page["id"], remaining


def notion_append_children(page_id: str, headers: Dict[str, str], children: List[Dict[str, Any]]) -> None:
    """
    Append children blocks to a page in batches of 100.
    """
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    MAX_BLOCKS = 100
    batch: List[Dict[str, Any]] = []
    for block in children:
        batch.append(block)
        if len(batch) >= MAX_BLOCKS:
            res = requests.patch(url, headers=headers, json={"children": batch})
            if res.status_code != 200:
                print(f"    [ERROR] Failed to append blocks batch: {res.status_code}")
                try:
                    error_data = res.json()
                    print(f"    Error details: {error_data}")
                    if 'code' in error_data and error_data['code'] == 'validation_error':
                        print(f"    This is likely a content validation error - check for invalid characters or formatting in the blocks.")
                except:
                    print(f"    Raw response: {res.text}")
            res.raise_for_status()
            batch = []
            time.sleep(0.3)
    if batch:
        res = requests.patch(url, headers=headers, json={"children": batch})
        if res.status_code != 200:
            print(f"    [ERROR] Failed to append final blocks: {res.status_code}")
            try:
                error_data = res.json()
                print(f"    Error details: {error_data}")
                if 'code' in error_data and error_data['code'] == 'validation_error':
                    print(f"    This is likely a content validation error - check for invalid characters or formatting in the blocks.")
            except:
                print(f"    Raw response: {res.text}")
        res.raise_for_status()


def notion_count_documents(database_id: str, headers: Dict[str, str]) -> int:
    """
    Count total number of pages/documents in the Notion database.
    """
    count = 0
    cursor = None

    while True:
        payload = {"page_size": 100}  # Maximum allowed by Notion API
        if cursor:
            payload["start_cursor"] = cursor

        try:
            data = notion_query_database(database_id, headers, payload)
            results = data.get("results", [])
            count += len(results)

            # Check if there are more pages
            if not data.get("has_more", False):
                break

            cursor = data.get("next_cursor")

        except requests.HTTPError as e:
            print(f"Error counting documents: {e}")
            break

    return count


def notion_get_existing_file_keys(database_id: str, headers: Dict[str, str]) -> Tuple[set[str], set[str]]:
    """
    Get existing file hashes and source filenames from Notion for fast pre-filtering.
    Returns (hashes, source_filenames_lower).
    """
    hashes: set[str] = set()
    source_filenames_lower: set[str] = set()
    cursor = None

    while True:
        payload = {"page_size": 100}  # Maximum allowed by Notion API
        if cursor:
            payload["start_cursor"] = cursor

        try:
            data = notion_query_database(database_id, headers, payload)
            results = data.get("results", [])

            for page in results:
                # File hash is the strongest dedupe key.
                file_hash_prop = _get_rich_text_plain(page, DEFAULT_PROPERTIES["File Hash"])
                if file_hash_prop:
                    hashes.add(file_hash_prop)
                # Filename lets us quickly skip obvious duplicates before hashing local files.
                source_filename = _get_rich_text_plain(page, DEFAULT_PROPERTIES["Source Filename"])
                if source_filename:
                    source_filenames_lower.add(source_filename.strip().lower())

            # Check if there are more pages
            if not data.get("has_more", False):
                break

            cursor = data.get("next_cursor")

        except requests.HTTPError as e:
            print(f"Error getting existing file keys: {e}")
            break

    return hashes, source_filenames_lower


def notion_find_page_id_by_rich_text_equals(
    database_id: str,
    headers: Dict[str, str],
    property_name: str,
    value: str,
) -> Optional[str]:
    """
    Returns first matching page id where a rich_text property equals value.
    """
    payload = {
        "page_size": 1,
        "filter": {"property": property_name, "rich_text": {"equals": value}},
    }
    try:
        data = notion_query_database(database_id, headers, payload)
        results = data.get("results", [])
        if not results:
            return None
        return results[0]["id"]
    except requests.HTTPError as e:
        # Common cause: filtering on a property that doesn't exist in the DB schema.
        resp = getattr(e, "response", None)
        if resp is not None and resp.status_code == 400:
            return None
        raise


# ----------------------------
# Notion block builders
# ----------------------------
def rt(text: str) -> List[Dict[str, Any]]:
    text = "" if text is None else str(text)
    return [{"type": "text", "text": {"content": text[:2000]}}]


def heading_2(text: str) -> Dict[str, Any]:
    return {"type": "heading_2", "heading_2": {"rich_text": rt(text)}}


def paragraph(text: str) -> Dict[str, Any]:
    return {"type": "paragraph", "paragraph": {"rich_text": rt(text)}}


def bulleted_item(text: str) -> Dict[str, Any]:
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rt(text)}}


def divider() -> Dict[str, Any]:
    return {"type": "divider", "divider": {}}


def chunk_text(text: str, size: int = 1800) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


# ----------------------------
# PDF extraction
# ----------------------------
@dataclass
class PdfExtractResult:
    base_text: str
    base_text_chars: int
    base_quality_score: float
    ocr_needed: bool
    text: str
    page_count: int
    used_ocr: bool
    ocr_attempted: bool
    ocr_error: Optional[str]
    pdf_metadata: Dict[str, Any]


@dataclass
class TextQuality:
    score: float  # 0..1 (higher is better)
    reason: str


def estimate_text_quality(text: str) -> TextQuality:
    """
    Heuristic quality estimate for extracted/OCR text.
    Used to flag likely-bad OCR (especially handwriting/scans).
    """
    if not text or not text.strip():
        return TextQuality(score=0.0, reason="no_text")

    s = text
    n = len(s)
    if n < 200:
        return TextQuality(score=0.15, reason="very_short_text")

    # Ratios
    letters = sum(1 for ch in s if ch.isalpha())
    digits = sum(1 for ch in s if ch.isdigit())
    spaces = sum(1 for ch in s if ch.isspace())
    question = s.count("?")
    replacement = s.count("\uFFFD")  # �
    punctuation = sum(1 for ch in s if ch in ".,;:!?\"'()[]{}-/\\")

    letters_ratio = letters / max(n, 1)
    digits_ratio = digits / max(n, 1)
    spaces_ratio = spaces / max(n, 1)
    question_ratio = question / max(n, 1)
    replacement_ratio = replacement / max(n, 1)
    punct_ratio = punctuation / max(n, 1)

    # Start from a baseline and penalize common OCR-garbage patterns
    score = 0.75
    reasons: List[str] = []

    if letters_ratio < 0.35:
        score -= 0.25
        reasons.append("low_letters")
    if spaces_ratio < 0.08:
        score -= 0.15
        reasons.append("low_spaces")
    if question_ratio > 0.02:
        score -= 0.2
        reasons.append("many_question_marks")
    if replacement_ratio > 0.001:
        score -= 0.2
        reasons.append("replacement_chars")
    if punct_ratio > 0.25 and letters_ratio < 0.45:
        score -= 0.1
        reasons.append("punct_heavy")
    if digits_ratio > 0.25 and letters_ratio < 0.35:
        score -= 0.1
        reasons.append("digit_heavy")

    # Clamp
    score = max(0.0, min(1.0, score))
    return TextQuality(score=score, reason=";".join(reasons) if reasons else "ok")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def clean_title_from_filename(filename: str) -> str:
    name = Path(filename).stem
    # Strip leading hash prefixes like "<hex>_"
    m = re.match(r"^[0-9a-fA-F]{16,}_(.+)$", name)
    if m:
        name = m.group(1)
    name = name.replace("_", " ").strip()
    # collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name if name else filename


def date_hint_from_filename(filename: str) -> Optional[str]:
    """
    Try to extract a date from filename; returns ISO date (YYYY-MM-DD) or None.
    """
    base = Path(filename).stem
    # yyyymmdd
    m = re.search(r"(19|20)\d{2}[01]\d[0-3]\d", base)
    if m:
        s = m.group(0)
        try:
            dt = datetime.strptime(s, "%Y%m%d").date()
            return dt.isoformat()
        except Exception:
            pass
    # yyyy-mm-dd or yyyy_mm_dd
    m = re.search(r"(19|20)\d{2}[-_][01]\d[-_][0-3]\d", base)
    if m:
        s = m.group(0).replace("_", "-")
        try:
            dt = datetime.strptime(s, "%Y-%m-%d").date()
            return dt.isoformat()
        except Exception:
            pass
    return None


def extract_text_pymupdf(pdf_path: Path, max_pages: int) -> Tuple[str, int, Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    try:
        page_count = len(doc)
        meta = doc.metadata or {}
        texts: List[str] = []
        for i in range(min(page_count, max_pages)):
            try:
                page = doc[i]
                t = page.get_text("text") or ""
                if t.strip():
                    texts.append(t)
            except Exception:
                continue
        return "\n\n".join(texts).strip(), page_count, meta
    finally:
        doc.close()


def render_pages_to_png_bytes(pdf_path: Path, page_numbers_1idx: List[int], dpi: int = 200) -> List[bytes]:
    doc = fitz.open(str(pdf_path))
    images: List[bytes] = []
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for pno in page_numbers_1idx:
            page = doc[pno - 1]
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
        return images
    finally:
        doc.close()


def ocr_pages_with_openai(
    client: OpenAI, model: str, image_bytes_list: List[bytes], page_numbers: List[int]
) -> str:
    image_contents: List[Dict[str, Any]] = []
    for b in image_bytes_list:
        b64 = base64.b64encode(b).decode("utf-8")
        image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    page_range = f"{page_numbers[0]}-{page_numbers[-1]}" if len(page_numbers) > 1 else str(page_numbers[0])
    prompt = (
        f"Extract (OCR) all text from the following PDF page image(s), pages {page_range}.\n"
        "Rules:\n"
        "- Do not add commentary.\n"
        "- Preserve headings/lists where possible.\n"
        "- If a page has no legible text, output an empty string for that page.\n"
        f'- Start each page with a line exactly like: "## Page X" where X is the page number.\n'
    )
    content = [{"type": "text", "text": prompt}] + image_contents
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return (resp.choices[0].message.content or "").strip()


def extract_pdf_text_with_optional_ocr(
    pdf_path: Path,
    max_pages_text: int,
    ocr_enabled: bool,
    ocr_pages: int,
    openai_client: Optional[OpenAI],
    vision_model: str,
    min_text_chars_for_no_ocr: int = 800,
) -> PdfExtractResult:
    text, page_count, meta = extract_text_pymupdf(pdf_path, max_pages=max_pages_text)
    used_ocr = False
    ocr_attempted = False
    ocr_error: Optional[str] = None

    # Decide whether OCR is needed (scanned PDFs / handwriting / garbled extraction).
    quality = estimate_text_quality(text or "")
    base_text = text or ""
    base_text_chars = len(base_text.strip())
    base_quality_score = float(quality.score)
    should_ocr = bool(base_text_chars < min_text_chars_for_no_ocr or base_quality_score < 0.30)

    if ocr_enabled and should_ocr:
        if openai_client is None:
            return PdfExtractResult(
                base_text=base_text,
                base_text_chars=base_text_chars,
                base_quality_score=base_quality_score,
                ocr_needed=should_ocr,
                text=text,
                page_count=page_count,
                used_ocr=False,
                ocr_attempted=False,
                ocr_error=None,
                pdf_metadata=meta,
            )

        used_ocr = True
        ocr_attempted = True
        pages = list(range(1, min(page_count, ocr_pages) + 1))
        images = render_pages_to_png_bytes(pdf_path, pages, dpi=200)
        try:
            ocr_text = ocr_pages_with_openai(openai_client, vision_model, images, pages)
            # Append OCR text (do not discard whatever PyMuPDF did extract)
            combined = (text + "\n\n" + ocr_text).strip() if text else ocr_text
            return PdfExtractResult(
                base_text=base_text,
                base_text_chars=base_text_chars,
                base_quality_score=base_quality_score,
                ocr_needed=should_ocr,
                text=combined,
                page_count=page_count,
                used_ocr=True,
                ocr_attempted=True,
                ocr_error=None,
                pdf_metadata=meta,
            )
        except Exception as e:
            # Continue upload even if OCR fails
            used_ocr = False
            ocr_error = f"{type(e).__name__}: {e}"
            return PdfExtractResult(
                base_text=base_text,
                base_text_chars=base_text_chars,
                base_quality_score=base_quality_score,
                ocr_needed=should_ocr,
                text=text,
                page_count=page_count,
                used_ocr=False,
                ocr_attempted=True,
                ocr_error=ocr_error,
                pdf_metadata=meta,
            )

    return PdfExtractResult(
        base_text=base_text,
        base_text_chars=base_text_chars,
        base_quality_score=base_quality_score,
        ocr_needed=should_ocr,
        text=text,
        page_count=page_count,
        used_ocr=used_ocr,
        ocr_attempted=ocr_attempted,
        ocr_error=ocr_error,
        pdf_metadata=meta,
    )


# ----------------------------
# Metadata extraction (OpenAI)
# ----------------------------
def extract_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    # Try to locate a JSON object in the response
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def safe_parse_iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # accept YYYY-MM-DD directly
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    try:
        dt = dateutil_parser.parse(s, fuzzy=True).date()
        return dt.isoformat()
    except Exception:
        return None


@dataclass
class DocMetadata:
    document_date: Optional[str]
    date_confidence: str
    author: Optional[str]
    topics: List[str]
    key_figures: List[str]
    summary_bullets: List[str]


def normalize_list(values: Any, max_items: int, max_len: int) -> List[str]:
    out: List[str] = []
    if isinstance(values, list):
        candidates = values
    elif isinstance(values, str):
        candidates = [v.strip() for v in re.split(r"[\n,;]+", values) if v.strip()]
    else:
        candidates = []
    for v in candidates:
        if not v:
            continue
        s = str(v).strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        if len(s) > max_len:
            s = s[: max_len - 1] + "…"
        if s not in out:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def extract_metadata_with_openai(
    client: OpenAI,
    model: str,
    title: str,
    filename: str,
    date_hint: Optional[str],
    pdf_metadata: Dict[str, Any],
    text: str,
) -> Optional[DocMetadata]:
    text_sample = text[:12000]
    prompt = f"""
You are extracting metadata for a PDF so it can be stored in a Notion database for search.

Return ONLY valid JSON with these keys:
- document_date: ISO date "YYYY-MM-DD" or null (use a date that appears in the document; if only filename date is available, you may use it but set date_confidence="low")
- date_confidence: one of ["high","medium","low"]
- author: string or null
- topics: array of 5-12 short topic tags (2-5 words each)
- key_figures: array of up to 15 people/org names mentioned as important in the document
- summary_bullets: array of 3-6 concise bullets

Context:
- title: {title}
- filename: {filename}
- filename_date_hint: {date_hint or "null"}
- pdf_metadata: {json.dumps(pdf_metadata, ensure_ascii=False)[:1500]}

Document text (may be partial):
{text_sample}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (resp.choices[0].message.content or "").strip()
    data = extract_first_json_object(content)
    if not isinstance(data, dict):
        return None

    document_date = safe_parse_iso_date(data.get("document_date"))
    date_confidence = str(data.get("date_confidence") or "low").lower()
    if date_confidence not in ("high", "medium", "low"):
        date_confidence = "low"

    author = data.get("author")
    author = str(author).strip() if author else None
    if author and len(author) > 200:
        author = author[:199] + "…"

    topics = normalize_list(data.get("topics"), max_items=12, max_len=60)
    key_figures = normalize_list(data.get("key_figures"), max_items=15, max_len=80)
    summary_bullets = normalize_list(data.get("summary_bullets"), max_items=6, max_len=240)
    if not summary_bullets:
        # fallback: try "summary" field if model used different name
        summary_bullets = normalize_list(data.get("summary"), max_items=6, max_len=240)

    # Optional second-pass cleanup for Topics + Key Figures quality.
    refined = refine_topics_and_key_figures_with_openai(
        client=client,
        model=model,
        title=title,
        filename=filename,
        text=text,
        topics=topics,
        key_figures=key_figures,
    )
    if refined:
        topics, key_figures = refined

    return DocMetadata(
        document_date=document_date,
        date_confidence=date_confidence,
        author=author,
        topics=topics,
        key_figures=key_figures,
        summary_bullets=summary_bullets,
    )


def refine_topics_and_key_figures_with_openai(
    client: OpenAI,
    model: str,
    title: str,
    filename: str,
    text: str,
    topics: List[str],
    key_figures: List[str],
) -> Optional[Tuple[List[str], List[str]]]:
    """
    Regenerate Topics + Key Figures with stricter rules to improve correctness.
    """
    text_sample = (text or "")[:12000]
    prompt = f"""
Clean and regenerate metadata lists for this document.
Return ONLY valid JSON with:
- topics: 4-8 topic tags (2-5 words each), thematic, no person names
- key_figures: up to 10 important people/org names (proper nouns only)

Rules:
- Remove vague items (e.g., "meeting", "notes", "discussion", "issue").
- Remove duplicates and near-duplicates.
- Keep specific, searchable labels.
- Do not invent names not grounded in the text.

Context:
- title: {title}
- filename: {filename}
- existing_topics: {json.dumps(topics, ensure_ascii=False)}
- existing_key_figures: {json.dumps(key_figures, ensure_ascii=False)}

Document text (may be partial):
{text_sample}
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
        new_topics = normalize_list(data.get("topics"), max_items=8, max_len=60)
        new_key_figures = normalize_list(data.get("key_figures"), max_items=10, max_len=80)
        # Only accept if at least one list is non-empty
        if not new_topics and not new_key_figures:
            return None
        return new_topics or topics, new_key_figures or key_figures
    except Exception:
        return None


def fallback_metadata(
    filename: str,
    pdf_metadata: Dict[str, Any],
) -> DocMetadata:
    author = pdf_metadata.get("author") or pdf_metadata.get("Author")
    author = str(author).strip() if author else None
    dt = date_hint_from_filename(filename)
    bullets = ["(No OpenAI metadata extraction enabled; add OPENAI_API_KEY to enrich.)"]
    return DocMetadata(
        document_date=dt,
        date_confidence="low" if dt else "low",
        author=author,
        topics=[],
        key_figures=[],
        summary_bullets=bullets,
    )


# ----------------------------
# Notion property builders
# ----------------------------
def prop_title(text: str) -> Dict[str, Any]:
    return {"title": [{"text": {"content": text[:2000]}}]}


def prop_rich_text(text: str) -> Dict[str, Any]:
    return {"rich_text": [{"text": {"content": (text or "")[:2000]}}]}


def prop_date(iso_date: Optional[str]) -> Dict[str, Any]:
    if not iso_date:
        return {"date": None}
    return {"date": {"start": iso_date}}


def prop_checkbox(value: bool) -> Dict[str, Any]:
    return {"checkbox": bool(value)}


def prop_number(value: Optional[float]) -> Dict[str, Any]:
    return {"number": value}


def prop_multi_select(values: List[str]) -> Dict[str, Any]:
    # Notion multi-select option names cannot contain commas.
    # Sanitize labels and dedupe before sending.
    cleaned: List[str] = []
    seen = set()
    for raw in values or []:
        if not raw:
            continue
        v = str(raw).strip()
        if not v:
            continue
        v = re.sub(r"\s+", " ", v)
        v = v.replace(",", " -")
        v = v.strip(" -")
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(v[:100])
    items = [{"name": v} for v in cleaned]
    return {"multi_select": items}


def _get_checkbox(page: Dict[str, Any], prop_name: str) -> Optional[bool]:
    try:
        p = (page.get("properties", {}) or {}).get(prop_name)
        if not p:
            return None
        if p.get("type") != "checkbox":
            return None
        return bool(p.get("checkbox"))
    except Exception:
        return None


def _get_number(page: Dict[str, Any], prop_name: str) -> Optional[float]:
    try:
        p = (page.get("properties", {}) or {}).get(prop_name)
        if not p:
            return None
        if p.get("type") != "number":
            return None
        return p.get("number")
    except Exception:
        return None


def _get_rich_text_plain(page: Dict[str, Any], prop_name: str) -> Optional[str]:
    try:
        p = (page.get("properties", {}) or {}).get(prop_name)
        if not p:
            return None
        if p.get("type") != "rich_text":
            return None
        parts = p.get("rich_text") or []
        text = "".join((x.get("plain_text") or "") for x in parts).strip()
        return text or None
    except Exception:
        return None


def should_update_as_unreadable(
    page: Dict[str, Any],
    available_property_names: set,
    text_chars_threshold: int,
) -> bool:
    """
    Decide if an existing Notion page should be reprocessed because it was unreadable / low-quality.
    Uses whatever properties exist in the database.
    """
    # Preferred signals (if you added them)
    needs_review_name = OPTIONAL_PROPERTIES.get("Needs Review")
    if needs_review_name and needs_review_name in available_property_names:
        v = _get_checkbox(page, needs_review_name)
        if v is True:
            return True

    # OCR error / attempt signals
    if DEFAULT_PROPERTIES["OCR Error"] in available_property_names:
        err = _get_rich_text_plain(page, DEFAULT_PROPERTIES["OCR Error"])
        if err:
            return True

    if DEFAULT_PROPERTIES["OCR Attempted"] in available_property_names:
        attempted = _get_checkbox(page, DEFAULT_PROPERTIES["OCR Attempted"])
        used = _get_checkbox(page, DEFAULT_PROPERTIES["OCR Used"]) if DEFAULT_PROPERTIES["OCR Used"] in available_property_names else None
        # attempted but not used is often a sign of trouble, re-try
        if attempted is True and used is False:
            return True

    # Fallback: very low extracted text length
    if DEFAULT_PROPERTIES["Text Chars"] in available_property_names:
        n = _get_number(page, DEFAULT_PROPERTIES["Text Chars"])
        if n is not None and n < float(text_chars_threshold):
            return True

    # Fallback: OCR used but quality flagged low
    oq = OPTIONAL_PROPERTIES.get("OCR Quality")
    if oq and oq in available_property_names:
        q = _get_number(page, oq)
        if q is not None and q < 0.45:
            return True

    return False


# ----------------------------
# Main
# ----------------------------
DEFAULT_PROPERTIES = {
    "Title": "Title",  # title
    "Source Filename": "Source Filename",  # rich_text
    "Source Path": "Source Path",  # rich_text
    "File Hash": "File Hash",  # rich_text
    "Document Date": "Document Date",  # date
    "Date Confidence": "Date Confidence",  # select OR rich_text (we'll write rich_text)
    "Author": "Author",  # rich_text
    "Topics": "Topics",  # multi_select
    "Key Figures": "Key Figures",  # multi_select
    "OCR Used": "OCR Used",  # checkbox
    "OCR Attempted": "OCR Attempted",  # checkbox
    "OCR Error": "OCR Error",  # rich_text
    "Page Count": "Page Count",  # number
    "Text Chars": "Text Chars",  # number
    "Processed At": "Processed At",  # date
}

OPTIONAL_PROPERTIES = {
    "Needs Review": "Needs Review",  # checkbox
    "OCR Quality": "OCR Quality",  # number (0..1)
    "OCR Quality Notes": "OCR Quality Notes",  # rich_text
}


def build_page_properties(
    title: str,
    source_filename: str,
    source_path: str,
    file_hash: str,
    meta: DocMetadata,
    extract: PdfExtractResult,
    processed_at_iso: str,
    available_property_names: Optional[set] = None,
    property_types: Optional[Dict[str, str]] = None,
    is_update: bool = False,
    include_multi_select: bool = True,
) -> Dict[str, Any]:
    # For updates, avoid modifying multi-select properties to prevent schema size issues
    props: Dict[str, Any] = {
        DEFAULT_PROPERTIES["Title"]: prop_title(title),
        DEFAULT_PROPERTIES["Source Filename"]: prop_rich_text(source_filename),
        DEFAULT_PROPERTIES["Source Path"]: prop_rich_text(source_path),
        DEFAULT_PROPERTIES["File Hash"]: prop_rich_text(file_hash),
        DEFAULT_PROPERTIES["Document Date"]: prop_date(meta.document_date),
        DEFAULT_PROPERTIES["Date Confidence"]: prop_rich_text(meta.date_confidence),
        DEFAULT_PROPERTIES["Author"]: prop_rich_text(meta.author or ""),
        DEFAULT_PROPERTIES["OCR Used"]: prop_checkbox(extract.used_ocr),
        DEFAULT_PROPERTIES["OCR Attempted"]: prop_checkbox(bool(extract.ocr_attempted)),
        DEFAULT_PROPERTIES["OCR Error"]: prop_rich_text(extract.ocr_error or ""),
        DEFAULT_PROPERTIES["Page Count"]: prop_number(float(extract.page_count)),
        DEFAULT_PROPERTIES["Text Chars"]: prop_number(float(len(extract.text or ""))),
        DEFAULT_PROPERTIES["Processed At"]: prop_date(processed_at_iso[:10]),
    }

    # Optional OCR quality / review flags (only if the DB has the properties)
    available = available_property_names or set()
    ptypes = property_types or {}

    # Topics are forced to rich_text to avoid schema-size growth from multi-select options.
    # Key figures can still be either multi-select or rich_text.
    if not is_update:
        topics_joined = ", ".join(meta.topics or [])
        figures_joined = ", ".join(meta.key_figures or [])

        if "Topics Text" in available:
            props["Topics Text"] = prop_rich_text(topics_joined)
        if "Key Figures Text" in available:
            props["Key Figures Text"] = prop_rich_text(figures_joined)

        topics_type = ptypes.get(DEFAULT_PROPERTIES["Topics"])
        figures_type = ptypes.get(DEFAULT_PROPERTIES["Key Figures"])

        if topics_type == "rich_text":
            props[DEFAULT_PROPERTIES["Topics"]] = prop_rich_text(topics_joined)

        if include_multi_select and figures_type == "multi_select":
            props[DEFAULT_PROPERTIES["Key Figures"]] = prop_multi_select(meta.key_figures)
        elif figures_type == "rich_text":
            props[DEFAULT_PROPERTIES["Key Figures"]] = prop_rich_text(figures_joined)

    quality = estimate_text_quality(extract.text or "")
    # Needs-review if OCR was used and still looks bad, OR if extracted text is generally very low quality.
    needs_review = bool((extract.used_ocr and quality.score < 0.45) or (quality.score < 0.30))

    if OPTIONAL_PROPERTIES["Needs Review"] in available:
        props[OPTIONAL_PROPERTIES["Needs Review"]] = prop_checkbox(needs_review)
    if OPTIONAL_PROPERTIES["OCR Quality"] in available:
        props[OPTIONAL_PROPERTIES["OCR Quality"]] = prop_number(float(round(quality.score, 3)))
    if OPTIONAL_PROPERTIES["OCR Quality Notes"] in available:
        props[OPTIONAL_PROPERTIES["OCR Quality Notes"]] = prop_rich_text(quality.reason)

    return props


def build_page_blocks(
    title: str,
    pdf_path: Path,
    source_path: str,
    file_hash: str,
    date_hint: Optional[str],
    meta: DocMetadata,
    extract: PdfExtractResult,
    extracted_text_for_page: str,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    blocks.append(heading_2("Metadata"))
    blocks.append(bulleted_item(f"Title: {title}"))
    blocks.append(bulleted_item(f"Source file: {pdf_path.name}"))
    blocks.append(bulleted_item(f"Source path: {source_path}"))
    blocks.append(bulleted_item(f"File hash (sha256): {file_hash}"))
    if date_hint:
        blocks.append(bulleted_item(f"Filename date hint: {date_hint}"))
    if meta.document_date:
        blocks.append(bulleted_item(f"Document date: {meta.document_date} ({meta.date_confidence})"))
    if meta.author:
        blocks.append(bulleted_item(f"Author: {meta.author}"))
    if meta.topics:
        blocks.append(bulleted_item(f"Topics: {', '.join(meta.topics)}"))
    if meta.key_figures:
        blocks.append(bulleted_item(f"Key figures: {', '.join(meta.key_figures)}"))
    blocks.append(bulleted_item(f"Pages: {extract.page_count}"))
    blocks.append(bulleted_item(f"OCR used: {extract.used_ocr}"))
    blocks.append(bulleted_item(f"OCR attempted: {extract.ocr_attempted}"))
    blocks.append(bulleted_item(f"OCR needed (auto-detect): {extract.ocr_needed}"))
    if extract.ocr_error:
        blocks.append(bulleted_item(f"OCR error: {extract.ocr_error}"))
    quality = estimate_text_quality(extract.text or "")
    blocks.append(bulleted_item(f"Text quality (heuristic): {quality.score:.2f} ({quality.reason})"))
    if (extract.used_ocr and quality.score < 0.45) or (quality.score < 0.30):
        blocks.append(bulleted_item("Needs review: likely low-quality OCR (common for handwriting/scans)."))

    blocks.append(divider())
    blocks.append(heading_2("Summary"))
    if meta.summary_bullets:
        for b in meta.summary_bullets:
            blocks.append(bulleted_item(b))
    else:
        blocks.append(paragraph("(No summary available.)"))

    blocks.append(divider())
    blocks.append(heading_2("Extracted Text"))
    if extracted_text_for_page.strip():
        for c in chunk_text(extracted_text_for_page, size=1800):
            blocks.append(paragraph(c))
    else:
        blocks.append(paragraph("(No extractable text found.)"))

    return blocks


def validate_database_schema(database: Dict[str, Any]) -> List[str]:
    """
    Returns a list of missing property names (expected by this script).
    """
    props = database.get("properties", {}) or {}
    missing = [p for p in DEFAULT_PROPERTIES.values() if p not in props]
    return missing


def build_reprocess_append_blocks(
    meta: DocMetadata,
    extract: PdfExtractResult,
    extracted_text_for_page: str,
) -> List[Dict[str, Any]]:
    """
    Appends a new section rather than overwriting existing page content.
    This makes re-OCR / reprocessing safe and keeps everything searchable.
    """
    blocks: List[Dict[str, Any]] = []
    quality = estimate_text_quality(extract.text or "")

    blocks.append(divider())
    blocks.append(heading_2("Reprocessed (AI)"))
    blocks.append(bulleted_item(f"OCR used: {extract.used_ocr}"))
    blocks.append(bulleted_item(f"Text quality (heuristic): {quality.score:.2f} ({quality.reason})"))

    blocks.append(heading_2("Reprocessed Summary"))
    if meta.summary_bullets:
        for b in meta.summary_bullets:
            blocks.append(bulleted_item(b))
    else:
        blocks.append(paragraph("(No summary available.)"))

    blocks.append(heading_2("Reprocessed Extracted Text"))
    if extracted_text_for_page.strip():
        for c in chunk_text(extracted_text_for_page, size=1800):
            blocks.append(paragraph(c))
    else:
        blocks.append(paragraph("(No extractable text found.)"))

    return blocks


# Default DB: Meeting Transcripts
# (your other DB is the Frozen Documents database)


def build_source_path(filename: str) -> str:
    name = Path(filename).name
    return str(PurePosixPath(SOURCE_PATH_PREFIX) / name)


def run_single_test_file_check(
    pdf_path: Path,
    args: argparse.Namespace,
    openai_client: Optional[OpenAI],
) -> None:
    """
    Preflight check for one file: extraction + metadata only (no Notion writes).
    """
    print("\n--- Single test file check ---")
    print(f"File: {pdf_path}")
    title = clean_title_from_filename(pdf_path.name)
    date_hint = date_hint_from_filename(pdf_path.name)

    extract = extract_pdf_text_with_optional_ocr(
        pdf_path=pdf_path,
        max_pages_text=args.max_pages_text,
        ocr_enabled=bool((args.auto_ocr or args.ocr) and openai_client is not None),
        ocr_pages=args.ocr_pages,
        openai_client=openai_client,
        vision_model=args.vision_model,
        min_text_chars_for_no_ocr=args.min_text_chars_for_no_ocr,
    )

    if openai_client:
        meta = extract_metadata_with_openai(
            client=openai_client,
            model=args.openai_model,
            title=title,
            filename=pdf_path.name,
            date_hint=date_hint,
            pdf_metadata=extract.pdf_metadata,
            text=extract.text or "",
        ) or fallback_metadata(pdf_path.name, extract.pdf_metadata)
    else:
        meta = fallback_metadata(pdf_path.name, extract.pdf_metadata)

    quality = estimate_text_quality(extract.text or "")
    print(f"Title: {title}")
    print(f"Date hint: {date_hint or '(none)'}")
    print(f"Extract chars: {len(extract.text or '')}")
    print(f"OCR used: {extract.used_ocr}")
    print(f"Text quality: {quality.score:.2f} ({quality.reason})")
    print(f"Document date: {meta.document_date or '(none)'} [{meta.date_confidence}]")
    print(f"Topics chosen: {len(meta.topics)}")
    print(f"Key figures chosen: {len(meta.key_figures)}")
    print(f"Topics: {', '.join(meta.topics[:8]) if meta.topics else '(none)'}")
    print(f"Key figures: {', '.join(meta.key_figures[:8]) if meta.key_figures else '(none)'}")
    bullets = meta.summary_bullets[:3] if meta.summary_bullets else []
    if bullets:
        print("Summary preview:")
        for b in bullets:
            print(f"  - {b}")
    else:
        print("Summary preview: (none)")

    if args.test_write_file:
        out_dir = Path(args.test_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdf_path.stem}__test_output.md"
        extracted_text_for_file = (extract.text or "")[: args.max_text_chars]
        lines: List[str] = []
        lines.append("# Single File Test Output")
        lines.append("")
        lines.append(f"- Source file: `{pdf_path.name}`")
        lines.append(f"- Title: {title}")
        lines.append(f"- Date hint: {date_hint or '(none)'}")
        lines.append(f"- Document date: {meta.document_date or '(none)'}")
        lines.append(f"- Date confidence: {meta.date_confidence}")
        lines.append(f"- OCR used: {extract.used_ocr}")
        lines.append(f"- Text quality: {quality.score:.2f} ({quality.reason})")
        lines.append(f"- Extract chars: {len(extract.text or '')}")
        lines.append("")
        lines.append("## Topics")
        lines.append(f"_Count: {len(meta.topics)}_")
        if meta.topics:
            for t in meta.topics:
                lines.append(f"- {t}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Key Figures")
        lines.append(f"_Count: {len(meta.key_figures)}_")
        if meta.key_figures:
            for k in meta.key_figures:
                lines.append(f"- {k}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Summary")
        if meta.summary_bullets:
            for b in meta.summary_bullets:
                lines.append(f"- {b}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Extracted Text")
        lines.append("")
        lines.append(extracted_text_for_file if extracted_text_for_file.strip() else "(No extractable text found.)")
        lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Test output written: {out_path}")
    print("--- End test file check ---\n")


def build_meeting_md_blocks(
    prepared: Dict[str, Any],
    raw_markdown: str,
    key_figures: Optional[List[str]] = None,
    max_chars: int = 120000,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    key_figures = key_figures or []

    blocks.append(heading_2("Metadata"))
    blocks.append(bulleted_item(f"Source file: {prepared.get('source_filename', '')}"))
    if prepared.get("document_date"):
        blocks.append(
            bulleted_item(
                f"Document date: {prepared.get('document_date')} ({prepared.get('date_confidence', 'low')})"
            )
        )
    if prepared.get("participants"):
        blocks.append(bulleted_item(f"Participants: {prepared.get('participants')}"))
    if prepared.get("topics"):
        blocks.append(bulleted_item(f"Topics: {', '.join(prepared.get('topics') or [])}"))
    if key_figures:
        blocks.append(bulleted_item(f"Key figures: {', '.join(key_figures)}"))
    if prepared.get("triage_legibility") is not None:
        blocks.append(bulleted_item(f"Triage legibility: {prepared.get('triage_legibility')}"))
    if prepared.get("triage_participant_identifiability") is not None:
        blocks.append(
            bulleted_item(
                f"Triage participant identifiability: {prepared.get('triage_participant_identifiability')}"
            )
        )
    if prepared.get("triage_subject_clarity") is not None:
        blocks.append(bulleted_item(f"Triage subject clarity: {prepared.get('triage_subject_clarity')}"))
    if prepared.get("triage_total") is not None:
        blocks.append(bulleted_item(f"Triage total: {prepared.get('triage_total')}"))

    blocks.append(divider())
    blocks.append(heading_2("Summary"))
    summary = (prepared.get("summary") or "").strip()
    if summary:
        pieces = [p.strip() for p in summary.split("|") if p.strip()]
        if pieces:
            for p in pieces[:8]:
                blocks.append(bulleted_item(p))
        else:
            blocks.append(paragraph(summary[:1800]))
    else:
        blocks.append(paragraph("(No summary available.)"))

    blocks.append(divider())
    blocks.append(heading_2("Transcript"))
    transcript = (prepared.get("transcript") or "").strip()
    transcript_to_store = transcript if transcript else raw_markdown
    transcript_to_store = transcript_to_store[:max_chars]
    for c in chunk_text(transcript_to_store, size=1800):
        blocks.append(paragraph(c))

    blocks.append(divider())
    blocks.append(heading_2("Original Markdown"))
    for c in chunk_text((raw_markdown or "")[:max_chars], size=1800):
        blocks.append(paragraph(c))

    return blocks


def build_meeting_properties(
    prepared: Dict[str, Any],
    md_path: Path,
    file_hash: str,
    processed_at_iso: str,
    available_property_names: set,
    property_types: Dict[str, str],
    include_multi_select: bool,
    key_figures: Optional[List[str]] = None,
) -> Dict[str, Any]:
    def _first_value(obj: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in obj and obj.get(k) is not None:
                return obj.get(k)
        return None

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    triage = prepared.get("triage") if isinstance(prepared.get("triage"), dict) else {}
    triage_legibility = _first_value(
        prepared,
        ["triage_legibility", "legibility"],
    )
    if triage_legibility is None:
        triage_legibility = _first_value(triage, ["legibility", "triage_legibility"])

    triage_participant = _first_value(
        prepared,
        ["triage_participant_identifiability", "participant_confidence"],
    )
    if triage_participant is None:
        triage_participant = _first_value(
            triage, ["participant_confidence", "triage_participant_identifiability"]
        )

    triage_subject = _first_value(
        prepared,
        ["triage_subject_clarity", "subject_confidence"],
    )
    if triage_subject is None:
        triage_subject = _first_value(triage, ["subject_confidence", "triage_subject_clarity"])

    triage_total = _first_value(
        prepared,
        ["triage_total", "total_score"],
    )
    if triage_total is None:
        triage_total = _first_value(triage, ["total_score", "triage_total"])

    props = build_frozen_document_properties_from_meeting(prepared)
    key_figures = key_figures or []
    participants_list = prepared.get("participants_list") or []
    participants_list = [str(p).strip() for p in participants_list if str(p).strip()]
    if not participants_list:
        participants_list = [p.strip(" -") for p in re.split(r"[\n,;|]+", str(prepared.get("participants") or "")) if p.strip(" -")]
    participants_joined = ", ".join(participants_list)

    # Keep only fields that exist in the target DB.
    props = {k: v for k, v in props.items() if k in available_property_names}

    # Rebuild topic/key-figure properties based on actual DB types.
    # Topics are forced to rich_text.
    topics = [str(t).strip() for t in (prepared.get("topics") or []) if str(t).strip()]
    topics_joined = ", ".join(topics)
    figures_joined = ", ".join(key_figures)
    props.pop("Topics", None)
    props.pop("Key Figures", None)

    if "Topics Text" in available_property_names:
        props["Topics Text"] = prop_rich_text(topics_joined)
    if "Key Figures Text" in available_property_names:
        props["Key Figures Text"] = prop_rich_text(figures_joined)

    topics_type = property_types.get("Topics")
    if topics_type == "rich_text":
        props["Topics"] = prop_rich_text(topics_joined)

    # Optional extra fields commonly present in frozen schema.
    if "Source Path" in available_property_names:
        props["Source Path"] = prop_rich_text(build_source_path(md_path.name))
    if "File Hash" in available_property_names:
        props["File Hash"] = prop_rich_text(file_hash)
    if "Processed At" in available_property_names:
        props["Processed At"] = prop_date(processed_at_iso[:10])
    if "Summary" in available_property_names and "Summary" not in props:
        props["Summary"] = prop_rich_text(str(prepared.get("summary") or ""))
    if "Participants Text" in available_property_names:
        props["Participants Text"] = prop_rich_text(participants_joined)
    participants_type = property_types.get("Participants")
    if include_multi_select and participants_type == "multi_select":
        props["Participants"] = prop_multi_select(participants_list[:20])
    elif participants_type == "rich_text" and "Participants" in available_property_names:
        props["Participants"] = prop_rich_text(participants_joined)
    figures_type = property_types.get("Key Figures")
    if figures_type == "rich_text":
        props["Key Figures"] = prop_rich_text(figures_joined)
    elif include_multi_select and key_figures and figures_type == "multi_select":
        props["Key Figures"] = prop_multi_select(key_figures)
    if "Triage Legibility" in available_property_names:
        props["Triage Legibility"] = prop_number(_to_float(triage_legibility))
    if "Triage Participant Identifiability" in available_property_names:
        props["Triage Participant Identifiability"] = prop_number(_to_float(triage_participant))
    if "Triage Subject Clarity" in available_property_names:
        props["Triage Subject Clarity"] = prop_number(_to_float(triage_subject))
    if "Triage Total" in available_property_names:
        props["Triage Total"] = prop_number(_to_float(triage_total))

    return props


def _compact_debug_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove very large text fields for readable debug output.
    """
    out = dict(obj or {})
    for k in [
        "transcript",
        "raw_markdown",
        "text",
        "base_text",
        "extracted_text_for_page",
        "original_markdown",
    ]:
        out.pop(k, None)
    # Keep lengths for context
    if "summary" in out and isinstance(out["summary"], str):
        out["summary_len"] = len(out["summary"])
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload PDFs from frozen/ into a Notion database with extracted metadata.")
    parser.add_argument(
        "--folder",
        default=DEFAULT_SOURCE_FOLDER,
        help=f"Folder containing PDFs (default: {DEFAULT_SOURCE_FOLDER})",
    )
    parser.add_argument(
        "--source-type",
        choices=["auto", "pdf", "md"],
        default="auto",
        help="Input source type in --folder. 'md' is for meeting_transcripts/transcripts.",
    )
    parser.add_argument("--database-id", default=NOTION_DATABASE_ID, help="Notion database ID")
    parser.add_argument("--notion-token", default=os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY") or read_notion_token_from_keys_file(), help="Notion integration token")
    parser.add_argument("--count-documents", action="store_true", help="Count and display the total number of documents in the database, then exit")

    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY") or read_openai_key_from_main_py(),
        help="OpenAI API key (enables metadata extraction + OCR fallback). Defaults to OPENAI_API_KEY or a best-effort fallback from main.py.",
    )
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_METADATA_MODEL") or "gpt-5.1", help="OpenAI model for metadata extraction")
    parser.add_argument("--vision-model", default=os.getenv("OPENAI_VISION_MODEL") or "gpt-5.1", help="OpenAI vision model for OCR fallback")

    parser.add_argument("--max-pages-text", type=int, default=50, help="Max pages to read text from (PyMuPDF) per PDF")
    parser.add_argument("--max-text-chars", type=int, default=120000, help="Max extracted characters to store into Notion page body")
    # Default behavior: auto-OCR unreadable documents if an OpenAI key is available.
    parser.add_argument(
        "--auto-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically OCR unreadable/garbled PDFs using OpenAI vision (default: enabled).",
    )
    # Backwards-compatible alias
    parser.add_argument("--ocr", action="store_true", help="Alias for --auto-ocr (deprecated).")
    parser.add_argument("--ocr-pages", type=int, default=6, help="How many first pages to OCR if needed")
    parser.add_argument(
        "--min-text-chars-for-no-ocr",
        type=int,
        default=800,
        help="If extracted text is shorter than this, OCR may be triggered (default: 800).",
    )

    parser.add_argument("--limit", type=int, default=0, help="Process only first N PDFs (0 = all)")
    parser.add_argument("--filter-filename", help="Only process PDFs with this exact filename (case-insensitive, without .pdf extension)")
    parser.add_argument(
        "--reverse-stop-on-existing-filename",
        action="store_true",
        help="PDF mode only: process files in reverse order and stop pre-filter scan at first filename that already exists in Notion.",
    )
    parser.add_argument(
        "--print-object",
        action="store_true",
        help="Print returned metadata object with large text fields removed.",
    )
    parser.add_argument(
        "--print-multiselect",
        action="store_true",
        help="Print Topics/Key Figures that will be sent to Notion before create/update.",
    )
    parser.add_argument(
        "--disable-multi-select",
        action="store_true",
        help="Do not write Topics/Key Figures multi-select properties (avoids Notion schema-size limit issues).",
    )
    parser.add_argument("--test-file", help="Run a single-file preflight check before processing. Accepts a path or filename under --folder.")
    parser.add_argument("--test-only", action="store_true", help="With --test-file, run only the test check and exit.")
    parser.add_argument("--test-write-file", action="store_true", help="With --test-file, write a local markdown test output.")
    parser.add_argument("--test-output-dir", default="_test_outputs", help="Output directory for --test-write-file (default: _test_outputs)")
    parser.add_argument("--dry-run", action="store_true", help="Do not create/update pages; just print what would happen")
    parser.add_argument("--update-existing", action="store_true", help="If File Hash already exists, update properties (does not rewrite page body)")
    parser.add_argument(
        "--update-body",
        action="store_true",
        help="When used with --update-existing, append refreshed summary/text to existing pages (useful for handwritten/scanned docs).",
    )
    parser.add_argument(
        "--update-only-unreadable",
        action="store_true",
        help="When used with --update-existing, only update pages that are flagged unreadable/needs-review (e.g., Needs Review checked, OCR Error present, or very low Text Chars).",
    )
    parser.add_argument(
        "--update-only-poor-extract",
        action="store_true",
        help="When used with --update-existing, ignore Notion flags and only update if the CURRENT extraction looks poor (short/garbled), triggering OCR + reprocessing.",
    )
    parser.add_argument("--sleep", type=float, default=0.35, help="Sleep between Notion writes (seconds)")
    parser.add_argument("--force", action="store_true", help="Proceed even if the Notion database schema is missing expected properties")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    if not args.database_id:
        raise SystemExit("Missing --database-id (or set NOTION_FROZEN_DATABASE_ID / NOTION_DATABASE_ID)")
    if not args.notion_token:
        raise SystemExit("Missing --notion-token (or set NOTION_TOKEN / NOTION_API_KEY, or put it in `keys`)")

    headers = notion_headers(args.notion_token)
    db = notion_get_database(args.database_id, headers)

    # Handle count documents request
    if args.count_documents:
        try:
            count = notion_count_documents(args.database_id, headers)
            print(f"Total documents in database: {count}")
        except Exception as e:
            print(f"Error counting documents: {e}")
        return

    db_properties = db.get("properties", {}) or {}
    available_property_names = set(db_properties.keys())
    property_types = {name: (info or {}).get("type", "") for name, info in db_properties.items()}

    openai_client: Optional[OpenAI] = None
    if args.openai_api_key:
        openai_client = OpenAI(api_key=args.openai_api_key)

    # Collect input files based on source type.
    source_type = args.source_type
    if source_type == "auto":
        has_md = any(p.is_file() and p.suffix.lower() == ".md" for p in folder.rglob("*"))
        has_pdf = any(p.is_file() and p.suffix.lower() == ".pdf" for p in folder.rglob("*"))
        if has_md and not has_pdf:
            source_type = "md"
        elif has_pdf:
            source_type = "pdf"
        else:
            source_type = "md" if has_md else "pdf"

    files_seen: set[str] = set()
    files: List[Path] = []
    wanted_ext = ".md" if source_type == "md" else ".pdf"
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() != wanted_ext:
            continue
        key = str(p.resolve()).lower()
        if key in files_seen:
            continue
        files_seen.add(key)
        files.append(p)
    files = sorted(files, reverse=True)
    if args.filter_filename:
        filter_lower = args.filter_filename.lower()
        files = [p for p in files if p.stem.lower() == filter_lower or p.name.lower() == filter_lower]
        if not files:
            print(f"No {wanted_ext} files found matching filename: {args.filter_filename}")
            return

    # Optional: one-file preflight check before full run.
    if args.test_file and source_type == "pdf":
        requested = Path(args.test_file)
        test_pdf: Optional[Path] = None
        if requested.exists() and requested.is_file() and requested.suffix.lower() == ".pdf":
            test_pdf = requested
        else:
            needle = requested.name.lower()
            for p in files:
                if p.name.lower() == needle:
                    test_pdf = p
                    break
                if p.stem.lower() == needle.replace(".pdf", ""):
                    test_pdf = p
                    break
        if not test_pdf:
            print(f"Test file not found: {args.test_file}")
            return
        run_single_test_file_check(test_pdf, args, openai_client)
        if args.test_only:
            print("Exiting after --test-only preflight check.")
            return

    # First, get database info to confirm we're connecting to the right database
    try:
        db_info = notion_get_database(args.database_id, headers)
        db_title = db_info.get("title", [{}])[0].get("plain_text", "Unknown")
        print(f"Connected to database: '{db_title}' (ID: {args.database_id[:8]}...)")
    except Exception as e:
        print(f"Warning: Could not get database info: {e}")

    file_hash_cache: Dict[Path, str] = {}
    if source_type == "pdf":
        missing = validate_database_schema(db)
        if missing:
            print("WARNING: Your Notion database is missing expected properties:")
            for m in missing:
                print(f"  - {m}")
            print("The script will likely fail until you add these properties (see FROZEN_TO_NOTION.md).")
            if not args.force:
                print("\nAborting. Create a new Notion database with the required properties, then re-run.")
                return

        # Pre-filter PDFs that already exist in the database
        print("Checking which PDFs are already in the database...")
        existing_hashes, existing_filenames_lower = notion_get_existing_file_keys(args.database_id, headers)
        print(f"Found {len(existing_hashes)} existing documents with file hashes in database")
        total_count = notion_count_documents(args.database_id, headers)
        print(f"Total documents in database: {total_count}")
        print(f"Documents without file hashes: {total_count - len(existing_hashes)}")

        files_to_process = []
        skipped_count = 0
        for pdf_path in files:
            if pdf_path.name.lower() in existing_filenames_lower:
                if args.reverse_stop_on_existing_filename:
                    print(f"  [STOP] {pdf_path.name} - filename exists in database; stopping scan in reverse mode")
                    break
                print(f"  [SKIP] {pdf_path.name} - filename already exists in database")
                skipped_count += 1
                continue
            file_hash = sha256_file(pdf_path)
            if file_hash in existing_hashes:
                print(f"  [SKIP] {pdf_path.name} - already exists in database")
                skipped_count += 1
            else:
                files_to_process.append(pdf_path)
                file_hash_cache[pdf_path] = file_hash
        files = files_to_process
        print(f"Pre-filtered {skipped_count} existing documents, {len(files)} PDFs to process")
    else:
        print("MD mode enabled: processing meeting transcript markdown files.")

    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"No {wanted_ext} files found.")
        return
    print(f"Found {len(files)} {wanted_ext} file(s) under: {folder}")

    processed_at = datetime.now(timezone.utc).date().isoformat()

    created = 0
    skipped = 0
    updated = 0
    failed = 0
    would_create = 0
    would_update = 0
    would_skip = 0
    updated_body = 0
    would_update_body = 0

    for pdf_path in tqdm(files, desc=f"Uploading {source_type.upper()} files"):
        try:
            if source_type == "md":
                raw_md = pdf_path.read_text(encoding="utf-8", errors="ignore")
                prepared = prepare_meeting_transcript_for_frozen_format(
                    file_path=str(pdf_path),
                    use_openai=bool(openai_client),
                    openai_api_key=args.openai_api_key,
                    openai_model=args.openai_model,
                    apply_acronyms_to_markdown=True,
                    write_back_markdown=True,
                )
                # Re-read in case acronym expansion updated transcript section.
                raw_md = pdf_path.read_text(encoding="utf-8", errors="ignore")
                key_figures: List[str] = []
                if openai_client:
                    md_meta = extract_metadata_with_openai(
                        client=openai_client,
                        model=args.openai_model,
                        title=str(prepared.get("title") or pdf_path.stem),
                        filename=pdf_path.name,
                        date_hint=str(prepared.get("document_date") or ""),
                        pdf_metadata={},
                        text=str(prepared.get("transcript") or raw_md),
                    )
                    if md_meta:
                        key_figures = md_meta.key_figures
                        if md_meta.topics:
                            prepared["topics"] = md_meta.topics

                if args.print_object:
                    debug_obj = _compact_debug_obj(prepared)
                    if key_figures:
                        debug_obj["key_figures"] = key_figures
                    print(f"  [OBJECT] {pdf_path.name}")
                    print(json.dumps(debug_obj, ensure_ascii=False, indent=2))

                if args.print_multiselect:
                    print(f"  [MULTI-SELECT] {pdf_path.name}")
                    print(f"    Participants ({len(prepared.get('participants_list') or [])}): {prepared.get('participants_list') or []}")
                    print(f"    Topics ({len(prepared.get('topics') or [])}): {prepared.get('topics') or []}")
                    print(f"    Key Figures ({len(key_figures)}): {key_figures}")

                file_hash = sha256_file(pdf_path)
                props = build_meeting_properties(
                    prepared=prepared,
                    md_path=pdf_path,
                    file_hash=file_hash,
                    processed_at_iso=processed_at,
                    available_property_names=available_property_names,
                    property_types=property_types,
                    include_multi_select=not args.disable_multi_select,
                    key_figures=key_figures,
                )
                blocks = build_meeting_md_blocks(
                    prepared=prepared,
                    raw_markdown=raw_md,
                    key_figures=key_figures,
                    max_chars=args.max_text_chars,
                )

                existing_page_id = notion_find_page_id_by_rich_text_equals(
                    database_id=args.database_id,
                    headers=headers,
                    property_name="Source Filename",
                    value=pdf_path.name,
                )
                if existing_page_id and not args.update_existing:
                    print(f"  [SKIP] {pdf_path.name} - already exists by Source Filename")
                    skipped += 1
                    continue
                if args.dry_run:
                    if existing_page_id and args.update_existing:
                        would_update += 1
                    else:
                        would_create += 1
                    continue

                if existing_page_id and args.update_existing:
                    notion_update_page(existing_page_id, headers, props)
                    updated += 1
                    if args.update_body:
                        notion_append_children(existing_page_id, headers, blocks)
                        updated_body += 1
                    time.sleep(args.sleep)
                    continue

                try:
                    page_id, remaining = notion_create_page(args.database_id, headers, props, blocks)
                except RuntimeError as e:
                    msg = str(e).lower()
                    # Safety fallback for Notion schema-size limit on multi-select options.
                    if "schema has exceeded the maximum size" in msg and not args.disable_multi_select:
                        print("  [RETRY] Schema limit hit; retrying without multi-select fields.")
                        props_retry = build_meeting_properties(
                            prepared=prepared,
                            md_path=pdf_path,
                            file_hash=file_hash,
                            processed_at_iso=processed_at,
                            available_property_names=available_property_names,
                            property_types=property_types,
                            include_multi_select=False,
                            key_figures=key_figures,
                        )
                        page_id, remaining = notion_create_page(args.database_id, headers, props_retry, blocks)
                    else:
                        raise
                if remaining:
                    notion_append_children(page_id, headers, remaining)
                created += 1
                time.sleep(args.sleep)
                continue

            file_hash = file_hash_cache.get(pdf_path) or sha256_file(pdf_path)

            title = clean_title_from_filename(pdf_path.name)
            date_hint = date_hint_from_filename(pdf_path.name)

            extract = extract_pdf_text_with_optional_ocr(
                pdf_path=pdf_path,
                max_pages_text=args.max_pages_text,
                ocr_enabled=bool((args.auto_ocr or args.ocr) and openai_client is not None),
                ocr_pages=args.ocr_pages,
                openai_client=openai_client,
                vision_model=args.vision_model,
                min_text_chars_for_no_ocr=args.min_text_chars_for_no_ocr,
            )

            # limit body size to avoid enormous Notion pages
            extracted_text_for_page = (extract.text or "")[: args.max_text_chars]

            if openai_client:
                meta = extract_metadata_with_openai(
                    client=openai_client,
                    model=args.openai_model,
                    title=title,
                    filename=pdf_path.name,
                    date_hint=date_hint,
                    pdf_metadata=extract.pdf_metadata,
                    text=extract.text or "",
                ) or fallback_metadata(pdf_path.name, extract.pdf_metadata)
            else:
                meta = fallback_metadata(pdf_path.name, extract.pdf_metadata)

            if args.print_object:
                debug_obj = _compact_debug_obj(asdict(meta))
                debug_obj.update(
                    {
                        "source_filename": pdf_path.name,
                        "page_count": extract.page_count,
                        "used_ocr": extract.used_ocr,
                        "ocr_attempted": extract.ocr_attempted,
                        "ocr_error": extract.ocr_error,
                        "text_chars": len(extract.text or ""),
                    }
                )
                print(f"  [OBJECT] {pdf_path.name}")
                print(json.dumps(debug_obj, ensure_ascii=False, indent=2))

            source_path = build_source_path(pdf_path.name)
            props = build_page_properties(
                title=title,
                source_filename=pdf_path.name,
                source_path=source_path,
                file_hash=file_hash,
                meta=meta,
                extract=extract,
                processed_at_iso=processed_at,
                available_property_names=available_property_names,
                property_types=property_types,
                is_update=False,  # We pre-filtered, so no existing pages
                include_multi_select=not args.disable_multi_select,
            )

            if args.print_multiselect:
                print(f"  [MULTI-SELECT] {pdf_path.name}")
                print(f"    Topics ({len(meta.topics)}): {meta.topics}")
                print(f"    Key Figures ({len(meta.key_figures)}): {meta.key_figures}")

            blocks = build_page_blocks(
                title=title,
                pdf_path=pdf_path,
                source_path=source_path,
                file_hash=file_hash,
                date_hint=date_hint,
                meta=meta,
                extract=extract,
                extracted_text_for_page=extracted_text_for_page,
            )

            if args.dry_run:
                would_create += 1
                continue

            # Since we pre-filtered, all remaining PDFs should be new
            try:
                print(f"  [CREATE] {pdf_path.name} - uploading to database")
            except UnicodeEncodeError:
                print(f"  [CREATE] {pdf_path.name.encode('ascii', 'ignore').decode('ascii')} - uploading to database")
            try:
                page_id, remaining = notion_create_page(args.database_id, headers, props, blocks)
            except RuntimeError as e:
                msg = str(e).lower()
                # Safety fallback for Notion multi-select schema growth limit.
                if "schema has exceeded the maximum size" in msg and not args.disable_multi_select:
                    print("  [RETRY] Schema limit hit; retrying without Topics/Key Figures multi-select fields.")
                    props_retry = build_page_properties(
                        title=title,
                        source_filename=pdf_path.name,
                        source_path=source_path,
                        file_hash=file_hash,
                        meta=meta,
                        extract=extract,
                        processed_at_iso=processed_at,
                        available_property_names=available_property_names,
                        property_types=property_types,
                        is_update=False,
                        include_multi_select=False,
                    )
                    page_id, remaining = notion_create_page(args.database_id, headers, props_retry, blocks)
                else:
                    raise
            if remaining:
                notion_append_children(page_id, headers, remaining)
            created += 1
            time.sleep(args.sleep)

        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {pdf_path.name}: {type(e).__name__}: {e}")
            continue

    print("\nDone.")
    if args.dry_run:
        print("  (dry-run: no pages were created)")
        print(f"  Would create: {would_create}")
        print(f"  Would update: {would_update}")
        print(f"  Failed: {failed}")
    else:
        print(f"  Created: {created}")
        print(f"  Updated: {updated}")
        if updated_body:
            print(f"  Updated body: {updated_body}")
        if skipped:
            print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
