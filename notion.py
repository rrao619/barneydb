import os
import re
import time
import csv
import json
import requests
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _read_notion_token_from_keys_file(keys_path="keys"):
    """
    Best-effort read of a Notion integration token from the repo `keys` file.

    Expected format (example):
      internal integration = ntn_...
    """
    try:
        if not os.path.exists(keys_path):
            return None
        with open(keys_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if "=" in raw:
            return raw.split("=", 1)[1].strip()
        return None
    except Exception:
        return None


# Backwards-compatible defaults, overridden by environment variables if present.
NOTION_API_KEY = (
    os.getenv("NOTION_API_KEY")
    or os.getenv("NOTION_TOKEN")
    or _read_notion_token_from_keys_file()
    or "ntn_V3932942242aq2cHlth4AO4W6OjgKDSY7Aa3xbZqkJqfx3"
)
DATABASE_ID = os.getenv("NOTION_DATABASE_ID") or "2cb0ea6befa280e681bef5b87e65a4c0"

HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

TRANSCRIPTS_DIR = "./transcripts"


# ---------- helpers ----------

def chunk_text(text, size=1800):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def load_abbreviation_glossary(csv_path="glossary_abbrev_with_and_without_period.csv"):
    """
    Load abbreviation glossary rows as (abbr, meaning, alternate_meaning).
    Supports both glossary CSV formats present in this repo.
    """
    rows = []
    if not os.path.exists(csv_path):
        return rows

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbr = (row.get("Abbreviation") or "").strip()
            meaning = (row.get("Meaning") or row.get("Meaning (to fill in)") or "").strip()
            alt = (row.get("AlternateMeaning") or row.get("Alternate meaning") or "").strip()
            if abbr:
                rows.append((abbr, meaning, alt))
    return rows


def _read_openai_key_from_main_py(main_py_path="main.py"):
    """
    Best-effort fallback for repos that keep the OpenAI key in main.py.
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


def _extract_first_json_object(text):
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def _extract_section_map(markdown_text):
    """
    Parse level-2 markdown sections and keep repeated headings as lists.
    """
    sections = {}
    current = None
    buffer = []

    def flush():
        nonlocal current, buffer
        if current is None:
            return
        sections.setdefault(current, []).append("\n".join(buffer).strip())
        buffer = []

    for line in markdown_text.splitlines():
        m = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m:
            flush()
            current = m.group(1).strip()
            continue
        if current is not None:
            buffer.append(line)
    flush()
    return sections


def _date_hint_from_filename(filename):
    """
    Extract YYYYMMDD date prefix from filename.
    Returns (iso_date, confidence, uncertain_flag).
    """
    m = re.search(r"(?<!\d)(\d{8})(?!\d)", filename)
    if not m:
        return None, "low", True
    raw = m.group(1)
    try:
        dt = datetime.strptime(raw, "%Y%m%d").date()
        return dt.isoformat(), "low", True
    except ValueError:
        return None, "low", True


def _parse_human_date(date_text):
    """
    Parse MM/DD/YY or MM/DD/YYYY style dates.
    """
    if not date_text:
        return None
    date_text = date_text.strip()
    if "uncertain" in date_text.lower() or "blank" in date_text.lower():
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_text, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _replace_markdown_section(markdown_text, heading, new_section_body):
    """
    Replace first `## <heading>` section body.
    """
    lines = markdown_text.splitlines()
    out = []
    i = 0
    replaced = False

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m and m.group(1).strip() == heading and not replaced:
            out.append(line)
            out.extend((new_section_body or "").splitlines())
            i += 1
            while i < len(lines):
                m2 = re.match(r"^\s*##\s+(.+?)\s*$", lines[i])
                if m2:
                    break
                i += 1
            replaced = True
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if markdown_text.endswith("\n") else "")


def _split_subjects_to_topics(subjects_text):
    tokens = [t.strip(" -") for t in re.split(r"[\n,;|]+", subjects_text or "") if t.strip(" -")]
    # Keep order, dedupe
    seen = set()
    out = []
    for t in tokens:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[:20]


def _infer_participants_from_text(summary, transcript, max_items=10):
    """
    Lightweight participant inference from speaker-like patterns.
    """
    candidates = []
    text = "\n".join([summary or "", transcript or ""])
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # "Name: ..." style
        m = re.match(r"^([A-Z][A-Za-z.\- ]{2,60}):\s+", s)
        if m:
            candidates.append(m.group(1).strip())
        # "Name - ..." or "Name — ..." style
        m = re.match(r"^([A-Z][A-Za-z.\- ]{2,60})\s+[—-]\s+", s)
        if m:
            candidates.append(m.group(1).strip())

    out = []
    seen = set()
    for c in candidates:
        key = c.lower()
        if key not in seen and len(c.split()) <= 6:
            seen.add(key)
            out.append(c)
        if len(out) >= max_items:
            break
    return out


def _normalize_participants_list(participants_text, max_items=20):
    values = [v.strip(" -") for v in re.split(r"[\n,;|]+", participants_text or "") if v.strip(" -")]
    out = []
    seen = set()
    for v in values:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= max_items:
            break
    return out


def _extract_meeting_metadata_with_openai(
    source_filename,
    summary,
    participants,
    subjects,
    transcript,
    openai_api_key=None,
    model="gpt-5.1",
):
    """
    OpenAI enrichment aligned with the other DB flow: summary + participants + topics + date/date confidence.
    """
    if OpenAI is None:
        return {}

    key = openai_api_key or os.getenv("OPENAI_API_KEY") or _read_openai_key_from_main_py()
    if not key:
        return {}

    client = OpenAI(api_key=key)

    prompt = f"""
Extract structured meeting metadata as strict JSON with keys:
- document_date: ISO date (YYYY-MM-DD) or null
- date_confidence: one of ["high","medium","low"]
- summary: concise string summary (3-6 bullets merged to one short paragraph)
- participants: array of participant names/entities (can include likely inferred entities)
- topics: array of 5-12 short topic tags

Rules:
- Route subject matter into `topics`.
- Keep uncertain values conservative.
- If date is only from filename convention, use confidence "low".
- Output JSON only.

Source filename: {source_filename}
Existing summary: {summary}
Existing participants: {participants}
Existing subjects: {subjects}

Transcript excerpt:
{(transcript or "")[:14000]}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        data = _extract_first_json_object(text) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        print(f"   [WARNING] OpenAI enrichment failed: {e}")
        return {}


def _expand_transcript_acronyms_conservative(transcript, glossary_rows):
    """
    Expand acronyms in transcript only when mapping is confident (single meaning, no alternate).
    """
    if not transcript:
        return transcript, []

    updated = transcript
    applied = []
    for abbr, meaning, alt in glossary_rows:
        short = (abbr or "").rstrip(".").strip()
        meaning = (meaning or "").strip()
        alt = (alt or "").strip()
        if not short or not meaning:
            continue
        if alt and alt.lower() != "nan":
            continue
        if not re.fullmatch(r"[A-Z0-9]{2,}", short):
            continue
        if len(short) < 3 and short not in {"UN", "US", "UK", "EU"}:
            continue
        if f"({short})" in updated:
            continue
        pat = re.compile(rf"(?<!\w){re.escape(short)}\.?(?!\w)")
        if pat.search(updated):
            updated = pat.sub(f"{meaning} ({short})", updated, count=1)
            applied.append(short)
    return updated, applied


def _select_best_summary(summary_sections):
    """
    Choose the best summary section when there are multiple '## Summary' blocks.
    """
    if not summary_sections:
        return ""
    cleaned = [s.strip() for s in summary_sections if s and s.strip()]
    if not cleaned:
        return ""
    # Canonical files usually have an early triage-summary and later real summary.
    # Prefer the last substantial section and strip synthetic title/date lines.
    preferred = cleaned[-1]
    lines = [ln for ln in preferred.splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].startswith("# ") and "CANONICAL_MEETING_TRANSCRIPT" in lines[0]:
        return ""
    return preferred


def _find_matching_acronyms(text, glossary_rows, max_items=30):
    """
    Return abbreviations found either by direct acronym hit or by matching meaning text.
    """
    found = []
    if not text:
        return found
    text_lower = text.lower()

    for abbr, meaning, alt in glossary_rows:
        normalized_abbr = abbr.rstrip(".")
        if not normalized_abbr:
            continue
        # Keep acronym-like tokens only (e.g., UN, USAID, ISI).
        if not re.fullmatch(r"[A-Z0-9]{2,}", normalized_abbr):
            continue
        if len(normalized_abbr) < 3 and normalized_abbr not in {"UN", "US", "UK", "EU"}:
            continue
        abbr_pat = rf"(?<!\w){re.escape(normalized_abbr)}\.?(?!\w)"
        meaning_hit = bool(meaning and meaning.lower() in text_lower)
        alt_hit = bool(alt and alt.lower() in text_lower)
        # Match acronym in uppercase form only to avoid false positives with common words.
        abbr_hit = re.search(abbr_pat, text) is not None
        if abbr_hit or meaning_hit or alt_hit:
            if normalized_abbr not in found:
                found.append(normalized_abbr)
        if len(found) >= max_items:
            break
    return found


def prepare_meeting_transcript_for_frozen_format(
    file_path,
    glossary_csv_path="glossary_abbrev_with_and_without_period.csv",
    use_openai=True,
    openai_api_key=None,
    openai_model="gpt-5.1",
    apply_acronyms_to_markdown=True,
    write_back_markdown=True,
):
    """
    Parse canonical meeting-transcript markdown and normalize it to frozen-doc style metadata.

    Returns a dict with:
      - title, source_filename, summary, participants, subjects, transcript
      - document_date, date_confidence
      - triage_legibility, triage_participant_identifiability, triage_subject_clarity, triage_total
      - topics (subjects routed to topics)
    """
    p = Path(file_path)
    content = p.read_text(encoding="utf-8", errors="ignore")
    sections = _extract_section_map(content)
    frontmatter: Dict[str, Any] = {}
    fm_match = re.match(r"^\s*---\s*\n(.*?)\n---\s*(?:\n|$)", content, flags=re.DOTALL)
    if fm_match:
        try:
            parsed_fm = yaml.safe_load(fm_match.group(1)) or {}
            if isinstance(parsed_fm, dict):
                frontmatter = parsed_fm
        except Exception:
            frontmatter = {}

    summary = _select_best_summary(sections.get("Summary", []))
    participants = (sections.get("Participants (inferred)", [""]) or [""])[0].strip()
    participants_list = _normalize_participants_list(participants)
    subjects = (sections.get("Subjects (inferred)", [""]) or [""])[0].strip()
    transcript = (sections.get("Transcript", [""]) or [""])[0].strip()

    triage_text = (sections.get("Triage Scores", [""]) or [""])[0]
    legibility = None
    participant_identifiability = None
    subject_clarity = None
    triage_total = None

    m = re.search(r"Legibility\s*:\s*(\d+)", triage_text, flags=re.IGNORECASE)
    if m:
        legibility = int(m.group(1))
    m = re.search(
        r"(?:Participant(?:\s+Identifiability|\s+Confidence|_confidence))\s*:\s*(\d+)",
        triage_text,
        flags=re.IGNORECASE,
    )
    if m:
        participant_identifiability = int(m.group(1))
    m = re.search(
        r"(?:Subject(?:\s+Clarity|\s+Confidence|_confidence))\s*:\s*(\d+)",
        triage_text,
        flags=re.IGNORECASE,
    )
    if m:
        subject_clarity = int(m.group(1))
    m = re.search(r"(?:Total\s+Score|total_score)\s*:\s*(\d+)", triage_text, flags=re.IGNORECASE)
    if m:
        triage_total = int(m.group(1))

    # Fallback for files where triage lives in YAML frontmatter.
    def _fm_int(*keys: str) -> Optional[int]:
        for k in keys:
            if k not in frontmatter:
                continue
            v = frontmatter.get(k)
            if v is None:
                continue
            try:
                return int(str(v).strip())
            except Exception:
                continue
        return None

    if legibility is None:
        legibility = _fm_int("legibility", "triage_legibility")
    if participant_identifiability is None:
        participant_identifiability = _fm_int(
            "participant_confidence",
            "triage_participant_identifiability",
        )
    if subject_clarity is None:
        subject_clarity = _fm_int("subject_confidence", "triage_subject_clarity")
    if triage_total is None:
        triage_total = _fm_int("total_score", "triage_total")

    filename_date_hint, date_confidence, _ = _date_hint_from_filename(p.name)

    document_date = None
    for date_block in sections.get("Date", []):
        first_line = (date_block.splitlines()[0] if date_block.splitlines() else "").strip()
        parsed = _parse_human_date(first_line)
        if parsed:
            document_date = parsed
            date_confidence = "medium"
            break
    if not document_date:
        document_date = filename_date_hint

    glossary_rows = load_abbreviation_glossary(glossary_csv_path)
    searchable_text = "\n".join([summary, participants, subjects, transcript[:12000]])
    _ = _find_matching_acronyms(searchable_text, glossary_rows)

    # Subjects are routed into Topics for Notion compatibility.
    topics = _split_subjects_to_topics(subjects)

    if use_openai:
        ai = _extract_meeting_metadata_with_openai(
            source_filename=p.name,
            summary=summary,
            participants=participants,
            subjects=subjects,
            transcript=transcript,
            openai_api_key=openai_api_key,
            model=openai_model,
        )
        ai_summary = str(ai.get("summary") or "").strip()
        if ai_summary:
            summary = ai_summary
        ai_participants = ai.get("participants")
        if isinstance(ai_participants, list):
            ai_participants = [str(x).strip() for x in ai_participants if str(x).strip()]
            if ai_participants:
                participants_list = ai_participants[:20]
                participants = ", ".join(ai_participants)
        ai_topics = ai.get("topics")
        if isinstance(ai_topics, list):
            ai_topics = [str(x).strip() for x in ai_topics if str(x).strip()]
            for t in ai_topics:
                if t.lower() not in {x.lower() for x in topics}:
                    topics.append(t)
            topics = topics[:20]
        ai_date = _parse_human_date(str(ai.get("document_date") or ""))
        if ai_date:
            document_date = ai_date
        ai_conf = str(ai.get("date_confidence") or "").strip().lower()
        if ai_conf in {"high", "medium", "low"}:
            date_confidence = ai_conf

    # Ensure participants is always populated when possible.
    if not (participants or "").strip():
        inferred = _infer_participants_from_text(summary, transcript)
        if inferred:
            participants_list = inferred[:20]
            participants = ", ".join(inferred)
    elif not participants_list:
        participants_list = _normalize_participants_list(participants)

    if apply_acronyms_to_markdown:
        expanded_transcript, acronym_expansions_applied = _expand_transcript_acronyms_conservative(transcript, glossary_rows)
        transcript = expanded_transcript
        if write_back_markdown and acronym_expansions_applied:
            updated_content = _replace_markdown_section(content, "Transcript", transcript)
            p.write_text(updated_content, encoding="utf-8")

    return {
        "title": p.stem,
        "source_filename": p.name,
        "summary": summary,
        "participants": participants,
        "participants_list": participants_list,
        "subjects": subjects,
        "topics": topics,
        "transcript": transcript,
        "document_date": document_date,
        "date_confidence": date_confidence,
        "triage_legibility": legibility,
        "triage_participant_identifiability": participant_identifiability,
        "triage_subject_clarity": subject_clarity,
        "triage_total": triage_total,
    }


def build_frozen_document_properties_from_meeting(prepared):
    """
    Build Notion properties payload for Frozen Documents style DB.
    Assumes:
      - Participants is rich_text
      - Topics is rich_text
      - Document Date is date
      - Date Confidence is rich_text
      - Triage fields are number
    """
    topics = prepared.get("topics") or []
    topics = [str(t).strip() for t in topics if str(t).strip()]

    props = {
        "Title": {
            "title": [{"text": {"content": str(prepared.get("title") or "Untitled")[:2000]}}]
        },
        "Source Filename": {
            "rich_text": [{"text": {"content": str(prepared.get("source_filename") or "")[:2000]}}]
        },
        "Summary": {
            "rich_text": [{"text": {"content": str(prepared.get("summary") or "")[:2000]}}]
        },
        "Participants": {
            "rich_text": [{"text": {"content": str(prepared.get("participants") or "")[:2000]}}]
        },
        "Topics": {
            "rich_text": [{"text": {"content": ", ".join(topics)[:2000]}}]
        },
        "Date Confidence": {
            "rich_text": [{"text": {"content": str(prepared.get("date_confidence") or "low")[:2000]}}]
        },
        "Triage Legibility": {
            "number": prepared.get("triage_legibility")
        },
        "Triage Participant Identifiability": {
            "number": prepared.get("triage_participant_identifiability")
        },
        "Triage Subject Clarity": {
            "number": prepared.get("triage_subject_clarity")
        },
        "Triage Total": {
            "number": prepared.get("triage_total")
        },
    }
    if prepared.get("document_date"):
        props["Document Date"] = {"date": {"start": prepared["document_date"]}}
    else:
        props["Document Date"] = {"date": None}
    return props


def parse_markdown(file_path):
    """
    Splits YAML frontmatter from body.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.startswith("---"):
        raise ValueError(f"No frontmatter found in {file_path}")

    _, frontmatter, body = content.split("---", 2)
    metadata = yaml.safe_load(frontmatter)

    return metadata, body.strip()


def markdown_to_notion_blocks(markdown_text):
    """
    Convert markdown text to Notion blocks, preserving structure.
    This makes the content fully searchable in Notion while maintaining formatting.
    """
    blocks = []
    lines = markdown_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip empty lines (but we'll add them as spacing between blocks)
        if not line.strip():
            i += 1
            continue
        
        # Parse headings (## Page X, ###, etc.)
        if line.startswith('## '):
            # Heading 2 (Page headers)
            content = line[3:].strip()
            if content:
                blocks.append({
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": content}}]}
                })
        elif line.startswith('### '):
            # Heading 3
            content = line[4:].strip()
            if content:
                blocks.append({
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"text": {"content": content}}]}
                })
        elif line.startswith('# '):
            # Heading 1
            content = line[2:].strip()
            if content:
                blocks.append({
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"text": {"content": content}}]}
                })
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet list item
            content = line[2:].strip()
            if content:
                blocks.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": content}}]}
                })
        elif line.startswith('1. ') or (line[0].isdigit() and '. ' in line[:5]):
            # Numbered list item
            # Extract number and content
            parts = line.split('. ', 1)
            if len(parts) == 2:
                content = parts[1].strip()
                if content:
                    blocks.append({
                        "type": "numbered_list_item",
                        "numbered_list_item": {"rich_text": [{"text": {"content": content}}]}
                    })
        else:
            # Regular paragraph
            # Collect consecutive non-empty lines into a paragraph
            paragraph_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith('#'):
                next_line = lines[i].rstrip()
                # Stop if we hit a list item
                if next_line.startswith('- ') or next_line.startswith('* ') or (next_line[0].isdigit() and '. ' in next_line[:5]):
                    break
                paragraph_lines.append(next_line)
                i += 1
            
            # Join lines with newlines, but limit to 2000 chars per block
            paragraph_text = '\n'.join(paragraph_lines)
            if len(paragraph_text) > 2000:
                # Split long paragraphs
                for chunk in chunk_text(paragraph_text, 2000):
                    blocks.append({
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": chunk}}]}
                    })
            else:
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": paragraph_text}}]}
                })
            continue  # Skip the i += 1 below since we already advanced
        
        i += 1
    
    return blocks


# ---------- Notion API ----------

def test_connection():
    """Test if the Notion API key and database connection are working."""
    print("=" * 60)
    print("Testing Notion API Connection...")
    print("=" * 60)
    
    # Test 1: Check API key by trying to access the database
    print(f"\n1. Testing API key and database access...")
    print(f"   Database ID: {DATABASE_ID}")
    print(f"   API Key: {NOTION_API_KEY[:20]}...")
    
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}"
    try:
        res = requests.get(url, headers=HEADERS)
        
        if res.status_code == 200:
            print("   [OK] API key is valid and database is accessible!")
            db_info = res.json()
            db_title = db_info.get("title", [{}])[0].get("plain_text", "Unknown")
            print(f"   [OK] Database name: {db_title}")
            return True
        elif res.status_code == 401:
            print("   [ERROR] Unauthorized - API key is invalid or expired")
            try:
                error = res.json()
                print(f"   Error details: {error}")
            except:
                print(f"   Error response: {res.text}")
            return False
        elif res.status_code == 404:
            print("   [ERROR] Database not found - Database ID might be incorrect")
            print("   Make sure:")
            print("   - The database ID is correct")
            print("   - The integration has access to this database")
            try:
                error = res.json()
                print(f"   Error details: {error}")
            except:
                print(f"   Error response: {res.text}")
            return False
        elif res.status_code == 403:
            print("   [ERROR] Forbidden - Integration doesn't have access to this database")
            print("   Make sure:")
            print("   - The integration is connected to the database")
            print("   - The integration has the correct permissions")
            try:
                error = res.json()
                print(f"   Error details: {error}")
            except:
                print(f"   Error response: {res.text}")
            return False
        else:
            print(f"   [ERROR] Unexpected status code {res.status_code}")
            try:
                error = res.json()
                print(f"   Error details: {error}")
            except:
                print(f"   Error response: {res.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Connection failed - {str(e)}")
        print("   Check your internet connection")
        return False
    except Exception as e:
        print(f"   [ERROR] Unexpected error - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_database_schema():
    """Retrieve and display the database schema to verify property names."""
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}"
    res = requests.get(url, headers=HEADERS)
    
    if res.status_code == 200:
        db_info = res.json()
        print("\nDatabase properties:")
        for prop_name, prop_info in db_info.get("properties", {}).items():
            prop_type = prop_info.get("type", "unknown")
            print(f"  - {prop_name} ({prop_type})")
        print()  # Add blank line
        return db_info.get("properties", {})
    else:
        print(f"[WARNING] Could not retrieve database schema: {res.status_code}")
        try:
            print(f"Error: {res.json()}")
        except:
            print(f"Error: {res.text}")
        return {}


def create_page_with_content(filename, summary, source_pdf=None, content_blocks=None, original_file_content=None):
    """
    Create a page in the database with all content included.
    Uses filename as the title.
    """
    # Use the markdown filename as the Notion Title (exactly)
    title = filename
    
    # Validate and sanitize inputs
    if not title or not isinstance(title, str):
        title = str(title) if title else "Untitled"
    if not filename or not isinstance(filename, str):
        filename = str(filename) if filename else "unknown"
    if not summary or not isinstance(summary, str):
        summary = str(summary) if summary else ""
    
    # Truncate to Notion limits (2000 chars for rich_text)
    title = title[:2000] if len(title) > 2000 else title
    filename = filename[:2000] if len(filename) > 2000 else filename
    summary = summary[:2000] if len(summary) > 2000 else summary
    if source_pdf:
        source_pdf = str(source_pdf)[:2000] if len(str(source_pdf)) > 2000 else str(source_pdf)
    
    # Build all blocks to add during page creation
    all_blocks = []
    
    # Summary section
    all_blocks.extend([
        {
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}
        },
        {
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": summary}}]}
        },
        {
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Full Transcript"}}]}
        },
    ])
    
    # Add content blocks if provided
    if content_blocks:
        all_blocks.extend(content_blocks)
    
    # Add original markdown file if provided
    if original_file_content:
        all_blocks.extend([
            {
                "type": "divider",
                "divider": {}
            },
            {
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Original Markdown File"}}]}
            },
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": "Below is the complete original markdown file for reference:"}}]}
            }
        ])
        
        # Add code blocks for original file (chunked if needed)
        for chunk in chunk_text(original_file_content, 2000):
            all_blocks.append({
                "type": "code",
                "code": {
                    "rich_text": [{"text": {"content": chunk}}],
                    "language": "markdown"
                }
            })
    
    # Create page with children (Notion allows up to 100 children in initial creation)
    MAX_INITIAL_CHILDREN = 100
    initial_blocks = all_blocks[:MAX_INITIAL_CHILDREN]
    remaining_blocks = all_blocks[MAX_INITIAL_CHILDREN:]
    
    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Title": {
                "title": [{"text": {"content": title}}]
            },
            "Filename": {
                "rich_text": [{"text": {"content": filename}}]
            },
            "Summary": {
                "rich_text": [{"text": {"content": summary}}]
            }
        },
        "children": initial_blocks  # Add initial blocks during creation
    }

    if source_pdf:
        payload["properties"]["Source File"] = {
            "rich_text": [{"text": {"content": source_pdf}}]
        }

    res = requests.post(
        "https://api.notion.com/v1/pages",
        headers=HEADERS,
        json=payload
    )
    
    # Better error handling (201 is also valid for POST requests)
    if res.status_code not in [200, 201]:
        try:
            error_data = res.json()
            print(f"Error response from Notion API (create_page): {error_data}")
        except:
            print(f"Error response (non-JSON): {res.text}")
        res.raise_for_status()
    
    page_data = res.json()
    page_id = page_data["id"]
    print(f"   Created page with ID: {page_id} (title: {title})")
    print(f"   Added {len(initial_blocks)} blocks during creation")
    
    # If there are remaining blocks, append them
    if remaining_blocks:
        print(f"   Appending {len(remaining_blocks)} additional blocks...")
        return page_id, remaining_blocks
    else:
        return page_id, []


def get_page(page_id):
    """Retrieve a page to verify it exists and is accessible."""
    page_id = str(page_id).strip().replace(" ", "")
    url = f"https://api.notion.com/v1/pages/{page_id}"
    res = requests.get(url, headers=HEADERS)
    if res.status_code == 200:
        return res.json()
    else:
        print(f"   [WARNING] Could not retrieve page: {res.status_code}")
        try:
            print(f"   Error: {res.json()}")
        except:
            print(f"   Error: {res.text}")
        return None


def append_remaining_blocks(page_id, blocks):
    """Append remaining blocks to a page that was created with initial content."""
    page_id = str(page_id).strip().replace(" ", "")
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    MAX_BLOCKS_PER_REQUEST = 100
    
    # Append in batches
    batch = []
    block_count = 0
    
    for block in blocks:
        batch.append(block)
        
        if len(batch) >= MAX_BLOCKS_PER_REQUEST:
            print(f"   Appending batch of {len(batch)} blocks...")
            # Notion "append block children" uses PATCH on this endpoint
            res = requests.patch(url, headers=HEADERS, json={"children": batch})
            time.sleep(0.3)
            
            if res.status_code != 200:
                try:
                    error_data = res.json()
                    print(f"Error response from Notion API (append_remaining): {error_data}")
                except:
                    print(f"Error response (non-JSON): {res.text}")
                res.raise_for_status()
            
            block_count += len(batch)
            batch = []
    
    # Append any remaining blocks
    if batch:
        print(f"   Appending final batch of {len(batch)} blocks...")
        res = requests.patch(url, headers=HEADERS, json={"children": batch})
        
        if res.status_code != 200:
            try:
                error_data = res.json()
                print(f"Error response from Notion API (append_remaining - final): {error_data}")
            except:
                print(f"Error response (non-JSON): {res.text}")
            res.raise_for_status()
        
        block_count += len(batch)
    
    print(f"   [OK] Appended {block_count} additional blocks")


def append_content(page_id, summary, full_text, original_file_content=None):
    # Ensure page_id is a string and properly formatted
    page_id = str(page_id).strip().replace(" ", "")
    
    # Verify page exists first
    print(f"   Verifying page {page_id} exists...")
    page_data = get_page(page_id)
    if not page_data:
        print(f"   [ERROR] Page {page_id} not found or not accessible")
        raise ValueError(f"Page {page_id} not accessible")
    
    # Notion page IDs should already be in UUID format with dashes
    # Just ensure it's clean and use it directly
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    
    # Notion API limit: max 100 blocks per request
    MAX_BLOCKS_PER_REQUEST = 100
    
    # Build initial blocks (summary section)
    initial_blocks = [
        {
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}
        },
        {
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": summary[:2000]}}]}  # Truncate to 2000 chars
        },
        {
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Full Transcript"}}]}
        },
    ]
    
    # Append initial blocks first
    print(f"   Appending summary section to page {page_id}...")
    print(f"   Using URL: {url}")
    
    # Try POST first (correct method for appending children)
    res = requests.patch(url, headers=HEADERS, json={"children": initial_blocks})
    
    if res.status_code not in [200, 201]:
        # If POST fails, try to get more details
        try:
            error_data = res.json()
            error_code = error_data.get("code", "unknown")
            error_msg = error_data.get("message", "Unknown error")
            print(f"Error response from Notion API (append_content - initial):")
            print(f"   Code: {error_code}")
            print(f"   Message: {error_msg}")
            print(f"   Full response: {error_data}")
            print(f"   URL used: {url}")
            print(f"   Page ID format: {page_id} (length: {len(page_id)})")
            
            # If it's an invalid URL error, the page might not be ready yet
            if error_code == "invalid_request_url":
                print(f"   [WARNING] Invalid URL error - page might not be ready. Waiting 2 seconds and retrying...")
                time.sleep(2)
                res = requests.patch(url, headers=HEADERS, json={"children": initial_blocks})
                if res.status_code != 200:
                    res.raise_for_status()
                else:
                    print(f"   [OK] Retry successful!")
        except Exception as e:
            print(f"Error response (non-JSON): {res.text}")
            print(f"Exception while handling error: {e}")
            res.raise_for_status()
    
    # Convert markdown to Notion blocks (preserves structure for better searchability)
    print(f"   Converting markdown to Notion blocks (preserving structure)...")
    markdown_blocks = markdown_to_notion_blocks(full_text)
    print(f"   Created {len(markdown_blocks)} blocks from markdown")
    
    # Append markdown blocks in batches
    block_count = len(initial_blocks)
    batch = []
    
    for block in markdown_blocks:
        batch.append(block)
        
        # Send batch when we reach the limit
        if len(batch) >= MAX_BLOCKS_PER_REQUEST:
            print(f"   Appending batch of {len(batch)} blocks ({block_count} total so far)...")
            res = requests.patch(url, headers=HEADERS, json={"children": batch})
            # Small delay between batches to avoid rate limiting
            time.sleep(0.3)
            
            if res.status_code != 200:
                try:
                    error_data = res.json()
                    print(f"Error response from Notion API (append_content - batch): {error_data}")
                    print(f"   URL used: {url}")
                except:
                    print(f"Error response (non-JSON): {res.text}")
                res.raise_for_status()
            
            block_count += len(batch)
            batch = []  # Reset for next batch
    
    # Append any remaining blocks
    if batch:
        print(f"   Appending final batch of {len(batch)} blocks ({block_count} total)...")
        res = requests.patch(url, headers=HEADERS, json={"children": batch})
        
        if res.status_code != 200:
            try:
                error_data = res.json()
                print(f"Error response from Notion API (append_content - final): {error_data}")
                print(f"   URL used: {url}")
            except:
                print(f"Error response (non-JSON): {res.text}")
            res.raise_for_status()
        
        block_count += len(batch)
    
    # Add original markdown file as a code block at the end (for reference)
    if original_file_content:
        print(f"   Adding original markdown file as code block...")
        # Split into chunks if needed (code blocks have 2000 char limit per block)
        original_chunks = list(chunk_text(original_file_content, 2000))
        
        # Add separator and heading
        footer_blocks = [
            {
                "type": "divider",
                "divider": {}
            },
            {
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Original Markdown File"}}]}
            },
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": "Below is the complete original markdown file for reference:"}}]}
            }
        ]
        
        # Add code blocks for each chunk
        for i, chunk in enumerate(original_chunks):
            footer_blocks.append({
                "type": "code",
                "code": {
                    "rich_text": [{"text": {"content": chunk}}],
                    "language": "markdown"
                }
            })
        
        # Append footer blocks in batches if needed
        footer_batch = []
        for block in footer_blocks:
            footer_batch.append(block)
            if len(footer_batch) >= MAX_BLOCKS_PER_REQUEST:
                res = requests.patch(url, headers=HEADERS, json={"children": footer_batch})
                if res.status_code != 200:
                    try:
                        error_data = res.json()
                        print(f"Error response from Notion API (append_content - footer): {error_data}")
                    except:
                        print(f"Error response (non-JSON): {res.text}")
                    res.raise_for_status()
                block_count += len(footer_batch)
                footer_batch = []
                time.sleep(0.3)
        
        # Append any remaining footer blocks
        if footer_batch:
            res = requests.patch(url, headers=HEADERS, json={"children": footer_batch})
            if res.status_code != 200:
                try:
                    error_data = res.json()
                    print(f"Error response from Notion API (append_content - footer final): {error_data}")
                except:
                    print(f"Error response (non-JSON): {res.text}")
                res.raise_for_status()
            block_count += len(footer_batch)
    
    print(f"   [OK] Successfully appended all content ({block_count} blocks total)")


# ---------- main loop ----------

def run():
    # Test connection first
    if not test_connection():
        print("\n" + "=" * 60)
        print("Connection test failed! Please fix the issues above before proceeding.")
        print("=" * 60)
        return
    
    # Check database schema to verify property names
    print("\n2. Retrieving database schema...")
    schema = get_database_schema()
    if schema:
        print("   [OK] Database schema retrieved\n")
    else:
        print("   [WARNING] Could not retrieve schema, but continuing anyway...\n")
    
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if not filename.endswith("_transcribed.md"):
            continue

        path = os.path.join(TRANSCRIPTS_DIR, filename)
        print(f"Uploading {filename}...")

        try:
            metadata, body = parse_markdown(path)
            
            # Read the full original file content (including frontmatter) for the code block
            with open(path, "r", encoding="utf-8") as f:
                original_file_content = f.read()

            source_pdf = metadata.get("source_pdf")
            summary = metadata.get("summary", "")
            
            # Handle summary - it might be None, empty, or a multiline string
            if summary is None:
                summary = ""
            elif isinstance(summary, str):
                summary = summary.strip()
            else:
                summary = str(summary).strip()

            # Convert markdown to Notion blocks
            print(f"   Converting markdown to Notion blocks...")
            markdown_blocks = markdown_to_notion_blocks(body)
            print(f"   Created {len(markdown_blocks)} blocks from markdown")
            
            # Create page with all content
            result = create_page_with_content(
                filename=filename,
                summary=summary,
                source_pdf=source_pdf,
                content_blocks=markdown_blocks,
                original_file_content=original_file_content
            )
            
            page_id, remaining_blocks = result
            
            # Append any remaining blocks if needed
            if remaining_blocks:
                append_remaining_blocks(page_id, remaining_blocks)

            print(f"[OK] Done: {filename}\n")
        except Exception as e:
            print(f"[ERROR] Error processing {filename}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    import sys
    
    # If --test flag is passed, just run the connection test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_connection()
        print("\n" + "=" * 60)
    else:
        run()
