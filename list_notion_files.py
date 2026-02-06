import argparse
import csv
import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

# Database IDs
TRANSCRIPTS_DATABASE_ID = "2cb0ea6befa280e681bef5b87e65a4c0"
FROZEN_DATABASE_ID = "2ed0ea6befa28092b93fc3a425e51204"

NOTION_VERSION = "2022-06-28"


def read_notion_token_from_keys_file(keys_path: str = "keys") -> Optional[str]:
    """Read Notion integration token from keys file."""
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


def notion_headers(notion_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def notion_query_database(
    database_id: str, headers: Dict[str, str], payload: Dict[str, Any]
) -> Dict[str, Any]:
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()


def _get_rich_text_plain(page: Dict[str, Any], prop_name: str) -> Optional[str]:
    """Extract plain text from a rich_text property."""
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


def _get_title_plain(page: Dict[str, Any], prop_name: str = "Title") -> Optional[str]:
    """Extract plain text from a title property."""
    try:
        p = (page.get("properties", {}) or {}).get(prop_name)
        if not p:
            return None
        if p.get("type") != "title":
            return None
        parts = p.get("title") or []
        text = "".join((x.get("plain_text") or "") for x in parts).strip()
        return text or None
    except Exception:
        return None


def get_all_files_from_database(
    database_id: str, headers: Dict[str, str], category: str
) -> List[Dict[str, str]]:
    """Get all files from a Notion database."""
    files = []
    cursor = None

    while True:
        payload = {"page_size": 100}  # Maximum allowed by Notion API
        if cursor:
            payload["start_cursor"] = cursor

        try:
            data = notion_query_database(database_id, headers, payload)
            results = data.get("results", [])

            for page in results:
                # Try different property names for filename
                filename = None
                
                # For transcripts database (uses "Filename" property)
                if category == "transcript":
                    filename = _get_rich_text_plain(page, "Filename")
                    if not filename:
                        # Fallback to Title
                        filename = _get_title_plain(page, "Title")
                
                # For frozen documents database (uses "Source Filename" property)
                elif category == "frozen":
                    filename = _get_rich_text_plain(page, "Source Filename")
                    if not filename:
                        # Fallback to Title
                        filename = _get_title_plain(page, "Title")
                
                if filename:
                    files.append({
                        "filename": filename,
                        "category": category,
                        "page_id": page.get("id", "")
                    })

            # Check if there are more pages
            if not data.get("has_more", False):
                break

            cursor = data.get("next_cursor")

        except requests.HTTPError as e:
            print(f"Error querying {category} database: {e}")
            break

    return files


def main():
    parser = argparse.ArgumentParser(
        description="List all files from Notion databases (transcripts and frozen documents) and export to CSV."
    )
    parser.add_argument(
        "--notion-token",
        default=os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY") or read_notion_token_from_keys_file(),
        help="Notion integration token",
    )
    parser.add_argument(
        "--output",
        default="notion_files.csv",
        help="Output CSV file path (default: notion_files.csv)",
    )
    args = parser.parse_args()

    if not args.notion_token:
        raise SystemExit(
            "Missing --notion-token (or set NOTION_TOKEN / NOTION_API_KEY, or put it in `keys`)"
        )

    headers = notion_headers(args.notion_token)

    print("Fetching files from Notion databases...")
    print(f"  - Transcripts database: {TRANSCRIPTS_DATABASE_ID}")
    print(f"  - Frozen documents database: {FROZEN_DATABASE_ID}")

    # Get files from both databases
    all_files = []

    # Get transcripts
    print("\nFetching transcripts...")
    try:
        transcript_files = get_all_files_from_database(
            TRANSCRIPTS_DATABASE_ID, headers, "transcript"
        )
        print(f"  Found {len(transcript_files)} transcript files")
        all_files.extend(transcript_files)
    except Exception as e:
        print(f"  Error fetching transcripts: {e}")
        print("  Continuing with frozen documents only...")

    # Get frozen documents
    print("\nFetching frozen documents...")
    try:
        frozen_files = get_all_files_from_database(
            FROZEN_DATABASE_ID, headers, "frozen"
        )
        print(f"  Found {len(frozen_files)} frozen document files")
        all_files.extend(frozen_files)
    except Exception as e:
        print(f"  Error fetching frozen documents: {e}")

    # Write to CSV
    print(f"\nWriting {len(all_files)} files to {args.output}...")
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "category"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for file_info in all_files:
            writer.writerow({
                "filename": file_info["filename"],
                "category": file_info["category"],
            })

    print(f"\nDone! Exported {len(all_files)} files to {args.output}")
    print(f"  - Transcripts: {len([f for f in all_files if f['category'] == 'transcript'])}")
    print(f"  - Frozen documents: {len([f for f in all_files if f['category'] == 'frozen'])}")


if __name__ == "__main__":
    main()
