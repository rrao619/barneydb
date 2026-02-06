import os
import io
import base64
import pandas as pd
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import fitz  # PyMuPDF
from PIL import Image
import multiprocessing
from multiprocessing import Pool
import traceback


def _openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment (or in a 'keys' file) to run OCR.")
    return OpenAI(api_key=key)


client = _openai_client()

# ---------------------------------------------------------
# LOAD ABBREVIATION GLOSSARY
# ---------------------------------------------------------
def load_glossary(csv_path="glossary_abbrev.csv"):
    csv_path = Path(csv_path)
    df = pd.read_csv(str(csv_path))
    glossary = []

    for _, row in df.iterrows():
        abbr = str(row["Abbreviation"]).strip()
        meaning = str(row["Meaning (to fill in)"]).strip()
        alt = str(row["Alternate meaning"]).strip()

        glossary.append((abbr, meaning, alt))

    return glossary


GLOSSARY = load_glossary()


def glossary_instructions():
    """Generate an instruction block for abbreviation expansion."""
    lines = ["Expand abbreviations using the following glossary:\n"]
    for abbr, meaning, alt in GLOSSARY:
        if alt and alt.lower() != "nan":
            lines.append(f"- '{abbr}' → '{meaning}' (or if context matches: '{alt}')")
        else:
            lines.append(f"- '{abbr}' → '{meaning}'")
    return "\n".join(lines)


# ---------------------------------------------------------
# GPT MODELS
# ---------------------------------------------------------
OCR_MODEL = "gpt-5.1"  # Vision-capable model
SUMMARY_MODEL = "gpt-5.1"


# ---------------------------------------------------------
# TRANSCRIBE MULTIPLE PAGES (BATCHED)
# ---------------------------------------------------------
def transcribe_pages(image_bytes_list, page_numbers, api_client=None):
    """
    Transcribe multiple pages in a single API call.
    This is faster than one page at a time.
    """
    if api_client is None:
        api_client = client
    
    # Convert all images to base64
    image_contents = []
    for idx, img_bytes in enumerate(image_bytes_list):
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        })
    
    page_range = f"{page_numbers[0]}-{page_numbers[-1]}" if len(page_numbers) > 1 else str(page_numbers[0])
    
    prompt = f"""
Transcribe the content of pages {page_range} with high accuracy.

IMPORTANT: For each page, start with "## Page X" (where X is the page number) before the transcription.

Rules:
- SUBSTITUTE ABBREVIATIONS: When you encounter abbreviations in the text, replace them with their full meanings from the glossary below. This is critical - do not leave abbreviations as-is.
- Choose alternate meaning if context fits better than the primary meaning.
- Correct OCR errors.
- Do NOT hallucinate material.
- Preserve structure and formatting.
- Clearly separate each page with "## Page X" headers.

ABBREVIATION GLOSSARY (use these to substitute abbreviations in the transcription):
{glossary_instructions()}

Remember: Actively look for and replace abbreviations with their full meanings throughout the transcription.
"""

    # Build content array: text prompt first, then all images
    content = [{"type": "text", "text": prompt}] + image_contents

    response = api_client.chat.completions.create(
        model=OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# SUMMARY GENERATION
# ---------------------------------------------------------
def generate_summary(full_text, api_client=None):
    if api_client is None:
        api_client = client
        
    prompt = f"""
Provide a concise summary (3–6 bullets) of this transcribed document.

Document:
{full_text}
"""

    response = api_client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# PROCESS ANY PDF (with error handling)
# ---------------------------------------------------------
def process_pdf(pdf_path):
    """
    Process a single PDF file. Returns (success, pdf_path, error_message, file_size)
    """
    pdf_path = Path(pdf_path)
    file_size = pdf_path.stat().st_size if pdf_path.exists() else 0
    
    try:
        # Check if markdown already exists
        out_path = pdf_path.with_stem(pdf_path.stem + "_transcribed").with_suffix(".md")
        if out_path.exists():
            print(f"\n[{os.getpid()}] Skipping {pdf_path.name} (markdown already exists)")
            return (True, str(pdf_path), None, file_size)
        
        print(f"\n[{os.getpid()}] Processing {pdf_path.name} ({file_size / 1024 / 1024:.2f} MB)...")

        # Create client for this process (thread-safe)
        process_client = _openai_client()

        # Open PDF with PyMuPDF (needs string path)
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        
        # Convert all pages to images first
        print(f"[{os.getpid()}]   Converting {num_pages} pages to images...")
        page_images = []
        for i in range(num_pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))  # 200 DPI
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            page_images.append(buffer.getvalue())
        
        doc.close()
        
        # Batch pages (max 20 per request to avoid API limits)
        MAX_PAGES_PER_BATCH = 20
        transcribed_pages = []
        
        for batch_start in range(0, num_pages, MAX_PAGES_PER_BATCH):
            batch_end = min(batch_start + MAX_PAGES_PER_BATCH, num_pages)
            batch_images = page_images[batch_start:batch_end]
            batch_page_nums = list(range(batch_start + 1, batch_end + 1))
            
            print(f"[{os.getpid()}]   Transcribing pages {batch_start + 1}-{batch_end} ({len(batch_images)} pages)...")
            batch_transcription = transcribe_pages(batch_images, batch_page_nums, process_client)
            
            # Parse response by splitting on "## Page" markers
            parts = batch_transcription.split("## Page ")
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                # Extract page number from first line
                lines = part.strip().split('\n', 1)
                if lines:
                    first_line = lines[0].strip()
                    # Try to extract page number (handle "Page X" or just "X")
                    try:
                        page_num = int(first_line.split()[0]) if first_line.split() else batch_page_nums[i] if i < len(batch_page_nums) else batch_start + i + 1
                    except:
                        page_num = batch_page_nums[i] if i < len(batch_page_nums) else batch_start + i + 1
                    
                    text = lines[1] if len(lines) > 1 else part.strip()
                    if text:
                        transcribed_pages.append((page_num, text))

        # Combine text
        full_text = "\n\n".join(f"## Page {p}\n{text}" for p, text in transcribed_pages)

        # Summary
        summary = generate_summary(full_text, process_client)

        # Output Markdown
        now = datetime.now().strftime("%Y-%m-%d")
        # out_path already defined at function start

        yaml_header = f"""---
title: "Transcription of {pdf_path.name}"
source_pdf: "{pdf_path.name}"
date_processed: "{now}"
pages: {num_pages}
tags: ["ocr", "transcription"]
glossary_used: true
summary: |
  {summary.replace('\n', '\n  ')}
---
"""

        with open(str(out_path), "w", encoding="utf-8") as f:
            f.write(yaml_header + "\n" + full_text)

        print(f"[{os.getpid()}]   ✓ Saved: {out_path.name}")
        return (True, str(pdf_path), None, file_size)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"[{os.getpid()}]   ✗ Error processing {pdf_path.name}: {str(e)}")
        return (False, str(pdf_path), error_msg, file_size)


# ---------------------------------------------------------
# FOLDER PROCESSING WITH MULTIPROCESSING
# ---------------------------------------------------------
def process_folder(folder_path, num_workers=None):
    folder_path = Path(folder_path)
    pdfs = list(folder_path.glob("*.pdf"))
    
    if not pdfs:
        print("No PDF files found.")
        return []
    
    # Sort by file size (largest first)
    pdfs_with_size = [(pdf, pdf.stat().st_size) for pdf in pdfs]
    pdfs_with_size.sort(key=lambda x: x[1], reverse=True)
    pdfs_sorted = [pdf for pdf, _ in pdfs_with_size]
    
    print(f"Found {len(pdfs_sorted)} PDF(s)")
    print(f"Processing order (largest to smallest):")
    for i, pdf in enumerate(pdfs_sorted[:10], 1):  # Show first 10
        size_mb = pdf.stat().st_size / 1024 / 1024
        print(f"  {i}. {pdf.name} ({size_mb:.2f} MB)")
    if len(pdfs_sorted) > 10:
        print(f"  ... and {len(pdfs_sorted) - 10} more")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(pdfs_sorted), 8)  # Max 8 workers
    
    print(f"\nProcessing with {num_workers} worker(s)...")
    
    # Process files in parallel
    failed_files = []
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_pdf, pdfs_sorted)
    
    # Collect failures
    for success, pdf_path, error_msg, file_size in results:
        if not success:
            failed_files.append({
                "file_name": Path(pdf_path).name,
                "file_path": str(pdf_path),
                "file_size_mb": file_size / 1024 / 1024 if file_size else 0,
                "error": error_msg
            })
    
    return failed_files


# ---------------------------------------------------------
# CONVERT GLOSSARY TO MARKDOWN
# ---------------------------------------------------------
def glossary_to_markdown(csv_path="glossary_abbrev.csv", out_path="glossary.md"):
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = pd.read_csv(str(csv_path))

    md = ["# Abbreviation Glossary\n"]

    for _, row in df.iterrows():
        abbr = row["Abbreviation"]
        meaning = str(row["Meaning (to fill in)"])
        alt = str(row["Alternate meaning"])


        md.append(f"## {abbr}")
        md.append(f"- **Meaning:** {meaning}")

        if isinstance(alt, str) and alt.strip() != "":
            md.append(f"- **Alternate Meaning:** {alt}")

        md.append("")  # blank line

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Glossary saved to {out_path}")


# ---------------------------------------------------------
# WRITE FAILURES TO EXCEL
# ---------------------------------------------------------
def write_failures_to_excel(failed_files, output_path="transcription_failures.xlsx"):
    """Write failed transcriptions to an Excel file."""
    if not failed_files:
        print("\n✓ No transcription failures to report!")
        return
    
    output_path = Path(output_path)
    df = pd.DataFrame(failed_files)
    
    # Reorder columns
    df = df[["file_name", "file_path", "file_size_mb", "error"]]
    
    # Write to Excel
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Failed Transcriptions', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Failed Transcriptions']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
    
    print(f"\n✗ Found {len(failed_files)} transcription failure(s)")
    print(f"  Details saved to: {output_path}")


# ---------------------------------------------------------
# RUN ALL
# ---------------------------------------------------------
if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    
    glossary_to_markdown()          # Convert glossary CSV → markdown
    
    folder_path = Path(r"C:\Users\rohit\Desktop\Barneydb\Candidate_A\Candidate_2_")
    failed_files = process_folder(folder_path)
    
    # Write failures to Excel
    write_failures_to_excel(failed_files)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
