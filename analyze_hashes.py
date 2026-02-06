import os
import hashlib
from pathlib import Path

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Analyze PDFs in frozen folder
frozen_dir = Path("frozen")
pdf_files = [f for f in frozen_dir.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']

print(f"Found {len(pdf_files)} PDF files")

# Calculate hashes and find duplicates
hashes = {}
for pdf in pdf_files:
    try:
        file_hash = sha256_file(pdf)
        if file_hash not in hashes:
            hashes[file_hash] = []
        hashes[file_hash].append(pdf.name)
    except Exception as e:
        print(f"Error hashing {pdf.name}: {e}")

# Analyze results
unique_hashes = len(hashes)
duplicate_groups = sum(1 for files in hashes.values() if len(files) > 1)
total_duplicates = sum(len(files) - 1 for files in hashes.values() if len(files) > 1)

print(f"Unique file hashes: {unique_hashes}")
print(f"Files with duplicates: {duplicate_groups}")
print(f"Total duplicate files: {total_duplicates}")
print(f"Expected database documents: {unique_hashes}")

# Show some examples of duplicates
print("\nDuplicate file examples:")
for hash_val, files in list(hashes.items())[:5]:  # Show first 5 groups
    if len(files) > 1:
        print(f"Hash {hash_val[:16]}...: {len(files)} files - {files[:3]}{'...' if len(files) > 3 else ''}")