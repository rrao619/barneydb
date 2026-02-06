import os

frozen_dir = "frozen"
pdf_count = len([f for f in os.listdir(frozen_dir) if f.lower().endswith(".pdf")])
print(f"PDFs in frozen folder: {pdf_count}")

# Also count documents in database
import subprocess
result = subprocess.run(["python", "upload_frozen_to_notion.py", "--count-documents"], capture_output=True, text=True)
if result.returncode == 0:
    for line in result.stdout.split('\n'):
        if 'Total documents in database:' in line:
            db_count = int(line.split(':')[1].strip())
            print(f"Documents in database: {db_count}")
            print(f"Documents to process: {pdf_count - db_count}")
            break