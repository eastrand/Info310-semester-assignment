#!/usr/bin/env python3
import re
from pathlib import Path
from pdfminer.high_level import extract_text
from concurrent.futures import ProcessPoolExecutor, as_completed

def count_words_in_pdf(path: Path) -> tuple[str, int]:
    """Return (filename, word_count) for a given PDF file."""
    try:
        text = extract_text(str(path)) or ""
        words = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
        return (path.name, words)
    except Exception:
        return (path.name, 0)

def main():
    pdfs = sorted(Path(".").glob("*.pdf"))
    if not pdfs:
        print("No PDF files found in current directory.")
        return

    total = 0
    results = []

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(count_words_in_pdf, p): p for p in pdfs}
        for future in as_completed(futures):
            name, count = future.result()
            results.append((name, count))
            print(f"{count:8d}  {name}")

    total = sum(c for _, c in results)
    print(f"\nTotal words: {total}")

if __name__ == "__main__":
    main()
