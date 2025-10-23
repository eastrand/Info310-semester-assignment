#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# deps: pip install pdfminer.six langdetect
from pdfminer.high_level import extract_text
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # deterministic

CHUNK_SIZE = 2000
MIN_TEXT_LEN = 200
EN_CODES = {"en", "en-us", "en-gb"}
MAX_WORKERS = 10

print_lock = Lock()


def majority_language(text: str):
    text = " ".join(text.split())
    if len(text) < MIN_TEXT_LEN:
        return None
    votes = {}
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i + CHUNK_SIZE]
        if len(chunk) < 25:
            continue
        try:
            lang = detect(chunk)
            votes[lang] = votes.get(lang, 0) + 1
        except LangDetectException:
            continue
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def is_english(lang_code: str | None) -> bool:
    if not lang_code:
        return False
    base = lang_code.split("-")[0].lower()
    return base == "en" or lang_code.lower() in EN_CODES


def process_pdf(pdf_path: Path):
    try:
        text = extract_text(str(pdf_path)) or ""
    except Exception as e:
        with print_lock:
            print(f"[SKIP] {pdf_path.name}: failed to extract text ({e})")
        return ("skip", False)

    if len(text.strip()) < MIN_TEXT_LEN:
        with print_lock:
            print(f"[SKIP] {pdf_path.name}: likely image-only (no text)")
        return ("skip", False)

    lang = majority_language(text)
    if lang and not is_english(lang):
        foreign_dir = (pdf_path.parent / ".." / "foreign").resolve()
        foreign_dir.mkdir(parents=True, exist_ok=True)
        dest = foreign_dir / pdf_path.name
        try:
            shutil.move(str(pdf_path), str(dest))
            with print_lock:
                print(f"[MOVED] {pdf_path.name} -> {dest} (lang={lang})")
            return ("moved", True)
        except Exception as e:
            with print_lock:
                print(f"[FAIL] {pdf_path.name}: could not move ({e})")
            return ("fail", False)
    else:
        with print_lock:
            print(f"[KEEP ] {pdf_path.name} (lang={lang or 'unknown'})")
        return ("keep", False)


def move_foreign_pdfs(folder: Path):
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(folder.glob("*.pdf"))
    moved = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_pdf, p): p for p in pdfs}
        for fut in as_completed(futures):
            _, did_move = fut.result()
            moved += 1 if did_move else 0

    print(f"\nDone. Checked: {len(pdfs)}, Moved (non-English): {moved}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python move_foreign_pdfs.py /path/to/folder", file=sys.stderr)
        sys.exit(2)
    move_foreign_pdfs(Path(sys.argv[1]))
