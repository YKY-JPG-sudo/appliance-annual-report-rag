from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "indexes"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PAGES_PATH = DATA_DIR / "pages.jsonl"
TABLES_PATH = DATA_DIR / "tables.jsonl"

