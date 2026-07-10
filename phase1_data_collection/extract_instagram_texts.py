#!/usr/bin/env python3
"""Extract comment text from all Instagram output files into one combined file."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "output" / "instagram"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "instagram_all_texts.txt"


def extract_texts(input_dir: Path, dedupe: bool) -> list[str]:
    txt_files = sorted(input_dir.glob("instagram_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nema instagram_*.txt fajlova u {input_dir}")

    texts: list[str] = []
    seen_ids: set[str] = set()

    for path in txt_files:
        if path.name == DEFAULT_OUTPUT.name:
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                comment_id = (row.get("id") or "").strip()
                if dedupe and comment_id:
                    if comment_id in seen_ids:
                        continue
                    seen_ids.add(comment_id)
                texts.append(text)

    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Izdvoji samo tekst komentara iz output/instagram fajlova."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder sa instagram_*.txt fajlovima (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Izlazni fajl (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Ne uklanjaj duplikate (ako isti komentar postoji u vise export fajlova)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        texts = extract_texts(args.input_dir, dedupe=not args.no_dedupe)
    except FileNotFoundError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    if not texts:
        print("Nema komentara za izvoz.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(texts) + "\n", encoding="utf-8")

    print(f"Procitano iz: {args.input_dir}")
    print(f"Sacuvano {len(texts)} tekstova -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
