#!/usr/bin/env python3
"""Assemble dataset_all.txt (Phase 3 training input) from every platform's
annotated files. Reads each input with proper CSV parsing (so embedded '|' or
'"' characters, however the source file happens to quote them, are handled
correctly), then writes plain unquoted 'text|url|label' lines — matching the
existing dataset_all.txt convention that common/data.py's simple line parser
expects (no header row, no CSV quoting).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANNOTATED_DIR = SCRIPT_DIR / "annotated"
DEFAULT_OUTPUT = ANNOTATED_DIR / "dataset_all.txt"
LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")

DEFAULT_INPUTS = [
    "ig_final_neutral_annotated.txt",
    "ig_final_za_vlast_annotated.txt",
    "ig_final_protiv_vlasti_annotated.txt",
    "x_final_neutral_annotated.txt",
    "x_final_za_vlast_annotated.txt",
    "x_final_protiv_vlasti_annotated.txt",
    "yt_final_annotated.txt",
]


def read_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            text, url, label = row[0].strip(), row[1].strip(), row[2].strip()
            if label not in LABELS or not text:
                continue  # skips header rows like "text|url|label" too
            rows.append((text, url, label))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spoji sve platform-anotirane fajlove u dataset_all.txt za Fazu 3."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Fajlovi (relativno na annotated/) za spajanje.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0

    for name in args.inputs:
        path = ANNOTATED_DIR / name
        if not path.is_file():
            print(f"Upozorenje: {path} ne postoji, preskacem.", file=sys.stderr)
            continue
        kept = 0
        for row in read_rows(path):
            key = (row[0], row[1])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            all_rows.append(row)
            kept += 1
        print(f"  {name}: {kept}")

    if not all_rows:
        print("Nema podataka za spajanje.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for text, url, label in all_rows:
            handle.write(f"{text}|{url}|{label}\n")

    if duplicates:
        print(f"Preskoceno {duplicates} duplikata (isti text+url)")

    counts = Counter(row[2] for row in all_rows)
    print(f"\nUkupno: {len(all_rows)} -> {args.output}")
    for label in LABELS:
        print(f"  {label}: {counts.get(label, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
