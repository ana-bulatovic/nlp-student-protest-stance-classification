#!/usr/bin/env python3
"""Merge annotated text|url|label batch files into one combined file,
then split the result by class (matching the ig_/x_ final_<class>_annotated.txt
naming convention used by the other platforms). Source batch files are never
modified or deleted.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")


def load_rows(input_dir: Path) -> list[list[str]]:
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nema .txt fajlova u {input_dir}")

    rows: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0

    for path in txt_files:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            for row in reader:
                text = (row.get("text") or "").strip()
                url = (row.get("url") or "").strip()
                label = (row.get("label") or "").strip()
                if not text or label not in LABELS:
                    continue
                key = (text, url)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                rows.append([text, url, label])

    if duplicates:
        print(f"Preskoceno {duplicates} duplikata (isti text+url u vise batch fajlova)")

    return rows


def write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerow(["text", "url", "label"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spoji anotirane batch fajlove (text|url|label) u jedan, pa podeli po klasi."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder sa anotiranim batch fajlovima (npr. annotated/youtube_batch)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Izlazni folder (default: parent od --input-dir)",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefiks za izlazne fajlove, npr. 'yt' -> yt_final_annotated.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir.parent

    try:
        rows = load_rows(args.input_dir)
    except FileNotFoundError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("Nema oznacenih redova za spajanje.", file=sys.stderr)
        return 1

    merged_path = output_dir / f"{args.prefix}_final_annotated.txt"
    write_rows(merged_path, rows)

    by_label: dict[str, list[list[str]]] = {label: [] for label in LABELS}
    for row in rows:
        by_label[row[2]].append(row)

    label_slug = {
        "NEUTRAL": "neutral",
        "ZA-VLAST": "za_vlast",
        "PROTIV-VLASTI": "protiv_vlasti",
    }
    for label in LABELS:
        slug = label_slug[label]
        out_path = output_dir / f"{args.prefix}_final_{slug}_annotated.txt"
        write_rows(out_path, by_label[label])

    counts = Counter(row[2] for row in rows)
    print(f"\nUkupno spojeno: {len(rows)} -> {merged_path}")
    for label in LABELS:
        print(f"  {label}: {counts.get(label, 0)} -> {args.prefix}_final_{label_slug[label]}_annotated.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
