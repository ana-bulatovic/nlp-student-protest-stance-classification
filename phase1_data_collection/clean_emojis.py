#!/usr/bin/env python3
"""Remove emojis from comment texts; drop comments that are emoji-only."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "output" / "instagram" / "instagram_all_texts.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "instagram" / "instagram_all_texts_clean.txt"

# Common emoji blocks + modifiers (ZWJ, variation selector, skin tones, flags)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U00002300-\U000023FF"
    "\U0001F1E6-\U0001F1FF"
    "\U0001F3FB-\U0001F3FF"
    "\u200d"
    "\ufe0f"
    "]+",
    flags=re.UNICODE,
)


def remove_emojis(text: str) -> str:
    cleaned = EMOJI_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_emoji_only(text: str) -> bool:
    return not remove_emojis(text)


def clean_text_line(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if is_emoji_only(text):
        return None
    cleaned = remove_emojis(text)
    return cleaned or None


def clean_lines_file(input_path: Path) -> tuple[list[str], int, int]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed_emoji_only = 0
    removed_empty = 0

    for line in lines:
        original = line.strip()
        if not original:
            removed_empty += 1
            continue
        if is_emoji_only(original):
            removed_emoji_only += 1
            continue
        cleaned = remove_emojis(original)
        if cleaned:
            kept.append(cleaned)
        else:
            removed_emoji_only += 1

    return kept, removed_emoji_only, removed_empty


def clean_structured_file(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    rows_out: list[list[str]] = []
    removed_emoji_only = 0
    removed_empty = 0

    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        header = next(reader)
        try:
            text_index = header.index("text")
        except ValueError as exc:
            raise ValueError(f"Kolona 'text' nije pronadjena u {input_path}") from exc

        for row in reader:
            if len(row) <= text_index:
                removed_empty += 1
                continue
            original = (row[text_index] or "").strip()
            if not original:
                removed_empty += 1
                continue
            if is_emoji_only(original):
                removed_emoji_only += 1
                continue
            cleaned = remove_emojis(original)
            if not cleaned:
                removed_emoji_only += 1
                continue
            row[text_index] = cleaned
            rows_out.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows_out)

    return len(rows_out), removed_emoji_only, removed_empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ukloni emoji iz komentara; obrisi komentare koji su samo emoji."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Ulazni fajl (default: {DEFAULT_INPUT.name}, jedan komentar po liniji)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Izlazni fajl (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Ulaz je pipe-separated CSV sa kolonom 'text' (instagram_*.txt export)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Obradi sve instagram_*.txt export fajlove u folderu (--structured)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.input_dir:
        if not args.structured:
            print("Greska: --input-dir zahteva --structured.", file=sys.stderr)
            return 1
        txt_files = sorted(args.input_dir.glob("instagram_*.txt"))
        txt_files = [
            p
            for p in txt_files
            if p.name not in {DEFAULT_INPUT.name, DEFAULT_OUTPUT.name, "instagram_all_texts_clean.txt"}
            and "clean" not in p.stem
        ]
        if not txt_files:
            print(f"Greska: nema export fajlova u {args.input_dir}", file=sys.stderr)
            return 1

        total_kept = total_removed = 0
        for path in txt_files:
            out_path = path.with_name(path.stem + "_clean.txt")
            kept, removed, _empty = clean_structured_file(path, out_path)
            total_kept += kept
            total_removed += removed
            print(f"  {path.name}: sacuvano {kept}, obrisano {removed} -> {out_path.name}")
        print(f"\nUkupno sacuvano: {total_kept}, obrisano (samo emoji): {total_removed}")
        return 0

    if not args.input.exists():
        print(f"Greska: fajl ne postoji: {args.input}", file=sys.stderr)
        return 1

    if args.structured:
        kept, removed, empty = clean_structured_file(args.input, args.output)
        print(f"Procitano iz: {args.input}")
        print(f"Sacuvano {kept} komentara -> {args.output}")
        print(f"Obrisano (samo emoji): {removed}, praznih: {empty}")
        return 0

    kept, removed, empty = clean_lines_file(args.input)
    if not kept:
        print("Nema komentara posle ciscenja.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(f"Procitano iz: {args.input}")
    print(f"Sacuvano {len(kept)} komentara -> {args.output}")
    print(f"Obrisano (samo emoji): {removed}, praznih linija: {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
