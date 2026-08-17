#!/usr/bin/env python3
"""Dopuni fb_final_*.txt fajlove novim komentarima iz anotiranog izvoza.

Cita facebook_all_texts_annotated.txt (format "text|url|label" koji izvozi
annotation_tool.html) i dopisuje samo komentare kojih jos nema u finalnim
fajlovima, u formatu "text|url|ANOTACIJA" (ZA-VLAST / PROTIV-VLASTI /
NEUTRAL). Postojeci redovi se nikada ne menjaju ni brisu.

Duplikati se prepoznaju po tekstu komentara (bez obzira na velika/mala slova
i visak razmaka), i to preko sva tri fajla - komentar koji je vec svrstan u
jednu kategoriju nece biti dodat u drugu.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATED = SCRIPT_DIR / "facebook_all_texts_annotated.txt"

# Oznaka iz alata za anotiranje -> finalni fajl. Oznaka se upisuje u trecu
# kolonu tacno onako kako je alat izvozi.
ANNOTATED_DIR = SCRIPT_DIR / "annotated"
LABEL_FILES = {
    "ZA-VLAST": ANNOTATED_DIR / "fb_final_za_vlast_annotated.txt",
    "PROTIV-VLASTI": ANNOTATED_DIR / "fb_final_protiv_vlasti_annotated.txt",
    "NEUTRAL": ANNOTATED_DIR / "fb_final_neutral_annotated.txt",
}

# Imena fajlova za --split (rucni pregled pre prepisivanja u finalne).
SPLIT_NAMES = {
    "ZA-VLAST": "fb_novo_za_vlast.txt",
    "PROTIV-VLASTI": "fb_novo_protiv_vlasti.txt",
    "NEUTRAL": "fb_novo_neutral.txt",
}


def norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def read_annotated(path: Path) -> list[tuple[str, str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        rows = list(reader)

    if not rows:
        return []

    start = 1 if rows[0] and rows[0][0].strip().lower() == "text" else 0
    result: list[tuple[str, str, str]] = []
    for row in rows[start:]:
        if len(row) < 3:
            continue
        text, url, label = row[0].strip(), row[1].strip(), row[2].strip().upper()
        if not text or label not in LABEL_FILES:
            continue
        result.append((text, url, label))
    return result


def read_existing(path: Path) -> tuple[list[str], set[str]]:
    """Vrati (postojeci redovi kakvi su, set normalizovanih tekstova)."""
    if not path.exists():
        return [], set()

    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    texts: set[str] = set()
    for line in raw_lines:
        try:
            fields = next(csv.reader(io.StringIO(line), delimiter="|"))
        except csv.Error:
            fields = line.split("|")
        if fields and fields[0].strip():
            texts.add(norm(fields[0]))
    return raw_lines, texts


def format_row(text: str, url: str, label: str) -> str:
    # Prelomi redova bi razbili jedan komentar na vise linija u fajlu.
    text = " ".join(text.split())
    buffer = io.StringIO()
    csv.writer(buffer, delimiter="|", quoting=csv.QUOTE_MINIMAL, lineterminator="").writerow(
        [text, url, label]
    )
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dopisi nove anotirane komentare u fb_final_*.txt fajlove."
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        default=DEFAULT_ANNOTATED,
        help=f"Anotirani fajl iz annotation_tool.html (default: {DEFAULT_ANNOTATED})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Samo prikazi sta bi bilo dodato, bez upisa u fajlove.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Ne diraj finalne fajlove. Umesto toga razvrstaj nove komentare u tri "
        "zasebna fajla, za rucni pregled pre prepisivanja u finalne.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=SCRIPT_DIR / "za_pregled",
        help="Folder u koji ide izlaz opcije --split (default: phase2_annotation/za_pregled).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.annotated.exists():
        print(f"Greska: nema fajla {args.annotated}", file=sys.stderr)
        return 1

    annotated = read_annotated(args.annotated)
    if not annotated:
        print(f"Greska: nema upotrebljivih redova u {args.annotated}", file=sys.stderr)
        return 1

    existing_lines: dict[str, list[str]] = {}
    already_seen: set[str] = set()
    for label, path in LABEL_FILES.items():
        lines, texts = read_existing(path)
        existing_lines[label] = lines
        already_seen |= texts

    additions: dict[str, list[str]] = {label: [] for label in LABEL_FILES}
    skipped_duplicates = 0

    for text, url, label in annotated:
        key = norm(text)
        if key in already_seen:
            skipped_duplicates += 1
            continue
        already_seen.add(key)
        additions[label].append(format_row(text, url, label))

    total_added = sum(len(v) for v in additions.values())
    print(f"Procitano iz: {args.annotated} ({len(annotated)} anotiranih komentara)")
    print(f"Preskoceno kao duplikat: {skipped_duplicates}")
    print()

    if args.split:
        args.split_dir.mkdir(parents=True, exist_ok=True)
        for label, name in SPLIT_NAMES.items():
            path = args.split_dir / name
            rows = additions[label]
            path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
            print(f"{path}: {len(rows)} komentara")
        print()
        print(f"Ukupno razvrstano: {total_added}. Finalni fajlovi nisu dirani.")
        return 0

    for label, path in LABEL_FILES.items():
        before = len(existing_lines[label])
        new_rows = additions[label]
        after = before + len(new_rows)
        print(f"{path.name}: {before} -> {after}  (+{len(new_rows)})")

        if args.dry_run or not new_rows:
            continue
        path.write_text("\n".join(existing_lines[label] + new_rows) + "\n", encoding="utf-8")

    print()
    if args.dry_run:
        print(f"DRY RUN - nista nije upisano. Ukupno bi bilo dodato: {total_added}")
    else:
        grand_total = sum(len(existing_lines[l]) + len(additions[l]) for l in LABEL_FILES)
        print(f"Dodato ukupno {total_added} novih komentara.")
        print(f"Ukupno u sva tri fajla: {grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
