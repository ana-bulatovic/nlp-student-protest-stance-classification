#!/usr/bin/env python3
"""Extract comment text (+ source url) from all YouTube output files into one combined file."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "output" / "youtube"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "_youtube_all_texts.txt"

# Raw files are named youtube_<videoId>_<YYYYMMDD_HHMMSS>.txt (see common.save_comments).
# Video IDs can contain underscores themselves, so anchor on the fixed-width
# timestamp at the end rather than splitting on "_".
TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})\.txt$")
FILENAME_RE = re.compile(r"^youtube_(.+)_(\d{8}_\d{6})$")
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def file_timestamp(path: Path) -> str | None:
    match = TIMESTAMP_RE.search(path.name)
    return match.group(1) if match else None


def file_video_id(path: Path) -> str | None:
    match = FILENAME_RE.match(path.stem)
    return match.group(1) if match else None


def extract_texts(
    input_dir: Path,
    dedupe: bool,
    after: str | None = None,
    video_ids: set[str] | None = None,
) -> list[list[str]]:
    txt_files = sorted(input_dir.glob("youtube_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nema youtube_*.txt fajlova u {input_dir}")

    rows: list[list[str]] = []
    seen_ids: set[str] = set()
    skipped_old = 0
    skipped_id = 0

    for path in txt_files:
        if path.name == DEFAULT_OUTPUT.name:
            continue
        if after is not None:
            ts = file_timestamp(path)
            # fixed-width YYYYMMDD_HHMMSS strings compare correctly as plain strings
            if ts is not None and ts <= after:
                skipped_old += 1
                continue
        if video_ids is not None:
            vid = file_video_id(path)
            if vid is None or vid not in video_ids:
                skipped_id += 1
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
                url = (row.get("url") or "").strip()
                # keep the source url so this file is ready to annotate
                # directly into text|url|LABEL, no need to look it up again
                rows.append([text, url])

    if after is not None and skipped_old:
        print(f"Preskoceno {skipped_old} starijih fajlova (pre --after {after})")
    if video_ids is not None and skipped_id:
        print(f"Preskoceno {skipped_id} fajlova van --video-ids liste")

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Izdvoji tekst + url komentara iz output/youtube fajlova."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder sa youtube_*.txt fajlovima (default: {DEFAULT_INPUT_DIR})",
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
    parser.add_argument(
        "--after",
        metavar="YYYYMMDD_HHMMSS",
        help=(
            "Ukljuci samo fajlove novije od ovog vremena (iz imena fajla), "
            "za merge samo nove serije preuzimanja. Koristi --print-now da dobijes trenutni format."
        ),
    )
    parser.add_argument(
        "--print-now",
        action="store_true",
        help="Ispisi trenutno vreme u --after formatu i izadji (pokreni pre novog preuzimanja)",
    )
    parser.add_argument(
        "--video-ids",
        metavar="ID1,ID2,...",
        help=(
            "Ukljuci samo fajlove za ove video ID-jeve (zarezom odvojeni). "
            "Pouzdanije od --after ako su stari video ID-jevi ponovo preuzeti "
            "(dobijaju novi timestamp pa --after ne moze da ih razlikuje)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.print_now:
        print(datetime.now().strftime(TIMESTAMP_FMT))
        return 0

    if args.after is not None and not re.fullmatch(r"\d{8}_\d{6}", args.after):
        print(
            f"Greska: --after mora biti u formatu YYYYMMDD_HHMMSS, dobio sam {args.after!r}",
            file=sys.stderr,
        )
        return 1

    video_ids = None
    if args.video_ids:
        video_ids = {v.strip() for v in args.video_ids.split(",") if v.strip()}

    try:
        rows = extract_texts(
            args.input_dir, dedupe=not args.no_dedupe, after=args.after, video_ids=video_ids
        )
    except FileNotFoundError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("Nema komentara za izvoz.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerow(["text", "url"])
        writer.writerows(rows)

    print(f"Procitano iz: {args.input_dir}")
    print(f"Sacuvano {len(rows)} komentara -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
