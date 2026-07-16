#!/usr/bin/env python3
"""
Jedan poziv: dohvati X (Twitter) replies -> sacuvaj punu tabelu ->
izvuci tekst -> ocisti emoji -> APPEND u x_all_texts_clean.txt
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clean_emojis import clean_text_line  # noqa: E402
from common import output_dir, save_comments  # noqa: E402
from download_x_comments import (  # noqa: E402
    DEFAULT_OUTPUT,
    create_client,
    iter_comments,
    load_unique_items,
    post_url,
)

DEFAULT_CLEAN_FILE = output_dir("x") / "x_all_texts_clean.txt"
DEFAULT_ALL_TEXTS = output_dir("x") / "x_all_texts.txt"


def append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline: download X replies -> full table -> "
            "extract text -> clean emojis -> APPEND to clean file."
        )
    )
    parser.add_argument("url", nargs="?", help="X/Twitter post URL")
    parser.add_argument("--url", dest="url_flag", help="X/Twitter post URL")
    parser.add_argument("--tweet-id", help="Tweet ID")
    parser.add_argument("--url-file", type=Path, help="Override default urls_x.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--username", help="X username (legacy login)")
    parser.add_argument("--email", help="Email naloga")
    parser.add_argument("--password", help="X lozinka")
    parser.add_argument("--cookies-file", type=Path, default=None)
    parser.add_argument("--refresh-session", action="store_true")
    parser.add_argument("--no-login", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--request-sleep", type=float, default=2.0)
    parser.add_argument("--max-comments", type=int, default=0)
    parser.add_argument(
        "--clean-file",
        type=Path,
        default=DEFAULT_CLEAN_FILE,
        help=f"APPEND ociscenih tekstova (default: {DEFAULT_CLEAN_FILE.name})",
    )
    parser.add_argument(
        "--all-texts-file",
        type=Path,
        default=DEFAULT_ALL_TEXTS,
        help=f"APPEND sirovih tekstova (default: {DEFAULT_ALL_TEXTS.name})",
    )
    parser.add_argument("--id", dest="direct_id", help=argparse.SUPPRESS)
    return parser.parse_args()


async def run_pipeline(args: argparse.Namespace) -> int:
    try:
        items = load_unique_items(args)
        client = await create_client(
            args.username,
            args.email,
            args.password,
            login=not args.no_login,
            cookies_file=args.cookies_file,
            refresh_session=args.refresh_session,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    clean_path = Path(args.clean_file)
    all_texts_path = Path(args.all_texts_file)

    total_raw = 0
    total_clean = 0
    total_dropped = 0
    errors = 0

    for tweet_id in items:
        try:
            print(f"\n=== {post_url(tweet_id)} ===")
            comments = [
                c
                async for c in iter_comments(
                    client,
                    tweet_id,
                    request_sleep=args.request_sleep,
                    max_comments=args.max_comments,
                )
            ]
            txt_path, _json_path = save_comments(
                comments,
                args.output_dir,
                "x",
                tweet_id,
                post_url(tweet_id),
            )
            print(f"  Puna tabela: {len(comments)} replies -> {txt_path.name}")

            raw_texts = [c.text.strip() for c in comments if c.text and c.text.strip()]
            append_lines(all_texts_path, raw_texts)
            total_raw += len(raw_texts)

            cleaned: list[str] = []
            dropped = 0
            for text in raw_texts:
                result = clean_text_line(text)
                if result is None:
                    dropped += 1
                else:
                    cleaned.append(result)

            append_lines(clean_path, cleaned)
            total_clean += len(cleaned)
            total_dropped += dropped
            print(
                f"  Tekst: +{len(raw_texts)} | ocisceno: +{len(cleaned)} | "
                f"obrisano (emoji-only): {dropped}"
            )
            print(f"  Append -> {clean_path.name}")
        except Exception as exc:
            errors += 1
            print(f"  Greska za {tweet_id}: {exc}", file=sys.stderr)

        if args.sleep > 0:
            await asyncio.sleep(args.sleep)

    print("\n--- Rezime ---")
    print(f"Sirovi tekstovi (append): {total_raw} -> {all_texts_path}")
    print(f"Ocisceni tekstovi (append): {total_clean} -> {clean_path}")
    print(f"Obrisano (samo emoji): {total_dropped}")
    if errors:
        print(f"Greske na {errors} post(ova).", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    return asyncio.run(run_pipeline(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
