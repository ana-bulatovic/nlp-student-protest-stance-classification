#!/usr/bin/env python3
"""
Jedan poziv: dohvati Instagram komentare -> sacuvaj punu tabelu ->
izvuci tekst -> ocisti emoji -> APPEND u instagram_all_texts_clean.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clean_emojis import clean_text_line  # noqa: E402
from common import output_dir, save_comments  # noqa: E402
from download_instagram_comments import (  # noqa: E402
    DEFAULT_OUTPUT,
    create_loader,
    iter_comments_via_iphone,
    load_unique_items,
    post_url,
)

DEFAULT_CLEAN_FILE = output_dir("instagram") / "instagram_all_texts_clean.txt"
DEFAULT_ALL_TEXTS = output_dir("instagram") / "instagram_all_texts.txt"


def append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline: download Instagram comments -> full table -> "
            "extract text -> clean emojis -> APPEND to clean file."
        )
    )
    parser.add_argument("url", nargs="?", help="Instagram post URL")
    parser.add_argument("--url", dest="url_flag", help="Instagram post URL")
    parser.add_argument("--shortcode", help="Post shortcode")
    parser.add_argument("--url-file", type=Path, help="Override default urls_instagram.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--username", help="Instagram username (legacy login)")
    parser.add_argument("--password", help="Instagram password (legacy login)")
    parser.add_argument("--sessionid", help="Instagram sessionid cookie")
    parser.add_argument(
        "--cookie",
        action="append",
        default=[],
        metavar="IME=VREDNOST",
        help="Dodatni cookie",
    )
    parser.add_argument("--cookies-file", type=Path, default=None)
    parser.add_argument(
        "--browser",
        choices=["chrome", "firefox", "edge", "brave", "chromium"],
    )
    parser.add_argument("--refresh-session", action="store_true")
    parser.add_argument("--no-login", action="store_true")
    parser.add_argument("--no-replies", action="store_true")
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--request-sleep", type=float, default=1.5)
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


def run_pipeline(args: argparse.Namespace) -> int:
    try:
        items = load_unique_items(args)
        loader = create_loader(
            args.username,
            args.password,
            login=not args.no_login,
            browser=args.browser,
            sessionid=args.sessionid or os.environ.get("INSTAGRAM_SESSIONID"),
            cookie_pairs=args.cookie,
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

    for shortcode in items:
        try:
            print(f"\n=== {post_url(shortcode)} ===")
            comments = list(
                iter_comments_via_iphone(
                    loader,
                    shortcode,
                    include_replies=not args.no_replies,
                    request_sleep=args.request_sleep,
                )
            )
            txt_path, _json_path = save_comments(
                comments,
                args.output_dir,
                "instagram",
                shortcode,
                post_url(shortcode),
            )
            print(f"  Puna tabela: {len(comments)} komentara -> {txt_path.name}")

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
            print(f"  Greska za {shortcode}: {exc}", file=sys.stderr)

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\n--- Rezime ---")
    print(f"Sirovi tekstovi (append): {total_raw} -> {all_texts_path}")
    print(f"Ocisceni tekstovi (append): {total_clean} -> {clean_path}")
    print(f"Obrisano (samo emoji): {total_dropped}")
    if errors:
        print(f"Greske na {errors} post(ova).", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
