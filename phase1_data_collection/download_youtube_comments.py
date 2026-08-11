#!/usr/bin/env python3
"""Download comments from public YouTube videos.

Uses the official YouTube Data API v3 (read-only, API key auth — no OAuth
login needed). See README.md for how to get an API key.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    CommentRecord,
    load_input_urls,
    output_dir,
    save_comments,
    session_dir,
    urls_file,
)

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print(
        "Error: google-api-python-client not installed. Run:\n"
        "  pip install google-api-python-client",
        file=sys.stderr,
    )
    sys.exit(1)


VIDEO_ID_PATTERNS = (
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=|youtube\.com/shorts/|youtube\.com/embed/|youtu\.be/)([A-Za-z0-9_-]{11})"),
)

DEFAULT_OUTPUT = output_dir("youtube")
DEFAULT_SESSION = session_dir("youtube")
DEFAULT_API_KEY_FILE = DEFAULT_SESSION / "api_key.txt"


def extract_video_id(value: str) -> str:
    value = value.strip()
    for pattern in VIDEO_ID_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    raise ValueError(f"Cannot parse YouTube video from: {value!r}")


def post_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def load_unique_items(args: argparse.Namespace) -> list[str]:
    items = load_input_urls(
        args.url,
        args.url_flag,
        args.url_file,
        args.video_id or args.direct_id,
        default_url_file=urls_file("youtube"),
    )
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        video_id = extract_video_id(item)
        if video_id not in seen:
            seen.add(video_id)
            unique.append(video_id)
    return unique


def resolve_api_key(api_key_arg: str | None) -> str:
    if api_key_arg:
        return api_key_arg.strip().strip('"').strip("'")

    env_key = os.environ.get("YOUTUBE_API_KEY")
    if env_key:
        return env_key.strip().strip('"').strip("'")

    if DEFAULT_API_KEY_FILE.exists():
        for line in DEFAULT_API_KEY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line.strip('"').strip("'")

    raise ValueError(
        "Nedostaje YouTube API key. Najlakse resenje:\n"
        "  1. https://console.cloud.google.com/ -> napravi/izaberi projekat\n"
        "  2. APIs & Services -> Library -> 'YouTube Data API v3' -> Enable\n"
        "  3. APIs & Services -> Credentials -> Create Credentials -> API key\n"
        f"  4. Sacuvaj kljuc u {DEFAULT_API_KEY_FILE} (jedna linija)\n\n"
        "Ili prosledi direktno: python download_youtube_comments.py --api-key \"VREDNOST\"\n"
        "Ili preko env promenljive: YOUTUBE_API_KEY=VREDNOST"
    )


def _comment_record(
    resource: dict,
    *,
    video_id: str,
    url: str,
    is_reply: bool,
    parent_comment_id: str | None,
) -> CommentRecord | None:
    snippet = resource.get("snippet") or {}
    text = (snippet.get("textOriginal") or snippet.get("textDisplay") or "").strip()
    if not text:
        return None
    return CommentRecord(
        id=str(resource.get("id") or ""),
        source="youtube",
        post_url=url,
        post_id=video_id,
        text=text,
        author=snippet.get("authorDisplayName", "") or "",
        created_at=snippet.get("publishedAt", "") or "",
        is_reply=is_reply,
        parent_comment_id=parent_comment_id,
    )


def iter_comments(
    youtube,
    video_id: str,
    include_replies: bool,
    max_comments: int,
) -> Iterator[CommentRecord]:
    url = post_url(video_id)
    fetched = 0
    page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            top_level = (item.get("snippet") or {}).get("topLevelComment") or {}
            record = _comment_record(
                top_level, video_id=video_id, url=url, is_reply=False, parent_comment_id=None
            )
            if record:
                fetched += 1
                yield record
                if max_comments and fetched >= max_comments:
                    return

            if include_replies:
                # commentThreads().list only returns a preview of replies
                # (not the full thread) — deeper pagination would need a
                # separate comments().list(parentId=...) call per thread.
                parent_id = str(top_level.get("id") or "")
                for reply in (item.get("replies") or {}).get("comments", []):
                    reply_record = _comment_record(
                        reply,
                        video_id=video_id,
                        url=url,
                        is_reply=True,
                        parent_comment_id=parent_id,
                    )
                    if reply_record:
                        fetched += 1
                        yield reply_record
                        if max_comments and fetched >= max_comments:
                            return

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    print(f"  Ukupno preuzeto: {fetched} komentara")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YouTube video comments.")
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument("--url", dest="url_flag", help="YouTube video URL")
    parser.add_argument("--video-id", help="11-character video ID")
    parser.add_argument("--url-file", type=Path, help="Override default urls_youtube.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", help="YouTube Data API v3 key")
    parser.add_argument("--no-replies", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0, help="Pauza izmedju videa (s)")
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Max komentara po videu (0 = bez limita)",
    )
    parser.add_argument("--id", dest="direct_id", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        items = load_unique_items(args)
        api_key = resolve_api_key(args.api_key)
    except ValueError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    youtube = build("youtube", "v3", developerKey=api_key)

    total = 0
    errors = 0
    for idx, video_id in enumerate(items):
        try:
            print(f"\n=== {post_url(video_id)} ===")
            comments = list(
                iter_comments(
                    youtube,
                    video_id,
                    include_replies=not args.no_replies,
                    max_comments=args.max_comments,
                )
            )
            txt, _js = save_comments(
                comments, args.output_dir, "youtube", video_id, post_url(video_id)
            )
            print(f"  Sacuvano {len(comments)} komentara -> {txt}")
            total += len(comments)
        except HttpError as exc:
            errors += 1
            reason = ""
            try:
                reason = exc.error_details[0].get("reason", "") if exc.error_details else ""
            except Exception:
                pass
            if reason == "commentsDisabled":
                print(f"  Komentari su iskljuceni za {video_id}, preskace se.", file=sys.stderr)
            elif reason == "quotaExceeded":
                print("  YouTube API kvota potrosena za danas.", file=sys.stderr)
            else:
                print(f"  Greska za {video_id}: {exc}", file=sys.stderr)
        except Exception as exc:
            errors += 1
            print(f"  Greska za {video_id}: {exc}", file=sys.stderr)
        if args.sleep > 0 and idx < len(items) - 1:
            time.sleep(args.sleep)

    print(f"\nUkupno sacuvano komentara: {total}")
    if errors:
        print(f"Greske na {errors} video(a).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
