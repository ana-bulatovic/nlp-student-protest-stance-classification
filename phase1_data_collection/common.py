"""Shared helpers for Phase 1 social media comment collection."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PHASE1_DIR = Path(__file__).resolve().parent


def output_dir(platform: str) -> Path:
    return PHASE1_DIR / "output" / platform


def session_dir(platform: str) -> Path:
    return PHASE1_DIR / "sessions" / platform


@dataclass
class CommentRecord:
    id: str
    source: str
    post_url: str
    post_id: str
    text: str
    author: str
    created_at: str
    is_reply: bool
    parent_comment_id: str | None


def urls_file(platform: str) -> Path:
    return PHASE1_DIR / f"urls_{platform}.txt"


def load_input_urls(
    positional_url: str | None,
    url_flag: str | None,
    url_file: Path | None,
    direct_id: str | None,
    *,
    default_url_file: Path | None = None,
) -> list[str]:
    items: list[str] = []

    for value in (positional_url, url_flag, direct_id):
        if value:
            items.append(value.strip())

    if url_file is None and not items and default_url_file is not None:
        url_file = default_url_file

    if url_file:
        if not url_file.exists():
            raise FileNotFoundError(
                f"URL file not found: {url_file}\n"
                f"Create it and add one post URL per line."
            )
        print(f"Loading URLs from: {url_file}")
        for line in url_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(line)

    if not items:
        hint = f" or add URLs to {default_url_file}" if default_url_file else ""
        raise ValueError(f"Provide a URL, --url, --id, or --url-file{hint}.")

    return items


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def save_comments(
    comments: list[CommentRecord],
    output_dir_path: Path,
    platform: str,
    post_id: str,
    post_url: str,
) -> tuple[Path, Path]:
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_post_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in post_id)

    txt_path = output_dir_path / f"{platform}_{safe_post_id}_{timestamp}.txt"
    json_path = output_dir_path / f"{platform}_{safe_post_id}_{timestamp}.json"

    with txt_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(
            ["id", "source", "url", "text", "author", "created_at", "is_reply", "parent_id"]
        )
        for comment in comments:
            writer.writerow(
                [
                    comment.id,
                    comment.source,
                    comment.post_url,
                    comment.text,
                    comment.author,
                    comment.created_at,
                    int(comment.is_reply),
                    comment.parent_comment_id or "",
                ]
            )

    payload = {
        "platform": platform,
        "post_id": post_id,
        "post_url": post_url,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "comment_count": len(comments),
        "comments": [asdict(comment) for comment in comments],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return txt_path, json_path


def add_common_cli_args(parser, default_output: Path) -> None:
    parser.add_argument("url", nargs="?", help="Post URL")
    parser.add_argument("--url", dest="url_flag", help="Post URL")
    parser.add_argument("--id", dest="direct_id", help="Post ID (tweet / video / post)")
    parser.add_argument(
        "--url-file",
        type=Path,
        default=None,
        help="TXT file with one URL or ID per line (overrides default urls file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output folder (default: {default_output})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Pause in seconds between posts (default: 2)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Max comments per post (0 = no limit)",
    )
    parser.add_argument(
        "--no-replies",
        action="store_true",
        help="Do not download replies to comments",
    )
