#!/usr/bin/env python3
"""Shared helpers for the Facebook collection scripts.

Kept separate from `common.py` (used by the Instagram/X/YouTube scripts)
because the Facebook pipeline needs a different output layout and comment
cleaning: Facebook renders a comment as one blob of text with the author,
timestamp and action buttons glued in, so it has to be stripped before the
text is usable, and the .txt export is written human-readable for annotation
rather than as one CSV row per comment.

Imported by `download_facebook_comments.py` and `extract_facebook_texts.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"
SESSION_ROOT = SCRIPT_DIR / "sessions"


@dataclass
class CommentRecord:
    """A single normalized comment, regardless of source platform."""

    id: str
    source: str
    post_url: str
    post_id: str
    text: str
    author: str
    created_at: str
    is_reply: bool = False
    parent_comment_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --- Comment text cleaning -------------------------------------------------
# Facebook renders comments in two different layouts: one where the author,
# timestamp and action buttons sit on separate lines, and a newer one where
# everything is glued into a single line ("Ana Ana · 2dTekstLikeReplyShare9").
# Both need stripping, so the logic lives here and is shared by the scraper
# and the export script.

_TIME_TOKEN = r"(?:\d+\s*[smhdwy]|just now|now)"

_AUTHOR_AGE_SUFFIX_RE = re.compile(
    r"\s+(?:a|an|\d+)\s+(?:second|minute|hour|day|week|month|year)s?\s+ago$", re.I
)

_UI_LINE_RE = re.compile(
    r"^(like|reply|share|comment|edited|see translation|see more|hide|delete|report|top fan|"
    r"prikaži prevod|prikaži više|sviđa mi se|odgovori|komentariši|deli|izmenjeno|"
    r"свиђа ми се|одговори|коментариши|подели|прикажи више|·)$",
    re.I,
)
_TIME_LINE_RE = re.compile(r"^\d+\s*[a-zšđčćžа-яё]{0,4}$", re.I)
_JUNK_LINE_RE = re.compile(r"^(giphy|tenor|follow|\.{2,}|…)$", re.I)
_MEDIA_LINE_RE = re.compile(r"^(media\d*\.)?(giphy|tenor)\.com$", re.I)

_TRAILING_UI_RE = re.compile(
    r"\s*(?:(?:like|reply|share|see translation|see more|edited|hide|top fan|giphy|tenor|"
    r"prikaži prevod|prikaži više|izmenjeno)\s*)+\d*\s*$",
    re.I,
)
_TRAILING_SEE_MORE_RE = re.compile(
    r"\s*[.…]*\s*(see more|prikaži više|прикажи више|vidi više)\s*$", re.I
)


def _strip_author_prefix(text: str, author: str) -> str:
    """Remove a leading "Author · 2d" run-in prefix from a comment's text."""
    if author:
        match = re.match(
            r"\s*" + re.escape(author) + r"\s*·\s*" + _TIME_TOKEN + r"\s*", text, flags=re.I
        )
        if match:
            return text[match.end():]
    match = re.match(r"\s*[^·\n]{1,60}·\s*" + _TIME_TOKEN + r"\s*", text, flags=re.I)
    if match:
        return text[match.end():]
    return text


def clean_comment_text(raw: str, author: str = "") -> str:
    """Strip author name, timestamps, action buttons and media placeholders."""
    # The author string may itself carry a relative timestamp ("Ana Ana a day
    # ago"), so normalize it before matching it against the comment body.
    author = _AUTHOR_AGE_SUFFIX_RE.sub("", author or "").strip()

    text = (raw or "").replace("\xa0", " ")
    text = _strip_author_prefix(text, author)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == author.lower():
            continue
        if (
            _UI_LINE_RE.match(stripped)
            or _TIME_LINE_RE.match(stripped)
            or _JUNK_LINE_RE.match(stripped)
            or _MEDIA_LINE_RE.match(stripped)
        ):
            continue
        lines.append(stripped)

    text = "\n".join(lines)
    text = _TRAILING_UI_RE.sub("", text)
    text = _TRAILING_SEE_MORE_RE.sub("", text)
    text = _TRAILING_UI_RE.sub("", text).strip()

    # A GIF/sticker-only comment can reduce to just its placeholder once the
    # run-in author and buttons are stripped, so re-check the final result.
    if _JUNK_LINE_RE.match(text) or _MEDIA_LINE_RE.match(text) or _UI_LINE_RE.match(text):
        return ""
    return text


def output_dir(platform: str) -> Path:
    """Default directory where downloaded comments for a platform are stored."""
    return OUTPUT_ROOT / platform


def session_dir(platform: str) -> Path:
    """Directory used for platform-specific session data (e.g. cookies)."""
    return SESSION_ROOT / platform


def urls_file(platform: str) -> Path:
    """Default file with one URL/ID per line, read when no other input is given."""
    return SCRIPT_DIR / f"urls_{platform}.txt"


def add_common_cli_args(parser: argparse.ArgumentParser, default_output: Path) -> None:
    """Register the CLI arguments shared by every download_<platform>_comments.py script."""
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="A single post URL or ID to fetch (optional).",
    )
    parser.add_argument(
        "--url",
        dest="url_flag",
        action="append",
        default=None,
        help="Additional post URL/ID; repeatable (--url a --url b).",
    )
    parser.add_argument(
        "--url-file",
        dest="url_file",
        type=Path,
        default=None,
        help="Path to a text file with one URL/ID per line.",
    )
    parser.add_argument(
        "--id",
        dest="direct_id",
        action="append",
        default=None,
        help="A raw numeric post ID; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=default_output,
        help=f"Directory to save results in (default: {default_output}).",
    )
    parser.add_argument(
        "--max-comments",
        dest="max_comments",
        type=int,
        default=0,
        help="Maximum comments to fetch per post (0 = no limit).",
    )
    parser.add_argument(
        "--no-replies",
        dest="no_replies",
        action="store_true",
        help="Skip replies to comments, only fetch top-level comments.",
    )
    parser.add_argument(
        "--sleep",
        dest="sleep",
        type=float,
        default=2.0,
        help="Seconds to wait between requests, to avoid rate limiting (default: 2.0).",
    )


def dedupe_preserve_order(items) -> list[str]:
    """Remove duplicates from an iterable while keeping first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_input_urls(
    url: str | None,
    url_flag: list[str] | None,
    url_file: Path | None,
    direct_id: list[str] | None,
    default_url_file: Path,
) -> list[str]:
    """Gather raw URL/ID strings from CLI args, a custom file, or the default file.

    Priority: explicit CLI input (positional url, --url, --id, --url-file) is
    combined together when given. If none of those are provided at all, fall
    back to reading `default_url_file`.
    """
    items: list[str] = []
    if url:
        items.append(url)
    if url_flag:
        items.extend(url_flag)
    if direct_id:
        items.extend(direct_id)

    explicit_file = url_file
    if explicit_file is not None:
        items.extend(_read_lines(explicit_file))

    if not items and explicit_file is None:
        if not default_url_file.exists():
            raise ValueError(
                f"No input provided and default file not found: {default_url_file}"
            )
        items.extend(_read_lines(default_url_file))

    if not items:
        raise ValueError("No input URLs/IDs were found.")

    return items


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


_POST_ID_PATTERNS = (
    r"/posts/(\w+)",
    r"story_fbid=(\d+)",
    r"[?&]fbid=(\d+)",
    r"/videos/(\d+)",
    r"/reels?/(\w+)",
    r"/share/[pvr]/(\w+)",
    r"/permalink/(\d+)",
    r"[?&]v=(\d+)",
)


def extract_post_id(post_url: str) -> str:
    """Stable identifier for one post, derived from its URL.

    Used both when naming output files and when matching a saved post back to
    the input link it came from.
    """
    for pattern in _POST_ID_PATTERNS:
        match = re.search(pattern, post_url)
        if match:
            return match.group(1)
    return re.sub(r"[^A-Za-z0-9]+", "_", post_url).strip("_")[-40:]


def _slugify(value: str, max_length: int = 60) -> str:
    """Turn an arbitrary string (e.g. a URL) into a safe filename fragment."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"^https?://", "", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return (value or "item")[:max_length]


def save_comments(
    comments: list[CommentRecord],
    out_dir: Path,
    source: str,
    post_ref: str,
    post_url: str,
) -> tuple[Path, Path]:
    """Save comments to a .txt (human-readable) and .json (structured) file.

    Returns the (txt_path, json_path) that were written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{source}_{_slugify(post_ref)}_{timestamp}"
    txt_path = out_dir / f"{base_name}.txt"
    json_path = out_dir / f"{base_name}.json"

    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(
            {
                "source": source,
                "post_url": post_url,
                "post_ref": post_ref,
                "fetched_at": timestamp,
                "comment_count": len(comments),
                "comments": [c.to_dict() for c in comments],
            },
            jf,
            ensure_ascii=False,
            indent=2,
        )

    with txt_path.open("w", encoding="utf-8") as tf:
        tf.write(f"Post: {post_url}\n")
        tf.write(f"Fetched: {timestamp}\n")
        tf.write(f"Total comments: {len(comments)}\n")
        tf.write("=" * 60 + "\n\n")
        for comment in comments:
            prefix = "  ↳ reply: " if comment.is_reply else ""
            tf.write(f"{prefix}[{comment.created_at}] {comment.author}:\n")
            tf.write(f"{comment.text}\n\n")

    return txt_path, json_path
