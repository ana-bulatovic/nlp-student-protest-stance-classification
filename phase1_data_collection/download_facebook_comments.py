#!/usr/bin/env python3
"""Download comments from public Facebook posts, pages, or groups.

Unlike scraping raw HTTP requests (which Facebook now blocks with a
"browser not supported" wall even with valid cookies), this drives a real
Chromium browser via Playwright, reusing a logged-in session saved by
facebook_login.py. Because it's a genuine browser, it renders the normal
www.facebook.com site (with JavaScript) instead of the old mbasic site.

Setup (see README instructions given in chat):
    pip install -r requirements.txt
    python3 -m playwright install chromium
    python3 facebook_login.py          # one-time interactive login
    python3 download_facebook_comments.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_facebook import (  # noqa: E402
    CommentRecord,
    add_common_cli_args,
    clean_comment_text,
    dedupe_preserve_order,
    extract_post_id,
    load_input_urls,
    output_dir,
    save_comments,
    session_dir,
    urls_file,
)

try:
    from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright
except ImportError:
    print("Error: playwright not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    print("Then run: python3 -m playwright install chromium", file=sys.stderr)
    sys.exit(1)


DEFAULT_OUTPUT = output_dir("facebook")
DEFAULT_STORAGE_STATE = session_dir("facebook") / "storage_state.json"

# URL patterns that point at one specific post/reel/video/photo (not a whole feed).
_POST_PATTERNS = (
    r"/posts/",
    r"/permalink/",
    r"permalink\.php",
    r"story_fbid=",
    r"/videos/",
    r"/photos/",
    r"/reel/",
    r"/reels/",
    r"/watch/?\?",
    r"/share/[pvr]/",
    r"[?&]fbid=",
)

# Path segments that are Facebook features, not page/account names. Without
# this guard a link like /reel/123 would be mistaken for a page called "reel"
# and the whole reels feed would get scraped instead of that single reel.
_RESERVED_PATH_SEGMENTS = {
    "groups", "profile.php", "permalink.php", "story.php", "photo", "photo.php",
    "reel", "reels", "watch", "video", "videos", "share", "stories", "story",
    "marketplace", "events", "media", "pg", "pages", "p", "posts", "gaming",
    "help", "settings", "notifications", "messages", "search", "hashtag",
}

# Confirmed by inspecting a real saved page (see chat history): Facebook tags
# every individual comment (top-level or reply) with an aria-label of the form
# "Comment by <Name> <relative time> ago". This is far more reliable than
# guessing a wrapping "Comments" container, so we select directly on it.
# Serbian-locale prefixes are included as a fallback in case the account
# language changes back.
_COMMENT_SELECTOR = (
    '[aria-label^="Comment by "], [aria-label^="Komentar od "], [aria-label^="Коментар од "]'
)

# NOTE: page.evaluate() requires the whole string passed in to be a single
# function expression (e.g. "() => { ... }"), not multiple concatenated
# top-level statements.
_EXTRACT_COMMENTS_JS = """
() => {
    const selector = '%s';
    const commentEls = Array.from(document.querySelectorAll(selector));
    return commentEls.map((el) => {
        const ariaLabel = el.getAttribute('aria-label') || '';
        const nested = Array.from(el.querySelectorAll(selector));
        const prevDisplay = nested.map((n) => n.style.display);
        nested.forEach((n) => { n.style.display = 'none'; });
        const fullText = el.innerText || '';
        nested.forEach((n, i) => { n.style.display = prevDisplay[i]; });
        const isReply = commentEls.some((other) => other !== el && other.contains(el));
        return { ariaLabel, fullText, isReply };
    });
}
""" % (_COMMENT_SELECTOR,)

_COUNT_COMMENTS_JS = """
() => document.querySelectorAll('%s').length
""" % (_COMMENT_SELECTOR,)

# Text patterns for "load more comments/replies" buttons, covering the
# English and Serbian (Latin + Cyrillic) phrasings we've seen so far.
_MORE_COMMENTS_RE = re.compile(
    r"(view|show|see|prikaž|прикаж).*(comment|komentar|коментар)", re.I
)
_MORE_REPLIES_RE = re.compile(
    r"(view|show|see|prikaž|прикаж).*(repl|odgovor|одговор)", re.I
)

# Facebook truncates long comments with a clickable "See more" link that
# expands the full text in place. We click through all of these before
# reading comment text, otherwise we'd save the cut-off version.
_SEE_MORE_RE = re.compile(r"^(see more|prikaži više|прикажи више|vidi više)$", re.I)

# Parses "Comment by <Name> <relative time> ago" (and the Serbian equivalents)
# into (author, relative_time).
_COMMENT_ARIA_RE = re.compile(
    r"^(?:Comment by|Komentar od|Коментар од)\s+(?P<author>.+?)\s+"
    r"(?P<time>\d[\w\s]*ago|just now|upravo sada|управо сада)$",
    re.I,
)


def _parse_comment_aria(aria_label: str) -> tuple[str, str]:
    match = _COMMENT_ARIA_RE.match(aria_label.strip())
    if match:
        return match.group("author").strip(), match.group("time").strip()
    author = re.sub(r"^(Comment by|Komentar od|Коментар од)\s+", "", aria_label, flags=re.I).strip()
    return author or "unknown", ""


def classify_target(value: str) -> tuple[str, str]:
    """Classify a raw input as a single post, a group feed, or a page/account feed.

    Returns a (kind, value) tuple where kind is one of "post", "group", "account".
    """
    value = value.strip()
    if re.fullmatch(r"\d+", value):
        return "post", value
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError(f"Cannot parse Facebook target from: {value!r}")

    if any(re.search(pattern, value) for pattern in _POST_PATTERNS):
        return "post", value

    group_match = re.search(r"facebook\.com/groups/([^/?#]+)", value)
    if group_match:
        return "group", group_match.group(1)

    profile_id_match = re.search(r"profile\.php\?id=(\d+)", value)
    if profile_id_match:
        return "account", profile_id_match.group(1)

    account_match = re.search(r"facebook\.com/(?:pg/|pages/)?([^/?#]+)", value)
    if account_match and account_match.group(1).lower() not in _RESERVED_PATH_SEGMENTS:
        return "account", account_match.group(1)

    raise ValueError(f"Cannot parse Facebook target from: {value!r}")


def _dismiss_cookie_banner(page: Page) -> None:
    for text in ("Allow all cookies", "Dozvoli sve kolačiće", "Prihvati sve kolačiće"):
        try:
            button = page.get_by_role("button", name=text, exact=False)
            if button.count() > 0:
                button.first.click(timeout=2000)
                page.wait_for_timeout(500)
                return
        except Exception:
            continue


def _clean_comment_text(full_text: str, author: str) -> str:
    return clean_comment_text(full_text, author)


def expand_comments(page: Page, max_comments: int, include_replies: bool, max_clicks: int = 25) -> None:
    """Repeatedly click "view more comments/replies" buttons to load the thread."""
    patterns = [_MORE_COMMENTS_RE]
    if include_replies:
        patterns.append(_MORE_REPLIES_RE)

    for _ in range(max_clicks):
        if max_comments:
            try:
                count = page.evaluate(_COUNT_COMMENTS_JS)
            except Exception:
                count = 0
            if count >= max_comments:
                break

        clicked = False
        try:
            buttons = page.get_by_role("button").all()
        except Exception:
            buttons = []
        for button in buttons:
            try:
                text = button.inner_text(timeout=500)
            except Exception:
                continue
            if any(pattern.search(text) for pattern in patterns):
                try:
                    button.scroll_into_view_if_needed(timeout=2000)
                    button.click(timeout=2000)
                    clicked = True
                    page.wait_for_timeout(900)
                    break  # DOM changed; re-scan buttons next loop
                except Exception:
                    continue
        if not clicked:
            break


def expand_truncated_comments(page: Page, max_clicks: int = 300) -> int:
    """Click every inline "See more" link so truncated comments load in full.

    Returns how many links were clicked.
    """
    clicks = 0
    for _ in range(max_clicks):
        locator = page.get_by_text(_SEE_MORE_RE)
        try:
            count = locator.count()
        except Exception:
            break
        if count == 0:
            break
        try:
            locator.first.scroll_into_view_if_needed(timeout=1500)
            locator.first.click(timeout=1500)
            clicks += 1
            page.wait_for_timeout(150)
        except Exception:
            # This particular match couldn't be clicked (e.g. stale/overlapping
            # element) - stop rather than looping forever on the same one.
            break
    return clicks


def fetch_post_comments(
    page: Page,
    post_url: str,
    max_comments: int,
    include_replies: bool,
    debug_html_dir: Path | None,
) -> tuple[str, str, list[CommentRecord]]:
    page.goto(post_url, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(2000)
    _dismiss_cookie_banner(page)

    for _ in range(4):
        page.mouse.wheel(0, 1800)
        page.wait_for_timeout(700)

    expand_comments(page, max_comments, include_replies)

    # All comments are loaded at this point; now expand any truncated
    # ("...See more") comment text before reading it.
    expand_truncated_comments(page)

    raw_comments = page.evaluate(_EXTRACT_COMMENTS_JS)
    post_id = extract_post_id(post_url)

    records: list[CommentRecord] = []
    for index, item in enumerate(raw_comments):
        author, created_at = _parse_comment_aria(item.get("ariaLabel") or "")
        text = _clean_comment_text(item.get("fullText") or "", author)
        if not text:
            continue
        records.append(
            CommentRecord(
                id=f"{post_id}_{index}",
                source="facebook",
                post_url=post_url,
                post_id=post_id,
                text=text,
                author=author,
                created_at=created_at,
                is_reply=bool(item.get("isReply")),
                parent_comment_id=None,
            )
        )
        if max_comments and len(records) >= max_comments:
            break

    if not records and debug_html_dir is not None:
        debug_html_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_html_dir / f"{post_id}_debug.html"
        debug_path.write_text(page.content(), encoding="utf-8")
        print(f"    No comments found - saved page HTML for inspection: {debug_path}", file=sys.stderr)

    return post_url, post_id, records


def collect_post_links_from_feed(page: Page, max_posts: int, max_scrolls: int = 30) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    stall = 0
    for _ in range(max_scrolls):
        hrefs = page.eval_on_selector_all("a", "els => els.map(e => e.href)")
        new_count = 0
        for href in hrefs:
            if not href or href in seen_set:
                continue
            if any(re.search(pattern, href) for pattern in _POST_PATTERNS):
                seen_set.add(href)
                seen.append(href)
                new_count += 1
        if max_posts and len(seen) >= max_posts:
            break
        if new_count == 0:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
        page.mouse.wheel(0, 2200)
        page.wait_for_timeout(1200)
    return seen[:max_posts] if max_posts else seen


def load_unique_items(args: argparse.Namespace) -> list[str]:
    raw = load_input_urls(
        args.url, args.url_flag, args.url_file, args.direct_id,
        default_url_file=urls_file("facebook"),
    )
    return dedupe_preserve_order(item.strip() for item in raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Facebook post/page/group comments.")
    add_common_cli_args(parser, DEFAULT_OUTPUT)
    parser.add_argument(
        "--storage-state",
        dest="storage_state",
        type=Path,
        default=None,
        help=f"Saved login session file from facebook_login.py (default: {DEFAULT_STORAGE_STATE}).",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=20,
        help="When a URL points to a whole page/group feed (not one post), "
        "max number of posts to walk through (default: 20).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser without a visible window (faster, but harder to debug).",
    )
    parser.add_argument(
        "--debug-html",
        action="store_true",
        help="If a post yields 0 comments, save the rendered page HTML for inspection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be fetched (and how each link was understood), "
        "then exit without opening a browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    storage_state_path = args.storage_state or DEFAULT_STORAGE_STATE
    if not storage_state_path.exists() and not args.dry_run:
        print(f"Error: no saved Facebook session found at {storage_state_path}.", file=sys.stderr)
        print("Run: python3 facebook_login.py   (one-time interactive login)", file=sys.stderr)
        return 1

    try:
        raw_items = load_unique_items(args)
        targets = dedupe_preserve_order(
            f"{kind}:{value}" for kind, value in (classify_target(item) for item in raw_items)
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        feeds = [t for t in targets if not t.startswith("post:")]
        print(f"Aktivnih linkova u ulazu: {len(raw_items)}")
        print(f"Za obradu (bez duplikata): {len(targets)}\n")
        for target in targets:
            kind, value = target.split(":", 1)
            print(f"  {kind:8} {value}")
        if feeds:
            print(
                f"\nPAZI: {len(feeds)} link(ova) se tumaci kao ceo feed stranice/grupe, "
                f"pa bi se skinulo do {args.max_posts} objava sa svakog."
            )
        else:
            print("\nSvi linkovi su pojedinacne objave - skinuce se samo one, nista drugo.")
        return 0

    debug_html_dir = args.output_dir / "_debug" if args.debug_html else None
    total = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        context = browser.new_context(storage_state=str(storage_state_path))
        page = context.new_page()

        for target in targets:
            kind, value = target.split(":", 1)
            print(f"\nFetching from {kind}: {value}")
            try:
                if kind == "post":
                    post_url = value if value.startswith("http") else f"https://www.facebook.com/{value}"
                    post_url, post_id, comments = fetch_post_comments(
                        page, post_url, args.max_comments, not args.no_replies, debug_html_dir
                    )
                    print(f"  Found {len(comments)} comments")
                    txt, _ = save_comments(comments, args.output_dir, "facebook", post_id, post_url)
                    print(f"  Saved -> {txt}")
                    total += len(comments)
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                else:
                    feed_url = (
                        f"https://www.facebook.com/groups/{value}"
                        if kind == "group"
                        else f"https://www.facebook.com/{value}"
                    )
                    page.goto(feed_url, wait_until="domcontentloaded", timeout=45000)
                    page.wait_for_timeout(2000)
                    _dismiss_cookie_banner(page)
                    post_links = collect_post_links_from_feed(page, args.max_posts)
                    print(f"  Found {len(post_links)} posts in feed")
                    for post_url in post_links:
                        post_url, post_id, comments = fetch_post_comments(
                            page, post_url, args.max_comments, not args.no_replies, debug_html_dir
                        )
                        print(f"  Post {post_id}: {len(comments)} comments")
                        txt, _ = save_comments(comments, args.output_dir, "facebook", post_id, post_url)
                        print(f"    Saved -> {txt}")
                        total += len(comments)
                        if args.sleep > 0:
                            time.sleep(args.sleep)
            except PlaywrightTimeoutError as exc:
                print(f"  Timed out for {value}: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"  Error for {value}: {exc}", file=sys.stderr)

        browser.close()

    print(f"\nTotal comments saved: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
