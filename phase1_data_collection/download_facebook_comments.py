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

# Reels are the one format where the page loads with no comments in the DOM at
# all: the player only shows a "Comment" button, and the panel behind it needs
# 10-15 seconds to populate (verified by inspecting a live reel). Normal posts
# already carry their comments on load, so this button is only ever pressed
# when the page turns up empty.
_COMMENT_BUTTON_RE = re.compile(r"^(comment|komentar|коментар)$", re.I)

# Text patterns for "load more comments/replies" buttons, covering the
# English and Serbian (Latin + Cyrillic) phrasings we've seen so far.
_MORE_COMMENTS_RE = re.compile(
    r"(view|show|see|prikaž|прикаж).*(comment|komentar|коментар)", re.I
)
_MORE_REPLIES_RE = re.compile(
    r"(view|show|see|prikaž|прикаж).*(repl|odgovor|одговор)", re.I
)

# Facebook defaults the comment section to "Most relevant", which hides a large
# share of the thread. The dropdown also offers "All comments"; switching costs
# one click and is the difference between a sample and the whole thread.
_SORT_BUTTON_RE = re.compile(r"^(most relevant|top comments|najrelevantnij|најрелевантниј)", re.I)
_ALL_COMMENTS_RE = re.compile(r"^all comments|^svi komentari|^сви коментари", re.I)

# The "view more" button doubles as a progress counter ("View more comments
# 6 of 24"), which is the only place Facebook states how many comments the
# post actually has.
_PROGRESS_RE = re.compile(r"(\d[\d.,]*)\s*of\s*(\d[\d.,]*)", re.I)

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


def open_comment_panel(page: Page, timeout_ms: int = 30000) -> bool:
    """Reveal comments on pages that load without any (reels).

    Returns True if comments are present by the end, either because they were
    there already or because opening the panel brought them in.
    """
    try:
        if page.evaluate(_COUNT_COMMENTS_JS) > 0:
            return True
    except Exception:
        pass

    # The panel is slow and sometimes swallows the first click outright, so a
    # single attempt is the difference between the whole thread and nothing.
    for attempt in range(3):
        try:
            page.get_by_role("button", name=_COMMENT_BUTTON_RE).first.click(timeout=5000)
        except Exception:
            page.wait_for_timeout(2000)
            continue

        try:
            page.wait_for_selector(_COMMENT_SELECTOR, state="attached", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            return True
        except PlaywrightTimeoutError:
            if attempt < 2:
                page.reload(wait_until="domcontentloaded")
                page.wait_for_timeout(4000)
    return False


def switch_to_all_comments(page: Page) -> bool:
    """Move the comment section off "Most relevant" onto "All comments"."""
    try:
        for button in page.get_by_role("button").all():
            try:
                text = button.inner_text(timeout=400).strip()
            except Exception:
                continue
            if not text or len(text) > 40 or not _SORT_BUTTON_RE.search(text):
                continue
            button.click(timeout=3000)
            page.wait_for_timeout(1200)
            option = page.get_by_role("menuitem", name=_ALL_COMMENTS_RE)
            if option.count() == 0:
                page.keyboard.press("Escape")
                return False
            option.first.click(timeout=3000)
            page.wait_for_timeout(3000)
            return True
    except Exception:
        pass
    return False


def reported_comment_total(page: Page) -> int | None:
    """How many comments Facebook says the post has, per the "N of M" counter."""
    try:
        buttons = page.get_by_role("button").all()
    except Exception:
        return None
    for button in buttons:
        try:
            text = button.inner_text(timeout=400)
        except Exception:
            continue
        if not text or not _MORE_COMMENTS_RE.search(text):
            continue
        match = _PROGRESS_RE.search(text)
        if match:
            try:
                return int(re.sub(r"[.,]", "", match.group(2)))
            except ValueError:
                return None
    return None


def _scroll_over_comments(page: Page, allow_page_scroll: bool) -> None:
    """Wheel over the comment list itself, so lazy loading kicks in.

    On a reel the wheel must never reach the page: the reel player treats a
    page-level scroll as "next reel" and silently swaps in a different video,
    after which we would be reading a stranger's comments.
    """
    try:
        box = page.locator(_COMMENT_SELECTOR).last.bounding_box(timeout=1500)
    except Exception:
        box = None
    if box:
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.wheel(0, 1000)
    elif allow_page_scroll:
        page.mouse.wheel(0, 1200)


# Matching is done inside the page rather than by walking Playwright locators.
# A long thread ends up with hundreds of buttons, and reading each one's text
# over the wire made every round slower than the last - the search itself, not
# the loading, was what capped a big reel at a few hundred comments.
_CLICK_MATCHING_BUTTON_JS = """
(pattern) => {
    const re = new RegExp(pattern, 'i');
    for (const el of document.querySelectorAll('[role="button"], button')) {
        const text = (el.innerText || '').trim();
        if (!text || text.length > 60 || !re.test(text)) continue;
        el.scrollIntoView({ block: 'center' });
        el.click();
        return text.slice(0, 40);
    }
    return null;
}
"""


def _click_first_match(page: Page, pattern: re.Pattern[str]) -> bool:
    """Click the first button whose label matches, if there is one."""
    try:
        clicked = page.evaluate(_CLICK_MATCHING_BUTTON_JS, pattern.pattern)
    except Exception:
        return False
    if not clicked:
        return False
    page.wait_for_timeout(1800)
    return True


def expand_comments(
    page: Page,
    max_comments: int,
    include_replies: bool,
    expected_post_id: str = "",
    allow_page_scroll: bool = True,
    deadline: float = 0.0,
    patience: int = 8,
    max_rounds: int = 600,
) -> int:
    """Load the whole thread by scrolling and clicking "view more" buttons.

    Facebook hands out comments in batches of roughly ten and the button
    disappears while a batch is in flight, so a single miss is not the end of
    the thread — hence `patience` rounds of no growth before giving up. A
    1500-comment reel therefore needs a few hundred rounds, which is what
    `deadline` is there to bound.

    Returns the number of comments loaded.
    """
    patterns = [_MORE_COMMENTS_RE]
    if include_replies:
        patterns.append(_MORE_REPLIES_RE)

    def count() -> int:
        try:
            return page.evaluate(_COUNT_COMMENTS_JS)
        except Exception:
            return 0

    stalled = 0
    last_report = 0
    for round_index in range(max_rounds):
        before = count()
        if max_comments and before >= max_comments:
            break
        if deadline and time.monotonic() > deadline:
            print(f"    (dostignuto vremensko ogranicenje na {before} komentara)")
            break
        if expected_post_id and extract_post_id(page.url) != expected_post_id:
            print(f"    (stranica je odlutala na {page.url} - prekidam)", file=sys.stderr)
            break

        _scroll_over_comments(page, allow_page_scroll)
        page.wait_for_timeout(900)

        # "View more comments" must be tried before "View N replies". Reply
        # buttons sit higher up the thread, so taking whichever matched first
        # meant expanding reply after reply and never reaching the next batch
        # of comments - a 844-comment reel stayed stuck at the first six.
        for pattern in patterns:
            if _click_first_match(page, pattern):
                break

        after = count()
        if after >= last_report + 50:
            last_report = after
            print(f"    ucitano {after} komentara...")

        if after == before:
            stalled += 1
            if stalled >= patience:
                break
        else:
            stalled = 0

    return count()


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
    post_timeout: float = 300.0,
) -> tuple[str, str, list[CommentRecord]]:
    page.goto(post_url, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(2000)
    _dismiss_cookie_banner(page)

    post_id = extract_post_id(post_url)
    is_reel = bool(re.search(r"/reels?/", post_url))

    # Reels have nothing below the fold, and scrolling the page would swap the
    # player over to the next reel in the feed.
    if not is_reel:
        for _ in range(4):
            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(700)

    open_comment_panel(page)
    switch_to_all_comments(page)
    reported = reported_comment_total(page)
    if reported is not None:
        print(f"  Facebook prijavljuje {reported} komentara")
    expand_comments(
        page,
        max_comments,
        include_replies,
        expected_post_id=post_id,
        allow_page_scroll=not is_reel,
        deadline=time.monotonic() + post_timeout if post_timeout else 0.0,
    )

    # All comments are loaded at this point; now expand any truncated
    # ("...See more") comment text before reading it.
    expand_truncated_comments(page)

    raw_comments = page.evaluate(_EXTRACT_COMMENTS_JS)

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
        "--post-timeout",
        dest="post_timeout",
        type=float,
        default=300.0,
        help="Koliko sekundi najvise da se dovlace komentari jedne objave "
        "(0 = bez ogranicenja; default: 300). Objava sa 1500 komentara traje "
        "i preko 10 minuta.",
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
                        page, post_url, args.max_comments, not args.no_replies, debug_html_dir,
                        args.post_timeout,
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
                            page, post_url, args.max_comments, not args.no_replies, debug_html_dir,
                            args.post_timeout,
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
