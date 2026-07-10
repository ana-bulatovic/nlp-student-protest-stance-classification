#!/usr/bin/env python3
"""Download comments from public Instagram posts."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

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
    import instaloader
    from instaloader import Instaloader, Post
    from instaloader.exceptions import ConnectionException
except ImportError:
    print("Error: instaloader not installed. Run: pip install instaloader", file=sys.stderr)
    sys.exit(1)


SHORTCODE_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)

DEFAULT_OUTPUT = output_dir("instagram")
DEFAULT_SESSION = session_dir("instagram")
DEFAULT_SESSIONID_FILE = DEFAULT_SESSION / "sessionid.txt"
DEFAULT_COOKIES_FILE = DEFAULT_SESSION / "cookies.txt"


def extract_shortcode(value: str) -> str:
    value = value.strip()
    match = SHORTCODE_PATTERN.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    raise ValueError(f"Cannot parse Instagram post from: {value!r}")


def post_url(shortcode: str) -> str:
    return f"https://www.instagram.com/p/{shortcode}/"


def load_unique_items(args: argparse.Namespace) -> list[str]:
    items = load_input_urls(
        args.url,
        args.url_flag,
        args.url_file,
        args.shortcode or args.direct_id,
        default_url_file=urls_file("instagram"),
    )
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        code = extract_shortcode(item)
        if code not in seen:
            seen.add(code)
            unique.append(code)
    return unique


def parse_cookie_pairs(pairs: list[str]) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Neispravan cookie format: {pair!r}. Koristi ime=vrednost")
        name, value = pair.split("=", 1)
        name = name.strip()
        value = unquote(value.strip().strip('"').strip("'"))
        if name:
            cookies[name] = value
    return cookies


def load_cookies_file(path: Path) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        cookies[name.strip()] = unquote(value.strip().strip('"').strip("'"))
    if not cookies:
        raise ValueError(f"U fajlu {path} nema cookies (format: ime=vrednost po liniji).")
    return cookies


def read_sessionid_file(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            if name.strip().lower() == "sessionid":
                return unquote(value.strip().strip('"').strip("'"))
            continue
        return unquote(line.strip('"').strip("'"))
    raise ValueError(f"U fajlu {path} nema sessionid.")


def resolve_sessionid(
    sessionid_arg: str | None,
    cookies_file: Path | None,
) -> tuple[str | None, bool]:
    """Vraca (sessionid, eksplicitno_prosledjen)."""
    if sessionid_arg:
        return unquote(sessionid_arg.strip().strip('"').strip("'")), True

    env_sessionid = os.environ.get("INSTAGRAM_SESSIONID")
    if env_sessionid:
        return unquote(env_sessionid.strip().strip('"').strip("'")), True

    if DEFAULT_SESSIONID_FILE.exists():
        return read_sessionid_file(DEFAULT_SESSIONID_FILE), False

    if cookies_file and cookies_file.exists():
        cookies = load_cookies_file(cookies_file)
        if cookies.get("sessionid"):
            return cookies["sessionid"], False

    if DEFAULT_COOKIES_FILE.exists():
        cookies = load_cookies_file(DEFAULT_COOKIES_FILE)
        if cookies.get("sessionid"):
            return cookies["sessionid"], False

    return None, False


def bootstrap_session_from_sessionid(loader: Instaloader, sessionid: str) -> str:
    """Dovoljan je samo sessionid — Instagram dopuni csrftoken/mid pri poseti."""
    session = loader.context._session
    session.cookies.update(
        {
            "sessionid": sessionid,
            "mid": "",
            "ig_pr": "1",
            "ig_vw": "1920",
            "csrftoken": "",
            "s_network": "",
            "ds_user_id": "",
        }
    )
    session.get("https://www.instagram.com/", timeout=30)
    cookies = session.cookies.get_dict()

    if cookies.get("csrftoken"):
        session.headers["X-CSRFToken"] = cookies["csrftoken"]

    username = loader.test_login()
    if not username:
        raise ValueError(
            "sessionid nije validan ili je istekao.\n"
            "Uloguj se u browser na instagram.com, kopiraj novi sessionid cookie "
            f"i sacuvaj ga u {DEFAULT_SESSIONID_FILE}"
        )

    loader.context.load_session(username, cookies)
    loader.context.username = username
    if cookies.get("ds_user_id"):
        loader.context.user_id = str(cookies["ds_user_id"])

    return username


def verify_loader_session(loader: Instaloader) -> bool:
    if not loader.context.is_logged_in:
        return False
    try:
        loader.test_login()
        return True
    except Exception:
        return False


def save_loader_session(loader: Instaloader, username: str) -> Path:
    DEFAULT_SESSION.mkdir(parents=True, exist_ok=True)
    session_file = DEFAULT_SESSION / f"session-{username}"
    loader.save_session_to_file(str(session_file))
    return session_file


def import_session_from_browser(loader: Instaloader, browser: str) -> str:
    try:
        import browser_cookie3
    except ImportError:
        raise ValueError(
            "Za ucitavanje iz browsera instaliraj: pip install browser-cookie3"
        ) from None

    browsers = {
        "chrome": browser_cookie3.chrome,
        "firefox": browser_cookie3.firefox,
        "edge": browser_cookie3.edge,
        "brave": browser_cookie3.brave,
        "chromium": browser_cookie3.chromium,
    }
    if browser not in browsers:
        raise ValueError(f"Nepodrzani browser: {browser}. Koristi: {', '.join(browsers)}")

    try:
        browser_cookies = list(browsers[browser]())
    except PermissionError as exc:
        raise ValueError(
            f"Windows ne dozvoljava citanje {browser} cookies bez admin prava.\n\n"
            "Koristi sessionid umesto browsera:\n"
            f"  1. F12 -> Application -> Cookies -> instagram.com -> sessionid\n"
            f"  2. Sacuvaj u {DEFAULT_SESSIONID_FILE}\n"
            "  3. python download_instagram_comments.py"
        ) from exc
    except Exception as exc:
        if "admin" in str(exc).lower():
            raise ValueError(
                f"Greska pri citanju {browser} cookies: {exc}\n"
                f"Koristi {DEFAULT_SESSIONID_FILE} umesto --browser."
            ) from exc
        raise

    cookies = {
        cookie.name: cookie.value
        for cookie in browser_cookies
        if "instagram.com" in cookie.domain
    }
    if not cookies.get("sessionid"):
        raise ValueError(
            f"Nema Instagram sessionid u browseru ({browser}). "
            "Prvo se uloguj na instagram.com."
        )

    return bootstrap_session_from_sessionid(loader, cookies["sessionid"])


def create_loader(
    username: str | None,
    password: str | None,
    login: bool,
    browser: str | None,
    sessionid: str | None,
    cookie_pairs: list[str],
    cookies_file: Path | None,
    refresh_session: bool,
) -> Instaloader:
    loader = Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=True,
        save_metadata=False,
        compress_json=False,
        quiet=False,
    )
    if not login:
        return loader

    DEFAULT_SESSION.mkdir(parents=True, exist_ok=True)
    resolved_sessionid, explicit_sessionid = resolve_sessionid(sessionid, cookies_file)
    if cookie_pairs:
        extra = parse_cookie_pairs(cookie_pairs)
        if extra.get("sessionid"):
            resolved_sessionid = extra["sessionid"]
            explicit_sessionid = True

    if not refresh_session and not explicit_sessionid and not browser and not cookie_pairs:
        session_files = sorted(
            DEFAULT_SESSION.glob("session-*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for session_file in session_files:
            session_username = session_file.name.removeprefix("session-")
            if username and username != session_username:
                continue
            try:
                loader.load_session_from_file(session_username, str(session_file))
                if verify_loader_session(loader):
                    print(f"Ucitana postojeca sesija za @{session_username}")
                    return loader
            except Exception:
                continue

    if resolved_sessionid:
        logged_in_as = bootstrap_session_from_sessionid(loader, resolved_sessionid)
        print(f"Ulogovano preko sessionid kao @{logged_in_as}")
        session_file = save_loader_session(loader, logged_in_as)
        print(f"Sesija sacuvana u: {session_file}")
        return loader

    if browser:
        logged_in_as = import_session_from_browser(loader, browser)
        print(f"Ulogovano preko browsera ({browser}) kao @{logged_in_as}")
        session_file = save_loader_session(loader, logged_in_as)
        print(f"Sesija sacuvana u: {session_file}")
        return loader

    username = username or os.environ.get("INSTAGRAM_USERNAME")
    if not username:
        raise ValueError(
            "Instagram login je obavezan. Najlakse resenje:\n"
            f"  1. Kopiraj sessionid cookie iz browsera\n"
            f"  2. Sacuvaj u {DEFAULT_SESSIONID_FILE}\n"
            "  3. python download_instagram_comments.py\n\n"
            "Ili: python download_instagram_comments.py --sessionid \"VREDNOST\""
        )

    session_file = DEFAULT_SESSION / f"session-{username}"
    try:
        loader.load_session_from_file(username, str(session_file))
        if verify_loader_session(loader):
            print(f"Ucitana postojeca sesija za @{username}")
            return loader
    except FileNotFoundError:
        pass

    password = password or os.environ.get("INSTAGRAM_PASSWORD")
    if not password:
        password = getpass.getpass(f"Instagram password for @{username}: ")

    try:
        loader.login(username, password)
    except instaloader.exceptions.TwoFactorAuthRequiredException:
        raise ValueError(
            "Instagram zahteva 2FA. Uloguj se u browser i koristi sessionid "
            f"u {DEFAULT_SESSIONID_FILE}."
        ) from None
    except instaloader.exceptions.BadCredentialsException:
        raise ValueError("Pogresno korisnicko ime ili lozinka.") from None
    except instaloader.exceptions.LoginException as exc:
        message = str(exc)
        if "Checkpoint required" in message or "checkpoint" in message.lower():
            raise ValueError(
                "Instagram trazi sigurnosnu proveru (checkpoint).\n"
                f"Kopiraj sessionid iz browsera u {DEFAULT_SESSIONID_FILE} i pokusaj ponovo."
            ) from exc
        raise ValueError(f"Instagram login greska: {message}") from exc

    save_loader_session(loader, username)
    print(f"Sesija sacuvana u: {session_file}")
    return loader


def iphone_json_with_retry(
    ctx,
    path: str,
    params: dict,
    *,
    max_attempts: int = 4,
    base_sleep: float = 3.0,
    request_sleep: float = 0.0,
) -> dict:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            if request_sleep > 0 and attempt == 0:
                time.sleep(request_sleep)
            return ctx.get_iphone_json(path, params)
        except ConnectionException as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                break
            wait = base_sleep * (attempt + 1)
            print(f"  Instagram API greska, ponovo za {wait:.0f}s...")
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def iter_comments_via_iphone(
    loader: Instaloader,
    shortcode: str,
    include_replies: bool,
    request_sleep: float,
) -> Iterator[CommentRecord]:
    """Preuzmi komentare direktno preko iPhone API-ja (bez GraphQL fallbacka)."""
    url = post_url(shortcode)
    mediaid = Post.shortcode_to_mediaid(shortcode)
    ctx = loader.context

    info = iphone_json_with_retry(
        ctx,
        f"api/v1/media/{mediaid}/info/",
        {},
        request_sleep=request_sleep,
    )
    items = info.get("items") or []
    if not items:
        raise ValueError(f"Post {shortcode} nije pronadjen ili nije dostupan.")
    total_expected = items[0].get("comment_count", 0)
    if total_expected:
        print(f"  Ocekivano komentara: ~{total_expected}")

    page = 0
    fetched = 0

    def _query(min_id: str | None = None) -> dict:
        pagination_params = {"min_id": min_id} if min_id is not None else {}
        return iphone_json_with_retry(
            ctx,
            f"api/v1/media/{mediaid}/comments/",
            {
                "can_support_threading": "true",
                "permalink_enabled": "false",
                **pagination_params,
            },
            request_sleep=request_sleep,
        )

    def _comment_record(
        node: dict,
        *,
        is_reply: bool,
        parent_comment_id: str | None,
    ) -> CommentRecord | None:
        text = (node.get("text") or "").strip()
        if not text:
            return None
        user = node.get("user") or {}
        created_at = datetime.fromtimestamp(node["created_at"], tz=timezone.utc).isoformat()
        return CommentRecord(
            id=str(node["pk"]),
            source="instagram",
            post_url=url,
            post_id=shortcode,
            text=text,
            author=user.get("username", ""),
            created_at=created_at,
            is_reply=is_reply,
            parent_comment_id=parent_comment_id,
        )

    def _iter_replies(comment_node: dict) -> Iterator[CommentRecord]:
        if not include_replies:
            return
        child_comment_count = comment_node.get("child_comment_count", 0)
        if child_comment_count == 0:
            return
        parent_pk = str(comment_node["pk"])
        preview = comment_node.get("preview_child_comments") or []
        if child_comment_count == len(preview):
            replies = preview
        else:
            answers_json = iphone_json_with_retry(
                ctx,
                f"api/v1/media/{mediaid}/comments/{comment_node['pk']}/child_comments/",
                {"max_id": ""},
                request_sleep=request_sleep,
            )
            replies = answers_json.get("child_comments") or []
        for child in replies:
            record = _comment_record(child, is_reply=True, parent_comment_id=parent_pk)
            if record:
                yield record

    def _paginated(comments_json: dict) -> Iterator[CommentRecord]:
        nonlocal page, fetched
        page += 1
        batch = comments_json.get("comments", [])
        for comment_node in batch:
            record = _comment_record(comment_node, is_reply=False, parent_comment_id=None)
            if record:
                fetched += 1
                yield record
            for reply in _iter_replies(comment_node):
                fetched += 1
                yield reply
        print(f"  Stranica {page}: +{len(batch)} komentara (ukupno {fetched})")
        next_min_id = comments_json.get("next_min_id")
        if next_min_id:
            yield from _paginated(_query(next_min_id))

    yield from _paginated(_query())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Instagram post comments.")
    parser.add_argument("url", nargs="?", help="Instagram post URL")
    parser.add_argument("--url", dest="url_flag", help="Instagram post URL")
    parser.add_argument("--shortcode", help="Post shortcode")
    parser.add_argument("--url-file", type=Path, help="Override default urls_instagram.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--username", help="Instagram username (legacy login)")
    parser.add_argument("--password", help="Instagram password (legacy login)")
    parser.add_argument(
        "--sessionid",
        help="Instagram sessionid cookie (dovoljan sam po sebi)",
    )
    parser.add_argument(
        "--cookie",
        action="append",
        default=[],
        metavar="IME=VREDNOST",
        help="Dodatni cookie, npr. --cookie sessionid=ABC",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help="Fajl sa cookies (ime=vrednost po liniji)",
    )
    parser.add_argument(
        "--browser",
        choices=["chrome", "firefox", "edge", "brave", "chromium"],
        help="Ucitaj sesiju iz browsera (na Windows-u cesto trazi admin)",
    )
    parser.add_argument(
        "--refresh-session",
        action="store_true",
        help="Ignorisi sacuvanu sesiju i ponovo ucitaj sessionid",
    )
    parser.add_argument("--no-login", action="store_true")
    parser.add_argument("--no-replies", action="store_true")
    parser.add_argument("--sleep", type=float, default=5.0, help="Pauza izmedju postova (s)")
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=1.5,
        help="Pauza izmedju Instagram API poziva (s)",
    )
    parser.add_argument("--id", dest="direct_id", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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
    except ValueError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    total = 0
    errors = 0
    for shortcode in items:
        try:
            print(f"\nFetching: {post_url(shortcode)}")
            comments = list(
                iter_comments_via_iphone(
                    loader,
                    shortcode,
                    include_replies=not args.no_replies,
                    request_sleep=args.request_sleep,
                )
            )
            txt, _js = save_comments(
                comments, args.output_dir, "instagram", shortcode, post_url(shortcode)
            )
            print(f"  Sacuvano {len(comments)} komentara -> {txt}")
            total += len(comments)
        except Exception as exc:
            errors += 1
            print(f"  Greska za {shortcode}: {exc}", file=sys.stderr)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\nUkupno sacuvano komentara: {total}")
    if errors:
        print(f"Greske na {errors} post(ova).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
