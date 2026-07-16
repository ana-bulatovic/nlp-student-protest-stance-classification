#!/usr/bin/env python3
"""Download comments (replies) from public X (Twitter) posts."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import re
import sys
from datetime import timezone
from pathlib import Path
from typing import AsyncIterator

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
    from twikit import Client
    from twikit.errors import TwitterException
except ImportError:
    print("Error: twikit not installed. Run: pip install twikit", file=sys.stderr)
    sys.exit(1)


def _patch_twikit_api_bugs() -> None:
    """Nuklearna globalna zakrpa.
    Presreće ceo Python json.loads mehanizam. Svaki dict koji twikit povuče sa mreže
    se u startu rađa kao SafeDict, čineći aplikaciju 100% imunom na nedostatak ključeva.
    """

    class SafeDict(dict):
        def __getitem__(self, key):
            try:
                val = super().__getitem__(key)
            except KeyError:
                return SafeDict()
            if isinstance(val, dict) and not isinstance(val, SafeDict):
                return SafeDict(val)
            return val

        def get(self, key, default=None):
            if key in self:
                val = self[key]
                if isinstance(val, dict) and not isinstance(val, SafeDict):
                    return SafeDict(val)
                return val
            return SafeDict() if default is None else default

        def __str__(self):
            return ""

        def __repr__(self):
            return ""

        def __int__(self):
            return 0

        def __float__(self):
            return 0.0

        def __bool__(self):
            return False

        # Zaštita od konkatenacije i matematičkih operacija
        def __add__(self, other):
            return str(self) + str(other)

        def __radd__(self, other):
            return str(other) + str(self)

        # Zaštita od iteracija kroz nepostojeće liste/rečnike
        def __iter__(self):
            return iter([])

        def append(self, *args, **kwargs):
            pass

        def extend(self, *args, **kwargs):
            pass

        def insert(self, *args, **kwargs):
            pass

        # Zaštita od tekstualnih metoda
        def strip(self, *args, **kwargs):
            return ""

        def lower(self, *args, **kwargs):
            return ""

        def upper(self, *args, **kwargs):
            return ""

        def split(self, *args, **kwargs):
            return []

        def replace(self, *args, **kwargs):
            return ""

        def startswith(self, *args, **kwargs):
            return False

        def endswith(self, *args, **kwargs):
            return False

    # Presretanje json.loads funkcije na nivou celog runtime-a
    orig_loads = json.loads

    def patched_loads(s, *args, **kwargs):
        if 'object_hook' not in kwargs:
            kwargs['object_hook'] = lambda d: SafeDict(d)
        res = orig_loads(s, *args, **kwargs)
        if isinstance(res, dict) and not isinstance(res, SafeDict):
            return SafeDict(res)
        return res

    json.loads = patched_loads


# Pokretanje svemoguće JSON zakrpe pre bilo kakvog API poziva
_patch_twikit_api_bugs()


def _patch_twikit_key_byte_indices() -> None:
    """Opciona zakrpa za poznati twikit bug: 'Couldn't get KEY_BYTE indices'."""
    import re as _re

    try:
        tx_mod = sys.modules.get("twikit.x_client_transaction.transaction")
        if tx_mod is None:
            import twikit.x_client_transaction.transaction as tx_mod  # type: ignore
    except ImportError:
        return

    on_demand_file_regex = _re.compile(
        r""",(\d+):["']ondemand\.s["']""", flags=(_re.VERBOSE | _re.MULTILINE)
    )
    on_demand_hash_pattern = r',{}:"([0-9a-f]+)"'
    indices_regex = _re.compile(r"\[(\w{1,2})\[(\d{1,2})\],\s*16\]")

    async def _patched_get_indices(self, home_page_response, session, headers):
        key_byte_indices: list[str] = []
        response = self.validate_response(home_page_response) or self.home_page_response
        response_str = str(response)

        on_demand_file = on_demand_file_regex.search(response_str)
        if on_demand_file:
            on_demand_file_index = on_demand_file.group(1)
            hash_regex = _re.compile(on_demand_hash_pattern.format(on_demand_file_index))
            hash_match = hash_regex.search(response_str)
            if hash_match:
                filename = hash_match.group(1)
                on_demand_file_url = (
                    "https://abs.twimg.com/responsive-web/client-web/"
                    f"ondemand.s.{filename}a.js"
                )
                on_demand_file_response = await session.request(
                    method="GET", url=on_demand_file_url, headers=headers
                )
                for item in indices_regex.finditer(str(on_demand_file_response.text)):
                    key_byte_indices.append(item.group(2))

        if not key_byte_indices:
            raise Exception("Couldn't get KEY_BYTE indices")

        key_byte_indices_int = list(map(int, key_byte_indices))
        return key_byte_indices_int[0], key_byte_indices_int[1:]

    tx_mod.ClientTransaction.get_indices = _patched_get_indices  # type: ignore[attr-defined]


if os.environ.get("TWIKIT_APPLY_KEY_BYTE_PATCH") == "1":
    _patch_twikit_key_byte_indices()

TWEET_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|mobile\.)?(?:twitter|x)\.com/[^/]+/status(?:es)?/(\d+)",
    re.IGNORECASE,
)

DEFAULT_OUTPUT = output_dir("x")
DEFAULT_SESSION = session_dir("x")
DEFAULT_COOKIES_FILE = DEFAULT_SESSION / "cookies.json"


def extract_tweet_id(value: str) -> str:
    value = value.strip()
    match = TWEET_URL_PATTERN.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", value):
        return value
    raise ValueError(f"Cannot parse X/Twitter post from: {value!r}")


def post_url(tweet_id: str) -> str:
    return f"https://x.com/i/status/{tweet_id}"


def load_unique_items(args: argparse.Namespace) -> list[str]:
    items = load_input_urls(
        args.url,
        args.url_flag,
        args.url_file,
        args.tweet_id or args.direct_id,
        default_url_file=urls_file("x"),
    )
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        tweet_id = extract_tweet_id(item)
        if tweet_id not in seen:
            seen.add(tweet_id)
            unique.append(tweet_id)
    return unique


async def create_client(
        username: str | None,
        email: str | None,
        password: str | None,
        login: bool,
        cookies_file: Path | None,
        refresh_session: bool,
) -> Client:
    client = Client("en-US")
    if not login:
        return client

    DEFAULT_SESSION.mkdir(parents=True, exist_ok=True)
    cookies_path = cookies_file or DEFAULT_COOKIES_FILE

    if not refresh_session and cookies_path.exists():
        try:
            with open(cookies_path, "r", encoding="utf-8") as f:
                cookie_data = json.load(f)

            if isinstance(cookie_data, list):
                print(f"Detektovan sirovi browser format u {cookies_path.name}. Vrši se automatska konverzija...")
                twikit_cookies = {}
                for cookie in cookie_data:
                    name = cookie.get("name")
                    value = cookie.get("value")
                    if name and value:
                        twikit_cookies[name] = value

                client.set_cookies(twikit_cookies)
                client.save_cookies(str(cookies_path))

            elif isinstance(cookie_data, dict):
                client.load_cookies(str(cookies_path))

            try:
                await client.user()
                print(f"Ucitana i verifikovana postojeca X sesija iz: {cookies_path}")
            except Exception:
                print(f"Ucitana X sesija iz: {cookies_path}")

            return client
        except Exception as exc:
            print(f"Sacuvana X sesija nije validna ili je Cloudflare blokira ({exc}), ponovo se logujem...")

    username = username or os.environ.get("X_USERNAME") or os.environ.get("TWITTER_USERNAME")
    email = email or os.environ.get("X_EMAIL") or os.environ.get("TWITTER_EMAIL")
    password = password or os.environ.get("X_PASSWORD") or os.environ.get("TWITTER_PASSWORD")

    if not username:
        raise ValueError(
            "X login je obavezan (X ne dozvoljava citanje repliesa bez naloga).\n"
            "Najsigurnije rešenje da izbegneš Cloudflare blokadu:\n"
            "  1. Uloguj se u browseru na x.com\n"
            "  2. Eksportuj kolačiće kao JSON preko ekstenzije (npr. Cookie-Editor)\n"
            f"  3. Sačuvaj taj JSON direktno u fajl: {cookies_path.resolve()}\n"
            "  4. Pokreni skriptu ponovo bez kredencijala."
        )

    if not password:
        password = getpass.getpass(f"X password for @{username}: ")

    try:
        await client.login(
            auth_info_1=username,
            auth_info_2=email or username,
            password=password,
            cookies_file=str(cookies_path),
        )
    except TwitterException as exc:
        raise ValueError(
            f"X login greska: {exc}\n"
            "Ako X trazi verifikacioni kod, uloguj se rucno u browseru, izvezi cookies\n"
            f"i sacuvaj ih kao JSON u {cookies_path}."
        ) from exc

    client.save_cookies(str(cookies_path))
    print(f"Ulogovano kao @{username}, sesija sacuvana u: {cookies_path}")
    return client


async def call_with_retry(coro_fn, *, max_attempts: int = 4, base_sleep: float = 5.0):
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except TwitterException as exc:
            last_exc = exc
            if getattr(exc, 'status', None) == 422 or "GRAPHQL_VALIDATION_FAILED" in str(exc):
                raise exc
            if attempt + 1 >= max_attempts:
                break
            wait = base_sleep * (attempt + 1)
            print(f"  X API greska ({exc}), ponovo za {wait:.0f}s...")
            await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


async def iter_comments(
        client: Client,
        tweet_id: str,
        request_sleep: float,
        max_comments: int = 0,
) -> AsyncIterator[CommentRecord]:
    url = post_url(tweet_id)

    async def _get_tweet():
        return await client.get_tweet_by_id(tweet_id)

    tweet = await call_with_retry(_get_tweet)
    total_expected = getattr(tweet, "reply_count", None)
    if total_expected:
        print(f"  Ocekivano replies: ~{total_expected}")

    replies = getattr(tweet, "replies", None)
    page = 0
    fetched = 0
    seen_comment_ids = set()

    # Faza 1: Pokušaj standardnog prikupljanja kroz Tweet detalje (Stranica 1)
    while replies:
        page += 1
        batch = list(replies)
        for reply in batch:
            reply_id = str(reply.id)
            text = (getattr(reply, "full_text", None) or getattr(reply, "text", None) or "").strip()
            if not text:
                continue
            author = getattr(reply, "user", None)
            created_at = ""
            dt = getattr(reply, "created_at_datetime", None)
            if dt is not None:
                created_at = dt.astimezone(timezone.utc).isoformat()

            seen_comment_ids.add(reply_id)
            yield CommentRecord(
                id=reply_id,
                source="x",
                post_url=url,
                post_id=tweet_id,
                text=text,
                author=getattr(author, "screen_name", "") if author else "",
                created_at=created_at,
                is_reply=True,
                parent_comment_id=tweet_id,
            )
            fetched += 1
            if max_comments and fetched >= max_comments:
                print(f"  Stranica {page}: +{len(batch)} replies (ukupno {fetched}, limit dostignut)")
                return
        print(f"  Stranica {page}: +{len(batch)} replies (ukupno {fetched})")

        if not batch:
            break

        if request_sleep > 0:
            await asyncio.sleep(request_sleep)

        async def _next():
            return await replies.next()

        try:
            replies = await call_with_retry(_next)
        except Exception as exc:
            if getattr(exc, 'status', None) == 422 or "GRAPHQL_VALIDATION_FAILED" in str(exc):
                print("  [Info] Kursor blokiran (422). Aktiviram Search API zamenu za preostale komentare...")
            else:
                print(f"  [Info] Paginacija prekinuta: {exc}")
            break

    # Faza 2: Rezervni plan (Search API Fallback) za duboko grebanje preostalih strana
    if not max_comments or fetched < max_comments:
        try:
            # Tražimo sve tvitove u okviru ove konverzacije (konverzacijska nit)
            search_query = f"conversation_id:{tweet_id}"
            search_results = await client.search_tweet(search_query, 'Latest')

            search_page = 0
            while search_results:
                search_batch = list(search_results)
                if not search_batch:
                    break

                search_page += 1
                new_fetched_in_batch = 0

                for reply in search_batch:
                    reply_id = str(reply.id)

                    # Preskačemo ako smo komentar već skinuli u Fazi 1 ili ako vrati sam korenski tvit
                    if reply_id in seen_comment_ids or reply_id == tweet_id:
                        continue

                    text = (getattr(reply, "full_text", None) or getattr(reply, "text", None) or "").strip()
                    if not text:
                        continue

                    author = getattr(reply, "user", None)
                    created_at = ""
                    dt = getattr(reply, "created_at_datetime", None)
                    if dt is not None:
                        created_at = dt.astimezone(timezone.utc).isoformat()

                    yield CommentRecord(
                        id=reply_id,
                        source="x",
                        post_url=url,
                        post_id=tweet_id,
                        text=text,
                        author=getattr(author, "screen_name", "") if author else "",
                        created_at=created_at,
                        is_reply=True,
                        parent_comment_id=tweet_id,
                    )
                    seen_comment_ids.add(reply_id)
                    fetched += 1
                    new_fetched_in_batch += 1

                    if max_comments and fetched >= max_comments:
                        print(
                            f"  Search Stranica {search_page}: +{new_fetched_in_batch} replies (ukupno {fetched}, limit dostignut)")
                        return

                if new_fetched_in_batch > 0:
                    print(
                        f"  Search Stranica {search_page}: +{new_fetched_in_batch} dodatnih replies (ukupno {fetched})")

                if request_sleep > 0:
                    await asyncio.sleep(request_sleep)

                async def _next_search():
                    return await search_results.next()

                try:
                    search_results = await call_with_retry(_next_search)
                except Exception:
                    break  # Kraj pretrage ili limit dostignut

        except Exception as search_exc:
            print(f"  [Info] Search zamena završila s radom ili nema više dubljih komentara: {search_exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download X (Twitter) post replies (comments).")
    parser.add_argument("url", nargs="?", help="X/Twitter post URL")
    parser.add_argument("--url", dest="url_flag", help="X/Twitter post URL")
    parser.add_argument("--tweet-id", help="Tweet ID")
    parser.add_argument("--url-file", type=Path, help="Override default urls_x.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--username", help="X username/handle")
    parser.add_argument("--email", help="Email naloga")
    parser.add_argument("--password", help="X lozinka")
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help=f"Fajl za cuvanje/ucitavanje sesije (JSON)",
    )
    parser.add_argument(
        "--refresh-session",
        action="store_true",
        help="Ignorisi sacuvanu sesiju i ponovo se uloguj",
    )
    parser.add_argument("--no-login", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sleep", type=float, default=5.0, help="Pauza izmedju postova (s)")
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=2.0,
        help="Pauza izmedju X API poziva (s)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Max replies po tweetu",
    )
    parser.add_argument("--id", dest="direct_id", help=argparse.SUPPRESS)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
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
    except ValueError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    total = 0
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
            txt, _js = save_comments(comments, args.output_dir, "x", tweet_id, post_url(tweet_id))
            print(f"  Sacuvano {len(comments)} replies -> {txt}")
            total += len(comments)
        except Exception as exc:
            errors += 1
            print(f"  Greska za {tweet_id}: {exc}", file=sys.stderr)
        if args.sleep > 0 and tweet_id != items[-1]:
            await asyncio.sleep(args.sleep)

    print(f"\nUkupno sacuvano replies: {total}")
    if errors:
        print(f"Greske na {errors} post(ova).", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())