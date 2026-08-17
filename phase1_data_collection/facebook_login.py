#!/usr/bin/env python3
"""One-time interactive Facebook login helper.

Opens a real Chromium browser (via Playwright) so you can log into Facebook
by hand - including solving any 2FA/checkpoint challenges - and then saves
the authenticated session to disk. download_facebook_comments.py reuses that
saved session, so you only need to run this again once it expires or you log
out.

Usage:
    python3 facebook_login.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import session_dir  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    print("Then run: python3 -m playwright install chromium", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    out_dir = session_dir("facebook")
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_state_path = out_dir / "storage_state.json"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.facebook.com/", wait_until="domcontentloaded")

        print("\nOtvoren je pravi Chrome prozor.")
        print("1. Uloguj se na svoj Facebook nalog u tom prozoru (reši i eventualni kod za potvrdu / 2FA ako se traži).")
        print("2. Kada vidiš svoj feed / naslovnu stranicu, vrati se ovde u terminal.")
        input("3. Pritisni ENTER ovde kada si ulogovana... ")

        context.storage_state(path=str(storage_state_path))
        browser.close()

    print(f"\nSesija je sačuvana u: {storage_state_path}")
    print("Ovaj fajl je osetljiv (kao lozinka) - nemoj ga deliti ni commit-ovati u git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
