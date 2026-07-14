#!/usr/bin/env python3
from twikit import Client
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Uvozimo centralizovanu definiciju foldera za sesije iz zajedničkog modula
from common import session_dir  # noqa: E402

# --- OVDE PASE-UJ VREDNOSTI KOJE SI IZVUKLA IZ BROWSERA ---
AUTH_TOKEN = "7c314b77a625bfa4631dd95bc7b5a0080ad7fb74"
CT0 = "9adf544a252a16ecbea705851e627964038d41a639af670848ed91db0bbd0de7b491a5ff05a483801cbb382e94857d15d0f1efe7948ee876ac891ea2a13a2e71722df9bb17e8ed5a3600b7f792f3fcff"


# ---------------------------------------------------------

def main():
    client = Client("en-US")

    # Ručno postavljamo kolačiće koje X koristi za aktivnu sesiju
    client.set_cookies({
        'auth_token': AUTH_TOKEN,
        'ct0': CT0
    })

    # Koristimo identičnu putanju koju koristi download skripta
    x_session_dir = session_dir("x")
    x_session_dir.mkdir(parents=True, exist_ok=True)
    cookies_file = x_session_dir / "cookies.json"

    # Twikit formatira i čuva kolačiće u svom internom JSON formatu
    client.save_cookies(str(cookies_file))
    print(f"\n[USPEH] Sesija je uspešno kreirana i sačuvana u:")
    print(f"-> {cookies_file.resolve()}")


if __name__ == "__main__":
    main()