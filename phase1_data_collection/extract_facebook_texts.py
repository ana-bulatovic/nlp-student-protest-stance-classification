#!/usr/bin/env python3
"""Spoji komentare iz output/facebook u jedan "text|url" fajl za anotiranje.

Za svaki facebook_*.txt fajl koristi se istoimeni .json (isti podaci, ali
pouzdaniji za citanje jer komentar moze imati vise redova). Ako .json ne
postoji, .txt se parsira direktno.

Izlaz je pipe-delimited fajl sa zaglavljem "text|url", u formatu koji cita
annotation_tool.html.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_facebook import clean_comment_text, extract_post_id  # noqa: E402

DEFAULT_INPUT_DIR = SCRIPT_DIR / "output" / "facebook"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "facebook_all_texts.txt"
DEFAULT_URLS_FILE = SCRIPT_DIR / "urls_facebook.txt"

# Fajlovi sa vec anotiranim komentarima (za --exclude-final). Zive u Fazi 2,
# jer ih tamo ocekuje build_dataset_all.py.
ANNOTATED_DIR = SCRIPT_DIR.parent / "phase2_annotation" / "annotated"
FINAL_FILES = (
    ANNOTATED_DIR / "fb_final_za_vlast_annotated.txt",
    ANNOTATED_DIR / "fb_final_protiv_vlasti_annotated.txt",
    ANNOTATED_DIR / "fb_final_neutral_annotated.txt",
)

# Zaglavlje komentara u .txt izvozu: "[vreme] Ime:" ili "  ↳ reply: [vreme] Ime:"
_TXT_COMMENT_HEADER_RE = re.compile(r"^\s*(?:↳\s*reply:\s*)?\[(?P<time>[^\]]*)\]\s*(?P<author>.+?):\s*$")


def read_json_file(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fallback_url = (data.get("post_url") or "").strip()
    pairs: list[tuple[str, str]] = []
    for comment in data.get("comments") or []:
        text = clean_comment_text(comment.get("text") or "", comment.get("author") or "")
        if not text:
            continue
        pairs.append((text, (comment.get("post_url") or fallback_url).strip()))
    return pairs


def read_txt_file(path: Path) -> list[tuple[str, str]]:
    post_url = ""
    pairs: list[tuple[str, str]] = []
    current: list[str] | None = None
    current_author = ""

    def flush() -> None:
        if current is None:
            return
        text = clean_comment_text("\n".join(current), current_author)
        if text:
            pairs.append((text, post_url))

    for line in path.read_text(encoding="utf-8").splitlines():
        if not post_url and line.startswith("Post:"):
            post_url = line[len("Post:"):].strip()
            continue
        if line.startswith(("Fetched:", "Total comments:")) or set(line.strip()) == {"="}:
            continue

        header = _TXT_COMMENT_HEADER_RE.match(line)
        if header:
            flush()
            current = []
            current_author = header.group("author").strip()
            continue
        if current is not None:
            current.append(line)

    flush()
    return pairs


def load_already_annotated() -> set[str]:
    """Normalizovani tekstovi komentara koji su vec u fb_final_*.txt fajlovima."""
    done: set[str] = set()
    for path in FINAL_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                fields = next(csv.reader(io.StringIO(line), delimiter="|"))
            except csv.Error:
                fields = line.split("|")
            if fields and fields[0].strip():
                done.add(" ".join(fields[0].lower().split()))
    return done


def load_active_post_ids(path: Path) -> dict[str, str]:
    """{post_id: link} za linkove koji nisu zakomentarisani u urls fajlu."""
    wanted: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        wanted[extract_post_id(line)] = line
    return wanted


def filter_to_active_urls(
    pairs: list[tuple[str, str]], urls_path: Path
) -> list[tuple[str, str]]:
    wanted = load_active_post_ids(urls_path)
    if not wanted:
        print(f"Greska: nema nijednog aktivnog linka u {urls_path}", file=sys.stderr)
        return []

    kept: list[tuple[str, str]] = []
    per_post: dict[str, int] = {post_id: 0 for post_id in wanted}
    for text, url in pairs:
        post_id = extract_post_id(url)
        if post_id in wanted:
            kept.append((text, url))
            per_post[post_id] += 1

    print(f"Aktivnih linkova u {urls_path.name}: {len(wanted)}")
    for post_id, link in wanted.items():
        print(f"  {per_post[post_id]:4}  {link}")
    empty = [link for post_id, link in wanted.items() if per_post[post_id] == 0]
    if empty:
        print(
            f"PAZI: {len(empty)} aktivnih linkova nema nijedan skinut komentar "
            "(objava jos nije preuzeta ili nema komentara)."
        )
    return kept


def collect_pairs(input_dir: Path, output_path: Path, dedupe: bool) -> list[tuple[str, str]]:
    txt_files = sorted(input_dir.glob("facebook_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nema facebook_*.txt fajlova u {input_dir}")

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for txt_path in txt_files:
        # Spojeni izvoz zivi u istom folderu i hvata ga isti glob.
        if txt_path.name.startswith("facebook_all") or txt_path.resolve() == output_path.resolve():
            continue
        json_path = txt_path.with_suffix(".json")
        try:
            file_pairs = read_json_file(json_path) if json_path.exists() else read_txt_file(txt_path)
        except (OSError, ValueError) as exc:
            print(f"Preskacem {txt_path.name}: {exc}", file=sys.stderr)
            continue

        for text, url in file_pairs:
            if dedupe:
                key = (" ".join(text.lower().split()), url)
                if key in seen:
                    continue
                seen.add(key)
            pairs.append((text, url))

    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spoji komentare iz output/facebook u jedan text|url fajl."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder sa facebook_*.txt fajlovima (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Izlazni fajl (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Ne uklanjaj duplikate (isti tekst na istoj objavi iz vise pokretanja).",
    )
    parser.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Zadrzi prelome redova unutar komentara (podrazumevano se spajaju u jedan red).",
    )
    parser.add_argument(
        "--exclude-final",
        action="store_true",
        help="Izostavi komentare koji su vec u fb_final_*.txt fajlovima "
        "(da ne anotiras dvaput iste).",
    )
    parser.add_argument(
        "--only-urls",
        nargs="?",
        const=DEFAULT_URLS_FILE,
        type=Path,
        default=None,
        metavar="FAJL",
        help="Uzmi samo komentare sa objava cijih linkova ima medju aktivnim "
        f"(nezakomentarisanim) redovima u {DEFAULT_URLS_FILE.name}, ili u zadatom fajlu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pairs = collect_pairs(args.input_dir, args.output, dedupe=not args.no_dedupe)
    except FileNotFoundError as exc:
        print(f"Greska: {exc}", file=sys.stderr)
        return 1

    if args.only_urls:
        if not args.only_urls.exists():
            print(f"Greska: nema fajla {args.only_urls}", file=sys.stderr)
            return 1
        pairs = filter_to_active_urls(pairs, args.only_urls)
        print()

    if args.exclude_final:
        done = load_already_annotated()
        before = len(pairs)
        pairs = [(t, u) for t, u in pairs if " ".join(t.lower().split()) not in done]
        print(f"Izostavljeno {before - len(pairs)} vec anotiranih komentara.")

    if not pairs:
        print("Nema komentara za izvoz.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text", "url"])
        for text, url in pairs:
            if not args.keep_newlines:
                text = " ".join(text.split())
            writer.writerow([text, url])

    urls = {url for _, url in pairs if url}
    print(f"Procitano iz: {args.input_dir}")
    print(f"Objava: {len(urls)}")
    print(f"Sacuvano {len(pairs)} komentara -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
