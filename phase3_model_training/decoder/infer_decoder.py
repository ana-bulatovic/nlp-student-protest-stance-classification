#!/usr/bin/env python3
"""Inferenca dekoderskog LLM (prompting) — jedan ili više komentara."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from decoder_prompts import LABELS, build_messages  # noqa: E402
from eval_decoder import (  # noqa: E402
    GeminiClient,
    OllamaClient,
    OpenAIClient,
    parse_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferenca: Ollama / ChatGPT / Gemini prompting za stance."
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai", "gemini"],
    )
    parser.add_argument("--ollama-model", default="llama2")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash")
    parser.add_argument("--lang", default="sr", choices=["sr", "en"])
    parser.add_argument("--style", default="detailed", choices=["short", "detailed"])
    parser.add_argument("--shot", default="few", choices=["zero", "few"])
    parser.add_argument("--text", "-t", action="append", default=[])
    parser.add_argument("--file", "-f", type=Path)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Ispisi prompt pre poziva API-ja",
    )
    return parser.parse_args()


def make_client(args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaClient(args.ollama_model, host=args.ollama_host), args.ollama_model
    if args.provider == "openai":
        return OpenAIClient(args.openai_model), args.openai_model
    return GeminiClient(args.gemini_model), args.gemini_model


def read_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        for line in args.file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                line = line.split("|", 1)[0].strip()
            if line:
                texts.append(line)
    return texts


def predict_one(client, text: str, lang: str, style: str, shot: str) -> dict:
    messages = build_messages(text, lang, style, shot)
    raw = client.complete(messages)
    label = parse_label(raw)
    return {
        "text": text,
        "label": label or "?",
        "raw": raw,
        "parsed_ok": label is not None,
    }


def print_row(row: dict) -> None:
    print(f"[{row['label']}] {row['text']}")
    if row["raw"] and row["raw"].strip().upper() != row["label"]:
        print(f"  raw: {row['raw']!r}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        client, model_name = make_client(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"Provider: {args.provider}/{model_name} | "
        f"lang={args.lang} style={args.style} shot={args.shot}-shot | "
        f"klase={', '.join(LABELS)}"
    )

    texts = read_texts(args)

    def run_one(text: str) -> None:
        if args.show_prompt:
            for m in build_messages(text, args.lang, args.style, args.shot):
                print(f"[{m['role']}]\n{m['content']}\n")
        print_row(predict_one(client, text, args.lang, args.style, args.shot))

    if args.interactive or not texts:
        print("Unesi komentar (prazan red za kraj):")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                break
            run_one(line)
        return 0

    for text in texts:
        run_one(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
