#!/usr/bin/env python3
"""Inferenca enkoderskog modela (Hugging Face / train_encoder.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "output" / "encoder_best"
LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferenca: BERTić/mBERT (HF Trainer).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text", "-t", action="append", default=[])
    parser.add_argument("--file", "-f", type=Path)
    parser.add_argument("--interactive", "-i", action="store_true")
    return parser.parse_args()


def load_model(model_dir: Path):
    import torch
    import st_compat
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    st_compat.apply()

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Nema modela: {model_dir}\n"
            "Prvo pokreni: python encoder/train_encoder.py --quick"
        )

    meta: dict = {}
    meta_path = model_dir / "stance_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    if use_cuda:
        model = model.to("cuda")
    return (model, tokenizer, meta, use_cuda)


def predict_texts(bundle, texts: list[str]) -> list[dict]:
    import numpy as np
    import torch

    model, tokenizer, meta, use_cuda = bundle
    max_length = int(meta.get("max_length", 128))
    labels = tuple(meta.get("labels") or LABELS)
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if use_cuda:
        enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()

    rows: list[dict] = []
    for text, arr in zip(texts, logits):
        pred_i = int(np.argmax(arr))
        label = labels[pred_i] if pred_i < len(labels) else labels[0]
        exp = np.exp(arr - np.max(arr))
        probs = exp / exp.sum()
        score_map = {
            labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))
        }
        score_map = dict(sorted(score_map.items(), key=lambda x: x[1], reverse=True))
        rows.append({"text": text, "label": label, "scores": score_map})
    return rows


def read_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        for line in args.file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                line = line.split("|", 1)[0].strip()
            if line:
                texts.append(line)
    return texts


def print_row(row: dict) -> None:
    print(f"[{row['label']}] {row['text']}")
    parts = [f"{k}={v:.3f}" for k, v in row["scores"].items()]
    print("  verovatnoce:", ", ".join(parts))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        bundle = load_model(args.model_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Greska pri ucitavanju modela: {exc}", file=sys.stderr)
        return 1

    model, _tok, meta, use_cuda = bundle
    cfg = meta.get("best_config") or {}
    print(
        f"Model: {args.model_dir.name} | HF Trainer | "
        f"{cfg.get('model_key', meta.get('model_key', '?'))} "
        f"epochs={cfg.get('epochs', meta.get('epochs', '?'))} "
        f"cv_macro_f1={meta.get('cv_macro_f1', '?')} | "
        f"device={'cuda' if use_cuda else 'cpu'}"
    )

    texts = read_texts(args)
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
            print_row(predict_texts(bundle, [line])[0])
        return 0

    for row in predict_texts(bundle, texts):
        print_row(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
