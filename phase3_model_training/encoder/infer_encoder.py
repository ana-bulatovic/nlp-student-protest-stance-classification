#!/usr/bin/env python3
"""Inferenca enkoderskog modela (Hugging Face / train_encoder.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_KEYS = ("bertic", "mbert")
LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")


def default_model_dir(model_key: str) -> Path:
    return SCRIPT_DIR / "output" / f"encoder_{model_key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferenca: BERTić / mBERT (HF Trainer).",
        epilog=(
            "Primeri:\n"
            '  python encoder/infer_encoder.py --model bertic -t "Pumpaj!"\n'
            '  python encoder/infer_encoder.py --model mbert -t "Pumpaj!"\n'
            "  python encoder/infer_encoder.py --model bertic -i\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=MODEL_KEYS,
        default="bertic",
        help="Koji sačuvani model (default: bertic → output/encoder_bertic)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Eksplicitni folder (overrides --model)",
    )
    parser.add_argument("--text", "-t", action="append", default=[])
    parser.add_argument("--file", "-f", type=Path)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Sacuvaj rezultate u CSV (npr. output/bertic_demo.csv)",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Stampaj preglednu tabelu (komentar | labela | verovatnoce)",
    )
    return parser.parse_args()


def resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir is not None:
        return args.model_dir
    return default_model_dir(args.model)


def load_model(model_dir: Path):
    import torch
    import st_compat
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    st_compat.apply()

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Nema modela: {model_dir}\n"
            "Prvo pokreni: python encoder/train_encoder.py --compare\n"
            "  (ili --model bertic / --model mbert)"
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
            if not line or line.startswith("#"):
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


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'#':<3} {'LABEL':<16} {'P(label)':>8}  KOMENTAR")
    print("-" * 100)
    for i, row in enumerate(rows, 1):
        p = row["scores"].get(row["label"], 0.0)
        text = row["text"].replace("\n", " ")
        if len(text) > 70:
            text = text[:67] + "..."
        print(f"{i:<3} {row['label']:<16} {p:>8.3f}  {text}")
    print("-" * 100)
    print(f"Ukupno: {len(rows)} komentara")


def save_csv(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    label_keys = list(LABELS)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["rbr", "komentar", "predikcija", "P_pred"] + [f"P_{k}" for k in label_keys])
        for i, row in enumerate(rows, 1):
            writer.writerow(
                [
                    i,
                    row["text"],
                    row["label"],
                    f"{row['scores'].get(row['label'], 0.0):.4f}",
                    *[f"{row['scores'].get(k, 0.0):.4f}" for k in label_keys],
                ]
            )
    print(f"CSV: {path}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    model_dir = resolve_model_dir(args)
    try:
        bundle = load_model(model_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Greska pri ucitavanju modela: {exc}", file=sys.stderr)
        return 1

    model, _tok, meta, use_cuda = bundle
    cfg = meta.get("best_config") or {}
    print(
        f"Model: {model_dir.name} | HF Trainer | "
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

    rows = predict_texts(bundle, texts)
    if args.table or args.csv is not None or args.file is not None:
        print_table(rows)
    else:
        for row in rows:
            print_row(row)

    if args.csv is not None:
        save_csv(rows, args.csv)
    elif args.file is not None:
        # podrazumevani CSV pored rezultata
        out = SCRIPT_DIR / "output" / f"infer_{args.model}_demo.csv"
        save_csv(rows, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
