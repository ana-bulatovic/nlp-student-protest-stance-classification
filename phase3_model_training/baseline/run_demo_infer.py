#!/usr/bin/env python3
"""Brzi demo inferencije na ~15 primera (kratki + duzi).

Pokretanje iz phase3_model_training/:

    python baseline/run_demo_infer.py
    python baseline/run_demo_infer.py --encoder
    python baseline/run_demo_infer.py --no-train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
ENCODER_DIR_PKG = PHASE3_DIR / "encoder"

SAMPLES = PHASE3_DIR / "samples" / "demo_comments.txt"
BASELINE_MODEL = SCRIPT_DIR / "output" / "baseline_model.joblib"
ENCODER_DIR = ENCODER_DIR_PKG / "output" / "encoder_best"


def load_samples(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        texts.append(line)
    return texts


def ensure_baseline(auto_train: bool) -> bool:
    if BASELINE_MODEL.is_file():
        return True
    if not auto_train:
        print(
            f"Nema modela: {BASELINE_MODEL}\n"
            "Pokreni: python baseline/train_baseline.py --quick",
            file=sys.stderr,
        )
        return False
    print("Nema baseline modela — pokrecem brzi trening (--quick)...\n")
    code = subprocess.call(
        [sys.executable, str(SCRIPT_DIR / "train_baseline.py"), "--quick"],
        cwd=str(SCRIPT_DIR),
    )
    return code == 0 and BASELINE_MODEL.is_file()


def run_baseline(texts: list[str]) -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from infer import load_bundle, predict_texts, _print_row

    bundle = load_bundle(BASELINE_MODEL)
    cfg = bundle.get("config") or {}
    print("=" * 60)
    print(
        f"BASELINE | {cfg.get('model', '?')} {cfg.get('weighting', '?')} "
        f"| macro_f1={cfg.get('macro_f1', float('nan')):.4f}"
        if isinstance(cfg.get("macro_f1"), (int, float))
        else f"BASELINE | {cfg.get('model', '?')}"
    )
    print("=" * 60)
    for i, row in enumerate(predict_texts(bundle, texts), 1):
        print(f"\n--- primer {i}/{len(texts)} ---")
        _print_row(row)


def run_encoder(texts: list[str]) -> None:
    if str(ENCODER_DIR_PKG) not in sys.path:
        sys.path.insert(0, str(ENCODER_DIR_PKG))
    from infer_encoder import load_model, predict_texts, print_row

    if not ENCODER_DIR.is_dir():
        print(
            f"Nema encoder modela: {ENCODER_DIR}\n"
            "Preskacem. (Opciono: python encoder/train_encoder.py --quick)",
            file=sys.stderr,
        )
        return
    model, meta, use_cuda = load_model(ENCODER_DIR)
    cfg = meta.get("best_config") or {}
    print("\n" + "=" * 60)
    print(
        f"ENCODER (Simple Transformers) | "
        f"{cfg.get('model_key', meta.get('model_key', '?'))} "
        f"epochs={cfg.get('epochs', meta.get('epochs', '?'))} | "
        f"device={'cuda' if use_cuda else 'cpu'}"
    )
    print("=" * 60)
    for i, row in enumerate(predict_texts(model, texts), 1):
        print(f"\n--- primer {i}/{len(texts)} ---")
        print_row(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo inferenca na samples/demo_comments.txt"
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=SAMPLES,
        help="Fajl sa komentarima (jedan po liniji)",
    )
    p.add_argument(
        "--encoder",
        action="store_true",
        help="Pokreni i encoder inferencu (ako postoji model)",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Ne pokreci automatski train_baseline.py --quick",
    )
    return p.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.samples.is_file():
        print(f"Nema fajla sa primerima: {args.samples}", file=sys.stderr)
        return 1

    texts = load_samples(args.samples)
    if not texts:
        print("Nema komentara u samples fajlu.", file=sys.stderr)
        return 1

    print(f"Ucitano {len(texts)} demo primera iz {args.samples.name}\n")

    if not ensure_baseline(auto_train=not args.no_train):
        return 1

    run_baseline(texts)

    if args.encoder:
        run_encoder(texts)

    print("\nGotovo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
