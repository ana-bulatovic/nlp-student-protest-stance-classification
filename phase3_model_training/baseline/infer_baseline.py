#!/usr/bin/env python3
"""Inferenca stance klasifikatora (baseline model sačuvan iz train_baseline.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import joblib
except ImportError:  # pragma: no cover
    from sklearn.externals import joblib  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PHASE3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_MODEL = SCRIPT_DIR / "output" / "baseline_model.joblib"


def load_bundle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Nema modela: {path}\n"
            "Prvo pokreni: python baseline/train_baseline.py --quick"
        )
    bundle = joblib.load(path)
    if "pipeline" not in bundle:
        raise ValueError(f"Neispravan model fajl: {path}")
    return bundle


def predict_texts(bundle: dict, texts: list[str]) -> list[dict]:
    pipe = bundle["pipeline"]
    preds = pipe.predict(texts)
    scores = None
    if hasattr(pipe, "decision_function"):
        try:
            scores = pipe.decision_function(texts)
        except Exception:
            scores = None
    elif hasattr(pipe, "predict_proba"):
        try:
            scores = pipe.predict_proba(texts)
        except Exception:
            scores = None

    classes = list(getattr(pipe, "classes_", bundle.get("labels", [])))
    out: list[dict] = []
    for i, (text, label) in enumerate(zip(texts, preds)):
        row: dict = {"text": text, "label": str(label)}
        if scores is not None:
            row_scores = scores[i]
            if hasattr(row_scores, "tolist"):
                row_scores = row_scores.tolist()
            if classes and len(classes) == len(row_scores):
                # decision_function: veći skor = jača klasa; prikaži po klasama
                paired = sorted(
                    zip(classes, row_scores), key=lambda x: x[1], reverse=True
                )
                row["scores"] = {str(c): float(s) for c, s in paired}
            else:
                row["scores"] = [float(s) for s in row_scores]
        out.append(row)
    return out


def read_input_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        for line in args.file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # dozvoli format text|url|label ili samo text
            if "|" in line:
                line = line.split("|", 1)[0].strip()
            if line:
                texts.append(line)
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferenca: predvidi stance za komentar(e)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Putanja do baseline_model.joblib",
    )
    parser.add_argument(
        "--text",
        "-t",
        action="append",
        default=[],
        help="Jedan komentar (moze se ponoviti)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Fajl sa komentarima (jedan po liniji)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interaktivni unos (prazan red = kraj)",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        bundle = load_bundle(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cfg = bundle.get("config") or {}
    print(
        f"Model: {args.model.name} | "
        f"{cfg.get('model', '?')} {cfg.get('weighting', '?')} "
        f"C={cfg.get('best_C', '?')} macro_f1={cfg.get('macro_f1', '?')}"
    )

    texts = read_input_texts(args)

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
            for row in predict_texts(bundle, [line]):
                _print_row(row)
        return 0

    for row in predict_texts(bundle, texts):
        _print_row(row)
    return 0


def _print_row(row: dict) -> None:
    print(f"[{row['label']}] {row['text']}")
    if "scores" in row and isinstance(row["scores"], dict):
        parts = [f"{k}={v:.3f}" for k, v in row["scores"].items()]
        print("  skorovi:", ", ".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
