#!/usr/bin/env python3
"""Faza 3.1 — osnovni (baseline) modeli za stance klasifikaciju.

Modeli: logistička regresija, Linear SVM, naivni Bajes (MultinomialNB).

Pretprocesiranje / reprezentacija (eksperimentalni faktori, bez podrazumevanog
„pobednika“): lowercasing, TF / IDF / TF-IDF, stemovanje, lematizacija.

Evaluacija: 10-slojna stratifikovana CV + ugnežđena CV za hiperparametre.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    import joblib
except ImportError:  # pragma: no cover
    from sklearn.externals import joblib  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PHASE3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.data import DEFAULT_DATA, LABELS, load_dataset  # noqa: E402
from text_preprocess import TextNormalizer  # noqa: E402

DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "baseline_results.json"
DEFAULT_MODEL = SCRIPT_DIR / "output" / "baseline_model.joblib"


@dataclass
class FoldResult:
    model: str
    weighting: str
    lowercase: bool
    normalize: str
    best_params: dict
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]


def make_vectorizer(weighting: str, lowercase: bool):
    """Jedna od tehnika reprezentacije: TF, IDF ili TF-IDF (ravnopravne opcije)."""
    common = dict(
        lowercase=lowercase,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    if weighting == "tf":
        return CountVectorizer(**common)
    if weighting == "tfidf":
        return TfidfVectorizer(**common, use_idf=True, norm="l2")
    if weighting == "idf":
        return TfidfVectorizer(**common, use_idf=True, binary=True, norm=None)
    raise ValueError(f"Nepoznato ponderisanje: {weighting}")


def make_classifier(model: str):
    if model == "lr":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="auto",
            random_state=42,
        )
    if model == "svm":
        return LinearSVC(max_iter=5000, dual="auto", random_state=42)
    if model == "nb":
        # MultinomialNB: klasičan naivni Bajes za tekst (najbolje uz TF brojače)
        return MultinomialNB()
    raise ValueError(f"Nepoznat model: {model}")


def param_grid_for(model: str, C_grid: list[float], alpha_grid: list[float]) -> dict:
    if model in {"lr", "svm"}:
        return {"clf__C": C_grid}
    if model == "nb":
        return {"clf__alpha": alpha_grid}
    raise ValueError(f"Nepoznat model: {model}")


def evaluate_config(
    texts: list[str],
    labels: list[str],
    model: str,
    weighting: str,
    lowercase: bool,
    normalize: str,
    outer_folds: int,
    inner_folds: int,
    C_grid: list[float],
    alpha_grid: list[float],
    seed: int,
) -> tuple[FoldResult, str, Pipeline]:
    y = np.array(labels)
    pipe = Pipeline(
        [
            ("norm", TextNormalizer(mode=normalize)),
            ("vec", make_vectorizer(weighting, lowercase)),
            ("clf", make_classifier(model)),
        ]
    )
    param_grid = param_grid_for(model, C_grid, alpha_grid)

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=inner,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )

    y_pred = cross_val_predict(search, texts, y, cv=outer, n_jobs=-1)

    search.fit(texts, y)
    best_params = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in search.best_params_.items()}
    fitted_pipe: Pipeline = search.best_estimator_

    acc = float(accuracy_score(y, y_pred))
    macro = float(f1_score(y, y_pred, average="macro", labels=list(LABELS)))
    weighted = float(f1_score(y, y_pred, average="weighted", labels=list(LABELS)))
    per = f1_score(y, y_pred, average=None, labels=list(LABELS))
    per_class = {lab: float(v) for lab, v in zip(LABELS, per)}

    report = classification_report(
        y, y_pred, labels=list(LABELS), digits=4, zero_division=0
    )
    result = FoldResult(
        model=model,
        weighting=weighting,
        lowercase=lowercase,
        normalize=normalize,
        best_params=best_params,
        accuracy=acc,
        macro_f1=macro,
        weighted_f1=weighted,
        per_class_f1=per_class,
    )
    return result, report, fitted_pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treniranje baseline modela (LR / SVM / Naive Bayes). "
            "TF/IDF/TF-IDF, lowercasing, stem i lema su pretprocesiranje."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "svm", "nb"],
        choices=["lr", "svm", "nb"],
        help="Osnovni modeli (nb = Multinomial Naive Bayes)",
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["tf", "idf", "tfidf"],
        choices=["tf", "idf", "tfidf"],
        help="Tehnike ponderisanja (sve ravnopravne; TF-IDF nije default)",
    )
    parser.add_argument(
        "--lowercase",
        nargs="+",
        type=int,
        default=[1, 0],
        help="1 = lowercasing uključen, 0 = isključen",
    )
    parser.add_argument(
        "--normalize",
        nargs="+",
        default=["none", "stem", "lemma"],
        choices=["none", "stem", "lemma"],
        help="Normalizacija tokena: none / stemovanje / lematizacija",
    )
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Brzi test: 3 folda; LR+SVM+NB; ponderisanje TF "
            "(ne TF-IDF); lowercase; normalize=none"
        ),
    )
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.data.is_file():
        print(f"Nema dataset fajla: {args.data}", file=sys.stderr)
        return 1

    texts, labels = load_dataset(args.data)
    print(f"Ucitano {len(texts)} primera iz {args.data}")
    print("Distribucija:", dict(Counter(labels)))

    models = list(args.models)
    weightings = list(args.weightings)
    lowercase_opts = [bool(x) for x in args.lowercase]
    normalize_opts = list(args.normalize)
    outer_folds = args.outer_folds
    inner_folds = args.inner_folds
    C_grid = [0.1, 1.0, 10.0]
    alpha_grid = [0.1, 0.5, 1.0]

    if args.quick:
        # TF-IDF nije podrazumevan: quick koristi TF + sva 3 modela
        models = ["lr", "svm", "nb"]
        weightings = ["tf"]
        lowercase_opts = [True]
        normalize_opts = ["none"]
        outer_folds = min(3, outer_folds)
        inner_folds = 2
        C_grid = [1.0, 10.0]
        alpha_grid = [0.5, 1.0]
        print("\n=== QUICK MODE (LR+SVM+NB, TF, bez stem/leme) ===")

    if "lemma" in normalize_opts:
        try:
            import simplemma  # noqa: F401
        except ImportError:
            print(
                "Upozorenje: simplemma nije instaliran — "
                "uklanjam 'lemma' iz --normalize. "
                "pip install simplemma",
                file=sys.stderr,
            )
            normalize_opts = [n for n in normalize_opts if n != "lemma"]
            if not normalize_opts:
                normalize_opts = ["none"]

    counts = Counter(labels)
    min_class = min(counts.values())
    if min_class < outer_folds:
        print(
            f"Upozorenje: najmanja klasa ima {min_class} primera, "
            f"a outer_folds={outer_folds}. Smanjujem outer_folds na {min_class}.",
            file=sys.stderr,
        )
        outer_folds = min_class

    results: list[dict] = []
    reports: list[str] = []
    fitted_by_key: dict[tuple, Pipeline] = {}

    total = len(models) * len(weightings) * len(lowercase_opts) * len(normalize_opts)
    print(
        f"Kombinacija: {len(models)} modela x {len(weightings)} ponderisanja x "
        f"{len(lowercase_opts)} lowercase x {len(normalize_opts)} normalize "
        f"= {total} konfiguracija"
    )

    for model in models:
        for weighting in weightings:
            for lowercase in lowercase_opts:
                for normalize in normalize_opts:
                    tag = (
                        f"{model.upper()} | {weighting.upper()} | "
                        f"lc={'da' if lowercase else 'ne'} | norm={normalize}"
                    )
                    print(f"\n--- {tag} ---")
                    result, report, fitted = evaluate_config(
                        texts=texts,
                        labels=labels,
                        model=model,
                        weighting=weighting,
                        lowercase=lowercase,
                        normalize=normalize,
                        outer_folds=outer_folds,
                        inner_folds=inner_folds,
                        C_grid=C_grid,
                        alpha_grid=alpha_grid,
                        seed=args.seed,
                    )
                    print(
                        f"best={result.best_params}  acc={result.accuracy:.4f}  "
                        f"macro_f1={result.macro_f1:.4f}"
                    )
                    print(report)
                    results.append(asdict(result))
                    reports.append(f"### {tag}\n\n{report}")
                    fitted_by_key[(model, weighting, lowercase, normalize)] = fitted

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": str(args.data),
        "n_samples": len(texts),
        "label_counts": dict(Counter(labels)),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "C_grid": C_grid,
        "alpha_grid": alpha_grid,
        "note": (
            "TF/IDF/TF-IDF, lowercasing, stem i lemma su tehnike pretprocesiranja; "
            "nijedna nije podrazumevani 'default model'."
        ),
        "results": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_path = args.output.with_suffix(".txt")
    report_path.write_text("\n\n".join(reports) + "\n", encoding="utf-8")

    ranked = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    print("\n=== Rangiranje (macro-F1) ===")
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. {r['model']} {r['weighting']} lc={r['lowercase']} "
            f"norm={r['normalize']}: macro_f1={r['macro_f1']:.4f} "
            f"acc={r['accuracy']:.4f}"
        )

    if not args.no_save_model and ranked:
        best = ranked[0]
        key = (
            best["model"],
            best["weighting"],
            best["lowercase"],
            best["normalize"],
        )
        best_pipe = fitted_by_key[key]
        bundle = {
            "pipeline": best_pipe,
            "labels": list(LABELS),
            "config": best,
            "data": str(args.data),
        }
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.model_out)
        print(f"Model za inferencu: {args.model_out}")

    print(f"\nRezultati: {args.output}")
    print(f"Izvestaji: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
