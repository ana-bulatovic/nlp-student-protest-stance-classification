#!/usr/bin/env python3
"""Faza 3.1 dopuna — uticaj opsega n-grama i frekvencijskog filtriranja.

Umesto ukrštanja pune mreže (model x weighting x lowercase x normalize x
ngram x filtriranje, što bi eksplodiralo na stotine nested-CV pokretanja),
ablacija se radi na POBEDNIČKOJ (weighting/lowercase/normalize) konfiguraciji
SVAKOG modela iz baseline_results.json, menjajući samo jednu osu odjednom:

  - opseg n-grama: unigram / unigram+bigram (trenutno) / +trigram
  - frekvencijsko filtriranje: bez / trenutno (min_df=2, max_df=0.95) / strože

Po model (lr/svm/nb) x 3 podešavanja x 2 ose = 18 dodatnih nested-CV pokretanja.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PHASE3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.data import DEFAULT_DATA, load_dataset  # noqa: E402
from train_baseline import evaluate_config  # noqa: E402

DEFAULT_BEST_JSON = SCRIPT_DIR / "output" / "baseline_results.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "ablation_ngram_freq_results.json"

NGRAM_SETTINGS: list[tuple[str, tuple[int, int]]] = [
    ("unigram", (1, 1)),
    ("unigram+bigram (trenutno)", (1, 2)),
    ("unigram+bigram+trigram", (1, 3)),
]
FREQ_SETTINGS: list[tuple[str, int, float]] = [
    ("bez filtriranja", 1, 1.0),
    ("trenutno (min_df=2, max_df=0.95)", 2, 0.95),
    ("strože (min_df=5, max_df=0.90)", 5, 0.90),
]


def best_configs_per_model(best_json: Path) -> dict[str, dict]:
    """Pobednička (weighting/lowercase/normalize) po modelu iz pune mreže."""
    payload = json.loads(best_json.read_text(encoding="utf-8"))
    best: dict[str, dict] = {}
    for r in payload["results"]:
        m = r["model"]
        if m not in best or r["macro_f1"] > best[m]["macro_f1"]:
            best[m] = r
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablacija opsega n-grama i frekvencijskog filtriranja, na "
            "pobedničkoj konfiguraciji svakog modela (ne cela mreža ponovo)."
        ),
        epilog=(
            "Primeri (iz phase3_model_training/):\n"
            "  python baseline/ablation_ngram_freq.py --quick\n"
            "  python baseline/ablation_ngram_freq.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--best-json",
        type=Path,
        default=DEFAULT_BEST_JSON,
        help="Rezultati pune mreže (train_baseline.py) — odatle se čita pobednik po modelu.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "svm", "nb"],
        choices=["lr", "svm", "nb"],
    )
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Brzi test: samo LR, 3 folda, po jedan C.",
    )
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
    if not args.best_json.is_file():
        print(
            f"Nema {args.best_json}.\n"
            "Prvo pokreni punu mrežu da bi se odredila pobednička "
            "konfiguracija po modelu:\n"
            "  python baseline/train_baseline.py",
            file=sys.stderr,
        )
        return 1

    texts, labels = load_dataset(args.data)
    print(f"Ucitano {len(texts)} primera iz {args.data}")
    print("Distribucija:", dict(Counter(labels)))

    best = best_configs_per_model(args.best_json)
    models = list(args.models)
    outer_folds = args.outer_folds
    inner_folds = args.inner_folds
    C_grid = [0.1, 1.0, 10.0]
    alpha_grid = [0.1, 0.5, 1.0]

    if args.quick:
        models = ["lr"]
        outer_folds = 3
        inner_folds = 2
        C_grid = [1.0]
        alpha_grid = [0.5]
        print("\n=== QUICK MODE (samo LR, 3 folda) ===")

    missing = [m for m in models if m not in best]
    if missing:
        print(
            f"Nema pobedničke konfiguracije za {missing} u {args.best_json}",
            file=sys.stderr,
        )
        return 1

    counts = Counter(labels)
    min_class = min(counts.values())
    if min_class < outer_folds:
        print(
            f"Upozorenje: najmanja klasa ima {min_class} primera, "
            f"outer_folds={outer_folds}. Smanjujem na {min_class}.",
            file=sys.stderr,
        )
        outer_folds = min_class

    ngram_rows: list[dict] = []
    freq_rows: list[dict] = []
    reports: list[str] = []

    for model in models:
        cfg = best[model]
        weighting, lowercase, normalize = (
            cfg["weighting"],
            cfg["lowercase"],
            cfg["normalize"],
        )
        print(
            f"\n=== {model.upper()} — pobednička konfiguracija: "
            f"{weighting}/lc={lowercase}/{normalize} "
            f"(macro_f1={cfg['macro_f1']:.4f} iz pune mreže) ===",
            flush=True,
        )

        print("  -- Opseg n-grama --", flush=True)
        for label, ngram_range in NGRAM_SETTINGS:
            print(f"    {label} {ngram_range} ...", flush=True)
            result, report, _ = evaluate_config(
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
                ngram_range=ngram_range,
            )
            print(
                f"      macro_f1={result.macro_f1:.4f} acc={result.accuracy:.4f}",
                flush=True,
            )
            row = asdict(result)
            row["ablation_axis"] = "ngram_range"
            row["setting_label"] = label
            ngram_rows.append(row)
            reports.append(f"### {model.upper()} — n-grami: {label}\n\n{report}")

        print("  -- Frekvencijsko filtriranje --", flush=True)
        for label, min_df, max_df in FREQ_SETTINGS:
            print(f"    {label} ...", flush=True)
            result, report, _ = evaluate_config(
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
                min_df=min_df,
                max_df=max_df,
            )
            print(
                f"      macro_f1={result.macro_f1:.4f} acc={result.accuracy:.4f}",
                flush=True,
            )
            row = asdict(result)
            row["ablation_axis"] = "freq_filter"
            row["setting_label"] = label
            freq_rows.append(row)
            reports.append(f"### {model.upper()} — filtriranje: {label}\n\n{report}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": str(args.data),
        "best_json": str(args.best_json),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "ngram_settings": [
            {"label": lab, "ngram_range": list(r)} for lab, r in NGRAM_SETTINGS
        ],
        "freq_settings": [
            {"label": lab, "min_df": mn, "max_df": mx}
            for lab, mn, mx in FREQ_SETTINGS
        ],
        "base_config_per_model": {
            m: {
                "weighting": best[m]["weighting"],
                "lowercase": best[m]["lowercase"],
                "normalize": best[m]["normalize"],
                "macro_f1": best[m]["macro_f1"],
            }
            for m in models
        },
        "ngram_ablation": ngram_rows,
        "freq_filter_ablation": freq_rows,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output.with_suffix(".txt").write_text(
        "\n\n".join(reports) + "\n", encoding="utf-8"
    )

    print(f"\nRezultati: {args.output}")
    print(f"Izvestaji: {args.output.with_suffix('.txt')}")

    print("\n=== Rangiranje — opseg n-grama (macro-F1) ===")
    for r in sorted(ngram_rows, key=lambda r: r["macro_f1"], reverse=True):
        print(f"  {r['model']:4s} {r['setting_label']:30s} macro_f1={r['macro_f1']:.4f}")
    print("\n=== Rangiranje — frekvencijsko filtriranje (macro-F1) ===")
    for r in sorted(freq_rows, key=lambda r: r["macro_f1"], reverse=True):
        print(f"  {r['model']:4s} {r['setting_label']:34s} macro_f1={r['macro_f1']:.4f}")

    try:
        from report_ablation_ngram_freq import generate as write_report

        print("\nGenerisem grafike i izveštaj za ablaciju ...", flush=True)
        write_report(args.output)
    except Exception as exc:
        print(f"Upozorenje: izveštaj nije napravljen ({exc})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
