#!/usr/bin/env python3
"""Izveštaj baseline eksperimenata (LR / SVM / NB × pretprocesiranje).

Čita baseline/output/baseline_results.json i pravi:
  - BASELINE_IZVESTAJ.md
  - output/baseline_analysis/*.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
DEFAULT_JSON = SCRIPT_DIR / "output" / "baseline_results.json"
OUT_DIR = PHASE3_DIR / "output" / "baseline_analysis"
REPORT = PHASE3_DIR / "BASELINE_IZVESTAJ.md"

LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")
LABEL_COLORS = {
    "NEUTRAL": "#6b7280",
    "ZA-VLAST": "#2563eb",
    "PROTIV-VLASTI": "#dc2626",
}
MODEL_ORDER = ("lr", "svm", "nb")
MODEL_NAMES = {
    "lr": "Logistička regresija",
    "svm": "Linear SVM",
    "nb": "Naivni Bajes",
}
WEIGHT_ORDER = ("tf", "idf", "tfidf")
NORM_ORDER = ("none", "stem", "lemma")
MODEL_COLORS = {"lr": "#0f766e", "svm": "#2563eb", "nb": "#d97706"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def save_fig(name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def cfg_label(r: dict) -> str:
    lc = "lc" if r["lowercase"] else "no-lc"
    return f"{r['model'].upper()} {r['weighting'].upper()} {lc} {r['normalize']}"


def mean_by(rows: list[dict], key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[str(r[key])].append(float(r["macro_f1"]))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def plot_ranking(ranked: list[dict], n: int = 15) -> Path:
    top = ranked[:n]
    labels = [cfg_label(r) for r in top][::-1]
    vals = [r["macro_f1"] for r in top][::-1]
    colors = [MODEL_COLORS.get(r["model"], "#64748b") for r in top][::-1]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(top))))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("macro-F1")
    ax.set_title(f"Baseline: top {len(top)} konfiguracija (macro-F1)")
    xmin = min(vals) - 0.02 if vals else 0
    ax.set_xlim(max(0, xmin), max(vals) * 1.04 if vals else 1)
    for y, v in enumerate(vals):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    return save_fig("01_ranking_macro_f1.png")


def plot_by_model(rows: list[dict]) -> Path:
    data, names, colors = [], [], []
    for m in MODEL_ORDER:
        vals = [r["macro_f1"] for r in rows if r["model"] == m]
        if vals:
            data.append(vals)
            names.append(MODEL_NAMES[m])
            colors.append(MODEL_COLORS[m])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    try:
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("macro-F1")
    ax.set_title("Raspodela macro-F1 po modelu (sve konfiguracije)")
    return save_fig("02_f1_by_model.png")


def plot_grouped(rows: list[dict], key: str, order: tuple[str, ...], title: str, fname: str) -> Path:
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for i, m in enumerate(MODEL_ORDER):
        vals = []
        for k in order:
            subset = [r["macro_f1"] for r in rows if r["model"] == m and str(r[key]) == str(k)]
            vals.append(float(np.mean(subset)) if subset else 0.0)
        ax.bar(x + (i - 1) * width, vals, width, label=MODEL_NAMES[m], color=MODEL_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([str(k).upper() if key != "lowercase" else ("da" if k in (True, "True", "1") else "ne") for k in order])
    if key == "lowercase":
        ax.set_xticklabels(["uključen", "isključen"])
    ax.set_ylabel("prosečan macro-F1")
    ax.set_title(title)
    ax.legend()
    return save_fig(fname)


def plot_heatmap(rows: list[dict]) -> Path:
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    weights = [w for w in WEIGHT_ORDER if any(r["weighting"] == w for r in rows)]
    mat = np.zeros((len(models), len(weights)))
    for i, m in enumerate(models):
        for j, w in enumerate(weights):
            vals = [r["macro_f1"] for r in rows if r["model"] == m and r["weighting"] == w]
            mat[i, j] = float(np.max(vals)) if vals else np.nan
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([w.upper() for w in weights])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_NAMES[m] for m in models])
    ax.set_title("Najbolji macro-F1: model × ponderisanje")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="macro-F1")
    return save_fig("05_heatmap_model_weighting.png")


def plot_best_per_class(best: dict) -> Path:
    labs = list(LABELS)
    vals = [best["per_class_f1"].get(l, 0.0) for l in labs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labs, vals, color=[LABEL_COLORS[l] for l in labs])
    ax.set_ylabel("F1")
    ax.set_title(f"Najbolja konfiguracija — F1 po klasi\n({cfg_label(best)})")
    ax.set_ylim(0, 1.05)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    return save_fig("06_best_per_class_f1.png")


def plot_best_per_model_class(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(LABELS))
    width = 0.25
    for i, m in enumerate(MODEL_ORDER):
        subset = [r for r in rows if r["model"] == m]
        if not subset:
            continue
        best = max(subset, key=lambda r: r["macro_f1"])
        vals = [best["per_class_f1"].get(l, 0.0) for l in LABELS]
        ax.bar(x + (i - 1) * width, vals, width, label=MODEL_NAMES[m], color=MODEL_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(list(LABELS))
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Najbolja konfiguracija svakog modela — F1 po klasi")
    ax.legend()
    return save_fig("07_best_model_per_class.png")


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(payload: dict, figs: list[Path]) -> str:
    rows: list[dict] = payload["results"]
    ranked = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    best = ranked[0]
    worst = ranked[-1]
    n = payload.get("n_samples", "?")
    labels = payload.get("label_counts", {})
    outer = payload.get("outer_folds", "?")
    inner = payload.get("inner_folds", "?")

    by_model = mean_by(rows, "model")
    by_w = mean_by(rows, "weighting")
    by_n = mean_by(rows, "normalize")
    by_lc = mean_by(rows, "lowercase")

    best_by_model = []
    for m in MODEL_ORDER:
        subset = [r for r in rows if r["model"] == m]
        if subset:
            best_by_model.append(max(subset, key=lambda r: r["macro_f1"]))

    lines = [
        "# Baseline izveštaj (Faza 3.1)",
        "",
        "Klasični modeli: logistička regresija, linear SVM, multinomialni naivni Bajes. "
        "Eksperimentalni faktori: ponderisanje (**TF / IDF / TF-IDF**), **lowercasing**, "
        "normalizacija tokena (**none / stem / lemma**). "
        "Evaluacija: ugnežđena stratifikovana unakrsna validacija, glavna metrika **macro-F1**.",
        "",
        "## 1. Postavka",
        "",
        *md_table(
            ["Stavka", "Vrednost"],
            [
                ["Skup", "`dataset_all.txt`"],
                ["Broj primera", str(n)],
                ["Klase", ", ".join(f"{k}={v}" for k, v in labels.items())],
                ["Spoljašnji foldovi", str(outer)],
                ["Unutrašnji foldovi (hiperparametri)", str(inner)],
                ["Broj konfiguracija", str(len(rows))],
                ["Hiperparametri LR/SVM", str(payload.get("C_grid", ""))],
                ["Hiperparametri NB (alpha)", str(payload.get("alpha_grid", ""))],
            ],
        ),
        "",
        payload.get("note", ""),
        "",
        "## 2. Najbolji rezultat",
        "",
        f"**Pobednik:** `{cfg_label(best)}`",
        "",
        *md_table(
            ["Metrika", "Vrednost"],
            [
                ["macro-F1", f"**{best['macro_f1']:.4f}**"],
                ["accuracy", f"{best['accuracy']:.4f}"],
                ["weighted-F1", f"{best['weighted_f1']:.4f}"],
                ["Najbolji hiperparametri", str(best.get("best_params", {}))],
                *[[f"F1 `{lab}`", f"{best['per_class_f1'].get(lab, 0):.4f}"] for lab in LABELS],
            ],
        ),
        "",
        f"Najslabija konfiguracija: `{cfg_label(worst)}` (macro-F1={worst['macro_f1']:.4f}). "
        f"Raspon: **{best['macro_f1'] - worst['macro_f1']:.3f}** poena macro-F1.",
        "",
        f"![Ranking](output/baseline_analysis/{figs[0].name})",
        "",
        "## 3. Top 10 konfiguracija",
        "",
        *md_table(
            ["#", "Model", "Ponder", "Lowercase", "Norm", "macro-F1", "acc", "F1 NEUTRAL", "F1 ZA-VLAST", "F1 PROTIV"],
            [
                [
                    str(i),
                    r["model"].upper(),
                    r["weighting"].upper(),
                    "da" if r["lowercase"] else "ne",
                    r["normalize"],
                    f"{r['macro_f1']:.4f}",
                    f"{r['accuracy']:.4f}",
                    f"{r['per_class_f1'].get('NEUTRAL', 0):.3f}",
                    f"{r['per_class_f1'].get('ZA-VLAST', 0):.3f}",
                    f"{r['per_class_f1'].get('PROTIV-VLASTI', 0):.3f}",
                ]
                for i, r in enumerate(ranked[:10], 1)
            ],
        ),
        "",
        "## 4. Uticaj faktora (prosečan macro-F1)",
        "",
        "### 4.1 Model",
        "",
        *md_table(
            ["Model", "Prosečan macro-F1", "Najbolji macro-F1"],
            [
                [
                    MODEL_NAMES[m],
                    f"{by_model.get(m, 0):.4f}",
                    f"{max(r['macro_f1'] for r in rows if r['model'] == m):.4f}",
                ]
                for m in MODEL_ORDER
                if any(r["model"] == m for r in rows)
            ],
        ),
        "",
        f"![Po modelu](output/baseline_analysis/{figs[1].name})",
        "",
        "### 4.2 Ponderisanje (TF / IDF / TF-IDF)",
        "",
        *md_table(
            ["Ponder", "Prosečan macro-F1"],
            [[w.upper(), f"{by_w.get(w, 0):.4f}"] for w in WEIGHT_ORDER if w in by_w],
        ),
        "",
        f"![Ponder](output/baseline_analysis/{figs[2].name})",
        "",
        f"![Heatmap](output/baseline_analysis/{figs[4].name})",
        "",
        "### 4.3 Normalizacija tokena",
        "",
        *md_table(
            ["Normalizacija", "Prosečan macro-F1"],
            [[k, f"{by_n.get(k, 0):.4f}"] for k in NORM_ORDER if k in by_n],
        ),
        "",
        f"![Norm](output/baseline_analysis/{figs[3].name})",
        "",
        "### 4.4 Lowercasing",
        "",
        *md_table(
            ["Lowercasing", "Prosečan macro-F1"],
            [
                ["uključen", f"{by_lc.get('True', by_lc.get(True, 0)):.4f}"],
                ["isključen", f"{by_lc.get('False', by_lc.get(False, 0)):.4f}"],
            ],
        ),
        "",
        "## 5. F1 po klasi",
        "",
        f"![Best class](output/baseline_analysis/{figs[5].name})",
        "",
        f"![Best model class](output/baseline_analysis/{figs[6].name})",
        "",
        "Najbolja konfiguracija po modelu:",
        "",
        *md_table(
            ["Model", "Konfiguracija", "macro-F1", "acc"],
            [
                [
                    MODEL_NAMES[r["model"]],
                    cfg_label(r),
                    f"{r['macro_f1']:.4f}",
                    f"{r['accuracy']:.4f}",
                ]
                for r in best_by_model
            ],
        ),
        "",
        "## 6. Zaključak za izveštaj",
        "",
        f"- Od **{len(rows)}** isprobanih konfiguracija najbolja je "
        f"**{MODEL_NAMES[best['model']]}** sa **{best['weighting'].upper()}**, "
        f"normalize=`{best['normalize']}`, lowercase={'da' if best['lowercase'] else 'ne'} "
        f"(macro-F1 = **{best['macro_f1']:.3f}**, acc = {best['accuracy']:.3f}).",
        f"- Najteža klasa po F1 kod pobednika: "
        f"**{min(LABELS, key=lambda l: best['per_class_f1'].get(l, 1.0))}** "
        f"({min(best['per_class_f1'].values()):.3f}).",
        "- TF-IDF nije unapred proglašen pobednikom: upoređeni su TF, IDF i TF-IDF ravnopravno.",
        "- Ove brojke su **donja granica** za Fazu 3; enkoder (BERTić / mBERT) treba da ih nadmaši po macro-F1.",
        "",
        "## 7. Fajlovi",
        "",
        "- Sirovi rezultati: `baseline/output/baseline_results.json`",
        "- Classification report-i: `baseline/output/baseline_results.txt`",
    ]
    for fig in figs:
        lines.append(f"- `output/baseline_analysis/{fig.name}`")
    lines += [
        "",
        "Regenerisanje (kad postoji JSON):",
        "",
        "```bash",
        "cd phase3_model_training",
        "python baseline/report_baseline.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def generate(json_path: Path) -> int:
    if not json_path.is_file():
        print(
            f"Nema rezultata: {json_path}\n"
            "Prvo pokreni puni trening:\n"
            "  python baseline/train_baseline.py",
            flush=True,
        )
        return 1
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("results") or []
    if not rows:
        print("JSON nema 'results' — trening nije upisao konfiguracije.", flush=True)
        return 1

    setup_style()
    ranked = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    lc_present = sorted({bool(r["lowercase"]) for r in rows}, reverse=True)

    figs = [
        plot_ranking(ranked),
        plot_by_model(rows),
        plot_grouped(
            rows, "weighting", WEIGHT_ORDER,
            "Prosečan macro-F1 po ponderisanju",
            "03_f1_by_weighting.png",
        ),
        plot_grouped(
            rows, "normalize", NORM_ORDER,
            "Prosečan macro-F1 po normalizaciji tokena",
            "04_f1_by_normalize.png",
        ),
        plot_heatmap(rows),
        plot_best_per_class(ranked[0]),
        plot_best_per_model_class(rows),
    ]
    if len(lc_present) > 1:
        plot_grouped(
            rows, "lowercase", (True, False),
            "Prosečan macro-F1: lowercasing",
            "08_f1_by_lowercase.png",
        )

    REPORT.write_text(build_report(payload, figs), encoding="utf-8")
    print(f"Izvestaj: {REPORT}")
    print(f"Grafike: {OUT_DIR}")
    print(f"Konfiguracija: {len(rows)}; najbolji macro-F1={ranked[0]['macro_f1']:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Grafike i Markdown iz baseline_results.json")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    return generate(args.json)


if __name__ == "__main__":
    raise SystemExit(main())
