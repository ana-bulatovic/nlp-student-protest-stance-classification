#!/usr/bin/env python3
"""Izveštaj enkoderskih eksperimenata (BERTić / mBERT).

Čita encoder/output/encoder_results*.json i pravi:
  - ENCODER_IZVESTAJ.md
  - output/encoder_analysis/*.png  (poređenje, F1 po klasi, foldovi, matrice)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
DEFAULT_JSON = SCRIPT_DIR / "output" / "encoder_results_compare_e4.json"
OUT_DIR = PHASE3_DIR / "output" / "encoder_analysis"
REPORT = PHASE3_DIR / "ENCODER_IZVESTAJ.md"

LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")
LABEL_COLORS = {
    "NEUTRAL": "#6b7280",
    "ZA-VLAST": "#2563eb",
    "PROTIV-VLASTI": "#dc2626",
}
MODEL_ORDER = ("bertic", "mbert")
MODEL_NAMES = {
    "bertic": "BERTić",
    "mbert": "mBERT",
}
MODEL_COLORS = {"bertic": "#0f766e", "mbert": "#7c3aed"}


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
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    return path


def best_per_model(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for key in MODEL_ORDER:
        subset = [r for r in rows if r.get("model_key") == key]
        if subset:
            out.append(max(subset, key=lambda r: float(r["macro_f1"])))
    if not out:
        # nepoznati ključevi — uzmi najbolji po svakom model_key
        seen: dict[str, dict] = {}
        for r in rows:
            k = str(r.get("model_key", "?"))
            if k not in seen or float(r["macro_f1"]) > float(seen[k]["macro_f1"]):
                seen[k] = r
        out = list(seen.values())
    return out


def plot_compare_metrics(bests: list[dict]) -> Path:
    metrics = ["macro_f1", "accuracy", "weighted_f1"]
    titles = ["macro-F1", "accuracy", "weighted-F1"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, r in enumerate(bests):
        key = r["model_key"]
        vals = [float(r[m]) for m in metrics]
        offset = (i - (len(bests) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=MODEL_NAMES.get(key, key),
            color=MODEL_COLORS.get(key, "#64748b"),
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("vrednost")
    ax.set_title("Poređenje enkodera (najbolja konfiguracija po modelu)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return save_fig("01_compare_metrics.png")


def plot_per_class(bests: list[dict]) -> Path:
    x = np.arange(len(LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, r in enumerate(bests):
        key = r["model_key"]
        vals = [float(r.get("per_class_f1", {}).get(lab, 0.0)) for lab in LABELS]
        offset = (i - (len(bests) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=MODEL_NAMES.get(key, key),
            color=MODEL_COLORS.get(key, "#64748b"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(LABELS))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title("F1 po klasi — BERTić vs mBERT")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    return save_fig("02_per_class_f1.png")


def plot_fold_macro(bests: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    data, names, colors = [], [], []
    for r in bests:
        folds = r.get("fold_macro_f1") or []
        if not folds:
            continue
        key = r["model_key"]
        data.append([float(v) for v in folds])
        names.append(MODEL_NAMES.get(key, key))
        colors.append(MODEL_COLORS.get(key, "#64748b"))
    if not data:
        ax.text(0.5, 0.5, "Nema fold_macro_f1 u JSON-u", ha="center", va="center")
        ax.set_axis_off()
        return save_fig("03_fold_macro_f1.png")
    try:
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("macro-F1 po foldu")
    ax.set_title("Raspodela macro-F1 po CV foldovima")
    return save_fig("03_fold_macro_f1.png")


def plot_confusion(r: dict, index: int) -> Path:
    key = r["model_key"]
    cm = np.array(r.get("confusion_matrix") or [], dtype=float)
    if cm.size == 0:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.text(0.5, 0.5, "Nema confusion_matrix", ha="center", va="center")
        ax.set_axis_off()
        return save_fig(f"0{3 + index}_cm_{key}.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(list(LABELS), rotation=30, ha="right")
    ax.set_yticklabels(list(LABELS))
    ax.set_xlabel("predikcija")
    ax.set_ylabel("stvarna klasa")
    ax.set_title(
        f"Matrica konfuzije — {MODEL_NAMES.get(key, key)} "
        f"(epochs={r.get('epochs', '?')})"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_fig(f"0{3 + index}_cm_{key}.png")


def plot_ranking(rows: list[dict]) -> Path:
    ranked = sorted(rows, key=lambda r: float(r["macro_f1"]))
    labels = [
        f"{MODEL_NAMES.get(r['model_key'], r['model_key'])} e={r['epochs']}"
        for r in ranked
    ]
    vals = [float(r["macro_f1"]) for r in ranked]
    colors = [MODEL_COLORS.get(r["model_key"], "#64748b") for r in ranked]
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.45 * len(ranked))))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("macro-F1")
    ax.set_title("Rangiranje encoder konfiguracija")
    for y, v in enumerate(vals):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=9)
    xmin = min(vals) - 0.03 if vals else 0
    ax.set_xlim(max(0, xmin), max(vals) * 1.06 if vals else 1)
    return save_fig("06_ranking_macro_f1.png")


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(payload: dict, figs: list[Path], bests: list[dict]) -> str:
    rows: list[dict] = payload["results"]
    ranked = sorted(rows, key=lambda r: float(r["macro_f1"]), reverse=True)
    winner = ranked[0]
    n = payload.get("n_samples", "?")
    labels = payload.get("label_counts", {})
    folds = payload.get("folds", "?")

    winner_name = MODEL_NAMES.get(winner["model_key"], winner["model_key"])
    lines = [
        "# Enkoder izveštaj (Faza 3.2a)",
        "",
        "Fine-tuning preko Hugging Face Trainer: **BERTić** (`classla/bcms-bertic`) "
        "i **mBERT** (`bert-base-multilingual-cased`). "
        "Evaluacija: stratifikovana **k-fold** unakrsna validacija; "
        "glavna metrika **macro-F1**. Posle CV-a svaki model se trenira na celom "
        "skupu i čuva u odvojenom folderu za inferencu.",
        "",
        "## 1. Postavka",
        "",
        *md_table(
            ["Stavka", "Vrednost"],
            [
                ["Skup", "`dataset_all.txt`"],
                ["Broj primera", str(n)],
                ["Klase", ", ".join(f"{k}={v}" for k, v in labels.items())],
                ["Foldovi (CV)", str(folds)],
                ["Batch size", str(payload.get("batch_size", "?"))],
                ["Learning rate", str(payload.get("lr", "?"))],
                ["Max length", str(payload.get("max_length", "?"))],
                ["Uređaj", str(payload.get("device", "?"))],
                ["Broj konfiguracija", str(len(rows))],
            ],
        ),
        "",
        "## 2. Ko je bolji?",
        "",
        f"**Pobednik (macro-F1):** **{winner_name}** "
        f"(epochs={winner.get('epochs')}, macro-F1 = **{winner['macro_f1']:.4f}**, "
        f"acc = {winner['accuracy']:.4f}).",
        "",
    ]

    if len(bests) >= 2:
        a, b = bests[0], bests[1]
        delta = float(a["macro_f1"]) - float(b["macro_f1"])
        lead = a if delta >= 0 else b
        trail = b if delta >= 0 else a
        lines += [
            *md_table(
                ["Model", "epochs", "macro-F1", "acc", "weighted-F1", "folder"],
                [
                    [
                        MODEL_NAMES.get(r["model_key"], r["model_key"]),
                        str(r.get("epochs", "?")),
                        f"{r['macro_f1']:.4f}",
                        f"{r['accuracy']:.4f}",
                        f"{r['weighted_f1']:.4f}",
                        f"`{Path(r.get('model_dir') or '').name or '—'}`",
                    ]
                    for r in bests
                ],
            ),
            "",
            f"Razlika macro-F1 ({MODEL_NAMES.get(lead['model_key'], lead['model_key'])} − "
            f"{MODEL_NAMES.get(trail['model_key'], trail['model_key'])}): "
            f"**{abs(float(lead['macro_f1']) - float(trail['macro_f1'])):.4f}**.",
            "",
        ]

    fig_names = {p.name: p.name for p in figs}
    lines += [
        f"![Poređenje metrika](output/encoder_analysis/{fig_names.get('01_compare_metrics.png', '01_compare_metrics.png')})",
        "",
        f"![Rangiranje](output/encoder_analysis/{fig_names.get('06_ranking_macro_f1.png', '06_ranking_macro_f1.png')})",
        "",
        "## 3. F1 po klasi",
        "",
        f"![F1 po klasi](output/encoder_analysis/{fig_names.get('02_per_class_f1.png', '02_per_class_f1.png')})",
        "",
        *md_table(
            ["Model", "F1 NEUTRAL", "F1 ZA-VLAST", "F1 PROTIV-VLASTI"],
            [
                [
                    MODEL_NAMES.get(r["model_key"], r["model_key"]),
                    f"{r.get('per_class_f1', {}).get('NEUTRAL', 0):.4f}",
                    f"{r.get('per_class_f1', {}).get('ZA-VLAST', 0):.4f}",
                    f"{r.get('per_class_f1', {}).get('PROTIV-VLASTI', 0):.4f}",
                ]
                for r in bests
            ],
        ),
        "",
        "## 4. Stabilnost po foldovima",
        "",
        f"![Foldovi](output/encoder_analysis/{fig_names.get('03_fold_macro_f1.png', '03_fold_macro_f1.png')})",
        "",
        *md_table(
            ["Model", "mean fold macro-F1", "std", "min", "max"],
            [
                [
                    MODEL_NAMES.get(r["model_key"], r["model_key"]),
                    f"{float(np.mean(r.get('fold_macro_f1') or [0])):.4f}",
                    f"{float(np.std(r.get('fold_macro_f1') or [0])):.4f}",
                    f"{float(np.min(r.get('fold_macro_f1') or [0])):.4f}",
                    f"{float(np.max(r.get('fold_macro_f1') or [0])):.4f}",
                ]
                for r in bests
                if r.get("fold_macro_f1")
            ],
        ),
        "",
        "## 5. Matrice konfuzije",
        "",
    ]
    for p in figs:
        if p.name.startswith("04_cm_") or p.name.startswith("05_cm_"):
            lines += [f"![{p.stem}](output/encoder_analysis/{p.name})", ""]

    lines += [
        "## 6. Zaključak za izveštaj",
        "",
        f"- Pobednik po macro-F1: **{winner_name}** "
        f"(epochs={winner.get('epochs')}, macro-F1=**{winner['macro_f1']:.3f}**).",
        f"- Najteža klasa kod pobednika: "
        f"**{min(LABELS, key=lambda l: winner.get('per_class_f1', {}).get(l, 1.0))}**.",
        "- Oba modela su sačuvana u odvojenim folderima; inferenca radi nezavisno.",
        "",
        "## 7. Fajlovi i inferenca",
        "",
        f"- Sirovi rezultati: `{Path(str(payload.get('_json_path', 'encoder/output/*.json'))).as_posix()}`",
        "- Classification report-i: uz JSON (`.txt`)",
        "- Modeli:",
    ]
    for r in bests:
        key = r["model_key"]
        md = r.get("model_dir") or f"encoder/output/encoder_{key}"
        lines.append(f"  - `{md}`")
    lines += [
        "",
        "```bash",
        "cd phase3_model_training",
        'python encoder/infer_encoder.py --model bertic -t "Pumpaj!"',
        'python encoder/infer_encoder.py --model mbert -t "Pumpaj!"',
        "python encoder/report_encoder.py",
        "```",
        "",
    ]
    for fig in figs:
        lines.append(f"- `output/encoder_analysis/{fig.name}`")
    lines.append("")
    return "\n".join(lines)


def generate(json_path: Path) -> int:
    if not json_path.is_file():
        # fallback: pronađi najnoviji encoder_results*.json
        cands = sorted(
            (SCRIPT_DIR / "output").glob("encoder_results*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            print(
                f"Nema rezultata: {json_path}\n"
                "Prvo pokreni:\n"
                "  python encoder/train_encoder.py --compare",
                flush=True,
            )
            return 1
        json_path = cands[0]
        print(f"Koristim: {json_path}", flush=True)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["_json_path"] = str(json_path)
    rows = payload.get("results") or []
    if not rows:
        print("JSON nema 'results'.", flush=True)
        return 1

    setup_style()
    bests = best_per_model(rows)
    figs: list[Path] = [
        plot_compare_metrics(bests),
        plot_per_class(bests),
        plot_fold_macro(bests),
    ]
    for i, r in enumerate(bests, start=1):
        figs.append(plot_confusion(r, i))
    figs.append(plot_ranking(rows))

    REPORT.write_text(build_report(payload, figs, bests), encoding="utf-8")
    ranked = sorted(rows, key=lambda r: float(r["macro_f1"]), reverse=True)
    print(f"Izvestaj: {REPORT}")
    print(f"Grafike: {OUT_DIR}")
    print(
        f"Pobednik: {ranked[0]['model_key']} macro_f1={ranked[0]['macro_f1']:.4f}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grafike, matrice i Markdown iz encoder_results JSON-a"
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    return generate(args.json)


if __name__ == "__main__":
    raise SystemExit(main())
