#!/usr/bin/env python3
"""Izveštaj ablacije opsega n-grama i frekvencijskog filtriranja.

Čita baseline/output/ablation_ngram_freq_results.json i pravi:
  - ABLATION_NGRAM_FREQ_IZVESTAJ.md
  - output/baseline_ablation_analysis/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
DEFAULT_JSON = SCRIPT_DIR / "output" / "ablation_ngram_freq_results.json"
OUT_DIR = PHASE3_DIR / "output" / "baseline_ablation_analysis"
REPORT = PHASE3_DIR / "ABLATION_NGRAM_FREQ_IZVESTAJ.md"

MODEL_ORDER = ("lr", "svm", "nb")
MODEL_NAMES = {
    "lr": "Logistička regresija",
    "svm": "Linear SVM",
    "nb": "Naivni Bajes",
}
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
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    return path


def plot_axis(rows: list[dict], settings: list[dict], title: str, fname: str) -> Path:
    """Grupisan bar-chart: po podešavanju te ose, jedan bar po modelu."""
    labels = [s["label"] for s in settings]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(MODEL_ORDER))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, m in enumerate(MODEL_ORDER):
        vals = []
        for lab in labels:
            match = [r for r in rows if r["model"] == m and r["setting_label"] == lab]
            vals.append(match[0]["macro_f1"] if match else 0.0)
        if not any(vals):
            continue
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_NAMES[m], color=MODEL_COLORS[m])
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("macro-F1")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    return save_fig(fname)


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def axis_section(
    rows: list[dict], settings: list[dict], title: str, fig: Path, baseline_label: str
) -> list[str]:
    lines = [f"## {title}", "", f"![{title}](output/baseline_ablation_analysis/{fig.name})", ""]
    for m in MODEL_ORDER:
        model_rows = [r for r in rows if r["model"] == m]
        if not model_rows:
            continue
        model_rows = sorted(model_rows, key=lambda r: [s["label"] for s in settings].index(r["setting_label"]))
        lines += md_table(
            ["Podešavanje", "macro-F1", "accuracy", "F1 NEUTRAL", "F1 ZA-VLAST", "F1 PROTIV-VLASTI"],
            [
                [
                    r["setting_label"],
                    f"{r['macro_f1']:.4f}",
                    f"{r['accuracy']:.4f}",
                    f"{r.get('per_class_f1', {}).get('NEUTRAL', 0):.4f}",
                    f"{r.get('per_class_f1', {}).get('ZA-VLAST', 0):.4f}",
                    f"{r.get('per_class_f1', {}).get('PROTIV-VLASTI', 0):.4f}",
                ]
                for r in model_rows
            ],
        )
        lines.append("")

        baseline = next((r for r in model_rows if r["setting_label"] == baseline_label), None)
        best = max(model_rows, key=lambda r: r["macro_f1"])
        if baseline is not None and best["setting_label"] != baseline_label:
            delta = best["macro_f1"] - baseline["macro_f1"]
            lines += [
                f"**{MODEL_NAMES[m]}**: najbolje podešavanje je „{best['setting_label']}"
                f"” (macro-F1={best['macro_f1']:.4f}), Δ={delta:+.4f} u odnosu na trenutno.",
                "",
            ]
        else:
            lines += [
                f"**{MODEL_NAMES[m]}**: trenutno podešavanje je i dalje najbolje "
                f"od testiranih (macro-F1={baseline['macro_f1']:.4f}).",
                "",
            ]
    return lines


def build_report(payload: dict, ngram_fig: Path, freq_fig: Path) -> str:
    ngram_rows = payload.get("ngram_ablation") or []
    freq_rows = payload.get("freq_filter_ablation") or []
    ngram_settings = payload.get("ngram_settings") or []
    freq_settings = payload.get("freq_settings") or []
    base_cfg = payload.get("base_config_per_model") or {}

    lines = [
        "# Ablacija: opseg n-grama i frekvencijsko filtriranje",
        "",
        "Dopuna baseline eksperimenata (Faza 3.1) — testira uticaj dve odlike "
        "vektorizacije koje su ranije bile fiksne u svim konfiguracijama "
        "(`ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`). Umesto pune mreže "
        "(što bi eksplodiralo kombinacije), svaka osa se menja pojedinačno na "
        "**pobedničkoj konfiguraciji svakog modela** iz pune mreže — tako se "
        "efekat vidi kroz sve tri porodice modela (LR/SVM/NB), a ne samo na "
        "jednoj tačci.",
        "",
        "## 1. Polazne (pobedničke) konfiguracije po modelu",
        "",
        *md_table(
            ["Model", "weighting", "lowercase", "normalize", "macro-F1 (puna mreža)"],
            [
                [
                    MODEL_NAMES.get(m, m),
                    cfg["weighting"],
                    str(cfg["lowercase"]),
                    cfg["normalize"],
                    f"{cfg['macro_f1']:.4f}",
                ]
                for m, cfg in base_cfg.items()
            ],
        ),
        "",
    ]

    baseline_ngram = next((s["label"] for s in ngram_settings if "trenutno" in s["label"]), None)
    baseline_freq = next((s["label"] for s in freq_settings if "trenutno" in s["label"]), None)

    lines += axis_section(
        ngram_rows, ngram_settings, "2. Uticaj opsega n-grama", ngram_fig, baseline_ngram or ""
    )
    lines += axis_section(
        freq_rows, freq_settings, "3. Uticaj frekvencijskog filtriranja", freq_fig, baseline_freq or ""
    )

    lines += [
        "## 4. Zaključak",
        "",
        "Efekat obe ose testiran je nezavisno na svakom modelu, uz sve ostalo "
        "fiksirano na pobedničku konfiguraciju tog modela — ako je pravac "
        "efekta konzistentan kod sva tri modela, to je jak signal da je "
        "opšti (ne samo za jedan model); ako se modeli razilaze, to ukazuje "
        "na interakciju između tipa modela i ove odlike vektorizacije.",
        "",
    ]
    return "\n".join(lines)


def generate(json_path: Path) -> int:
    if not json_path.is_file():
        print(f"Nema rezultata: {json_path}\nPrvo pokreni: python baseline/ablation_ngram_freq.py", flush=True)
        return 1

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    ngram_rows = payload.get("ngram_ablation") or []
    freq_rows = payload.get("freq_filter_ablation") or []
    if not ngram_rows and not freq_rows:
        print("JSON nema ablacione rezultate.", flush=True)
        return 1

    setup_style()
    ngram_fig = plot_axis(
        ngram_rows, payload.get("ngram_settings") or [], "Uticaj opsega n-grama na macro-F1", "01_ngram_ablation.png"
    )
    freq_fig = plot_axis(
        freq_rows, payload.get("freq_settings") or [], "Uticaj frekvencijskog filtriranja na macro-F1", "02_freq_filter_ablation.png"
    )

    REPORT.write_text(build_report(payload, ngram_fig, freq_fig), encoding="utf-8")
    print(f"Izvestaj: {REPORT}")
    print(f"Grafike: {OUT_DIR}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Grafike i Markdown iz ablation_ngram_freq_results.json")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    return generate(args.json)


if __name__ == "__main__":
    raise SystemExit(main())
