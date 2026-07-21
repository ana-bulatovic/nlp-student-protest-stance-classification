#!/usr/bin/env python3
"""Analiza podataka Faza 1 (prikupljanje) i Faza 2 (anotacija).

Generiše grafike (PNG) i Markdown izveštaj.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PHASE1_IG = ROOT / "phase1_data_collection" / "output" / "instagram"
PHASE2_ANN = ROOT / "phase2_annotation" / "annotated"
OUT_DIR = SCRIPT_DIR / "output" / "data_analysis"
REPORT = SCRIPT_DIR / "ANALIZA_FAZA1_FAZA2.md"

LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")
LABEL_COLORS = {
    "NEUTRAL": "#6b7280",
    "ZA-VLAST": "#2563eb",
    "PROTIV-VLASTI": "#dc2626",
}


def load_annotated(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        text, url, label = line.rsplit("|", 2)
        rows.append((text.strip(), url.strip(), label.strip()))
    return rows


def phase1_post_stats() -> dict[str, int]:
    """Broj komentara po shortcode-u (max po export fajlu)."""
    post_counts: dict[str, int] = {}
    for path in sorted(PHASE1_IG.glob("instagram_*.txt")):
        if path.name.startswith("instagram_all"):
            continue
        parts = path.stem.split("_")
        short = parts[1] if len(parts) > 1 else path.stem
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            n = sum(1 for row in reader if (row.get("text") or "").strip())
        post_counts[short] = max(post_counts.get(short, 0), n)
    return post_counts


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def top_tokens(texts: list[str], n: int = 15) -> list[tuple[str, int]]:
    stop = {
        "i", "u", "na", "je", "se", "da", "su", "za", "od", "to", "a", "o", "sa",
        "ne", "li", "ali", "kao", "sve", "ovo", "taj", "ta", "te", "ti", "mi",
        "po", "iz", "do", "ako", "kad", "kada", "jos", "još", "vec", "već",
        "the", "and", "of", "in", "to", "is", "ja", "smo", "ste", "sam",
        "bih", "bi", "će", "ce", "nije", "nismo", "nisu", "koji", "koja", "koje",
        "ovaj", "ova", "ovo", "tako", "samo", "ima", "biti", "bio", "bila",
    }
    cnt: Counter[str] = Counter()
    for text in texts:
        for tok in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
            if len(tok) < 3 or tok in stop or tok.isdigit():
                continue
            cnt[tok] += 1
    return cnt.most_common(n)


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


def plot_funnel(raw: int, clean: int, annotated: int) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    stages = ["Sirovi tekstovi\n(instagram_all_texts)", "Očišćeni\n(_clean)", "Anotirani final\n(dataset_all)"]
    vals = [raw, clean, annotated]
    colors = ["#94a3b8", "#64748b", "#0f766e"]
    bars = ax.bar(stages, vals, color=colors, width=0.55)
    ax.set_ylabel("Broj komentara")
    ax.set_title("Faza 1 → Faza 2: tok filtriranja podataka")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01, str(v),
                ha="center", va="bottom", fontweight="bold")
    return save_fig("01_funnel_phase1_to_phase2.png")


def plot_label_pie(counts: Counter) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = list(LABELS)
    sizes = [counts[l] for l in labels]
    colors = [LABEL_COLORS[l] for l in labels]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p/100*sum(sizes)))})",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Faza 2: raspodela klasa (N=500)")
    return save_fig("02_label_distribution.png")


def plot_label_bars(counts: Counter) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(LABELS)
    vals = [counts[l] for l in labels]
    bars = ax.bar(labels, vals, color=[LABEL_COLORS[l] for l in labels])
    ax.set_ylabel("Broj primera")
    ax.set_title("Faza 2: broj anotiranih komentara po klasi")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", fontweight="bold")
    # balance line (ideal equal)
    ideal = sum(vals) / 3
    ax.axhline(ideal, color="#9ca3af", linestyle="--", label=f"idealno uravnoteženo ({ideal:.0f})")
    ax.legend()
    return save_fig("03_label_bars.png")


def plot_length_hist(by_label: dict[str, list[int]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, 55, 3)
    for lab in LABELS:
        ax.hist(
            by_label[lab],
            bins=bins,
            alpha=0.45,
            label=f"{lab} (med={np.median(by_label[lab]):.0f})",
            color=LABEL_COLORS[lab],
        )
    ax.set_xlabel("Broj tokena (reči) po komentaru")
    ax.set_ylabel("Frekvencija")
    ax.set_title("Faza 2: dužina komentara po klasi")
    ax.legend()
    return save_fig("04_length_by_label.png")


def plot_length_box(by_label: dict[str, list[int]]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [by_label[l] for l in LABELS]
    bp = ax.boxplot(data, labels=list(LABELS), patch_artist=True)
    for patch, lab in zip(bp["boxes"], LABELS):
        patch.set_facecolor(LABEL_COLORS[lab])
        patch.set_alpha(0.55)
    ax.set_ylabel("Broj tokena")
    ax.set_title("Faza 2: raspodela dužine (boxplot)")
    return save_fig("05_length_boxplot.png")


def plot_posts_phase1(post_counts: dict[str, int]) -> Path:
    items = sorted(post_counts.items(), key=lambda x: -x[1])
    shorts = [s for s, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(shorts)), vals, color="#0ea5e9")
    ax.set_xticks(range(len(shorts)))
    ax.set_xticklabels(shorts, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Broj komentara")
    ax.set_title("Faza 1: komentari po Instagram objavi (unique shortcode)")
    return save_fig("06_phase1_comments_per_post.png")


def plot_annotated_per_url(rows: list[tuple[str, str, str]]) -> Path:
    # shortcode from url
    def short(url: str) -> str:
        m = re.search(r"/p/([^/]+)", url)
        return m.group(1) if m else url[-12:]

    by_post: dict[str, Counter] = defaultdict(Counter)
    for _t, url, lab in rows:
        by_post[short(url)][lab] += 1

    posts = sorted(by_post.keys(), key=lambda p: -sum(by_post[p].values()))
    bottoms = np.zeros(len(posts))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(posts))
    for lab in LABELS:
        vals = np.array([by_post[p][lab] for p in posts])
        ax.bar(x, vals, bottom=bottoms, label=lab, color=LABEL_COLORS[lab])
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(posts, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Broj anotiranih komentara")
    ax.set_title("Faza 2: anotirani primeri po izvornoj objavi (stacked po klasi)")
    ax.legend()
    return save_fig("07_annotated_by_source_stacked.png")


def plot_top_words(by_label_texts: dict[str, list[str]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for ax, lab in zip(axes, LABELS):
        tops = top_tokens(by_label_texts[lab], n=12)
        if not tops:
            continue
        words, freqs = zip(*tops[::-1])
        ax.barh(words, freqs, color=LABEL_COLORS[lab])
        ax.set_title(lab)
        ax.set_xlabel("frekvencija")
    fig.suptitle("Faza 2: najčešći tokeni po klasi (bez stop-reči)", y=1.02)
    return save_fig("08_top_tokens_by_label.png")


def plot_balance_vs_sources(rows: list[tuple[str, str, str]]) -> Path:
    """Udeo klasa u top 5 izvora."""
    def short(url: str) -> str:
        m = re.search(r"/p/([^/]+)", url)
        return m.group(1) if m else url

    url_counts = Counter(short(u) for _, u, _ in rows)
    top5 = [u for u, _ in url_counts.most_common(5)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(top5))
    width = 0.25
    for i, lab in enumerate(LABELS):
        vals = []
        for p in top5:
            subset = [r for r in rows if short(r[1]) == p]
            n = len(subset) or 1
            vals.append(100 * sum(1 for r in subset if r[2] == lab) / n)
        ax.bar(x + (i - 1) * width, vals, width, label=lab, color=LABEL_COLORS[lab])
    ax.set_xticks(x)
    ax.set_xticklabels(top5, rotation=30, ha="right")
    ax.set_ylabel("Udeo klase (%)")
    ax.set_title("Faza 2: udeo klasa unutar top-5 izvora")
    ax.legend()
    ax.set_ylim(0, 100)
    return save_fig("09_class_share_top_sources.png")


def main() -> int:
    setup_style()
    rows = load_annotated(PHASE2_ANN / "dataset_all.txt")
    label_counts = Counter(r[2] for r in rows)
    by_label_lens: dict[str, list[int]] = {l: [] for l in LABELS}
    by_label_texts: dict[str, list[str]] = {l: [] for l in LABELS}
    for text, _url, lab in rows:
        by_label_lens[lab].append(word_count(text))
        by_label_texts[lab].append(text)

    raw_n = count_lines(PHASE1_IG / "instagram_all_texts.txt")
    clean_n = count_lines(PHASE1_IG / "instagram_all_texts_clean.txt")
    ann_n = len(rows)
    post_counts = phase1_post_stats()

    figs = [
        plot_funnel(raw_n, clean_n, ann_n),
        plot_label_pie(label_counts),
        plot_label_bars(label_counts),
        plot_length_hist(by_label_lens),
        plot_length_box(by_label_lens),
        plot_posts_phase1(post_counts),
        plot_annotated_per_url(rows),
        plot_top_words(by_label_texts),
        plot_balance_vs_sources(rows),
    ]

    # length summary table
    length_stats = {}
    for lab in LABELS:
        arr = np.array(by_label_lens[lab])
        length_stats[lab] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "p90": float(np.percentile(arr, 90)),
            "max": int(arr.max()),
        }

    # imbalance ratio
    max_c = max(label_counts.values())
    min_c = min(label_counts.values())
    imbalance = max_c / min_c if min_c else float("inf")

    summary = {
        "phase1": {
            "unique_posts": len(post_counts),
            "comments_in_exports_max_per_post": sum(post_counts.values()),
            "all_texts_lines": raw_n,
            "clean_texts_lines": clean_n,
            "comments_per_post": post_counts,
        },
        "phase2": {
            "annotated_total": ann_n,
            "label_counts": dict(label_counts),
            "imbalance_max_over_min": round(imbalance, 2),
            "length_stats_tokens": length_stats,
            "nema_urls": sum(1 for r in rows if r[1] == "NEMA"),
            "unique_source_urls": len({r[1] for r in rows}),
        },
        "figures": [p.name for p in figs],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown report
    lines = [
        "# Analiza podataka — Faza 1 i Faza 2",
        "",
        "Izvor: Instagram komentari o studentskim protestima; finalni anotirani skup `dataset_all.txt`.",
        "",
        "## 1. Pregled (što smo dobili)",
        "",
        "| Etapa | Broj |",
        "|-------|------|",
        f"| Instagram objave (unique) | **{len(post_counts)}** |",
        f"| Komentari u exportima (max po objavi) | **{sum(post_counts.values())}** |",
        f"| `instagram_all_texts.txt` (sa mogućim duplikatima exporta) | **{raw_n}** |",
        f"| `instagram_all_texts_clean.txt` | **{clean_n}** |",
        f"| Finalni anotirani skup (Faza 2) | **{ann_n}** |",
        "",
        f"Od sirovog ka finalnom: zadržano **{100*ann_n/raw_n:.1f}%** linija iz `all_texts` "
        f"(ili **{100*ann_n/clean_n:.1f}%** od clean skupa) — očekivano, jer se ručno bira "
        "kvalitetan, uravnoteženiji podskup za učenje modela.",
        "",
        f"![Funnel](output/data_analysis/{figs[0].name})",
        "",
        "## 2. Faza 1 — prikupljanje",
        "",
        "- Platforma u ovom izveštaju: **Instagram** (pipe `|` TXT + JSON export po objavi).",
        f"- Broj jedinstvenih objava: **{len(post_counts)}**.",
        "- Raspodela komentara po objavi je **jako neuravnotežena** (nekoliko viralnih postova "
        "daje većinu komentara).",
        "",
        f"![Po objavi](output/data_analysis/{figs[5].name})",
        "",
        "### Top objave po broju komentara (Faza 1)",
        "",
        "| Shortcode | Komentara |",
        "|-----------|-----------|",
    ]
    for short, n in sorted(post_counts.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"| `{short}` | {n} |")

    lines += [
        "",
        "## 3. Faza 2 — anotacija",
        "",
        "### 3.1 Raspodela klasa",
        "",
        "| Klasa | Broj | Udeo |",
        "|-------|------|------|",
    ]
    for lab in LABELS:
        c = label_counts[lab]
        lines.append(f"| `{lab}` | {c} | {100*c/ann_n:.1f}% |")
    lines += [
        "",
        f"**Neuravnoteženost** (max/min): **{imbalance:.2f}×** "
        f"(`PROTIV-VLASTI` je najveća klasa). "
        "Zato u Fazi 3 koristimo **macro-F1**, ne samo accuracy.",
        "",
        f"![Pie](output/data_analysis/{figs[1].name})",
        "",
        f"![Bars](output/data_analysis/{figs[2].name})",
        "",
        "### 3.2 Dužina komentara (broj tokena)",
        "",
        "| Klasa | n | prosek | medijana | std | p90 | max |",
        "|-------|---|--------|----------|-----|-----|-----|",
    ]
    for lab in LABELS:
        s = length_stats[lab]
        lines.append(
            f"| `{lab}` | {s['n']} | {s['mean']:.1f} | {s['median']:.0f} | "
            f"{s['std']:.1f} | {s['p90']:.0f} | {s['max']} |"
        )
    lines += [
        "",
        "Komentari su uglavnom **kratki** (medijana ~8–10 tokena) — tipično za društvene mreže. "
        "Klase su slične po dužini; `PROTIV-VLASTI` je malo kraća u proseku.",
        "",
        f"![Hist](output/data_analysis/{figs[3].name})",
        "",
        f"![Box](output/data_analysis/{figs[4].name})",
        "",
        "### 3.3 Izvori (URL / objava)",
        "",
        f"- Jedinstvenih URL izvora u anotiranom skupu: **{len({r[1] for r in rows})}**",
        f"- Redova sa `NEMA` URL: **{sum(1 for r in rows if r[1]=='NEMA')}**",
        "",
        "Anotirani podskup **ne prati** proporciju sirovih komentara 1:1 — biraju se "
        "primeri po klasama. Ipak, i u finalu dominiraju neke objave:",
        "",
        f"![Stacked](output/data_analysis/{figs[6].name})",
        "",
        f"![Share](output/data_analysis/{figs[8].name})",
        "",
        "### 3.4 Leksički signal (top tokeni)",
        "",
        "Najčešći tokeni (grubo, bez stop-reči) daju uvid u to šta model može da „nauči“ "
        "preko bag-of-words / TF-IDF:",
        "",
        f"![Tokens](output/data_analysis/{figs[7].name})",
        "",
        "## 4. Poređenje Faza 1 vs Faza 2",
        "",
        "| Dimenzija | Faza 1 | Faza 2 |",
        "|-----------|--------|--------|",
        f"| Cilj | što više javnih komentara | kvalitetan **označen** skup |",
        f"| Obim | ~{sum(post_counts.values())} (export) / {raw_n} linija all | **{ann_n}** primera |",
        f"| Oznake | nema | 3 klase |",
        f"| URL metapodatak | u export TXT/JSON | u annotated fajlovima |",
        f"| Balans klasa | N/A | neuravnotežen ({imbalance:.2f}×) |",
        "",
        "**Zaključak za modele (Faza 3):**",
        "",
        "1. Skup je mali (500) → baseline i enkoder mogu overfittovati; CV je obavezna.",
        "2. Klasa `NEUTRAL` je najmanja i često najteža (nejasan signal).",
        "3. Kratki tekstovi → n-grami i kontekst enkodera pomažu više od dugih dokumenata.",
        "4. Domacija pojedinih objava → oprez od „curenja“ stila jednog thread-a u train/test "
        "(stratifikacija po klasi pomaže, ali ne rešava source bias potpuno).",
        "",
        "## 5. Fajlovi grafika",
        "",
    ]
    for fig in figs:
        lines.append(f"- `output/data_analysis/{fig.name}`")
    lines.append("")
    lines.append(f"Numerički rezime: `output/data_analysis/summary.json`")
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Grafika: {OUT_DIR}")
    print(f"Izvestaj: {REPORT}")
    print(json.dumps(summary["phase2"]["label_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
