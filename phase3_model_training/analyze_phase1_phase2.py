#!/usr/bin/env python3
"""Analiza podataka Faza 1 (prikupljanje) i Faza 2 (anotacija).

Generiše grafike (PNG) i Markdown izveštaj za ceo skup:
Instagram, X, YouTube, Facebook.
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
PHASE1 = ROOT / "phase1_data_collection" / "output"
PHASE2_ANN = ROOT / "phase2_annotation" / "annotated"
OUT_DIR = SCRIPT_DIR / "output" / "data_analysis"
REPORT = SCRIPT_DIR / "ANALIZA_FAZA1_FAZA2.md"

LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")
LABEL_COLORS = {
    "NEUTRAL": "#6b7280",
    "ZA-VLAST": "#2563eb",
    "PROTIV-VLASTI": "#dc2626",
}
PLATFORMS = ("Instagram", "X", "YouTube", "Facebook", "Nepoznato")
PLATFORM_COLORS = {
    "Instagram": "#e1306c",
    "X": "#111827",
    "YouTube": "#dc2626",
    "Facebook": "#2563eb",
    "Nepoznato": "#9ca3af",
}


def load_annotated(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit("|", 2)
        if len(parts) != 3:
            continue
        text, url, label = parts
        label = label.strip()
        if label not in LABELS:
            continue
        rows.append((text.strip(), url.strip(), label))
    return rows


def detect_platform(url: str) -> str:
    u = (url or "").lower()
    if "instagram.com" in u:
        return "Instagram"
    if "x.com" in u or "twitter.com" in u:
        return "X"
    if "youtube.com" in u or "youtu.be" in u:
        return "YouTube"
    if "facebook.com" in u or "fb.com" in u or "fb.watch" in u:
        return "Facebook"
    return "Nepoznato"


def source_id(url: str) -> str:
    u = url or ""
    m = re.search(r"instagram\.com/(?:p|reel)/([^/?]+)", u, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"(?:x|twitter)\.com/.+/status/(\d+)", u, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"(?:x|twitter)\.com/i/status/(\d+)", u, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{6,})", u, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"facebook\.com/.+?/posts/([^/?]+)", u, flags=re.I)
    if m:
        return m.group(1)[:18]
    m = re.search(r"facebook\.com/reel/([^/?]+)", u, flags=re.I)
    if m:
        return m.group(1)[:18]
    if not u or u.upper() == "NEMA":
        return "NEMA"
    return u.rstrip("/").split("/")[-1][:18] or "NEMA"


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def count_export_comments(folder: Path, prefix: str) -> dict[str, int]:
    """Max broj komentara po post-id iz export TXT tabela."""
    post_counts: dict[str, int] = {}
    skip = {"all_texts", "final", "clean", "batch"}
    for path in sorted(folder.glob(f"{prefix}*.txt")):
        stem_l = path.stem.lower()
        if any(s in stem_l for s in skip):
            continue
        parts = path.stem.split("_")
        post_id = parts[1] if len(parts) > 1 else path.stem
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="|")
                if reader.fieldnames and "text" in reader.fieldnames:
                    n = sum(1 for row in reader if (row.get("text") or "").strip())
                else:
                    n = max(0, count_lines(path) - 1)
        except Exception:
            n = max(0, count_lines(path) - 1)
        post_counts[post_id] = max(post_counts.get(post_id, 0), n)
    return post_counts


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def top_tokens(texts: list[str], n: int = 15) -> list[tuple[str, int]]:
    stop = {
        "i", "u", "na", "je", "se", "da", "su", "za", "od", "to", "a", "o", "sa",
        "ne", "li", "ali", "kao", "sve", "ovo", "taj", "ta", "te", "ti", "mi",
        "po", "iz", "do", "ako", "kad", "kada", "jos", "još", "vec", "već",
        "the", "and", "of", "in", "is", "ja", "smo", "ste", "sam",
        "bih", "bi", "će", "ce", "nije", "nismo", "nisu", "koji", "koja", "koje",
        "ovaj", "ova", "tako", "samo", "ima", "biti", "bio", "bila",
        "vas", "nam", "nas", "vam", "ga", "ih", "mu", "joj", "njih",
        "što", "sto", "šta", "sta", "jer", "pa", "ili", "nema", "imao",
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
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    stages = ["Sirovi tekstovi\n(all_texts)", "Očišćeni\n(_clean)", "Anotirani final\n(dataset_all)"]
    vals = [raw, clean, annotated]
    colors = ["#94a3b8", "#64748b", "#0f766e"]
    bars = ax.bar(stages, vals, color=colors, width=0.55)
    ax.set_ylabel("Broj komentara")
    ax.set_title("Faza 1 → Faza 2: tok filtriranja (sve platforme)")
    ymax = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + ymax * 0.01,
            str(v),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    return save_fig("01_funnel_phase1_to_phase2.png")


def plot_label_pie(counts: Counter, n: int) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sizes = [counts[l] for l in LABELS]
    colors = [LABEL_COLORS[l] for l in LABELS]
    _wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=list(LABELS),
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p / 100 * sum(sizes)))})",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title(f"Faza 2: raspodela klasa (N={n})")
    return save_fig("02_label_distribution.png")


def plot_label_bars(counts: Counter) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [counts[l] for l in LABELS]
    bars = ax.bar(list(LABELS), vals, color=[LABEL_COLORS[l] for l in LABELS])
    ax.set_ylabel("Broj primera")
    ax.set_title("Faza 2: broj anotiranih komentara po klasi")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, str(v), ha="center", fontweight="bold")
    ideal = sum(vals) / 3
    ax.axhline(ideal, color="#9ca3af", linestyle="--", label=f"idealno uravnoteženo ({ideal:.0f})")
    ax.legend()
    return save_fig("03_label_bars.png")


def plot_length_hist(by_label: dict[str, list[int]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_lens = [x for lab in LABELS for x in by_label[lab]]
    xmax = int(np.percentile(all_lens, 99)) + 5 if all_lens else 50
    bins = np.arange(0, max(xmax, 20) + 3, 3)
    for lab in LABELS:
        arr = by_label[lab]
        if not arr:
            continue
        ax.hist(
            arr,
            bins=bins,
            alpha=0.45,
            label=f"{lab} (med={np.median(arr):.0f})",
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
    try:
        bp = ax.boxplot(data, tick_labels=list(LABELS), patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=list(LABELS), patch_artist=True)
    for patch, lab in zip(bp["boxes"], LABELS):
        patch.set_facecolor(LABEL_COLORS[lab])
        patch.set_alpha(0.55)
    ax.set_ylabel("Broj tokena")
    ax.set_title("Faza 2: raspodela dužine (boxplot)")
    return save_fig("05_length_boxplot.png")


def plot_phase1_by_platform(raw_by: dict[str, int], clean_by: dict[str, int]) -> Path:
    names = [p for p in ("Instagram", "X", "YouTube", "Facebook") if raw_by.get(p) or clean_by.get(p)]
    x = np.arange(len(names))
    width = 0.38
    raw_vals = [raw_by.get(p, 0) for p in names]
    clean_vals = [clean_by.get(p, 0) for p in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    b1 = ax.bar(x - width / 2, raw_vals, width, label="sirovi (all_texts)", color="#94a3b8")
    b2 = ax.bar(x + width / 2, clean_vals, width, label="očišćeni (_clean)", color="#0ea5e9")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Broj komentara")
    ax.set_title("Faza 1: prikupljeni komentari po platformi")
    ax.legend()
    ymax = max(raw_vals + clean_vals + [1])
    for bars in (b1, b2):
        for bar in bars:
            v = int(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.01, str(v), ha="center", va="bottom", fontsize=8)
    return save_fig("06_phase1_by_platform.png")


def plot_annotated_by_platform(rows: list[tuple[str, str, str]]) -> Path:
    by_plat: dict[str, Counter] = defaultdict(Counter)
    for _t, url, lab in rows:
        by_plat[detect_platform(url)][lab] += 1
    names = [p for p in PLATFORMS if sum(by_plat[p].values())]
    bottoms = np.zeros(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(names))
    for lab in LABELS:
        vals = np.array([by_plat[p][lab] for p in names], dtype=float)
        ax.bar(x, vals, bottom=bottoms, label=lab, color=LABEL_COLORS[lab])
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Broj anotiranih komentara")
    ax.set_title("Faza 2: anotirani primeri po platformi (stacked po klasi)")
    ax.legend()
    for i, p in enumerate(names):
        total = int(sum(by_plat[p].values()))
        ax.text(i, total + max(bottoms) * 0.01, str(total), ha="center", va="bottom", fontweight="bold")
    return save_fig("07_annotated_by_platform_stacked.png")


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


def plot_class_share_by_platform(rows: list[tuple[str, str, str]]) -> Path:
    names = [p for p in ("Instagram", "X", "YouTube", "Facebook")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names))
    width = 0.25
    for i, lab in enumerate(LABELS):
        vals = []
        for p in names:
            subset = [r for r in rows if detect_platform(r[1]) == p]
            n = len(subset) or 1
            vals.append(100 * sum(1 for r in subset if r[2] == lab) / n)
        ax.bar(x + (i - 1) * width, vals, width, label=lab, color=LABEL_COLORS[lab])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Udeo klase (%)")
    ax.set_title("Faza 2: udeo klasa unutar svake platforme")
    ax.legend()
    ax.set_ylim(0, 100)
    return save_fig("09_class_share_by_platform.png")


def plot_length_by_platform(rows: list[tuple[str, str, str]]) -> Path:
    names = [p for p in ("Instagram", "X", "YouTube", "Facebook")]
    data = []
    used = []
    for p in names:
        lens = [word_count(t) for t, u, _ in rows if detect_platform(u) == p]
        if lens:
            data.append(lens)
            used.append(p)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    try:
        bp = ax.boxplot(data, tick_labels=used, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=used, patch_artist=True)
    for patch, p in zip(bp["boxes"], used):
        patch.set_facecolor(PLATFORM_COLORS[p])
        patch.set_alpha(0.5)
    ax.set_ylabel("Broj tokena")
    ax.set_title("Faza 2: dužina komentara po platformi")
    return save_fig("10_length_by_platform.png")


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main() -> int:
    setup_style()
    rows = load_annotated(PHASE2_ANN / "dataset_all.txt")
    if not rows:
        print("Nema primera u dataset_all.txt", flush=True)
        return 1

    label_counts = Counter(r[2] for r in rows)
    plat_counts = Counter(detect_platform(r[1]) for r in rows)
    by_label_lens: dict[str, list[int]] = {l: [] for l in LABELS}
    by_label_texts: dict[str, list[str]] = {l: [] for l in LABELS}
    for text, _url, lab in rows:
        by_label_lens[lab].append(word_count(text))
        by_label_texts[lab].append(text)

    raw_by = {
        "Instagram": count_lines(PHASE1 / "instagram" / "instagram_all_texts.txt"),
        "X": count_lines(PHASE1 / "x" / "x_all_texts.txt"),
        "YouTube": count_lines(PHASE1 / "youtube" / "_youtube_all_texts.txt"),
        "Facebook": count_lines(PHASE1 / "facebook" / "facebook_all_texts.txt"),
    }
    clean_by = {
        "Instagram": count_lines(PHASE1 / "instagram" / "instagram_all_texts_clean.txt"),
        "X": count_lines(PHASE1 / "x" / "x_all_texts_clean.txt"),
        "YouTube": count_lines(PHASE1 / "youtube" / "_youtube_all_texts_clean.txt"),
        "Facebook": count_lines(PHASE1 / "facebook" / "facebook_all_texts_clean.txt")
        or raw_by["Facebook"],
    }
    raw_n = sum(raw_by.values())
    clean_n = sum(clean_by.values())
    ann_n = len(rows)

    posts_ig = count_export_comments(PHASE1 / "instagram", "instagram_")
    posts_x = count_export_comments(PHASE1 / "x", "x_")
    posts_yt = count_export_comments(PHASE1 / "youtube", "youtube_")
    posts_fb = count_export_comments(PHASE1 / "facebook", "facebook_")
    unique_posts = {
        "Instagram": len(posts_ig),
        "X": len(posts_x),
        "YouTube": len(posts_yt),
        "Facebook": len(posts_fb),
    }

    missing_url = sum(1 for r in rows if not r[1] or r[1].upper() == "NEMA")
    unique_urls = len({r[1] for r in rows if r[1] and r[1].upper() != "NEMA"})

    figs = [
        plot_funnel(raw_n, clean_n, ann_n),
        plot_label_pie(label_counts, ann_n),
        plot_label_bars(label_counts),
        plot_length_hist(by_label_lens),
        plot_length_box(by_label_lens),
        plot_phase1_by_platform(raw_by, clean_by),
        plot_annotated_by_platform(rows),
        plot_top_words(by_label_texts),
        plot_class_share_by_platform(rows),
        plot_length_by_platform(rows),
    ]

    length_stats = {}
    for lab in LABELS:
        arr = np.array(by_label_lens[lab], dtype=float)
        length_stats[lab] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()) if len(arr) else 0.0,
            "median": float(np.median(arr)) if len(arr) else 0.0,
            "std": float(arr.std()) if len(arr) else 0.0,
            "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
            "max": int(arr.max()) if len(arr) else 0,
        }

    max_c = max(label_counts.values())
    min_c = min(label_counts.values())
    imbalance = max_c / min_c if min_c else float("inf")
    biggest = max(LABELS, key=lambda l: label_counts[l])
    smallest = min(LABELS, key=lambda l: label_counts[l])

    plat_label = {
        p: {lab: sum(1 for r in rows if detect_platform(r[1]) == p and r[2] == lab) for lab in LABELS}
        for p in ("Instagram", "X", "YouTube", "Facebook", "Nepoznato")
    }

    summary = {
        "phase1": {
            "raw_by_platform": raw_by,
            "clean_by_platform": clean_by,
            "unique_posts": unique_posts,
            "all_texts_lines": raw_n,
            "clean_texts_lines": clean_n,
        },
        "phase2": {
            "annotated_total": ann_n,
            "label_counts": dict(label_counts),
            "platform_counts": dict(plat_counts),
            "platform_by_label": plat_label,
            "imbalance_max_over_min": round(imbalance, 2),
            "length_stats_tokens": length_stats,
            "missing_urls": missing_url,
            "unique_source_urls": unique_urls,
        },
        "figures": [p.name for p in figs],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pct_raw = 100 * ann_n / raw_n if raw_n else 0
    pct_clean = 100 * ann_n / clean_n if clean_n else 0
    all_lens = [x for lab in LABELS for x in by_label_lens[lab]]
    med_all = float(np.median(all_lens)) if all_lens else 0

    lines = [
        "# Analiza podataka — Faza 1 i Faza 2",
        "",
        "Izvor: javni komentari o studentskim protestima u Srbiji "
        "(Instagram, X, YouTube, Facebook). "
        "Finalni anotirani skup: `phase2_annotation/annotated/dataset_all.txt`.",
        "",
        "Grafike su u `phase3_model_training/output/data_analysis/` "
        "(PNG, spremne za git / izveštaj).",
        "",
        "## 1. Pregled",
        "",
        *md_table(
            ["Etapa", "Broj"],
            [
                ["Sirovi tekstovi (sve platforme, `all_texts`)", f"**{raw_n}**"],
                ["Očišćeni tekstovi (`_clean`)", f"**{clean_n}**"],
                ["Finalni anotirani skup (Faza 2)", f"**{ann_n}**"],
                ["Jedinstvenih URL izvora u Fazi 2", f"**{unique_urls}**"],
                ["Redova bez URL-a", f"**{missing_url}**"],
            ],
        ),
        "",
        f"Od sirovog ka finalnom zadržano je **{pct_raw:.1f}%** linija iz `all_texts` "
        f"(**{pct_clean:.1f}%** od clean skupa). To je očekivano: ručno se bira "
        "kvalitetan, uravnoteženiji podskup za učenje modela.",
        "",
        f"![Funnel](output/data_analysis/{figs[0].name})",
        "",
        "## 2. Faza 1 — prikupljanje",
        "",
        "Komentari su prikupljeni sa četiri platforme. Broj sirovih linija zavisi od "
        "exporta (mogući duplikati pri ponovnom preuzimanju iste objave).",
        "",
        *md_table(
            ["Platforma", "Sirovi", "Očišćeni", "Jedinstvene objave (export)"],
            [
                [
                    p,
                    str(raw_by[p]),
                    str(clean_by[p]),
                    str(unique_posts[p]),
                ]
                for p in ("Instagram", "X", "YouTube", "Facebook")
            ],
        ),
        "",
        f"![Po platformi](output/data_analysis/{figs[5].name})",
        "",
        "## 3. Faza 2 — anotacija",
        "",
        "### 3.1 Raspodela klasa",
        "",
        *md_table(
            ["Klasa", "Broj", "Udeo"],
            [
                [f"`{lab}`", str(label_counts[lab]), f"{100 * label_counts[lab] / ann_n:.1f}%"]
                for lab in LABELS
            ],
        ),
        "",
        f"**Neuravnoteženost** (max/min): **{imbalance:.2f}×** "
        f"(najveća `{biggest}`, najmanja `{smallest}`). "
        "Zato u Fazi 3 koristimo **macro-F1**, ne samo accuracy.",
        "",
        f"![Pie](output/data_analysis/{figs[1].name})",
        "",
        f"![Bars](output/data_analysis/{figs[2].name})",
        "",
        "### 3.2 Raspodela po platformi",
        "",
        *md_table(
            ["Platforma", "n", "NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI", "Udeo u skupu"],
            [
                [
                    p,
                    str(plat_counts.get(p, 0)),
                    str(plat_label[p]["NEUTRAL"]),
                    str(plat_label[p]["ZA-VLAST"]),
                    str(plat_label[p]["PROTIV-VLASTI"]),
                    f"{100 * plat_counts.get(p, 0) / ann_n:.1f}%",
                ]
                for p in ("Instagram", "X", "YouTube", "Facebook", "Nepoznato")
                if plat_counts.get(p, 0)
            ],
        ),
        "",
        f"![Stacked platform](output/data_analysis/{figs[6].name})",
        "",
        f"![Share platform](output/data_analysis/{figs[8].name})",
        "",
        "### 3.3 Dužina komentara (broj tokena)",
        "",
        *md_table(
            ["Klasa", "n", "prosek", "medijana", "std", "p90", "max"],
            [
                [
                    f"`{lab}`",
                    str(s["n"]),
                    f"{s['mean']:.1f}",
                    f"{s['median']:.0f}",
                    f"{s['std']:.1f}",
                    f"{s['p90']:.0f}",
                    str(s["max"]),
                ]
                for lab, s in ((lab, length_stats[lab]) for lab in LABELS)
            ],
        ),
        "",
        f"Komentari su uglavnom **kratki** (ukupna medijana ≈ {med_all:.0f} tokena) — "
        "tipično za društvene mreže. Klase su slične po dužini.",
        "",
        f"![Hist](output/data_analysis/{figs[3].name})",
        "",
        f"![Box](output/data_analysis/{figs[4].name})",
        "",
        f"![Len platform](output/data_analysis/{figs[9].name})",
        "",
        "### 3.4 Leksički signal (top tokeni)",
        "",
        "Najčešći tokeni (bez stop-reči) pokazuju šta bag-of-words / TF-IDF baseline "
        "može da nauči po klasama.",
        "",
        f"![Tokens](output/data_analysis/{figs[7].name})",
        "",
        "## 4. Zaključak za modele (Faza 3)",
        "",
        f"1. Skup ima **{ann_n}** primera — dovoljno za baseline; enkoder i dalje treba "
        "**stratifikovanu CV** da se smanji overfitting.",
        f"2. Klasa `{smallest}` je najmanja; macro-F1 je prava glavna metrika.",
        "3. Kratki tekstovi → n-grami (baseline) i kontekst enkodera pomažu više od "
        "modela rađenih za duge dokumente.",
        "4. Četiri platforme unose različit žargon i dužinu; model treba da generalizuje "
        "preko izvora, ne samo unutar jednog threada.",
        "",
        "## 5. Fajlovi grafika",
        "",
    ]
    for fig in figs:
        lines.append(f"- `output/data_analysis/{fig.name}`")
    lines += [
        "",
        "Numerički rezime: `output/data_analysis/summary.json`",
        "",
        "Regenerisanje:",
        "",
        "```bash",
        "cd phase3_model_training",
        "python analyze_phase1_phase2.py",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Grafika: {OUT_DIR}")
    print(f"Izvestaj: {REPORT}")
    print(json.dumps(summary["phase2"]["label_counts"], ensure_ascii=False))
    print(json.dumps(summary["phase2"]["platform_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
