"""Zajednički podaci za Fazu 3: oznake i učitavanje skupa."""

from __future__ import annotations

from pathlib import Path

LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")

PHASE3_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PHASE3_DIR.parent
DEFAULT_DATA = REPO_ROOT / "phase2_annotation" / "annotated" / "dataset_all.txt"


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Loš format u {path.name}, linija {line_no}: {line[:80]!r}")
        text, _url, label = parts
        label = label.strip()
        if label not in LABELS:
            raise ValueError(f"Nepoznata oznaka {label!r} (linija {line_no})")
        if not text.strip():
            continue
        texts.append(text.strip())
        labels.append(label)
    if not texts:
        raise ValueError(f"Nema primera u {path}")
    return texts, labels
