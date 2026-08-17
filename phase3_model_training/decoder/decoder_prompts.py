"""Prompt šabloni za dekoderske LLM (Faza 3.2b).

Varijante po PDF-u / predlogu projekta:
  - jezik prompta: srpski / engleski
  - stil: kratak (short) / detaljan (detailed)
  - režim: zero-shot / few-shot
"""

from __future__ import annotations

import sys
from pathlib import Path

_PHASE3 = Path(__file__).resolve().parent.parent
if str(_PHASE3) not in sys.path:
    sys.path.insert(0, str(_PHASE3))

from common.data import LABELS  # noqa: E402

# Ručno sastavljeni few-shot primeri (nisu iz eval skupa → nema curenja oznaka).
FEWSHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "Živeo Vučić, jedini koji radi za Srbiju!",
        "ZA-VLAST",
    ),
    (
        "Stop blokadama, hoćemo normalan život i da deca idu u škole.",
        "ZA-VLAST",
    ),
    (
        "Bravo studenti, jedini imate hrabrosti da se borite protiv korupcije!",
        "PROTIV-VLASTI",
    ),
    (
        "Dosta više ovog režima, sistematski uništavaju obrazovanje.",
        "PROTIV-VLASTI",
    ),
    (
        "Gde mogu da pogledam celu emisiju od juče naveče?",
        "NEUTRAL",
    ),
    (
        "Ne znam šta da mislim o svemu ovome.",
        "NEUTRAL",
    ),
]

LABEL_HELP_SR = (
    "NEUTRAL — bez jasnog političkog stava, pitanje, informacija ili van teme\n"
    "ZA-VLAST — podrška vlasti / kritika studenata, blokada ili protesta\n"
    "PROTIV-VLASTI — kritika vlasti / podrška studentskom protestu"
)

LABEL_HELP_EN = (
    "NEUTRAL — no clear political stance, a question, information, or off-topic\n"
    "ZA-VLAST — support for the government / criticism of students, blockades, or protests\n"
    "PROTIV-VLASTI — criticism of the government / support for the student protest"
)


def _fewshot_block(lang: str) -> str:
    lines: list[str] = []
    for text, label in FEWSHOT_EXAMPLES:
        if lang == "sr":
            lines.append(f"Komentar: {text}\nOznaka: {label}")
        else:
            lines.append(f"Comment: {text}\nLabel: {label}")
    return "\n\n".join(lines)


def build_system_prompt(lang: str, style: str) -> str:
    """Sistemska / uvodna instrukcija (bez konkretnog komentara)."""
    if lang == "sr":
        if style == "short":
            return (
                "Klasifikuj politički stav Instagram komentara o studentskim "
                "protestima u Srbiji. Odgovori ISKLJUČIVO jednom oznakom: "
                "NEUTRAL, ZA-VLAST ili PROTIV-VLASTI."
            )
        return (
            "Ti si anotator za stance klasifikaciju komentara o studentskim "
            "protestima u Srbiji.\n\n"
            f"Klase:\n{LABEL_HELP_SR}\n\n"
            "Pravila:\n"
            "- Ocjenjuj STAV prema temi (vlast vs. studenti/protest), ne samo sentiment.\n"
            "- Pozitivan ton može ići uz bilo koju klasu.\n"
            "- Ako je stav nejasan, van teme ili samo pitanje → NEUTRAL.\n"
            "- Odgovori ISKLJUČIVO jednom od tri oznake, bez objašnjenja."
        )

    # engleski
    if style == "short":
        return (
            "Classify the political stance of an Instagram comment about student "
            "protests in Serbia. Reply with EXACTLY one label: "
            "NEUTRAL, ZA-VLAST, or PROTIV-VLASTI."
        )
    return (
        "You are an annotator for stance classification of comments about "
        "student protests in Serbia.\n\n"
        f"Classes:\n{LABEL_HELP_EN}\n\n"
        "Rules:\n"
        "- Judge STANCE toward the topic (government vs. students/protest), not mere sentiment.\n"
        "- Positive tone can appear with any class.\n"
        "- If unclear, off-topic, or only a question → NEUTRAL.\n"
        "- Reply with EXACTLY one of the three labels, no explanation."
    )


def build_user_prompt(
    comment: str,
    lang: str,
    style: str,
    shot: str,
) -> str:
    """Korisnički prompt sa (opciono) few-shot primerima + komentarom."""
    parts: list[str] = []

    if shot == "few":
        if lang == "sr":
            parts.append("Primeri:")
        else:
            parts.append("Examples:")
        parts.append(_fewshot_block(lang))
        parts.append("")

    if lang == "sr":
        if style == "detailed" and shot == "zero":
            parts.append(f"Klase:\n{LABEL_HELP_SR}\n")
        parts.append(f"Komentar: {comment}")
        parts.append("Oznaka:")
    else:
        if style == "detailed" and shot == "zero":
            parts.append(f"Classes:\n{LABEL_HELP_EN}\n")
        parts.append(f"Comment: {comment}")
        parts.append("Label:")

    return "\n".join(parts)


def build_messages(
    comment: str,
    lang: str,
    style: str,
    shot: str,
) -> list[dict[str, str]]:
    """Poruke u chat formatu (system + user)."""
    return [
        {"role": "system", "content": build_system_prompt(lang, style)},
        {"role": "user", "content": build_user_prompt(comment, lang, style, shot)},
    ]


def config_key(provider: str, model: str, lang: str, style: str, shot: str) -> str:
    return f"{provider}|{model}|{lang}|{style}|{shot}"
