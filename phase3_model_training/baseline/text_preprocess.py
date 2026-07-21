#!/usr/bin/env python3
"""Pretprocesiranje teksta za baseline modele: lowercasing, stemovanje, lematizacija.

Napomena (preporuka profesora): TF / IDF / TF-IDF, lowercasing, stem i lema
su tehnike pretprocesiranja / reprezentacije — ne „podrazumevani model“.
"""

from __future__ import annotations

import re
from functools import lru_cache

from sklearn.base import BaseEstimator, TransformerMixin

# Jednostavni sufiksi za grubo stemovanje srpskog (latinica + česta ćirilica
# se prethodno ne transliteriše ovde — radi na tokenima kakvi jesu).
_STEM_SUFFIXES = tuple(
    sorted(
        [
            "ijama",
            "ovima",
            "evima",
            "ovima",
            "ama",
            "ima",
            "ome",
            "oga",
            "ome",
            "emu",
            "ima",
            "ama",
            "ovi",
            "evi",
            "ski",
            "ska",
            "sko",
            "ost",
            "anja",
            "anje",
            "iti",
            "ati",
            "eti",
            "uti",
            "om",
            "em",
            "im",
            "og",
            "eg",
            "ih",
            "ih",
            "u",
            "e",
            "a",
            "i",
            "o",
        ],
        key=len,
        reverse=True,
    )
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def stem_token(token: str) -> str:
    """Grubo stemovanje: skida česte nastavke ako ostane dovoljno korena."""
    t = token.lower()
    if len(t) <= 4:
        return t
    for suf in _STEM_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t


@lru_cache(maxsize=1)
def _get_lemmatize():
    try:
        from simplemma import lemmatize

        return lemmatize
    except ImportError:
        return None


def lemma_token(token: str) -> str:
    """Lematizacija preko simplemma (sr). Fallback: lowercased token."""
    lemmatize = _get_lemmatize()
    t = token.lower()
    if lemmatize is None:
        return t
    try:
        return lemmatize(t, lang="sr") or t
    except Exception:
        return t


def normalize_text(text: str, mode: str) -> str:
    """mode: none | stem | lemma"""
    tokens = _TOKEN_RE.findall(text)
    if mode == "none":
        return " ".join(tokens)
    if mode == "stem":
        return " ".join(stem_token(tok) for tok in tokens)
    if mode == "lemma":
        return " ".join(lemma_token(tok) for tok in tokens)
    raise ValueError(f"Nepoznat normalize mode: {mode}")


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: lista tekstova -> lista normalizovanih tekstova."""

    def __init__(self, mode: str = "none"):
        if mode not in {"none", "stem", "lemma"}:
            raise ValueError("mode mora biti none|stem|lemma")
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode == "lemma" and _get_lemmatize() is None:
            raise ImportError(
                "Za lematizaciju instaliraj: pip install simplemma"
            )
        return self

    def transform(self, X):
        return [normalize_text(str(text), self.mode) for text in X]
