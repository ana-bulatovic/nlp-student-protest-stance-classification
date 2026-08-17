#!/usr/bin/env python3
"""Faza 3.2b — dekoderski / generativni LLM putem prompt inženjeringa.

Zero-shot i few-shot klasifikacija (Ollama lokalno / ChatGPT / Gemini), poređenje:
  - jezika prompta (srpski vs engleski)
  - formata instrukcija (kratak vs detaljan)
Evaluacija na celom anotiranom skupu (prema predlogu projekta).

Podrazumevano: Ollama (nema API ključa) — zahteva `ollama serve` + povučen model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PHASE3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from decoder_prompts import (  # noqa: E402
    LABELS,
    build_messages,
    config_key,
)
from common.data import DEFAULT_DATA, load_dataset  # noqa: E402

DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "decoder_results.json"
DEFAULT_CACHE = SCRIPT_DIR / "output" / "decoder_cache"

# Alias-i koje model može vratiti (mapiraju se na naše LABELS).
_LABEL_ALIASES: dict[str, str] = {
    "NEUTRAL": "NEUTRAL",
    "NEUTRALAN": "NEUTRAL",
    "NEUTRALNO": "NEUTRAL",
    "ZA-VLAST": "ZA-VLAST",
    "ZA_VLAST": "ZA-VLAST",
    "ZAVLAST": "ZA-VLAST",
    "PRO-VLAST": "ZA-VLAST",
    "PRO_VLAST": "ZA-VLAST",
    "PROVLAST": "ZA-VLAST",
    "PROTIV-VLASTI": "PROTIV-VLASTI",
    "PROTIV_VLASTI": "PROTIV-VLASTI",
    "PROTIVVLASTI": "PROTIV-VLASTI",
    "PROTIV-VLAST": "PROTIV-VLASTI",
    "PROTIV_VLAST": "PROTIV-VLASTI",
    "PROTIVVLAST": "PROTIV-VLASTI",
    "PRO-STUDENT": "PROTIV-VLASTI",
    "PRO_STUDENT": "PROTIV-VLASTI",
    "PROSTUDENT": "PROTIV-VLASTI",
    "PRO-STUDENTI": "PROTIV-VLASTI",
}


@dataclass
class DecoderResult:
    provider: str
    model: str
    lang: str
    style: str
    shot: str
    n_samples: int
    n_parsed: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    parse_fail_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dekoderski LLM (prompting): zero/few-shot, SR/EN, short/detailed."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["ollama"],
        choices=["ollama", "openai", "gemini"],
        help="Provajderi (podrazumevano: ollama — lokalno, bez API ključa)",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama2",
        help="Ollama model (npr. llama2, tinyllama, mistral, qwen2.5)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model (npr. gpt-4o-mini, gpt-4o)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash",
        help="Gemini model (npr. gemini-2.0-flash, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["sr", "en"],
        choices=["sr", "en"],
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["short", "detailed"],
        choices=["short", "detailed"],
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        default=["zero", "few"],
        choices=["zero", "few"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ograniči broj primera (0 = ceo skup). Korisno uz --quick.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 20 primera, 1 konfiguracija (sr/short/zero)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Samo ispiši promptove, bez API poziva",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Pauza između API poziva (sekunde); za cloud API probaj 0.15",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignoriši keš i uvek zovi API",
    )
    return parser.parse_args()


def parse_label(raw: str) -> str | None:
    """Izvuci jednu od tri oznake iz slobodnog odgovora modela."""
    if not raw:
        return None
    text = raw.strip().upper()
    # direktan match
    compact = re.sub(r"\s+", "", text)
    compact = compact.replace("—", "-").replace("–", "-")
    for alias, label in _LABEL_ALIASES.items():
        if compact == alias or compact.startswith(alias + ".") or compact.startswith(
            alias + ","
        ):
            return label

    # traži alias kao celu reč / token u tekstu
    normalized = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("_", "-")
        .replace(" ", "")
    )
    # prioritet: duži alias-i prvi
    for alias in sorted(_LABEL_ALIASES.keys(), key=len, reverse=True):
        if alias in normalized:
            return _LABEL_ALIASES[alias]

    # labavi match sa crtama
    for label in LABELS:
        if label in text.replace("_", "-"):
            return label
    return None


def _cache_path(cache_dir: Path, key: str, comment: str) -> Path:
    h = hashlib.sha256(f"{key}\n{comment}".encode("utf-8")).hexdigest()[:24]
    return cache_dir / key.replace("|", "__") / f"{h}.json"


def load_cached(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_cached(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class OllamaClient:
    """Lokalni Ollama HTTP API — bez API ključa."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")
        self._check_alive()

    def _check_alive(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama nije dostupna na {self.host}.\n"
                "Pokreni: ollama serve\n"
                f"Zatim: ollama pull {self.model}"
            ) from exc
        names = {m.get("name", "").split(":")[0] for m in payload.get("models", [])}
        names |= {m.get("name", "") for m in payload.get("models", [])}
        short = self.model.split(":")[0]
        if self.model not in names and short not in names:
            available = sorted(names) or ["(nema modela)"]
            raise RuntimeError(
                f"Model {self.model!r} nije u Ollami.\n"
                f"Dostupno: {', '.join(available)}\n"
                f"Pokreni: ollama pull {self.model}"
            )

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 32,
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        msg = payload.get("message") or {}
        return (msg.get("content") or "").strip()


class OpenAIClient:
    def __init__(self, model: str) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Nedostaje OPENAI_API_KEY. Koristi --providers ollama "
                "(lokalno, bez ključa) ili postavi env varijablu."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Nedostaje paket openai. Instaliraj: pip install openai"
            ) from exc
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def complete(self, messages: list[dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=32,
        )
        return (resp.choices[0].message.content or "").strip()


class GeminiClient:
    def __init__(self, model: str) -> None:
        api_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError(
                "Nedostaje GEMINI_API_KEY. Koristi --providers ollama "
                "(lokalno, bez ključa) ili postavi env varijablu."
            )
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "Nedostaje paket google-generativeai. "
                "Instaliraj: pip install google-generativeai"
            ) from exc
        genai.configure(api_key=api_key)
        self.model_name = model
        self._genai = genai
        self.model = genai.GenerativeModel(model)

    def complete(self, messages: list[dict[str, str]]) -> str:
        system = ""
        user_parts: list[str] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_parts.append(m["content"])
        prompt = (system + "\n\n" + "\n\n".join(user_parts)).strip()
        resp = self.model.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=32,
            ),
        )
        return (getattr(resp, "text", None) or "").strip()


def make_client(
    provider: str,
    *,
    openai_model: str,
    gemini_model: str,
    ollama_model: str,
    ollama_host: str,
):
    if provider == "ollama":
        return OllamaClient(ollama_model, host=ollama_host), ollama_model
    if provider == "openai":
        return OpenAIClient(openai_model), openai_model
    if provider == "gemini":
        return GeminiClient(gemini_model), gemini_model
    raise ValueError(f"Nepoznat provider: {provider}")


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = list(LABELS)
    per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "per_class_f1": {lab: float(v) for lab, v in zip(labels, per)},
        "report": classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        ),
    }


def evaluate_config(
    client,
    provider: str,
    model: str,
    texts: list[str],
    labels: list[str],
    lang: str,
    style: str,
    shot: str,
    cache_dir: Path,
    use_cache: bool,
    sleep_s: float,
    dry_run: bool,
) -> tuple[DecoderResult, str, list[dict]]:
    key = config_key(provider, model, lang, style, shot)
    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict] = []
    n_fail = 0

    for i, (text, gold) in enumerate(zip(texts, labels), 1):
        messages = build_messages(text, lang, style, shot)
        raw = ""
        pred: str | None = None
        cached = False

        if dry_run:
            if i <= 2:
                print(f"\n--- dry-run primer {i} [{key}] ---")
                for m in messages:
                    print(f"[{m['role']}]\n{m['content']}\n")
            pred = None
        else:
            cpath = _cache_path(cache_dir, key, text)
            hit = load_cached(cpath) if use_cache else None
            if hit and "raw" in hit:
                raw = hit["raw"]
                pred = hit.get("parsed") or parse_label(raw)
                cached = True
            else:
                try:
                    raw = client.complete(messages)
                except Exception as exc:  # noqa: BLE001
                    print(f"  API greska na #{i}: {exc}", file=sys.stderr)
                    raw = ""
                pred = parse_label(raw)
                if use_cache:
                    save_cached(
                        cpath,
                        {
                            "provider": provider,
                            "model": model,
                            "lang": lang,
                            "style": style,
                            "shot": shot,
                            "text": text,
                            "gold": gold,
                            "raw": raw,
                            "parsed": pred,
                        },
                    )
                if sleep_s > 0:
                    time.sleep(sleep_s)

        if pred is None:
            n_fail += 1
            # za metrike: tretiraj neparsiran odgovor kao pogrešan (NEUTRAL fallback
            # bi veštački podigao F1 za tu klasu — bolje isključiti iz skorova)
            status = "parse_fail"
        else:
            y_true.append(gold)
            y_pred.append(pred)
            status = "ok" if pred == gold else "wrong"

        rows.append(
            {
                "text": text,
                "gold": gold,
                "pred": pred,
                "raw": raw,
                "status": status,
                "cached": cached,
            }
        )

        if i % 50 == 0 or i == len(texts):
            print(f"  [{key}] {i}/{len(texts)} (parse_fail={n_fail})")

    if dry_run:
        result = DecoderResult(
            provider=provider,
            model=model,
            lang=lang,
            style=style,
            shot=shot,
            n_samples=len(texts),
            n_parsed=0,
            accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            per_class_f1={lab: 0.0 for lab in LABELS},
            parse_fail_rate=1.0,
        )
        return result, "(dry-run — nema metrika)", rows

    if not y_pred:
        raise RuntimeError(f"Nijedan odgovor nije parsiran za {key}")

    metrics = compute_metrics(y_true, y_pred)
    result = DecoderResult(
        provider=provider,
        model=model,
        lang=lang,
        style=style,
        shot=shot,
        n_samples=len(texts),
        n_parsed=len(y_pred),
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        per_class_f1=metrics["per_class_f1"],
        parse_fail_rate=float(n_fail / max(len(texts), 1)),
    )
    return result, metrics["report"], rows


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

    providers = list(args.providers)
    langs = list(args.langs)
    styles = list(args.styles)
    shots = list(args.shots)
    limit = args.limit

    if args.quick:
        providers = [providers[0]]
        langs = ["sr"]
        styles = ["short"]
        shots = ["zero"]
        limit = limit or 20
        print("\n=== QUICK MODE (decoder smoke test) ===")

    if limit and limit > 0:
        texts = texts[:limit]
        labels = labels[:limit]
        print(f"Limit: {len(texts)} primera")

    if args.dry_run:
        print("DRY-RUN: nema API poziva")

    results: list[dict] = []
    reports: list[str] = []
    all_predictions: dict[str, list[dict]] = {}

    for provider in providers:
        client = None
        model_name = {
            "ollama": args.ollama_model,
            "openai": args.openai_model,
            "gemini": args.gemini_model,
        }.get(provider, provider)
        if not args.dry_run:
            try:
                client, model_name = make_client(
                    provider,
                    openai_model=args.openai_model,
                    gemini_model=args.gemini_model,
                    ollama_model=args.ollama_model,
                    ollama_host=args.ollama_host,
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1

        for lang in langs:
            for style in styles:
                for shot in shots:
                    tag = f"{provider}/{model_name} | {lang} | {style} | {shot}-shot"
                    print(f"\n=== {tag} ===")
                    result, report, rows = evaluate_config(
                        client=client,
                        provider=provider,
                        model=model_name,
                        texts=texts,
                        labels=labels,
                        lang=lang,
                        style=style,
                        shot=shot,
                        cache_dir=args.cache_dir,
                        use_cache=not args.no_cache,
                        sleep_s=args.sleep,
                        dry_run=args.dry_run,
                    )
                    if not args.dry_run:
                        print(
                            f"acc={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}  "
                            f"parsed={result.n_parsed}/{result.n_samples}  "
                            f"parse_fail={result.parse_fail_rate:.2%}"
                        )
                        print(report)
                    results.append(asdict(result))
                    reports.append(f"### {tag}\n\n{report}")
                    all_predictions[config_key(provider, model_name, lang, style, shot)] = (
                        rows
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "decoder_prompting",
        "data": str(args.data),
        "n_samples": len(texts),
        "label_counts": dict(Counter(labels)),
        "providers": providers,
        "langs": langs,
        "styles": styles,
        "shots": shots,
        "dry_run": args.dry_run,
        "results": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = args.output.with_suffix(".txt")
    report_path.write_text("\n\n".join(reports) + "\n", encoding="utf-8")

    pred_path = args.output.with_name("decoder_predictions.json")
    pred_path.write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if results and not args.dry_run:
        ranked = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
        print("\n=== Rangiranje (macro-F1) ===")
        for i, r in enumerate(ranked, 1):
            print(
                f"{i}. {r['provider']}/{r['model']} {r['lang']}/{r['style']}/"
                f"{r['shot']}: macro_f1={r['macro_f1']:.4f} acc={r['accuracy']:.4f}"
            )

    print(f"\nRezultati: {args.output}")
    print(f"Izvestaji: {report_path}")
    print(f"Predikcije: {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
