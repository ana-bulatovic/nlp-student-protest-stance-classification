#!/usr/bin/env python3
"""Faza 3.2a — enkoderski LLM preko Simple Transformers.

Fine-tune BERTić (mono) i mBERT (multi) za stance klasifikaciju.
Interfejs: Simple Transformers ClassificationModel (preporuka profesora).
Evaluacija: stratifikovana CV + poređenje po broju epoha.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# Na Windows+CPU: sklearn PRE torch/transformers može da sruši proces (0xC0000005).
# Zato prvo st_compat (učitava torch), pa tek onda sklearn.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PHASE3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

print("Pokretanje train_encoder.py ...", flush=True)
print("Ucitavanje torch/transformers (st_compat) ...", flush=True)
import st_compat  # noqa: E402

st_compat.apply()
print("OK - torch spreman. Ucitavanje sklearn/pandas ...", flush=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from common.data import DEFAULT_DATA, LABELS, load_dataset  # noqa: E402

DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "encoder_results.json"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "output" / "encoder_best"

# BERTić = ELECTRA arhitektura; mBERT = BERT
MODEL_PRESETS = {
    "bertic": {"model_type": "electra", "model_name": "classla/bcms-bertic"},
    "mbert": {"model_type": "bert", "model_name": "bert-base-multilingual-cased"},
}


@dataclass
class EncoderResult:
    model_key: str
    model_name: str
    epochs: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    fold_macro_f1: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning enkodera (Simple Transformers): BERTić / mBERT."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bertic", "mbert"],
        choices=list(MODEL_PRESETS.keys()),
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Varijante po broju epoha (PDF)",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 folda, 1 epoha, jedan model",
    )
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def compute_metrics_arrays(y_true: list[str], y_pred: list[str]) -> dict:
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


def make_args(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
):
    st_compat.apply()
    from simpletransformers.classification import ClassificationArgs

    args = ClassificationArgs()
    args.num_train_epochs = epochs
    args.learning_rate = lr
    args.max_seq_length = max_length
    args.train_batch_size = batch_size
    args.eval_batch_size = batch_size
    args.overwrite_output_dir = True
    args.reprocess_input_data = True
    args.use_multiprocessing = False
    args.use_multiprocessing_for_evaluation = False
    args.save_eval_checkpoints = False
    args.save_model_every_epoch = False
    args.save_steps = -1
    args.save_best_model = False
    args.evaluate_during_training = False
    args.manual_seed = seed
    args.output_dir = str(output_dir)
    args.best_model_dir = str(output_dir / "best")
    args.labels_list = list(LABELS)
    args.silent = True
    return args


def build_model(
    model_key: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    use_cuda: bool,
):
    st_compat.apply()
    from simpletransformers.classification import ClassificationModel

    preset = MODEL_PRESETS[model_key]
    args = make_args(output_dir, epochs, batch_size, lr, max_length, seed)
    return ClassificationModel(
        preset["model_type"],
        preset["model_name"],
        num_labels=len(LABELS),
        args=args,
        use_cuda=use_cuda,
    )


def train_one_fold(
    model_key: str,
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    work_dir: Path,
    use_cuda: bool,
) -> list[str]:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        model_key, work_dir, epochs, batch_size, lr, max_length, seed, use_cuda
    )
    train_df = pd.DataFrame({"text": train_texts, "labels": train_labels})
    # Simple Transformers: nekoliko linija za trening
    model.train_model(train_df)

    preds, _raw = model.predict(test_texts)
    # preds mogu biti string (labels_list) ili int
    out: list[str] = []
    for p in preds:
        if isinstance(p, str) and p in LABELS:
            out.append(p)
        else:
            out.append(LABELS[int(p)])
    return out


def evaluate_encoder_config(
    model_key: str,
    texts: list[str],
    labels: list[str],
    epochs: int,
    folds: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    scratch_dir: Path,
    use_cuda: bool,
) -> tuple[EncoderResult, str]:
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_true: list[str] = []
    all_pred: list[str] = []
    fold_scores: list[float] = []
    model_name = MODEL_PRESETS[model_key]["model_name"]

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(texts, y), 1):
        print(f"  fold {fold_i}/{folds} ...")
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        work = scratch_dir / f"{model_key}_e{epochs}_fold{fold_i}"
        preds = train_one_fold(
            model_key=model_key,
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_length=max_length,
            seed=seed,
            work_dir=work,
            use_cuda=use_cuda,
        )
        fold_macro = float(
            f1_score(
                test_labels, preds, average="macro", labels=list(LABELS), zero_division=0
            )
        )
        fold_scores.append(fold_macro)
        all_true.extend(test_labels)
        all_pred.extend(preds)
        print(f"    fold macro-F1={fold_macro:.4f}")

    metrics = compute_metrics_arrays(all_true, all_pred)
    result = EncoderResult(
        model_key=model_key,
        model_name=model_name,
        epochs=epochs,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        per_class_f1=metrics["per_class_f1"],
        fold_macro_f1=fold_scores,
    )
    return result, metrics["report"]


def train_full_and_save(
    model_key: str,
    texts: list[str],
    labels: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    out_dir: Path,
    use_cuda: bool,
    best_meta: dict,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        model_key, out_dir, epochs, batch_size, lr, max_length, seed, use_cuda
    )
    train_df = pd.DataFrame({"text": texts, "labels": labels})
    model.train_model(train_df)

    meta = {
        "framework": "simpletransformers",
        "model_key": model_key,
        "model_type": MODEL_PRESETS[model_key]["model_type"],
        "model_name": MODEL_PRESETS[model_key]["model_name"],
        "epochs": epochs,
        "labels": list(LABELS),
        "max_length": max_length,
        "best_config": best_meta,
        "cv_macro_f1": best_meta.get("macro_f1"),
    }
    (out_dir / "stance_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _check_compat() -> None:
    """Deprecated: logika je u st_compat.apply()."""
    return


def _patch_simpletransformers_compat() -> None:
    st_compat.apply()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    if not args.data.is_file():
        print(f"Nema dataset fajla: {args.data}", file=sys.stderr)
        return 1

    print("Ucitavanje Simple Transformers ...", flush=True)
    try:
        import torch
        from simpletransformers.classification import ClassificationModel  # noqa: F401
    except ImportError:
        print(
            "Nedostaju paketi. Instaliraj:\n"
            "  pip install simpletransformers pandas\n"
            "  pip install -r phase3_model_training/requirements.txt",
            file=sys.stderr,
        )
        return 1
    print(f"OK — torch {torch.__version__}", flush=True)

    texts, labels = load_dataset(args.data)
    print(f"Ucitano {len(texts)} primera iz {args.data}")
    print("Distribucija:", dict(Counter(labels)))
    use_cuda = torch.cuda.is_available()
    print(f"Uredjaj: {'cuda' if use_cuda else 'cpu'}")
    print("Framework: Simple Transformers (ClassificationModel)")
    if not use_cuda:
        print(
            "Upozorenje: nema GPU — fine-tuning ce biti spor. "
            "Za puni eksperiment preporucen je Google Colab / Azure.",
            flush=True,
        )

    models = list(args.models)
    epochs_list = list(args.epochs)
    folds = args.folds
    batch_size = args.batch_size

    if args.quick:
        models = [models[0]]
        epochs_list = [1]
        folds = min(2, folds)
        batch_size = min(4, batch_size)
        print("\n=== QUICK MODE (Simple Transformers smoke test) ===")

    counts = Counter(labels)
    min_class = min(counts.values())
    if min_class < folds:
        print(
            f"Upozorenje: najmanja klasa ima {min_class} primera, "
            f"folds={folds}. Smanjujem folds na {min_class}.",
            file=sys.stderr,
        )
        folds = min_class

    scratch = SCRIPT_DIR / "output" / "_encoder_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    reports: list[str] = []

    for model_key in models:
        model_name = MODEL_PRESETS[model_key]["model_name"]
        for epochs in epochs_list:
            tag = f"{model_key} ({model_name}) | epochs={epochs}"
            print(f"\n=== {tag} ===")
            result, report = evaluate_encoder_config(
                model_key=model_key,
                texts=texts,
                labels=labels,
                epochs=epochs,
                folds=folds,
                batch_size=batch_size,
                lr=args.lr,
                max_length=args.max_length,
                seed=args.seed,
                scratch_dir=scratch,
                use_cuda=use_cuda,
            )
            print(
                f"acc={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}  "
                f"fold_mean={float(np.mean(result.fold_macro_f1)):.4f}"
            )
            print(report)
            results.append(asdict(result))
            reports.append(f"### {tag}\n\n{report}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": "simpletransformers",
        "data": str(args.data),
        "n_samples": len(texts),
        "label_counts": dict(Counter(labels)),
        "folds": folds,
        "batch_size": batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "device": "cuda" if use_cuda else "cpu",
        "results": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = args.output.with_suffix(".txt")
    report_path.write_text("\n\n".join(reports) + "\n", encoding="utf-8")

    ranked = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    print("\n=== Rangiranje (macro-F1) ===")
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. {r['model_key']} epochs={r['epochs']}: "
            f"macro_f1={r['macro_f1']:.4f} acc={r['accuracy']:.4f}"
        )

    if not args.no_save_model and ranked:
        best = ranked[0]
        print(
            f"\nTreniram finalni model na celom skupu: "
            f"{best['model_key']} epochs={best['epochs']} ..."
        )
        train_full_and_save(
            model_key=best["model_key"],
            texts=texts,
            labels=labels,
            epochs=int(best["epochs"]),
            batch_size=batch_size,
            lr=args.lr,
            max_length=args.max_length,
            seed=args.seed,
            out_dir=args.model_dir,
            use_cuda=use_cuda,
            best_meta=best,
        )
        print(f"Model za inferencu: {args.model_dir}")

    print(f"\nRezultati: {args.output}")
    print(f"Izvestaji: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
