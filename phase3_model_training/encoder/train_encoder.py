#!/usr/bin/env python3
"""Faza 3.2a — enkoderski modeli: BERTić i mBERT.

Fine-tune preko Hugging Face Trainer (isti modeli kao Simple Transformers).
Simple Transformers ClassificationModel na Windows+CUDA često ugasi proces
bez traceback-a (fp16, multiprocessing, putanje sa razmakom).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PHASE3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

print("Pokretanje train_encoder.py ...", flush=True)
print("Ucitavanje torch/transformers (st_compat) ...", flush=True)
import st_compat  # noqa: E402

st_compat.apply()
print("OK - torch spreman. Ucitavanje sklearn ...", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from common.data import DEFAULT_DATA, LABELS, load_dataset  # noqa: E402

DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "encoder_results.json"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "output" / "encoder_best"
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

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


class StanceDataset(Dataset):
    def __init__(self, encodings: dict, label_ids: list[int]):
        self.encodings = encodings
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[idx])
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tuning enkodera (HF Trainer): jedan model + jedan broj epoha "
            "po pokretanju (BERTić ili mBERT)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Primeri (iz phase3_model_training/):\n"
            "  python encoder/train_encoder.py --model bertic --epochs 3\n"
            "  python encoder/train_encoder.py --model mbert --epochs 2\n"
            "  python encoder/train_encoder.py --model bertic --epochs 4 --final-only\n"
            "  python encoder/train_encoder.py --quick\n"
            "  python encoder/train_encoder.py --all   # opcioni puni grid\n"
        ),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON izlaz (default: output/encoder_results_<model>_e<epochs>.json)",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--model",
        type=str,
        default="bertic",
        choices=list(MODEL_PRESETS.keys()),
        help="Koji encoder trenirati (jedan po pokretanju). Default: bertic",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Broj epoha za ovaj run (jedan broj). Default: 3",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Puni eksperiment: bertic+mbert × epohe 2,3,4 (ignoriše --model/--epochs)",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 folda, 1 epoha (isti --model)",
    )
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Preskoči CV: istreniraj na celom skupu i sačuvaj model za inferencu.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Mešana preciznost (brže; na Windows+CUDA može da ruši proces).",
    )
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


def encode_labels(labels: list[str]) -> list[int]:
    return [LABEL2ID[lab] for lab in labels]


def make_training_args(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    use_cuda: bool,
    fp16: bool,
):
    from transformers import TrainingArguments

    common = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=seed,
        fp16=bool(fp16 and use_cuda),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        overwrite_output_dir=True,
        disable_tqdm=False,
    )
    try:
        return TrainingArguments(**common, eval_strategy="no")
    except TypeError:
        return TrainingArguments(**common, evaluation_strategy="no")


def load_backbone(model_key: str, max_length: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = MODEL_PRESETS[model_key]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return tokenizer, model, max_length


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> dict:
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
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
    fp16: bool = False,
) -> list[str]:
    from transformers import Trainer

    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("    ucitavam model ...", flush=True)
    tokenizer, model, max_length = load_backbone(model_key, max_length)
    train_ds = StanceDataset(
        tokenize_texts(tokenizer, train_texts, max_length),
        encode_labels(train_labels),
    )
    print(f"    treniram {len(train_ds)} primera, {epochs} epoha ...", flush=True)
    trainer = Trainer(
        model=model,
        args=make_training_args(work_dir, epochs, batch_size, lr, seed, use_cuda, fp16),
        train_dataset=train_ds,
    )
    trainer.train()
    print("    predikcija ...", flush=True)

    model.eval()
    device = next(model.parameters()).device
    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            chunk = test_texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            ids = logits.argmax(dim=-1).tolist()
            preds.extend(ID2LABEL[int(j)] for j in ids)

    del trainer, model
    if use_cuda:
        torch.cuda.empty_cache()
    shutil.rmtree(work_dir, ignore_errors=True)
    return preds


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
    fp16: bool = False,
) -> tuple[EncoderResult, str]:
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_true: list[str] = []
    all_pred: list[str] = []
    fold_scores: list[float] = []
    model_name = MODEL_PRESETS[model_key]["model_name"]

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(texts, y), 1):
        print(f"  fold {fold_i}/{folds} ...", flush=True)
        if fold_i == 1:
            print(
                "    (prvi fold skida model sa Hugging Face ako nije u kesu)",
                flush=True,
            )
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
            fp16=fp16,
        )
        fold_macro = float(
            f1_score(
                test_labels, preds, average="macro", labels=list(LABELS), zero_division=0
            )
        )
        fold_scores.append(fold_macro)
        all_true.extend(test_labels)
        all_pred.extend(preds)
        print(f"    fold macro-F1={fold_macro:.4f}", flush=True)

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
    fp16: bool = False,
) -> None:
    from transformers import Trainer

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, max_length = load_backbone(model_key, max_length)
    train_ds = StanceDataset(
        tokenize_texts(tokenizer, texts, max_length),
        encode_labels(labels),
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(out_dir / "_train", epochs, batch_size, lr, seed, use_cuda, fp16),
        train_dataset=train_ds,
    )
    trainer.train()
    tokenizer.save_pretrained(out_dir)
    trainer.save_model(out_dir)
    shutil.rmtree(out_dir / "_train", ignore_errors=True)

    meta = {
        "framework": "transformers.Trainer",
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


def scratch_root() -> Path:
    # Putanja projekta ima razmak ("Ana Bulatovic") — HF/ST to ne voli na Windows.
    base = Path(tempfile.gettempdir()) / "opj_encoder_scratch"
    base.mkdir(parents=True, exist_ok=True)
    return base


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()
    if not args.data.is_file():
        print(f"Nema dataset fajla: {args.data}", file=sys.stderr)
        return 1

    print("Ucitavanje transformers Trainer ...", flush=True)
    try:
        from transformers import AutoTokenizer  # noqa: F401
        from transformers import Trainer  # noqa: F401
    except ImportError:
        print(
            "Nedostaju paketi. Instaliraj:\n"
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
    print("Framework: Hugging Face Trainer (BERTić / mBERT)")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print("fp16: iskljucen. Ukljuci sa --fp16 ako zelis.", flush=True)
    else:
        print(
            "Upozorenje: PyTorch ne vidi CUDA.\n"
            f"  torch={torch.__version__}  torch.version.cuda={torch.version.cuda}",
            flush=True,
        )

    if args.all:
        models = ["bertic", "mbert"]
        epochs_list = [2, 3, 4]
        print("\n=== FULL GRID: bertic+mbert × epochs 2,3,4 ===", flush=True)
    else:
        models = [args.model]
        epochs_list = [int(args.epochs)]

    folds = args.folds
    batch_size = args.batch_size

    if args.quick:
        models = [models[0]]
        epochs_list = [1]
        folds = min(2, folds)
        batch_size = min(8, batch_size)
        print("\n=== QUICK MODE (HF Trainer smoke test) ===")

    if args.output is None:
        if args.all:
            args.output = DEFAULT_OUTPUT
        else:
            args.output = (
                SCRIPT_DIR
                / "output"
                / f"encoder_results_{models[0]}_e{epochs_list[0]}.json"
            )

    n_combos = len(models) * len(epochs_list)
    print(
        f"Plan: model(i)={models}  epohe={epochs_list}  folds={folds}  "
        f"→ {n_combos} kombinacija "
        f"({n_combos * folds} treninga + eventualni final)",
        flush=True,
    )

    if args.final_only:
        model_key = models[0]
        epochs = int(epochs_list[0])
        print(
            f"\n=== FINAL-ONLY: {model_key} epochs={epochs} na celom skupu ===",
            flush=True,
        )
        train_full_and_save(
            model_key=model_key,
            texts=texts,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=args.lr,
            max_length=args.max_length,
            seed=args.seed,
            out_dir=args.model_dir,
            use_cuda=use_cuda,
            best_meta={
                "model_key": model_key,
                "epochs": epochs,
                "macro_f1": None,
                "note": "final-only (nema CV u ovom pokretanju)",
            },
            fp16=args.fp16,
        )
        print(f"Model za inferencu: {args.model_dir}")
        print("Dalje: python encoder/infer_encoder.py -t \"Pumpaj!\"")
        return 0

    counts = Counter(labels)
    min_class = min(counts.values())
    if min_class < folds:
        print(
            f"Upozorenje: najmanja klasa ima {min_class} primera, "
            f"folds={folds}. Smanjujem folds na {min_class}.",
            file=sys.stderr,
        )
        folds = min_class

    scratch = scratch_root()
    results: list[dict] = []
    reports: list[str] = []

    def save_partial() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "framework": "transformers.Trainer",
            "data": str(args.data),
            "n_samples": len(texts),
            "label_counts": dict(Counter(labels)),
            "folds": folds,
            "batch_size": batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "device": "cuda" if use_cuda else "cpu",
            "results": results,
            "partial": True,
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        args.output.with_suffix(".txt").write_text(
            "\n\n".join(reports) + "\n", encoding="utf-8"
        )

    try:
        for model_key in models:
            model_name = MODEL_PRESETS[model_key]["model_name"]
            for epochs in epochs_list:
                tag = f"{model_key} ({model_name}) | epochs={epochs}"
                print(f"\n=== {tag} ===", flush=True)
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
                    fp16=args.fp16,
                )
                print(
                    f"acc={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}  "
                    f"fold_mean={float(np.mean(result.fold_macro_f1)):.4f}",
                    flush=True,
                )
                print(report, flush=True)
                results.append(asdict(result))
                reports.append(f"### {tag}\n\n{report}")
                save_partial()
                print(f"Sacuvano (delimicno): {args.output}", flush=True)
    except KeyboardInterrupt:
        print("\nPrekinuto. Cuva se to sto je do sada zavrseno ...", flush=True)
        if results:
            save_partial()
            print(f"Delimicni rezultati: {args.output}", flush=True)
            print(f"Izvestaj: {args.output.with_suffix('.txt')}", flush=True)
        else:
            print(
                "Nijedna kombinacija nije stigla do kraja 10 foldova — nema JSON-a.",
                flush=True,
            )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": "transformers.Trainer",
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
            fp16=args.fp16,
        )
        print(f"Model za inferencu: {args.model_dir}")

    print(f"\nRezultati: {args.output}")
    print(f"Izvestaji: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
