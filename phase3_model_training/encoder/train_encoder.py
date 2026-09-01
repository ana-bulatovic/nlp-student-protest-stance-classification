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
from dataclasses import asdict, dataclass, field
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
    confusion_matrix,
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


def default_model_dir(model_key: str) -> Path:
    return SCRIPT_DIR / "output" / f"encoder_{model_key}"


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
    confusion_matrix: list[list[int]]
    model_dir: str | None = None
    # Macro-F1 (i accuracy) posle SVAKE epohe, usrednjeno preko foldova.
    # Obavezan deo svakog CV pokretanja — ne opciono, jer izveštaj mora da
    # obrazloži izbor broja epoha (v. UPUTSTVA_ANOTACIJA/ENCODER_IZVESTAJ §5).
    epoch_curve: list[dict] = field(default_factory=list)


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
            "Fine-tuning enkodera (HF Trainer): BERTić / mBERT. "
            "Preporučeno: --compare (oba modela, 5 epoha, CV, odvojeni folderi + izveštaj)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Primeri (iz phase3_model_training/):\n"
            "  python encoder/train_encoder.py --compare\n"
            "      # bertic pa mbert, 5 epoha, 10-fold; čuva encoder_bertic/ i encoder_mbert/\n"
            "  python encoder/train_encoder.py --model bertic --epochs 5\n"
            "  python encoder/train_encoder.py --model mbert --epochs 5 --final-only\n"
            "  python encoder/train_encoder.py --compare --quick\n"
        ),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON izlaz (default zavisi od moda)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Folder za sačuvani model (default: output/encoder_<model>)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bertic",
        choices=list(MODEL_PRESETS.keys()),
        help="Jedan model (ignorisano sa --compare). Default: bertic",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Broj epoha. Default: 5",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="BERTić pa mBERT (isti --epochs), CV, oba foldera + poređenje/izveštaj",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help=(
            "Nastavi prekinuti run: učitaj postojeći --output JSON, "
            "preskoči CV za modele koji već imaju rezultat; "
            "ako folder modela nedostaje, uradi samo finalni trening."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Puni grid: bertic+mbert × epohe 2,3,4 (retko; za poređenje epoha)",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Default 32 (RTX 3090, 24GB VRAM); smanji za manje GPU-ove ili CPU.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 folda, 1 epoha",
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
        help="Mešana preciznost fp16 (brže; na Windows+CUDA može da ruši proces).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help=(
            "Mešana preciznost bf16 — preporučeno na Ampere+ (RTX 3090): "
            "isti raspon eksponenta kao fp32, bez gubitka skaliranja."
        ),
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Isključi TF32 matmul/cudnn (podrazumevano uključeno na Ampere+ GPU).",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=None,
        help="Broj DataLoader worker procesa (default: 4 na CUDA, 0 na CPU).",
    )
    return parser.parse_args()


def compute_metrics_arrays(y_true: list[str], y_pred: list[str]) -> dict:
    labels = list(LABELS)
    per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "per_class_f1": {lab: float(v) for lab, v in zip(labels, per)},
        "confusion_matrix": cm.astype(int).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        ),
    }


def encode_labels(labels: list[str]) -> list[int]:
    return [LABEL2ID[lab] for lab in labels]


def build_compute_metrics():
    label_ids = list(range(len(LABELS)))

    def compute_metrics(eval_pred) -> dict:
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0)
            ),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    return compute_metrics


def make_training_args(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    use_cuda: bool,
    fp16: bool,
    bf16: bool = False,
    dataloader_workers: int | None = None,
    eval_strategy: str = "no",
):
    from transformers import TrainingArguments

    workers = dataloader_workers if dataloader_workers is not None else (4 if use_cuda else 0)
    common = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        # Nema gradijenata pri evaluaciji — veći eval batch bez dodatne VRAM cene.
        per_device_eval_batch_size=batch_size * 2 if use_cuda else batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=seed,
        fp16=bool(fp16 and use_cuda and not bf16),
        bf16=bool(bf16 and use_cuda),
        dataloader_num_workers=workers,
        dataloader_pin_memory=use_cuda,
        overwrite_output_dir=True,
        disable_tqdm=False,
    )
    # Spreči Trainer da forsira safetensors pri eventualnom save-u.
    try:
        return TrainingArguments(
            **common, eval_strategy=eval_strategy, save_safetensors=False
        )
    except TypeError:
        try:
            return TrainingArguments(
                **common, evaluation_strategy=eval_strategy, save_safetensors=False
            )
        except TypeError:
            try:
                return TrainingArguments(**common, eval_strategy=eval_strategy)
            except TypeError:
                return TrainingArguments(**common, evaluation_strategy=eval_strategy)


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


def extract_epoch_curve(trainer) -> list[dict]:
    """Macro-F1/accuracy zabeleženi na kraju svake epohe (eval_strategy='epoch').

    Obavezan deo svakog fold treninga — ovo je izvor podataka za diskusiju
    "uticaj broja epoha" u izveštaju (koliko epoha zaista doprinosi rezultatu).
    """
    curve: list[dict] = []
    for entry in trainer.state.log_history:
        if "eval_macro_f1" not in entry:
            continue
        curve.append(
            {
                "epoch": int(round(entry["epoch"])),
                "macro_f1": float(entry["eval_macro_f1"]),
                "accuracy": float(entry.get("eval_accuracy", float("nan"))),
            }
        )
    return curve


def train_one_fold(
    model_key: str,
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    work_dir: Path,
    use_cuda: bool,
    fp16: bool = False,
    bf16: bool = False,
    dataloader_workers: int | None = None,
) -> tuple[list[str], list[dict]]:
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
    # Held-out fold služi i kao eval skup — omogućava praćenje macro-F1 posle
    # svake epohe u OVOM istom treningu, bez odvojenih punih pokretanja po
    # broju epoha (obavezno, v. extract_epoch_curve).
    eval_ds = StanceDataset(
        tokenize_texts(tokenizer, test_texts, max_length),
        encode_labels(test_labels),
    )
    print(f"    treniram {len(train_ds)} primera, {epochs} epoha ...", flush=True)
    trainer = Trainer(
        model=model,
        args=make_training_args(
            work_dir, epochs, batch_size, lr, seed, use_cuda, fp16,
            bf16=bf16, dataloader_workers=dataloader_workers, eval_strategy="epoch",
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=build_compute_metrics(),
    )
    trainer.train()
    epoch_curve = extract_epoch_curve(trainer)
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
    return preds, epoch_curve


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
    bf16: bool = False,
    dataloader_workers: int | None = None,
) -> tuple[EncoderResult, str]:
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_true: list[str] = []
    all_pred: list[str] = []
    fold_scores: list[float] = []
    fold_curves: list[list[dict]] = []
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
        preds, epoch_curve = train_one_fold(
            model_key=model_key,
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_length=max_length,
            seed=seed,
            work_dir=work,
            use_cuda=use_cuda,
            fp16=fp16,
            bf16=bf16,
            dataloader_workers=dataloader_workers,
        )
        fold_curves.append(epoch_curve)
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
    epoch_curve_summary = aggregate_epoch_curves(fold_curves)
    print(f"  Uticaj broja epoha ({model_key}, usrednjeno preko {folds} foldova):", flush=True)
    for row in epoch_curve_summary:
        print(
            f"    epoha {row['epoch']}: macro-F1={row['macro_f1_mean']:.4f} "
            f"(std={row['macro_f1_std']:.4f}, min={row['macro_f1_min']:.4f}, "
            f"max={row['macro_f1_max']:.4f})",
            flush=True,
        )
    result = EncoderResult(
        model_key=model_key,
        model_name=model_name,
        epochs=epochs,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        per_class_f1=metrics["per_class_f1"],
        fold_macro_f1=fold_scores,
        confusion_matrix=metrics["confusion_matrix"],
        epoch_curve=epoch_curve_summary,
    )
    return result, metrics["report"]


def aggregate_epoch_curves(fold_curves: list[list[dict]]) -> list[dict]:
    """Usrednjava per-epoch macro-F1/accuracy preko foldova.

    Obavezan izlaz svakog CV pokretanja (v. EncoderResult.epoch_curve) —
    daje podatke za obrazloženje broja epoha u izveštaju, bez posebnih
    dodatnih pokretanja treninga po broju epoha.
    """
    by_epoch: dict[int, list[dict]] = {}
    for curve in fold_curves:
        for row in curve:
            by_epoch.setdefault(row["epoch"], []).append(row)
    summary: list[dict] = []
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        f1_vals = [r["macro_f1"] for r in rows]
        acc_vals = [r["accuracy"] for r in rows]
        summary.append(
            {
                "epoch": epoch,
                "n_folds": len(rows),
                "macro_f1_mean": float(np.mean(f1_vals)),
                "macro_f1_std": float(np.std(f1_vals)),
                "macro_f1_min": float(np.min(f1_vals)),
                "macro_f1_max": float(np.max(f1_vals)),
                "accuracy_mean": float(np.mean(acc_vals)),
            }
        )
    return summary


def make_weights_contiguous(model) -> None:
    """Safetensors / neki HF save putevi padaju na non-contiguous Electra težinama."""
    for p in model.parameters():
        if p.data is not None and not p.data.is_contiguous():
            p.data = p.data.contiguous()
    for b in model.buffers():
        if b.data is not None and not b.data.is_contiguous():
            b.data = b.data.contiguous()


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
    bf16: bool = False,
    dataloader_workers: int | None = None,
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
        args=make_training_args(
            out_dir / "_train", epochs, batch_size, lr, seed, use_cuda, fp16,
            bf16=bf16, dataloader_workers=dataloader_workers,
        ),
        train_dataset=train_ds,
    )
    trainer.train()
    make_weights_contiguous(model)
    tokenizer.save_pretrained(out_dir)
    # Direktno na model: Trainer.save_model ponekad forsira safetensors=True.
    model.save_pretrained(out_dir, safe_serialization=False)
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

    is_ampere_plus = False
    if use_cuda:
        major, _minor = torch.cuda.get_device_capability(0)
        is_ampere_plus = major >= 8  # RTX 3090 = sm_86
        print(f"GPU: {torch.cuda.get_device_name(0)} (compute capability {major}.{_minor})", flush=True)
        if not args.no_tf32 and is_ampere_plus:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32: ukljucen (Ampere+ GPU detektovan).", flush=True)
        torch.backends.cudnn.benchmark = True
        if args.bf16 and not is_ampere_plus:
            print(
                "Upozorenje: --bf16 trazen ali GPU nije Ampere+; "
                "moze biti sporije ili nepodrzano.",
                flush=True,
            )
        if args.fp16 and args.bf16:
            print("Napomena: i --fp16 i --bf16 prosledjeni — koristi se bf16.", flush=True)
        if not args.fp16 and not args.bf16:
            print(
                "Mesana preciznost iskljucena. Ukljuci sa --bf16 (preporuceno na RTX 3090+) "
                "ili --fp16.",
                flush=True,
            )
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
    elif args.compare:
        models = ["bertic", "mbert"]
        epochs_list = [int(args.epochs)]
        print(
            f"\n=== COMPARE: bertic pa mbert | epochs={epochs_list[0]} | CV ===",
            flush=True,
        )
    else:
        models = [args.model]
        epochs_list = [int(args.epochs)]

    folds = args.folds
    batch_size = args.batch_size

    if args.quick:
        if not args.compare and not args.all:
            models = [models[0]]
        epochs_list = [1]
        folds = min(2, folds)
        batch_size = min(8, batch_size)
        print("\n=== QUICK MODE (HF Trainer smoke test) ===")

    if args.output is None:
        if args.compare:
            args.output = (
                SCRIPT_DIR / "output" / f"encoder_results_compare_e{epochs_list[0]}.json"
            )
        elif args.all:
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
        out_dir = args.model_dir or default_model_dir(model_key)
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
            out_dir=out_dir,
            use_cuda=use_cuda,
            best_meta={
                "model_key": model_key,
                "epochs": epochs,
                "macro_f1": None,
                "note": "final-only (nema CV u ovom pokretanju)",
            },
            fp16=args.fp16,
            bf16=args.bf16,
            dataloader_workers=args.dataloader_workers,
        )
        print(f"Model za inferencu: {out_dir}")
        print(
            f'Dalje: python encoder/infer_encoder.py --model {model_key} -t "Pumpaj!"'
        )
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
    # Jedan finalni model po model_key (ako --all ima više epoha, uzima najbolju)
    best_by_model: dict[str, dict] = {}

    if args.continue_run and args.output.is_file():
        try:
            prev = json.loads(args.output.read_text(encoding="utf-8"))
            for row in prev.get("results") or []:
                results.append(row)
                key = row.get("model_key")
                if key and (
                    key not in best_by_model
                    or float(row["macro_f1"]) > float(best_by_model[key]["macro_f1"])
                ):
                    best_by_model[key] = row
            txt_path = args.output.with_suffix(".txt")
            if txt_path.is_file():
                reports.append(txt_path.read_text(encoding="utf-8").rstrip())
            print(
                f"Nastavak: ucitano {len(results)} rezultata iz {args.output} "
                f"(modeli: {sorted(best_by_model)})",
                flush=True,
            )
        except Exception as exc:
            print(f"Upozorenje: --continue nije ucitao JSON ({exc})", flush=True)

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
            "labels": list(LABELS),
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
            already = model_key in best_by_model and args.continue_run
            if already:
                print(
                    f"\n=== PRESKACEM CV: {model_key} vec u JSON "
                    f"(macro_f1={best_by_model[model_key]['macro_f1']:.4f}) ===",
                    flush=True,
                )
            else:
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
                        bf16=args.bf16,
                        dataloader_workers=args.dataloader_workers,
                    )
                    print(
                        f"acc={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}  "
                        f"fold_mean={float(np.mean(result.fold_macro_f1)):.4f}",
                        flush=True,
                    )
                    print(report, flush=True)
                    row = asdict(result)
                    results.append(row)
                    reports.append(f"### {tag}\n\n{report}")
                    prev = best_by_model.get(model_key)
                    if prev is None or row["macro_f1"] > prev["macro_f1"]:
                        best_by_model[model_key] = row
                    save_partial()
                    print(f"Sacuvano (delimicno): {args.output}", flush=True)

            if not args.no_save_model and model_key in best_by_model:
                best = best_by_model[model_key]
                out_dir = (
                    args.model_dir
                    if args.model_dir and len(models) == 1
                    else default_model_dir(model_key)
                )
                meta_ok = (out_dir / "stance_meta.json").is_file()
                weights_ok = any(out_dir.glob("pytorch_model*.bin")) or any(
                    out_dir.glob("model.safetensors")
                )
                if args.continue_run and meta_ok and weights_ok:
                    print(
                        f"Preskacem finalni trening: model vec postoji u {out_dir}",
                        flush=True,
                    )
                    for r in results:
                        if r["model_key"] == model_key and r["epochs"] == best["epochs"]:
                            r["model_dir"] = str(out_dir)
                    best["model_dir"] = str(out_dir)
                    save_partial()
                    continue

                print(
                    f"\n=== Finalni model na celom skupu: {model_key} "
                    f"epochs={best['epochs']} → {out_dir} ===",
                    flush=True,
                )
                train_full_and_save(
                    model_key=model_key,
                    texts=texts,
                    labels=labels,
                    epochs=int(best["epochs"]),
                    batch_size=batch_size,
                    lr=args.lr,
                    max_length=args.max_length,
                    seed=args.seed,
                    out_dir=out_dir,
                    use_cuda=use_cuda,
                    best_meta=best,
                    fp16=args.fp16,
                    bf16=args.bf16,
                    dataloader_workers=args.dataloader_workers,
                )
                for r in results:
                    if r["model_key"] == model_key and r["epochs"] == best["epochs"]:
                        r["model_dir"] = str(out_dir)
                best["model_dir"] = str(out_dir)
                save_partial()
                print(f"Model za inferencu: {out_dir}", flush=True)
    except KeyboardInterrupt:
        print("\nPrekinuto. Cuva se to sto je do sada zavrseno ...", flush=True)
        if results:
            save_partial()
            print(f"Delimicni rezultati: {args.output}", flush=True)
            print(f"Izvestaj: {args.output.with_suffix('.txt')}", flush=True)
        else:
            print(
                "Nijedna kombinacija nije stigla do kraja foldova — nema JSON-a.",
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
        "labels": list(LABELS),
        "results": results,
        "compare": bool(args.compare),
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

    if len(ranked) >= 2:
        winner = ranked[0]
        print(
            f"\n>>> Bolji po macro-F1: {winner['model_key']} "
            f"(epochs={winner['epochs']}, macro_f1={winner['macro_f1']:.4f})",
            flush=True,
        )

    print(f"\nRezultati: {args.output}")
    print(f"Izvestaji: {report_path}")
    for key, best in best_by_model.items():
        md = best.get("model_dir") or default_model_dir(key)
        print(
            f'Inferenca {key}: python encoder/infer_encoder.py --model {key} -t "Pumpaj!"'
        )
        print(f"  folder: {md}")

    try:
        from report_encoder import generate as write_encoder_report

        print("\nGenerisem grafike i izvestaj za encoder ...", flush=True)
        write_encoder_report(args.output)
    except Exception as exc:
        print(f"Upozorenje: encoder izvestaj nije napravljen ({exc})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
