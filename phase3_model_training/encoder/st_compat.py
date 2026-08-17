#!/usr/bin/env python3
"""Kompatibilnost Simple Transformers ↔ transformers 4.4x / torch 2.0.

Pozovi `apply()` PRE `from simpletransformers...`, i idealno PRE sklearn importa
(na nekim Windows+CPU setupima sklearn-pa-transformers pravi ACCESS_VIOLATION).
"""

from __future__ import annotations

# Stara imena koja Simple Transformers i dalje uvozi; u transformers>=4.44
# postoje kao SequenceSummary.
_SEQUENCE_SUMMARY_ALIASES = (
    ("transformers.models.xlnet.modeling_xlnet", "XLNetSequenceSummary"),
    ("transformers.models.xlm.modeling_xlm", "XLMSequenceSummary"),
    ("transformers.models.flaubert.modeling_flaubert", "FlaubertSequenceSummary"),
)


def apply() -> None:
    # 1) Prvo torch/transformers — pre sklearn ako je moguće
    try:
        import torch
        import transformers
        from transformers.modeling_utils import SequenceSummary
    except ImportError:
        return

    torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    tf_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tf_ver >= (5, 0) and torch_ver < (2, 4):
        raise SystemExit(
            "Nekompatibilne verzije: transformers "
            f"{transformers.__version__} zahteva torch>=2.4, a instaliran je "
            f"torch {torch.__version__}.\n"
            "Popravi sa:\n"
            '  pip install "transformers>=4.36,<4.45" "huggingface-hub>=0.23,<1" '
            '"tokenizers>=0.19,<0.20"\n'
        )

    # 2) Alias *SequenceSummary na pravim modulima
    for module_path, old_name in _SEQUENCE_SUMMARY_ALIASES:
        _alias_sequence_summary(module_path, old_name, SequenceSummary)

    # 3) Simple Transformers ne prosledjuje safe_serialization=False, a neke
    # arhitekture (npr. ELECTRA/BERTic) imaju non-contiguous tenzore koje
    # safetensors odbija da sacuva ("You are trying to save a non contiguous
    # tensor"). Vrati na stari torch.save format koji to dozvoljava.
    _patch_save_pretrained()


def _patch_save_pretrained() -> None:
    from transformers.modeling_utils import PreTrainedModel

    if getattr(PreTrainedModel.save_pretrained, "_stance_patched", False):
        return

    original_save_pretrained = PreTrainedModel.save_pretrained

    def patched_save_pretrained(self, *args, **kwargs):
        kwargs.setdefault("safe_serialization", False)
        return original_save_pretrained(self, *args, **kwargs)

    patched_save_pretrained._stance_patched = True
    PreTrainedModel.save_pretrained = patched_save_pretrained


def _alias_sequence_summary(module_path: str, old_name: str, sequence_summary_cls) -> None:
    try:
        mod = __import__(module_path, fromlist=["*"])
    except Exception:
        return
    if not hasattr(mod, old_name):
        setattr(mod, old_name, sequence_summary_cls)
