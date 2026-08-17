# Phase 3: Obučavanje i evaluacija modela

Stance klasifikacija komentara (3 klase): `NEUTRAL`, `ZA-VLAST`, `PROTIV-VLASTI`.

```text
phase3_model_training/
  baseline/     # LR, SVM, Naive Bayes
  encoder/      # BERTić, mBERT (Simple Transformers)
  decoder/      # Ollama / ChatGPT / Gemini prompting
  common/       # zajedničko učitavanje skupa
  samples/
```

**Napomena:** TF-IDF **nije** podrazumevani model, već jedna od tehnika pretprocesiranja.

Detalji: [`DOCS_TRAINING_INFERENCE.md`](DOCS_TRAINING_INFERENCE.md)

## Instalacija

```bash
pip install -r phase3_model_training/requirements.txt
```

Komande ispod pokreći iz `phase3_model_training/`.

Dataset: `../phase2_annotation/annotated/dataset_all.txt` (učitava se automatski).

## Analiza podataka (grafike za izveštaj)

```bash
python analyze_phase1_phase2.py
```

Izlaz: `output/data_analysis/*.png` + `ANALIZA_FAZA1_FAZA2.md`

## Baseline

Na **drugom računaru**: `git pull`, pa iz korena projekta:

```bash
pip install -r phase3_model_training/requirements.txt
cd phase3_model_training

# 1) brzi test da sve radi (~minut)
python baseline/train_baseline.py --quick

# 2) puni eksperiment (LR + SVM + NB, TF/IDF/TF-IDF, stem/lema, 10-fold)
python baseline/train_baseline.py
```

Izlaz: `baseline/output/` (`baseline_results.json`, `.txt`, i `.joblib` model — joblib se ne commituje).

Inferenca:

```bash
python baseline/infer_baseline.py -t "Pumpaj!"
```

## Enkoder (Simple Transformers)

**Važno (Windows + torch 2.0):** ne instaliraj `transformers` 5.x — koristi `requirements.txt`.

Pokreni **posle** baseline-a, iz `phase3_model_training/`:

```bash
# 1) smoke test (1 epoha, 2 folda, jedan model) — da vidiš da torch radi
python encoder/train_encoder.py --quick

# 2) puni eksperiment (BERTić + mBERT, 2/3/4 epohe, 10-fold CV)
python encoder/train_encoder.py
```

Bez GPU-a je sporo; za puni encoder bolje Colab / mašina sa NVIDIA karticom.

Izlaz: `encoder/output/` (rezultati JSON/TXT da; folder `encoder_best/` je prevelik za git).

## Dekoder (prompting)

Podrazumevano: **Ollama** (lokalno, bez API ključa). Već imaš `llama2` i `tinyllama`.

```powershell
# Ollama mora da radi u pozadini
ollama serve   # ako već nije pokrenut

cd phase3_model_training

# brzi test (20 primera)
python decoder/eval_decoder.py --quick

# drugi lokalni model
python decoder/eval_decoder.py --quick --ollama-model tinyllama

# pun eksperiment (sr/en × short/detailed × zero/few)
python decoder/eval_decoder.py

# inferenca
python decoder/infer_decoder.py -t "Pumpaj!"
python decoder/infer_decoder.py -i --shot few --lang sr
```

Opciono cloud (ako kasnije dobiješ ključ):

```powershell
$env:OPENAI_API_KEY = "sk-..."
python decoder/eval_decoder.py --providers openai --quick
```

Izlaz: `decoder/output/` (+ keš u `decoder/output/decoder_cache/`)

## Metrike

Glavna: **macro-F1**.
