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

## Baseline

```bash
python baseline/train_baseline.py --quick
python baseline/train_baseline.py
python baseline/train_baseline.py --models nb --normalize stem --weightings tf idf tfidf
python baseline/infer.py -t "Pumpaj!"
python baseline/run_demo_infer.py
```

Izlaz: `baseline/output/`

## Enkoder (Simple Transformers)

**Važno (Windows + torch 2.0):** ne instaliraj `transformers` 5.x — koristi `requirements.txt`.

```bash
python encoder/train_encoder.py --quick
python encoder/train_encoder.py
python encoder/infer_encoder.py -t "Pumpaj!"
```

Izlaz: `encoder/output/`

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
