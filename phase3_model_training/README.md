# Phase 3: Obučavanje i evaluacija modela

Stance klasifikacija komentara (3 klase): `NEUTRAL`, `ZA-VLAST`, `PROTIV-VLASTI`.

```text
phase3_model_training/
  baseline/     # LR, SVM, Naive Bayes
  encoder/      # BERTić, mBERT (HF Trainer)
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

Posle punog treninga skripta sama pravi statistiku za izveštaj:

- `BASELINE_IZVESTAJ.md`
- `output/baseline_analysis/*.png`

Ručno (ako JSON već postoji):

```bash
python baseline/report_baseline.py
```

Inferenca:

```bash
python baseline/infer_baseline.py -t "Pumpaj!"
```

## Enkoder (Hugging Face Trainer)

**Važno (Windows + torch 2.0):** ne instaliraj `transformers` 5.x — koristi `requirements.txt`.

Pokreni iz `phase3_model_training/`. Preporučeni tok: **BERTić pa mBERT**, **4 epohe**, **10-fold CV**, modeli u **dva foldera**, pa automatsko poređenje + grafike.

```bash
# 1) brzi smoke test (oba modela, 2 folda, 1 epoha)
python encoder/train_encoder.py --compare --quick

# 2) puni uporedni trening (bertic → mbert, 4 epohe, 10-fold; čuva oba modela)
python encoder/train_encoder.py --compare

# samo jedan model (ako treba odvojeno)
python encoder/train_encoder.py --model bertic --epochs 4
python encoder/train_encoder.py --model mbert --epochs 4

# samo sačuvaj model bez CV
python encoder/train_encoder.py --model bertic --epochs 4 --final-only
```

Posle `--compare` skripta:

1. radi CV za **bertic**, pa ga trenira na celom skupu → `encoder/output/encoder_bertic/`
2. isto za **mbert** → `encoder/output/encoder_mbert/`
3. štampa ko je bolji (macro-F1)
4. generiše izveštaj i grafike (ili ručno: `python encoder/report_encoder.py`)

Izlaz za izveštaj:

- `ENCODER_IZVESTAJ.md`
- `output/encoder_analysis/*.png` — poređenje metrika, F1 po klasi, foldovi, matrice konfuzije, ranking
- JSON: `encoder/output/encoder_results_compare_e4.json` (+ `.txt` classification report)

Inferenca (oba modela):

```bash
python encoder/infer_encoder.py --model bertic -t "Pumpaj!"
python encoder/infer_encoder.py --model mbert -t "Pumpaj!"
python encoder/infer_encoder.py --model bertic -i
```

Bez GPU-a je sporo. Folderi modela (`encoder_bertic/`, `encoder_mbert/`) su preveliki za git.

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
