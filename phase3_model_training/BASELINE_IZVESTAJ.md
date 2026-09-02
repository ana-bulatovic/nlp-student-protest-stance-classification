# Baseline izveštaj (Faza 3.1)

Klasični modeli: logistička regresija, linear SVM, multinomialni naivni Bajes. Eksperimentalni faktori: ponderisanje (**TF / IDF / TF-IDF**), **lowercasing**, normalizacija tokena (**none / stem / lemma**). Evaluacija: ugnežđena stratifikovana unakrsna validacija, glavna metrika **macro-F1**.

## 1. Postavka

| Stavka | Vrednost |
|---|---|
| Skup | `dataset_all.txt` |
| Broj primera | 2874 |
| Klase | NEUTRAL=668, ZA-VLAST=1038, PROTIV-VLASTI=1168 |
| Spoljašnji foldovi | 10 |
| Unutrašnji foldovi (hiperparametri) | 3 |
| Broj konfiguracija | 54 |
| Hiperparametri LR/SVM | [0.1, 1.0, 10.0] |
| Hiperparametri NB (alpha) | [0.1, 0.5, 1.0] |

TF/IDF/TF-IDF, lowercasing, stem i lemma su tehnike pretprocesiranja; nijedna nije podrazumevani 'default model'.

## 2. Najbolji rezultat

**Pobednik:** `LR TF lc stem`

| Metrika | Vrednost |
|---|---|
| macro-F1 | **0.5532** |
| accuracy | 0.5825 |
| weighted-F1 | 0.5791 |
| Najbolji hiperparametri | {'clf__C': 1.0} |
| F1 `NEUTRAL` | 0.3806 |
| F1 `ZA-VLAST` | 0.6451 |
| F1 `PROTIV-VLASTI` | 0.6340 |

Najslabija konfiguracija: `SVM IDF no-lc none` (macro-F1=0.4646). Raspon: **0.089** poena macro-F1.

![Ranking](output/baseline_analysis/01_ranking_macro_f1.png)

## 3. Top 10 konfiguracija

| # | Model | Ponder | Lowercase | Norm | macro-F1 | acc | F1 NEUTRAL | F1 ZA-VLAST | F1 PROTIV |
|---|---|---|---|---|---|---|---|---|---|
| 1 | LR | TF | da | stem | 0.5532 | 0.5825 | 0.381 | 0.645 | 0.634 |
| 2 | LR | TF | ne | stem | 0.5532 | 0.5825 | 0.381 | 0.645 | 0.634 |
| 3 | LR | IDF | da | stem | 0.5485 | 0.5734 | 0.390 | 0.637 | 0.619 |
| 4 | LR | IDF | ne | stem | 0.5485 | 0.5734 | 0.390 | 0.637 | 0.619 |
| 5 | SVM | TF | da | stem | 0.5475 | 0.5846 | 0.356 | 0.643 | 0.643 |
| 6 | SVM | TF | ne | stem | 0.5475 | 0.5846 | 0.356 | 0.643 | 0.643 |
| 7 | LR | TFIDF | da | stem | 0.5404 | 0.5800 | 0.345 | 0.635 | 0.641 |
| 8 | LR | TFIDF | ne | stem | 0.5404 | 0.5800 | 0.345 | 0.635 | 0.641 |
| 9 | NB | IDF | da | stem | 0.5377 | 0.5682 | 0.353 | 0.639 | 0.622 |
| 10 | NB | IDF | ne | stem | 0.5377 | 0.5682 | 0.353 | 0.639 | 0.622 |

## 4. Uticaj faktora (prosečan macro-F1)

### 4.1 Model

| Model | Prosečan macro-F1 | Najbolji macro-F1 |
|---|---|---|
| Logistička regresija | 0.5215 | 0.5532 |
| Linear SVM | 0.5096 | 0.5475 |
| Naivni Bajes | 0.5197 | 0.5377 |

![Po modelu](output/baseline_analysis/02_f1_by_model.png)

### 4.2 Ponderisanje (TF / IDF / TF-IDF)

| Ponder | Prosečan macro-F1 |
|---|---|
| TF | 0.5217 |
| IDF | 0.5158 |
| TFIDF | 0.5132 |

![Ponder](output/baseline_analysis/03_f1_by_weighting.png)

![Heatmap](output/baseline_analysis/05_heatmap_model_weighting.png)

### 4.3 Normalizacija tokena

| Normalizacija | Prosečan macro-F1 |
|---|---|
| none | 0.5004 |
| stem | 0.5379 |
| lemma | 0.5125 |

![Norm](output/baseline_analysis/04_f1_by_normalize.png)

### 4.4 Lowercasing

| Lowercasing | Prosečan macro-F1 |
|---|---|
| uključen | 0.5210 |
| isključen | 0.5129 |

## 5. F1 po klasi

![Best class](output/baseline_analysis/06_best_per_class_f1.png)

![Best model class](output/baseline_analysis/07_best_model_per_class.png)

Najbolja konfiguracija po modelu:

| Model | Konfiguracija | macro-F1 | acc |
|---|---|---|---|
| Logistička regresija | LR TF lc stem | 0.5532 | 0.5825 |
| Linear SVM | SVM TF lc stem | 0.5475 | 0.5846 |
| Naivni Bajes | NB IDF lc stem | 0.5377 | 0.5682 |

## 6. Ablacija: n-grami i filtriranje rečnika (Faza 3.1b)

Nakon osnovnih 54 konfiguracije (Poglavlja 1–6), pobedničke konfiguracije po modelu — **LR TF lc stem**, **SVM TF lc stem**, **NB IDF lc stem** — dodatno su testirane variranjem dva faktora vektorizacije, dok su ponderisanje, lowercasing i normalizacija ostali fiksirani na najbolje pronađene vrednosti:

- **Opseg n-grama**: unigram / unigram+bigram (trenutno) / unigram+bigram+trigram
- **Filtriranje rečnika**: bez filtriranja (min_df=1, max_df=1.0) / trenutno (min_df=2, max_df=0.95) / strože (min_df=5, max_df=0.90)

Za svaki model testirano je po 6 konfiguracija (3 za n-grame, 3 za filtriranje); konfiguracija „trenutno" je zajednička referentna tačka za oba faktora, pa se njen rezultat ponavlja u oba bloka. Ukupno **18 pokretanja**, bez punog faktorijalnog ukrštanja (3×3=9 po modelu, 27 ukupno) i bez ponavljanja cele mreže od 54 baznih konfiguracija (što bi dalo 300+ pokretanja).

### 6.1 Rezultati

| Model | Faktor | Vrednost | macro-F1 | acc | F1 NEUTRAL | F1 ZA-VLAST | F1 PROTIV |
|---|---|---|---|---|---|---|---|
| LR | n-grami | unigram | 0.5367 | 0.5647 | 0.365 | 0.634 | 0.611 |
| LR | n-grami | unigram+bigram (trenutno) | **0.5535** | 0.5828 | 0.381 | 0.646 | 0.634 |
| LR | n-grami | unigram+bigram+trigram | 0.5502 | 0.5779 | 0.383 | 0.640 | 0.628 |
| LR | filtriranje | bez filtriranja | 0.5474 | 0.5755 | 0.370 | 0.651 | 0.622 |
| LR | filtriranje | trenutno (min_df=2, max_df=0.95) | **0.5535** | 0.5828 | 0.381 | 0.646 | 0.634 |
| LR | filtriranje | strože (min_df=5, max_df=0.90) | 0.5147 | 0.5418 | 0.348 | 0.610 | 0.586 |
| SVM | n-grami | unigram | 0.5406 | 0.5765 | 0.346 | 0.648 | 0.628 |
| SVM | n-grami | unigram+bigram (trenutno) | **0.5475** | 0.5846 | 0.356 | 0.643 | 0.643 |
| SVM | n-grami | unigram+bigram+trigram | 0.5471 | 0.5842 | 0.357 | 0.645 | 0.640 |
| SVM | filtriranje | bez filtriranja | 0.5473 | 0.5713 | 0.382 | 0.655 | 0.605 |
| SVM | filtriranje | trenutno (min_df=2, max_df=0.95) | **0.5475** | 0.5846 | 0.356 | 0.643 | 0.643 |
| SVM | filtriranje | strože (min_df=5, max_df=0.90) | 0.5187 | 0.5550 | 0.329 | 0.619 | 0.608 |
| NB | n-grami | unigram | **0.5440** | 0.5755 | 0.354 | 0.655 | 0.623 |
| NB | n-grami | unigram+bigram (trenutno) | 0.5377 | 0.5682 | 0.353 | 0.639 | 0.622 |
| NB | n-grami | unigram+bigram+trigram | 0.5329 | 0.5651 | 0.341 | 0.632 | 0.626 |
| NB | filtriranje | bez filtriranja | 0.5437 | 0.5811 | 0.340 | 0.658 | 0.633 |
| NB | filtriranje | trenutno (min_df=2, max_df=0.95) | 0.5377 | 0.5682 | 0.353 | 0.639 | 0.622 |
| NB | filtriranje | strože (min_df=5, max_df=0.90) | 0.5389 | 0.5665 | 0.363 | 0.637 | 0.617 |

### 6.2 Zapažanja

- **LR i SVM**: trenutna konfiguracija (unigram+bigram, min_df=2/max_df=0.95) ostaje najbolja ili suštinski izjednačena sa najboljom alternativom (SVM bez filtriranja: 0.5473 vs 0.5475 — razlika zanemarljiva). Trigrami ne donose dosledan dobitak.
- **NB**: odstupa od ostalih — čisti unigrami (0.5440) i izostanak filtriranja (0.5437) blago nadmašuju trenutnu konfiguraciju (0.5377), za ~0.006–0.007 macro-F1. Razlika je mala i verovatno unutar šuma, ali sugeriše da NB manje profitira od bigrama.
- **Strože filtriranje (min_df=5, max_df=0.90)** dosledno pogoršava macro-F1 kod sva tri modela (za 0.02–0.04 u odnosu na trenutnu konfiguraciju) i ne preporučuje se.
- Zaključak: trenutna konfiguracija (unigram+bigram, min_df=2, max_df=0.95) se zadržava kao podrazumevana za sva tri modela radi konzistentnosti; NB unigram varijanta može se pomenuti kao alternativa od sekundarnog značaja.

## 7. Zaključak za izveštaj

- Od **54** isprobanih konfiguracija najbolja je **Logistička regresija** sa **TF**, normalize=`stem`, lowercase=da (macro-F1 = **0.553**, acc = 0.582).
- Najteža klasa po F1 kod pobednika: **NEUTRAL** (0.381).
- TF-IDF nije unapred proglašen pobednikom: upoređeni su TF, IDF i TF-IDF ravnopravno.
- Ove brojke su **donja granica** za Fazu 3; enkoder (BERTić / mBERT) treba da ih nadmaši po macro-F1.
- Dodatna ablacija n-grama i filtriranja rečnika (Poglavlje 6) potvrđuje da trenutna podešavanja (unigram+bigram, min_df=2, max_df=0.95) ostaju najbolji ili skoro najbolji izbor za sva tri modela.

## 8. Fajlovi

- Sirovi rezultati: `baseline/output/baseline_results.json`
- Classification report-i: `baseline/output/baseline_results.txt`
- `output/baseline_analysis/01_ranking_macro_f1.png`
- `output/baseline_analysis/02_f1_by_model.png`
- `output/baseline_analysis/03_f1_by_weighting.png`
- `output/baseline_analysis/04_f1_by_normalize.png`
- `output/baseline_analysis/05_heatmap_model_weighting.png`
- `output/baseline_analysis/06_best_per_class_f1.png`
- `output/baseline_analysis/07_best_model_per_class.png`
- Ablacija n-grama/filtriranja: `baseline\output\ablation_ngram_freq_results.txt`

Regenerisanje (kad postoji JSON):

```bash
cd phase3_model_training
python baseline/report_baseline.py
```
