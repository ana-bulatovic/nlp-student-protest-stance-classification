# Phase 2: Anotacija podataka

Ručno anotirani komentari sa Instagrama o studentskim protestima u Srbiji
(stance / stav prema vlasti u kontekstu protesta).

## Šema oznaka

| Oznaka | Značenje |
|--------|----------|
| `NEUTRAL` | Neutralan / nejasan / van teme stav |
| `ZA-VLAST` | Podrška vlasti / kritika blokada i studenata |
| `PROTIV-VLASTI` | Kritika vlasti / podrška studentskom protestu |

Format zapisa (UTF-8 TXT, `|` kao separator):

```text
komentar|url_izvora|oznaka
```

Ako URL izvora nije pouzdano nađen: u koloni URL stoji `NEMA`.

## Struktura

```text
phase2_annotation/
  annotated/
    ig_final_neutral_annotated.txt
    ig_final_pro_vlast_annotated.txt
    ig_final_pro_student_annotated.txt
    dataset_all.txt                 # spojeno (za treniranje)
  README.md
```

## Broj primera (Instagram, trenutno)

| Oznaka | Broj |
|--------|------|
| NEUTRAL | 118 |
| ZA-VLAST | 150 |
| PROTIV-VLASTI | 232 |
| **Ukupno** | **500** |

Izvor: kopije finalnih anotiranih fajlova iz `phase1_data_collection/output/instagram/`.

## Napomena za dokumentaciju (Faza 2)

Prema propozicijama projekta, u ovoj fazi treba još dopuniti:

1. Uputstva za anotaciju (definicije + problematični slučajevi)
2. Kalibracioni skup (~10%, paralelna anotacija članova grupe)
3. Analizu saglasnosti anotatora
4. Deskriptivnu statistiku finalnog skupa

Trenutno su ovde sačuvani finalni Instagram anotirani podaci spremni za Fazu 3.

## Analiza skupa

Deskriptivna statistika Faza 1 + Faza 2 (grafike + izveštaj):

```bash
cd phase3_model_training
python analyze_phase1_phase2.py
```

Izveštaj: `phase3_model_training/ANALIZA_FAZA1_FAZA2.md`  
Grafike: `phase3_model_training/output/data_analysis/`
