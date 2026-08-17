# Phase 2: Anotacija podataka

Ručna anotacija komentara (Instagram + X) o studentskim protestima u Srbiji —
klasifikacija stava (stance) prema vlasti u kontekstu protesta.

## Uputstva i kalibracija

Glavni dokument za rad anotatora:

- **[UPUTSTVA_ANOTACIJA.md](UPUTSTVA_ANOTACIJA.md)** — definicije oznaka, granični slučajevi, tok rada i **procedura kalibracije**

Folder za kalibraciju (paralelna anotacija ~10% skupa):

```text
calibration/
  calibration_set.txt           # uzorak za nezavisnu anotaciju
  ann_<ime>.txt                 # predaje anotatora (napravite vi)
  disagreements_resolved.md     # dogovoreni sporni slučajevi
  agreement_report.md           # κ / α i rezime
```

**Redosled:** pročitati uputstva → uraditi kalibraciju → tek onda finalna anotacija ostatka skupa.

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

Ako URL izvora nije pouzdano nađen: kolona URL ostaje prazna ili `NEMA`.

## Struktura

```text
phase2_annotation/
  UPUTSTVA_ANOTACIJA.md
  README.md
  calibration/
  annotated/
    ig_final_*_annotated.txt
    x_final_*_annotated.txt
    dataset_all.txt                 # spojeno (za treniranje)
```

## Napomena za dokumentaciju (Faza 2)

Prema propozicijama projekta, u ovoj fazi treba:

1. ~~Uputstva za anotaciju (definicije + problematični slučajevi)~~ → `UPUTSTVA_ANOTACIJA.md`
2. Kalibracioni skup (~10%, paralelna anotacija članova grupe) → folder `calibration/`
3. Analizu saglasnosti anotatora → `calibration/agreement_report.md`
4. Deskriptivnu statistiku finalnog skupa → vidi analizu ispod

## Analiza skupa

Deskriptivna statistika Faza 1 + Faza 2 (grafike + izveštaj):

```bash
cd phase3_model_training
python analyze_phase1_phase2.py
```

Izveštaj: `phase3_model_training/ANALIZA_FAZA1_FAZA2.md`  
Grafike: `phase3_model_training/output/data_analysis/`
