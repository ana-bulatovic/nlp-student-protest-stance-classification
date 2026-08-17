# Uputstva za anotaciju i kalibraciju

**Projekat:** klasifikacija stava (stance) u komentarima o studentskim protestima u Srbiji  
**Faza:** 2 — anotacija  
**Oznake:** `NEUTRAL` | `ZA-VLAST` | `PROTIV-VLASTI`

Ovaj dokument služi da svi anotatori rade **po istim pravilima**, pa da se posle toga uradi **kalibracija** (paralelna anotacija + mera saglasnosti) pre finalnog obeležavanja ostatka skupa.

---

## 1. Šta anotiramo

Anotira se **jedan komentar** (jedna linija teksta), u kontekstu studentskih protesta / blokada i političke situacije u Srbiji.

**Cilj oznake:** kakav je **stav autora komentara** prema **vlasti** (i, posredno, prema studentskom protestu), a ne:

- da li se tebi sviđa komentar,
- da li je komentar „pametan“ ili „glup“,
- da li je gramatički tačan,
- ko je autor u stvarnom životu (osim ako to eksplicitno piše u tekstu).

**Format zapisa** (UTF-8, separator `|`):

```text
tekst_komentara|url_izvora|OZNAKA
```

- `url_izvora` — URL objave sa koje je komentar (Instagram / X). Ako nije pouzdano nađen: ostavi prazno ili `NEMA`.
- `OZNAKA` — tačno jedna od tri vrednosti ispod (velika slova, crtica kako piše).

---

## 2. Definicije oznaka

### `PROTIV-VLASTI`

Komentar **kritizuje vlast** (Vučić, SNS, institucije pod kontrolom vlasti, policiju u službi režima, režimske medije…) **i/ili podržava studente / blokade / protest**.

Tipični signali:

- podrška studentima, blokadama, šetnjama, zahtevima;
- kritika korupcije, nasilja, cenzure, laži vlasti;
- pozitivni apel studentima („bravo“, „uz vas smo“, „napred“), kada je jasno da je u kontekstu protesta;
- negativan odnos prema predsedniku / vladajućoj stranci.

**Primeri (ilustrativno):**

| Komentar | Zašto |
|----------|--------|
| „Dokle će Vučić da se sprda sa narodom?“ | direktna kritika vlasti |
| „Uz studente smo, samo napred“ | podrška protestu |
| „Suditi za izdaju“ | napad na vlast / sistem |

### `ZA-VLAST`

Komentar **podržava vlast** **i/ili napada / omalovažava studente, blokade, protest**.

Tipični signali:

- podrška Vučiću / SNS / „Srbija pobeđuje“;
- tvrdnje da su studenti plaćeni, nasilni, „blokaderi“, sekta, strani agenti;
- poziv da se vrate na fakultet, da prestanu da „maltretiraju narod“;
- odbrana institucija u kontekstu napada na protest.

**Primeri:**

| Komentar | Zašto |
|----------|--------|
| „Živeo Aleksandar Vučić!“ | podrška vlasti |
| „Vratite se na fakultet, narod vas neće“ | napad na studente/protest |
| „Blokaderi hoće nasiljem na vlast“ | diskreditacija protesta |

### `NEUTRAL`

Komentar **nema jasan stance** prema vlasti / protestu, ili je:

- van teme,
- pitanje / tehnička informacija,
- dvosmislen,
- čista emocija bez mete (bez jasnog „za“ ili „protiv“),
- previše kratak / nejasan da bi se pouzdano klasifikovao.

**Primeri:**

| Komentar | Zašto |
|----------|--------|
| „U koliko sati?“ | informativno pitanje |
| „Gde pogledati celu emisiju?“ | van stava |
| „Joj kako vas sve volim!!!“ | emocija bez jasnog političkog stava (osim ako kontekst jasno kaže kome) |

---

## 3. Pravila odlučivanja (redosled)

Kad nisi siguran, idi ovim redosledom:

1. **Da li komentar eksplicitno podržava vlast ili napada studente/protest?** → `ZA-VLAST`
2. **Da li eksplicitno kritikuje vlast ili podržava studente/protest?** → `PROTIV-VLASTI`
3. **Inače** → `NEUTRAL`

### Dodatna pravila

1. **Samo tekst komentara.** Ne nagađaj stav na osnovu profila, lajkova ili drugih komentara (osim ako je odgovor jasan samo uz kratak kontekst roditeljskog komentara — i tada beleži šta si pretpostavio).
2. **Jedna dominantna poruka.** Ako ima i kritiku vlasti i kritiku studenata, uzmi **jači / glavni** smisao. Ako su ravnopravni i zbuni te — `NEUTRAL`, ili označi kao sporan (vidi §5).
3. **Sarkazam / ironija.** Ako je ironija **jasna** iz teksta, anotiraj **stvarni** stav (npr. „Bravo, predsedniče…“ u očigledno negativnom tonu → `PROTIV-VLASTI`). Ako nisi siguran da je ironija — `NEUTRAL` ili sporan slučaj.
4. **Uvreda sama po sebi nije oznaka.** „Idioti“ bez mete → `NEUTRAL`. „Vučić idiot“ → `PROTIV-VLASTI`. „Studenti idioti" → `ZA-VLAST`.
5. **Emoji / uzvici.** „🔥👏“ uz jasan kontekst podrške studentima može biti `PROTIV-VLASTI`; sami emoji bez sadržaja → `NEUTRAL` (ili se izbacuju iz skupa ako su prazni).
6. **Mešavina pisama / greške.** Ignoriši pravopis; gledaj značenje.
7. **Ne „popravljaj“ tekst** pri anotaciji. Ostavi komentar kako jeste; samo dodaj `|url|OZNAKA`.

---

## 4. Problematični / granični slučajevi

| Situacija | Preporuka |
|-----------|-----------|
| Podrška studentima + kritika opozicije | `PROTIV-VLASTI` ako je fokus podrška protestu; ako samo „opozicija je loša“ bez stava o vlasti/protestu → `NEUTRAL` ili sporan |
| „Srbija“ / patriotizam bez politike | `NEUTRAL`, osim ako jasno glorifikuje vlast ili napada protest |
| „Nasilje nije rešenje“ bez mete | `NEUTRAL` (može biti kritika i jednih i drugih) |
| Verski / moralni apel bez politike | `NEUTRAL` |
| Link / „pogledajte ovo“ bez stava | `NEUTRAL` |
| Odgovor tipa „tačno“, „laž“, „bravo“ | gledaj **na šta** se odnosi; ako nije jasno iz samog teksta → `NEUTRAL` / sporan |
| Veoma kratko: „Bravo“, „MRŠ“, „Naravno“ | često dvosmisleno → radije `NEUTRAL` ili sporan, osim ako je meta očigledna |
| Komentar o medijima (N1, Informer, RTS…) | ako napada režimske medije / brani kritičke → često `PROTIV-VLASTI`; obrnuto → `ZA-VLAST`; nejasno → `NEUTRAL` |

**Sporan slučaj:** ako i posle pravila nisi siguran, stavi privremeno `NEUTRAL` **ili** označi komentar oznakom `#SPORNO` na kraju teksta / u posebnom fajlu `sporni.txt`, da se reši na kalibracionom sastanku.

---

## 5. Tok rada anotacije (pojedinačno)

1. Uzmi ulazni fajl (npr. listu komentara iz Faze 1 ili `*_final_*.txt`).
2. Za svaki komentar dodeli tačno jednu oznaku.
3. Upisuj u formatu `tekst|url|OZNAKA`.
4. Ne briši komentare zbog neslaganja — ili ih stavi u `NEUTRAL`, ili označi kao sporne.
5. Na kraju: broj po klasama + lista spornih.

**Preporuka tempa:** bolje sporije i konzistentnije nego brzo i nasumično. Posle kalibracije radi se brže.

---

## 6. Kalibracija (obavezno pre finalne anotacije)

Cilj kalibracije: da članovi tima **usklade razumevanje oznaka**, izmere **saglasnost**, i dogovore rešenja za granične slučajeve.

### 6.1 Priprema kalibracionog skupa

1. Nasumično izaberi **oko 10%** komentara iz radnog skupa koji ćete anotirati  
   (npr. ako planirate ~700 finalnih, kalibracija ≈ **70** komentara; minimum razumno **50–100**).
2. Skup treba da bude **mešan** (kratki + dugi, IG + X ako imaš oba, različite objave).
3. Sačuvaj ga kao poseban fajl, npr.:

```text
phase2_annotation/calibration/calibration_set.txt
```

Format ulaza (bez oznake):

```text
tekst|url
```

ili samo `tekst` po liniji.

4. **Ne deli „tačne“ oznake** unapred. Svaki anotator radi **nezavisno**.

### 6.2 Paralelna anotacija

1. Svaki član grupe anotira **isti** `calibration_set.txt` u svoj fajl, npr.:

```text
phase2_annotation/calibration/ann_ana.txt
phase2_annotation/calibration/ann_ime.txt
```

Format izlaza:

```text
tekst|url|OZNAKA
```

2. Bez dogovaranja tokom anotacije (chat / gledanje tuđih oznaka zabranjeno do kraja kruga).
3. Vreme: dogovorite rok (npr. 1–2 dana).

### 6.3 Merenje saglasnosti

Posle što svi predaju fajlove, izračunaj:

| Metrika | Šta pokazuje | Cilj (orijentir) |
|---------|--------------|------------------|
| **Procenat slaganja** (pairwise) | koliko često se dvoje slažu | ≥ 80% poželjno |
| **Cohen’s κ** (za 2 anotatora) | slaganje iznad slučajnosti | ≥ 0.6 prihvatljivo, ≥ 0.7 dobro |
| **Krippendorff’s α** (2+ anotatora) | generalnija mera | ≥ 0.67 prihvatljivo za mnoge NLP zadatke |

**Gde se računa:** kratka skripta u `phase2_annotation/calibration/` (ili ručno u tabeli). Za izveštaj navedi: broj primera, broj anotatora, κ / α, i matrice zabune (koju klasu ko meša sa kojom).

Ako je κ / α **nisko** (< 0.6):

1. ne nastavljaj odmah na veliki skup;
2. uradi sastanak (§6.4);
3. po potrebi **ponovi** kalibraciju na novom uzorku (~5–10%) dok mera ne poraste.

### 6.4 Kalibracioni sastanak

1. Izlistaj sve komentare gde se anotatori **ne slažu**.
2. Za svaki: pročitaj tekst, primeni §2–§4, dogovorite **zlatnu oznaku** (gold label).
3. Dopunite ovaj dokument novim primerima u §4 ako se ponavlja isti tip neslaganja.
4. Odaberite **glavnog anotatora** (ili par) za ostatak skupa — radi po usklađenim pravilima.
5. Sporne zlatne oznake zabeležite u:

```text
phase2_annotation/calibration/disagreements_resolved.md
```

### 6.5 Posle kalibracije — finalna anotacija

1. Anotiraj ostatak skupa po usklađenim uputstvima.
2. Drži isti format `tekst|url|OZNAKA`.
3. Na kraju spoji fajlove u `annotated/dataset_all.txt` (ili ažuriraj postojeći).
4. U README / izveštaju Faze 2 navedi:
   - veličinu kalibracionog skupa,
   - broj anotatora,
   - κ / α,
   - kratak zaključak („posle kalibracije nastavljeno sa jednim anotatorom…“).

---

## 7. Predložena struktura foldera

```text
phase2_annotation/
  README.md
  UPUTSTVA_ANOTACIJA.md          ← ovaj dokument
  annotated/                     ← finalni anotirani fajlovi
  calibration/                   ← napravi pri kalibraciji
    calibration_set.txt          ← uzorak bez oznaka (ili sa gold posle)
    ann_<ime>.txt                ← nezavisne anotacije
    disagreements_resolved.md    ← dogovoreni sporni slučajevi
    agreement_report.md          ← κ / α i rezime
```

---

## 8. Brzi ček-lista za anotatora

- [ ] Koristim samo `NEUTRAL` / `ZA-VLAST` / `PROTIV-VLASTI`
- [ ] Gledam stav prema **vlasti / protestu**, ne lični utisak
- [ ] Sarkazam označavam samo kad je jasan
- [ ] Kratko i dvosmisleno → radije `NEUTRAL` ili `#SPORNO`
- [ ] Format: `tekst|url|OZNAKA`
- [ ] Pre velikog skupa: urađena kalibracija (§6)

---

## 9. Šta sledi posle Faze 2

Finalni anotirani skup ide u **Fazu 3** (treniranje / evaluacija modela).  
Kvalitet oznaka direktno utiče na macro-F1 — zato kalibracija nije opcioni „papir“, nego deo metodologije.
