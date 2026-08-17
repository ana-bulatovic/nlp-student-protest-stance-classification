# Faza 3 — Opis treniranja i inferencije (baseline + enkoderi + dekoderi)

Ovaj dokument detaljno opisuje **teorijsku osnovu** (sa primerima), tok rada i bitne funkcije u skriptama:

- `train_baseline.py` / `infer.py` — osnovni modeli (LR, SVM, **Naive Bayes**)
- `train_encoder.py` / `infer_encoder.py` — enkoderski LLM (BERTić, mBERT)
- `eval_decoder.py` / `infer_decoder.py` — dekoderski LLM (ChatGPT / Gemini, prompting)
- `text_preprocess.py` — stemovanje / lematizacija

Usklađen je sa **Fazom 3** i komentarom profesora:

- stavka 1: baseline klasifikatori + pretprocesiranje (TF/IDF/TF-IDF ravnopravno, **TF-IDF nije default**; stem/lema; nested / 10-fold CV)
- stavka 2a: fine-tuning enkodera + poređenje po epohama
- stavka 2b: dekoderski modeli + prompt engineering (zero/few-shot, SR/EN)

Teorijski pojmovi (SVM, TF-IDF, metrike, nested CV, Transformer…) detaljno su u **§2**.

---

## 1. Zadatak i podaci

### 1.1 NLP zadatak

**Stance klasifikacija** komentara sa Instagrama u kontekstu studentskih protesta u Srbiji.

Model za dati tekst treba da predvidi jednu od tri oznake:

| Oznaka | Značenje |
|--------|----------|
| `NEUTRAL` | Neutralan, nejasan ili van teme stav |
| `ZA-VLAST` | Podrška vlasti / kritika blokada i studenata |
| `PROTIV-VLASTI` | Kritika vlasti / podrška studentskom protestu |

### 1.2 Format ulaznog skupa

Podaci dolaze iz Faze 2:

`phase2_annotation/annotated/dataset_all.txt`

Svaka linija:

```text
komentar|url_izvora|oznaka
```

Primer:

```text
Znanje je moć!|https://www.instagram.com/p/DUBlHK6Db4n/|NEUTRAL
```

### 1.3 Distribucija (trenutni skup)

| Klasa | Broj primera |
|-------|--------------|
| NEUTRAL | 118 |
| ZA-VLAST | 150 |
| PROTIV-VLASTI | 232 |
| **Ukupno** | **500** |

Skup je **neuravnotežen** (najviše `PROTIV-VLASTI`). Zato se za poređenje modela koristi **macro-F1**, a ne samo accuracy.

---

## 2. Teorijska osnova (detaljno)

Ovaj odeljak objašnjava **svaki bitan pojam** koji se koristi u baseline i enkoder delu projekta, uz jednostavne primere iz domena studentskih protesta / stance klasifikacije.

### 2.1 Klasifikacija teksta i „stance“

**Klasifikacija** = zadatak u kome model dobija ulaz **x** (ovde: komentar) i vraća oznaku **y** iz konačnog skupa klasa.

U našem projektu oznaka **y** može biti jedna od tri vrednosti: `NEUTRAL`, `ZA-VLAST` ili `PROTIV-VLASTI`.

**Stance (stav)** nije isto što i sentiment:

| Pojam | Pitanje | Primer |
|-------|---------|--------|
| Sentiment | Da li je tekst pozitivan/negativan? | „Odlično!“ → pozitivno |
| Stance | Kakav je **stav prema temi/cilju**? | „Odlično što studenti blokiraju!“ → `PROTIV-VLASTI` |

Dva pozitivna komentara mogu imati **suprotan** stance:

- „Odličan govor predsednika!“ → `ZA-VLAST`
- „Odličan protest studenata!“ → `PROTIV-VLASTI`

Zato je ovo **stance klasifikacija**, ne običan sentiment analysis.

---

### 2.2 Zašto „osnovni“ (baseline) modeli?

Prema PDF-u projekta, Faza 3 ima:

1. **Osnovne modele** — LR, Linear SVM i **naivni Bajes**, sa eksplicitnim odlikama
2. **Velike jezičke modele** — enkoderski (BERTić, mBERT) i/ili dekoderski (prompting)

**Baseline** = jednostavan, jak referentni sistem. Cilj:

- dati **donju granicu** performansi (LLM treba da bude bar bolji)
- biti brz na CPU-u i lako interpretabilan
- omogućiti analizu pretprocesiranja: lowercasing, **TF / IDF / TF-IDF** (ravnopravno; **TF-IDF nije default**), **stemovanje i/ili lematizacija**

Ako baseline daje macro-F1 ≈ 0.45, a enkoder ≈ 0.60, u izveštaju jasno pokazuješ dobit od Transformer modela.

---

### 2.3 Odlike (features) i vektor prostora

Računar ne „čita“ tekst kao čovek. Svaki komentar pretvaramo u **vektor brojeva** **x = (x1, x2, ..., xd)**.

- **d** = dimenzija = broj različitih tokena/n-grama u rečniku
- **xi** = težina i-tog tokena u tom komentaru (0 ako ga nema)

**Primer (veoma pojednostavljeno)**  
Rečnik: `[vucic, blokada, pumpaj, predsednik]`

| Komentar | Vektor |
|----------|--------|
| „Živeo Vučić“ | `(1, 0, 0, 0)` |
| „Stop blokadama“ | `(0, 1, 0, 0)` |
| „Pumpaj!“ | `(0, 0, 1, 0)` |

Linearni klasifikator uči težine **w** i računa skor tipa **w · x + b**. Veći skor za neku klasu → ta klasa se bira.

---

### 2.4 Tokenizacija, unigrami i bigrami

**Token** ≈ jedinica teksta (obično reč posle čišćenja interpunkcije).

- **Unigram** = jedna reč: `vucic`, `blokada`
- **Bigram** = dve uzastopne reči: `ziveo vucic`, `stop blokadama`

Zašto bigrami?

Sam unigram `stop` je nejasan. Bigram `stop blokadama` nosi jasan stance signal (`ZA-VLAST`).

U kodu: `ngram_range=(1, 2)` → koriste se i unigrami i bigrami.

**Filtriranje rečnika:**

| Parametar | Značenje | Zašto |
|-----------|----------|-------|
| `min_df=2` | token mora u ≥ 2 dokumenta | uklanja retke greške / šum |
| `max_df=0.95` | izbaci tokene u >95% dokumenata | uklanja suviše opšte reči |

---

### 2.5 Lowercasing (normalizacija na mala slova)

**Lowercasing** pretvara sva slova u mala: `Vučić` → `vučić`, `PUMPAJ` → `pumpaj`.

**Prednosti:**

- manji rečnik (`Vucic`, `vucic`, `VUCIC` postaju jedna odlika)
- manje preklapanja zbog nasumičnog CAPS pisanja na mrežama

**Mane:**

- ponekad CAPS nosi emociju/signal (`SRAMota!!!`)
- u srpskom/latinici i ćirilici lowercasing ne rešava problem pisma (to je posebna normalizacija)

Zato u punom baseline eksperimentu testiramo **sa i bez** lowercasing-a.

---


### 2.5b Stemovanje i lematizacija

Preporuka profesora: razmotriti **stemovanje** i/ili **lematizaciju** kao deo pretprocesiranja.

| Tehnika | Šta radi | U projektu |
|---------|----------|------------|
| **Stemovanje** | skida nastavke → grubi „koren“ (`blokadama` → `blokad`) | pravila u `text_preprocess.py` |
| **Lematizacija** | svodi na rečnički oblik / lemu (`blokadama` → `blokada`) | `simplemma` (jezik `sr`) |

Primer:

```text
original:   Studenti blokiraju fakultete
stem:       student blokir fakultet
lemma:      student blokirati fakultet   (zavisi od lematizera)
```

Cilj: smanjiti dimenzionalnost rečnika i spojiti morfološke varijante iste reči. U eksperimentima se porede `none` / `stem` / `lemma`.

---

### 2.6 TF — term frequency (frekvencija termina)

**TF** broji koliko se puta token javlja u **jednom** dokumentu.

Primer komentar:  
`„Blokade, blokade, blokade — dosta više!“`

| Token | TF |
|-------|----|
| blokade | 3 |
| dosta | 1 |
| vise | 1 |

U kodu: `CountVectorizer` → sirovi brojevi pojavljivanja.

**Intuicija:** što se reč češće ponavlja u komentaru, to je „važnija“ za taj komentar.  
**Problem:** česte opšte reči (`je`, `da`, `ovo`) mogu dominirati ako ih ne filtriramo / ne ponderišemo IDF-om.

---

### 2.7 IDF — inverse document frequency

**IDF** meri koliko je token **retak u celom korpusu**.

Klasična formula (konceptualno):

```text
IDF(t) = log(N / df(t))
```

- **N** = broj dokumenata
- **df(t)** = u koliko dokumenata se token t javlja

**Primer** (N = 1000 komentara):

| Token | df | IDF (grubo) | Tumačenje |
|-------|----|-------------|-----------|
| `je` | 900 | malo | skoro svuda, malo informativan |
| `blokada` | 80 | srednje | koristan signal |
| `cacilend` | 5 | veliko | retka, jako karakteristična reč |

**IDF režim u našem kodu:** token se tretira binarnо (prisutan/odsutan), pa se množi IDF težinom — naglašava **prisustvo retkih** reči, bez brojanja ponavljanja.

---

### 2.8 TF-IDF

**TF-IDF** = TF × IDF (često uz L2 normalizaciju vektora).

```text
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

**Šta time dobijamo?**

- reč česta **u dokumentu**, a retka **u korpusu** → velika težina
- reč česta svuda (`i`, `da`) → mala težina

**Mini-primer**

Dokument A: `„Pumpaj bagru!“`  
Dokument B: `„Živeo predsednik!“`

Token `pumpaj` ima visok IDF ako se retko javlja → u A dobija veliku TF-IDF težinu → model lako povezuje sa `PROTIV-VLASTI`.

**L2 normalizacija** (`norm="l2"`): deli vektor njegovom dužinom, da dugački komentari ne dominiraju samo zbog više tokena.

U praksi je **TF-IDF najčešći** izbor za LR/SVM baseline na tekstu.

---

### 2.9 Pipeline (cev obrade)

**Pipeline** = niz koraka koji se uvek izvršavaju istim redosledom:

```text
sirov tekst  →  vektorizer (TF-IDF)  →  klasifikator (LR/SVM)  →  oznaka
```

U sklearn-u:

```python
Pipeline([
    ("vec", TfidfVectorizer(...)),
    ("clf", LinearSVC(...)),
])
```

**Zašto je važno:** pri unakrsnoj validaciji vektorizer se fituje **samo na train fold**, pa transformiše test fold. Time sprečavamo **curenje informacija** (data leakage) iz testa u treniranje.

---

### 2.10 Regularizacija i hiperparametar `C`

Linearni modeli lako **preprilagode** (overfit) šum u podacima ako su težine prevelike.

**Regularizacija** kažnjava velike težine → model ostaje jednostavniji i bolje generalizuje.

U LR i SVM parametar **`C`** kontroliše kompromis:

| `C` | Efekat |
|-----|--------|
| malo `C` (npr. 0.1) | jača regularizacija, jednostavniji model |
| veliko `C` (npr. 10) | slabija regularizacija, model se više „lepi“ za trening |

Zato `C` **ne biramo „odoka“**, već unutrašnjom unakrsnom validacijom (`GridSearchCV`).

---

### 2.11 Logistička regresija (LR) — detaljno

#### Ideja

LR pretpostavlja da se verovatnoća klase može modelovati glatkom S-krivom (**logistička / sigmoid** funkcija) nad linearnom kombinacijom odlika.

Za binarni slučaj (2 klase):

```text
P(y=1 | x) = sigma(w · x + b) = 1 / (1 + e^(-(w · x + b)))
```

gde je **sigma** logistička (S) funkcija.

- ako je skor **w · x + b** veliki pozitivan → verovatnoća blizu 1
- ako je veliki negativan → verovatnoća blizu 0

Za **3 klase** sklearn koristi proširenje (one-vs-rest ili multinomial/softmax, zavisno od solvera). Kod nas: `solver="lbfgs"`, `multi_class="auto"`.

#### Primer (intuitivno)

Pretpostavimo 2 odlike posle TF-IDF: `pumpaj` i `ziveo_vucic`.

Model nauči npr.:

- velika pozitivna težina za `pumpaj` → ka `PROTIV-VLASTI`
- velika pozitivna težina za `ziveo_vucic` → ka `ZA-VLAST`

Komentar „Pumpaj!!!“ ima veliki skor za `PROTIV-VLASTI`.

#### Prednosti / mane

| + | − |
|---|---|
| brz, stabilan, dobre verovatnoće | linearan — ne hvata složene interakcije |
| koeficijenti se mogu tumačiti | slabiji na ironiji / dugom kontekstu |
| dobar baseline | zahteva dobre odlike (TF-IDF) |

---

### 2.12 Metoda potpornih vektora (SVM) — detaljno

#### Ideja

**SVM (Support Vector Machine / metoda potpornih vektora)** traži granicu između klasa koja **maksimizuje marginu** — rastojanje od granice do najbližih tačaka obe klase.

Te najbliže tačke zovu se **potporni vektori** (*support vectors*): samo one „drže“ granicu. Ostale tačke dalje od granice ne utiču na njenu poziciju.

#### Geometrijska intuicija (2D)

Zamisli dve grupe tačaka:

```text
  ZA-VLAST:     o  o
                 o     |          x  x   PROTIV-VLASTI
                    o  |        x
                       |     x
                 <--- margina --->
```

SVM bira pravu `|` tako da razmak (margina) bude što veći. To često bolje generalizuje od „bilo koje“ razdvajajuće prave.

#### Linearni SVM (`LinearSVC`)

Kod nas je **linearni** kernel: granica je hiperravan u TF-IDF prostoru.

```text
f(x) = w · x + b
```

- **f(x) > 0** → jedna strana
- **f(x) < 0** → druga strana

Za više klasa koristi se one-vs-rest: za svaku klasu poseban „ova klasa vs ostale“ klasifikator, pa se bira klasa sa najvećim skorom (`decision_function`).

#### Primer na komentarima

Posle TF-IDF, komentari su tačke u visokodimenzionalnom prostoru.

- Komentari tipa „Živeo Vučić“, „Bravo predsedniče“ grupišu se u jednoj oblasti → `ZA-VLAST`
- „Pumpaj“, „Lopovi“, „Izdajnici“ u drugoj → `PROTIV-VLASTI`
- Neutralni („Lepa slika“, „Gde gledati?“) su često bliže sredini / mešoviti → teže se razdvajaju (zato je `NEUTRAL` najslabija klasa)

SVM uči hiperravan koja razdvaja te oblasti uz maksimalnu marginu.

#### Soft margin i `C`

U realnim podacima klase se **preklapaju** (isti reči mogu u obe klase). Soft-margin SVM dozvoljava greške, a `C` kontroliše koliko strogo kažnjava prekršaje margine:

- malo `C` → šira margina, više tolerancije na greške
- veliko `C` → uža margina, manje grešaka na treningu (rizik overfita)

#### Prednosti / mane

| + | − |
|---|---|
| često jak na tekstu + TF-IDF | skorovi nisu prave verovatnoće |
| otporan na neke šumove (margina) | sporiji od LR na ogromnim skupovima (kod nas OK) |
| radi u veoma visokim dimenzijama | linearni SVM ne modeluje nelinearne obrasce bez kernela |

**Napomena:** `decision_function` vraća **skorove**, ne verovatnoće. Veći skor = jača preferencija klase. (Kod enkodera / LR sa `predict_proba` dobijaš prave verovatnoće.)

---


### 2.12b Naivni Bajesov klasifikator (MultinomialNB)

**Naivni Bajes** pretpostavlja da su odlike **uslovno nezavisne** date klase:

```text
P(klasa | dokument) ~ P(klasa) * proizvod tokena P(token | klasa)
```

Za tekst se koristi **MultinomialNB** (frekvencije tokena; prirodno uz **TF**).

Hiperparametar **alpha** = smoothing (Laplace/Lidstone) — sprečava nulte verovatnoće za neviđene tokene.

Dodat je na preporuku profesora, uz LR i SVM. Nije vezan isključivo za TF-IDF: TF-IDF ostaje samo jedna od opcija ponderisanja.

---

### 2.13 Unakrsna validacija (cross-validation, CV)

#### Problem train/test jedne podele

Ako jednom podeliš 80/20, rezultat zavisi od sreće te podele.

#### k-fold CV

Podeli skup na **k** delova (foldova):

```text
Fold 1: TEST | train | train | ...
Fold 2: train | TEST | train | ...
...
Fold k: train | ... | TEST
```

Svaki primer tačno jednom bude u testu. Finalna ocena = prosek / spojene predikcije.

Kod nas spoljašnja CV podrazumevano **k = 10** (zahtev PDF-a).

#### Stratifikovana CV

**Stratifikacija** = u svakom foldu čuva se **slična proporcija klasa** kao u celom skupu.

Bez stratifikacije, neki fold može imati skoro 0 `NEUTRAL` primera → loša i nestabilna ocena.

Primer (500 primera: 118 / 150 / 232):  
U 10-fold stratifikaciji svaki test fold ima otprilike 12 + 15 + 23 primera po klasama.

---

### 2.14 Ugnežđena (nested) unakrsna validacija

#### Zašto treba?

Ako na **istim** foldovima:

1. biraš najbolji `C`, i
2. prijaviš F1,

onda si `C` donekle „prilagodio“ i test podatcima → **optimistično pristrasna** ocena.

#### Kako radi nested CV?

```text
Za svaki OUTER fold i:
  train_i / test_i

  Na train_i pokreni INNER CV:
    isprobaj C ∈ {0.1, 1, 10}
    izaberi C* sa najboljim macro-F1

  Treniraj model sa C* na celom train_i
  Evaluiraj na test_i   ← test_i NIJE korišćen za izbor C
```

**Outer petlja** = poštena evaluacija  
**Inner petlja** = selekcija hiperparametra

U kodu:

- inner: `GridSearchCV(..., cv=inner)`
- outer: `cross_val_predict(search, ..., cv=outer)`

---

### 2.15 Metrike klasifikacije — detaljno

Za jednu klasu **X** (npr. `ZA-VLAST`):

| Simbol | Značenje |
|--------|----------|
| TP | stvarno X, predviđeno X |
| FP | nije X, a predviđeno X |
| FN | stvarno X, a predviđeno nešto drugo |
| TN | nije X i nije predviđeno X |

#### Precision (preciznost)

```text
Precision = TP / (TP + FP)
```

„Od svega što sam označio kao `ZA-VLAST`, koliko je stvarno to?“

Visok precision → malo lažnih alarma za tu klasu.

#### Recall (odziv / osetljivost)

```text
Recall = TP / (TP + FN)
```

„Od svih stvarnih `ZA-VLAST` komentara, koliko sam uhvatio?“

Visok recall → malo propuštenih primera te klase.

#### F1

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Harmonijska sredina — kažnjava ako je jedna od dve metrika slaba.

**Primer:**  
Precision=1.0, Recall=0.1 → model retko predviđa klasu, ali kad predvidi pogodi. F1 je nizak (~0.18), što je fer.

#### Accuracy

```text
Accuracy = (broj tacnih) / (ukupan broj)
```

Na neuravnoteženom skupu varljiva: model koji uvek glasa `PROTIV-VLASTI` (232/500) ima accuracy ≈ 46% bez ikakvog učenja.

#### Macro-F1 vs Weighted-F1

| Metrika | Kako se računa | Kada |
|---------|----------------|------|
| **Macro-F1** | prosek F1 preko 3 klase (jednaka težina) | **glavna** za projekat — ne favorizuje većinsku klasu |
| **Weighted-F1** | prosek F1 ponderisan brojem primera klase | bliže overall utisku na neuravnoteženom skupu |

Zato rangiramo konfiguracije po **macro-F1**.

---

### 2.16 Overfitting i underfitting

| Pojam | Simptom | Tipičan uzrok |
|-------|---------|----------------|
| **Underfitting** | loš i train i test | previše jaka regularizacija, premalo odlika/epoha |
| **Overfitting** | odličan train, loš test | preveliko `C`, previše epoha, mali skup |

Nested CV i regularizacija smanjuju rizik lažno dobrih rezultata.

---

### 2.17 Enkoderski Transformer modeli (BERTić / mBERT) — teorija

Baseline koristi **ručne** odlike (n-grami). Enkoderi uče **reprezentacije** teksta iz ogromnog korpusa, pa ih **fino podešavamo** (fine-tuning) na naših 500 komentara.

#### Šta je enkoder?

**Encoder Transformer** (BERT porodica) čita ceo ulaz odjednom (obostrani kontekst) i pravi vektori za tokene / [CLS] reprezentaciju rečenice.

Razlika:

| Tip | Primer | Tipična upotreba |
|-----|--------|------------------|
| Enkoder | BERT, BERTić, mBERT | klasifikacija, NER |
| Dekoder | GPT, Gemini | generisanje teksta / prompting |

#### Fine-tuning

1. Uzmeš već istrenirani jezički model
2. Na vrh dodaš klasifikacionu glavu (3 izlaza)
3. Nastaviš treniranje na anotiranim komentarima (par epoha)

**Epoha** = jedan prolaz kroz ceo trening skup.  
PDF traži da uporedimo varijante po broju epoha (npr. 2, 3, 4): premalo → underfitting; previše → overfitting na malom skupu.

#### BERTić vs mBERT

| Model | Tip | Zašto u projektu |
|-------|-----|------------------|
| **BERTić** (`classla/bcms-bertic`) | monolingvalni BCMS | bolje „razume“ srpski/BCMS idiom |
| **mBERT** (`bert-base-multilingual-cased`) | multilingvalni | PDF zahtev: bar jedan multi model; poređenje domena/jezika |

#### Softmax verovatnoće

Na izlazu enkodera dobijaš 3 logita → **softmax** ih pretvara u verovatnoće koje se sabiraju na 1:

```text
P_i = e^(z_i) / sum_j( e^(z_j) )
```

Zato `infer_encoder.py` ispisuje npr. `PROTIV-VLASTI=0.72, ZA-VLAST=0.19, NEUTRAL=0.09`.

---

### 2.18 Sažetak pojmova (cheat-sheet)

| Pojam | Jedna rečenica |
|-------|----------------|
| Stance | Stav prema temi, ne samo sentiment |
| TF | Koliko puta se reč javlja u dokumentu |
| IDF | Koliko je reč retka u korpusu |
| TF-IDF | TF × IDF — frekvencija ponderisana informativnošću |
| N-gram | N uzastopnih tokena (1=reč, 2=par reči) |
| Lowercasing | Sve u mala slova |
| LR | Linearni model verovatnoće klase |
| SVM | Maksimalna margina između klasa; potporni vektori definišu granicu |
| `C` | Kompromis fitovanja vs regularizacije |
| Stratified k-fold | k testova uz očuvanu proporciju klasa |
| Nested CV | Outer = ocena, inner = izbor hiperparametra |
| Precision / Recall / F1 | Tačnost predikcija klase / pokrivenost klase / njihov spoj |
| Macro-F1 | Prosek F1 preko klasa (jednako važne) |
| Stemovanje | Skidanje nastavaka do grubog korena reči |
| Lematizacija | Svođenje na lemu / osnovni oblik |
| Naivni Bajes | Klasifikator uz pretpostavku nezavisnosti odlika; alpha = smoothing |
| Fine-tuning | Naknadna obuka prettreniranog Transformer-a na našem zadatku |
| Epoha | Jedan prolaz kroz trening podatke |
| Zero-shot | Klasifikacija samo iz instrukcije, bez primera u promptu |
| Few-shot | Klasifikacija uz nekoliko primerâ (komentar → oznaka) u promptu |
| Prompt engineering | Dizajn instrukcija koje usmeravaju generativni model |

---

### 2.19 Dekoderski / generativni LLM (prompting) — teorija

Za razliku od enkodera (fine-tuning težina), **dekoder** (GPT, Gemini) koristimo **bez obuke na našem skupu**: šaljemo mu tekstualnu instrukciju (**prompt**) i čitamo odgovor.

#### Zero-shot vs few-shot

| Režim | Šta model vidi | Prednost | Mana |
|-------|----------------|----------|------|
| **Zero-shot** | samo definicija zadatka + komentar | jeftinije, kraći prompt | slabije na graničnim slučajevima |
| **Few-shot** | definicija + nekoliko (komentar, oznaka) + novi komentar | bolje kalibrisan stil odgovora | duži prompt, rizik loših primera |

U projektu few-shot primeri su **ručno sastavljeni** (ne iz eval skupa), da ne curi oznaka u evaluaciju.

#### Jezik prompta (sr vs en)

Isti komentari i iste klase, ali instrukcija na **srpskom** ili **engleskom**. Cilj: da li model bolje prati uputstvo na engleskom (često jači „instruction following“) ili na srpskom (bliže domenu teksta).

#### Kratak vs detaljan prompt

| Stil | Sadržaj |
|------|---------|
| **short** | jedna rečenica + traži samo oznaku |
| **detailed** | definicije klasa, pravila (stance ≠ sentiment), zabranjena objašnjenja |

#### Parsiranje odgovora

Model ponekad vrati `"Oznaka: ZA-VLAST"` ili engleski alias. `parse_label` normalizuje alias-e (`PRO-STUDENT` → `PROTIV-VLASTI`, `PRO-VLAST` → `ZA-VLAST`, …). Neparsirani odgovori se beleže kao `parse_fail` i **ne ulaze** u F1 (da NEUTRAL fallback ne veštački podigne metriku).

#### Zašto nema CV?

Predlog projekta za dekoder traži **evaluaciju na celom anotiranom skupu** (nema fine-tuning-a, nema hiperparametara tipa `C`). Poređenje ide konfiguracija × konfiguracija (provider × jezik × stil × shot) i vs. baseline/enkoder po macro-F1.

---

## 3. Struktura foldera

```text
phase3_model_training/
  train_baseline.py       # treniranje + evaluacija + čuvanje modela
  infer.py                # inferenca baseline
  train_encoder.py        # fine-tuning BERTić / mBERT
  infer_encoder.py        # inferenca enkodera
  decoder_prompts.py      # šabloni promptova (SR/EN, short/detailed, few-shot)
  eval_decoder.py         # evaluacija ChatGPT / Gemini
  infer_decoder.py        # inferenca dekodera
  requirements.txt        # sklearn + torch + transformers + openai/gemini
  README.md               # kratko uputstvo
  DOCS_TRAINING_INFERENCE.md   # ovaj dokument
  output/
    baseline_results.json
    baseline_model.joblib
    encoder_results.json
    encoder_best/
    decoder_results.json
    decoder_cache/
```

---

## 4. Kako se trenira (tok rada)

### 4.1 Instalacija

```bash
cd nlp-student-protest-stance-classification
pip install -r phase3_model_training/requirements.txt
```

### 4.2 Brzi test (`--quick`)

Namenjen proveri da sve radi (manje foldova; **LR+SVM+NB**; ponderisanje **TF** — ne TF-IDF; bez stem/leme):

```bash
cd phase3_model_training
python train_baseline.py --quick
```

Šta `--quick` menja:

| Parametar | Puno treniranje | Quick |
|-----------|-----------------|-------|
| Modeli | LR + SVM | LR + SVM |
| Ponderisanje | tf, idf, tfidf | samo tfidf |
| Lowercase | da i ne | samo da |
| Outer folds | 10 | 3 |
| Inner folds | 3 | 2 |
| C grid | 0.1, 1.0, 10.0 | 1.0, 10.0 |

### 4.3 Puno treniranje (prema specifikaciji)

```bash
python train_baseline.py
```

To prolazi sve kombinacije:

**model × ponderisanje × lowercase**  
= 2 × 3 × 2 = **12 konfiguracija**

Za svaku radi nested CV i na kraju rangira po macro-F1.

### 4.4 Korisne CLI opcije (`train_baseline.py`)

| Opcija | Značenje |
|--------|----------|
| `--data PUTANJA` | Drugi dataset fajl |
| `--models lr svm` | Koje modele trenirati |
| `--weightings tf idf tfidf` | Koja ponderisanja |
| `--lowercase 1 0` | 1=uključeno, 0=isključeno |
| `--outer-folds 10` | Broj spoljašnjih foldova |
| `--inner-folds 3` | Broj unutrašnjih foldova |
| `--model-out PUTANJA` | Gde sačuvati `.joblib` model |
| `--no-save-model` | Ne čuvaj model za inferencu |
| `--seed 42` | Seed za reproduktivnost |

### 4.5 Šta se dobija posle treniranja

1. **`output/baseline_results.json`** — struktura sa:
   - putanjom do podataka
   - brojem primera i brojem po klasama
   - brojem foldova i `C` mrežom
   - listom rezultata po konfiguraciji (`accuracy`, `macro_f1`, `per_class_f1`, `best_C`…)

2. **`output/baseline_results.txt`** — tekstualni `classification_report` po konfiguraciji

3. **`output/baseline_model.joblib`** — bundle za inferencu:
   - `pipeline` — fitted `Pipeline(vec + clf)`
   - `labels` — lista klasa
   - `config` — metapodaci najbolje konfiguracije
   - `data` — putanja do skupa na kom je treniran

Najbolji model = onaj sa **najvećim macro-F1** u nested CV evaluaciji. On se zatim dodatno fituje na **celom** skupu (radi produkcione inferencije).

---

## 5. Detaljan opis `train_baseline.py`

### 5.1 Importi i konstante

```python
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = ... / "phase2_annotation" / "annotated" / "dataset_all.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "baseline_results.json"
DEFAULT_MODEL = SCRIPT_DIR / "output" / "baseline_model.joblib"
LABELS = ("NEUTRAL", "ZA-VLAST", "PROTIV-VLASTI")
```

- `SCRIPT_DIR` — folder u kom je skripta (nezavisno od cwd)
- `DEFAULT_*` — podrazumevane putanje
- `LABELS` — dozvoljene oznake (provera pri učitavanju)

Biblioteke:

- `sklearn` — vektorizacija, modeli, CV, metrike
- `joblib` — serijalizacija fitted pipeline-a
- `numpy` — niz oznaka
- `argparse` — CLI

### 5.2 `FoldResult` (dataclass)

```python
@dataclass
class FoldResult:
    model: str
    weighting: str
    lowercase: bool
    best_C: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
```

**Čemu služi:** jedinstvena struktura rezultata jedne konfiguracije.  
**Šta se dobija:** lako se pretvara u dict (`asdict`) i upisuje u JSON.

### 5.3 `load_dataset(path)`

**Ulaz:** putanja do UTF-8 TXT fajla `tekst|url|oznaka`.  
**Izlaz:** `(texts, labels)` — dve paralelne liste.

Bitni delovi:

1. `rsplit("|", 2)` — deli sa desna na najviše 3 dela, da `|` unutar komentara ne pokvari parsiranje
2. provera da oznaka ∈ `LABELS`
3. preskakanje praznih linija / praznog teksta
4. `ValueError` ako format nije ispravan ili nema primera

**Čemu služi:** jedinstveno mesto za učitavanje anotiranih podataka u memoriju.

### 5.4 `make_vectorizer(weighting, lowercase)`

Pravi odgovarajući sklearn vektorizer.

Zajednički parametri:

```python
lowercase=lowercase
analyzer="word"
ngram_range=(1, 2)
min_df=2
max_df=0.95
```

Grananje:

| `weighting` | Objekat |
|-------------|---------|
| `"tf"` | `CountVectorizer` |
| `"tfidf"` | `TfidfVectorizer(use_idf=True, norm="l2")` |
| `"idf"` | `TfidfVectorizer(use_idf=True, binary=True, norm=None)` |

**Čemu služi:** izoluje logiku pretprocesiranja od modela (lakše eksperimentisanje).

### 5.5 `make_classifier(model)`

| `model` | Klasifikator |
|---------|--------------|
| `"lr"` | `LogisticRegression(max_iter=2000, solver="lbfgs", ...)` |
| `"svm"` | `LinearSVC(max_iter=5000, dual="auto", ...)` |

`C` se **ne** postavlja ovde trajno — bira ga `GridSearchCV` kroz `clf__C`.

**Čemu služi:** fabrika klasifikatora; isti interfejs za LR i SVM unutar `Pipeline`.

### 5.6 `evaluate_config(...)` — srce evaluacije

**Ulaz:** tekstovi, oznake, izbor modela/ponderisanja/lowercase, broj foldova, `C` mreža, seed.  
**Izlaz:** `(FoldResult, classification_report_string, fitted_pipeline)`.

Koraci u funkciji:

1. **Pipeline**
   ```python
   Pipeline([
       ("vec", make_vectorizer(...)),
       ("clf", make_classifier(...)),
   ])
   ```
   Jedan objekat radi: `fit` na train → `transform` + `predict` na test.

2. **Outer i inner foldovi**
   ```python
   outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
   inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
   ```

3. **GridSearchCV (unutrašnja CV)**
   ```python
   GridSearchCV(pipe, param_grid={"clf__C": C_grid},
                cv=inner, scoring="f1_macro", n_jobs=-1, refit=True)
   ```
   - bira `C` po **macro-F1**
   - `refit=True` — posle izbora fituje najbolji estimator na celom (unutrašnjem) train skupu

4. **Nested predikcije**
   ```python
   y_pred = cross_val_predict(search, texts, y, cv=outer, n_jobs=-1)
   ```
   Za svaki outer fold: treniraj `search` na train delu, predvidi test deo.  
   Na kraju `y_pred` ima po jednu predikciju za **svaki** primer, dobijenu kad je taj primer bio u test foldu → poštena ocena.

5. **Finalni fit na celom skupu**
   ```python
   search.fit(texts, y)
   fitted_pipe = search.best_estimator_
   ```
   Ovo **nije** za metrike nested CV-a, već da se sačuva model spreman za produkcionu inferencu.

6. **Računanje metrika** na `(y, y_pred)` iz nested CV-a:
   - accuracy
   - macro / weighted F1
   - F1 po klasi
   - `classification_report`

**Šta se dobija:** uporedivi rezultati konfiguracije + report + pipeline za čuvanje.

### 5.7 `parse_args()`

Definiše CLI interfejs (vidi tabelu u §4.4).  
**Čemu služi:** omogućava pokretanje bez menjanja koda.

### 5.8 `main()` — orkestracija

Tok:

1. UTF-8 stdout/stderr (Windows konzola)
2. provera da dataset postoji
3. `load_dataset`
4. postavljanje listi modela / ponderisanja / lowercase / foldova / `C_grid`
5. ako je `--quick` → smanjeni režim
6. zaštita: ako najmanja klasa ima manje primera od `outer_folds`, smanji broj foldova
7. trostruka petlja nad konfiguracijama → `evaluate_config`
8. upis JSON + TXT rezultata
9. rangiranje po `macro_f1`
10. čuvanje najboljeg pipeline-a u `.joblib` (osim ako je `--no-save-model`)
11. `return 0` (uspeh) ili `1` (greška)

---

## 6. Detaljan opis `infer.py`

### 6.1 Svrha

Uzeti sačuvani model i predvideti oznaku za:

- jedan ili više komentara sa komandne linije (`-t`)
- linije iz fajla (`-f`)
- interaktivni unos (`-i`)

### 6.2 `load_bundle(path)`

Učitava `joblib` fajl.

Provere:

- fajl postoji (inače poruka da prvo pokreneš treniranje)
- u bundle-u postoji ključ `"pipeline"`

**Izlaz:** dict sa `pipeline`, `labels`, `config`, `data`.

### 6.3 `predict_texts(bundle, texts)`

1. `pipe.predict(texts)` → oznake
2. pokušaj skorova:
   - prvo `decision_function` (SVM, i neki LR režimi)
   - inače `predict_proba` (ako postoji)
3. mapira skorove na imena klasa i sortira opadajuće

**Izlaz:** lista dict-ova:

```python
{
  "text": "...",
  "label": "ZA-VLAST",
  "scores": {"ZA-VLAST": 0.056, "PROTIV-VLASTI": -0.053, "NEUTRAL": -1.067}
}
```

Napomena za SVM `decision_function`: skorovi **nisu** verovatnoće; veći skor = jača preferencija klase.

### 6.4 `read_input_texts(args)`

Skuplja tekstove iz:

- `--text` / `-t` (može više puta)
- `--file` / `-f` (jedan komentar po liniji)

Ako linija izgleda kao anotirani format (`tekst|url|oznaka`), uzima se samo prva kolona (tekst).

### 6.5 `parse_args()` (infer)

| Opcija | Značenje |
|--------|----------|
| `--model` | Putanja do `.joblib` |
| `-t / --text` | Komentar (ponavljivo) |
| `-f / --file` | Fajl sa komentarima |
| `-i / --interactive` | Režim unosa sa tastature |

### 6.6 `main()` (infer)

1. učitaj bundle
2. ispiši kratke metapodatke modela (tip, C, macro-F1 iz treninga)
3. ako nema tekstova ili je `-i` → interaktivna petlja
4. inače predvidi sve zadate tekstove
5. `_print_row` ispisuje `[OZNAKA] tekst` + skorove

### 6.7 `_print_row(row)`

Formatira jedan rezultat za konzolu.

---

## 7. Kako pokrenuti inferencu

Najpre mora postojati model:

```bash
python train_baseline.py --quick
```

Zatim:

```bash
# jedan komentar
python infer.py -t "Ziveo Vucic i SNS"

# vise komentara
python infer.py -t "Pumpaj!" -t "Ne znam sta da mislim o ovome"

# fajl
python infer.py -f ../phase2_annotation/annotated/ig_final_neutral_annotated.txt

# interaktivno
python infer.py -i
```

Primer izlaza:

```text
Model: baseline_model.joblib | svm tfidf C=1.0 macro_f1=0.4498...
[ZA-VLAST] Ziveo Vucic i SNS
  skorovi: ZA-VLAST=0.056, PROTIV-VLASTI=-0.053, NEUTRAL=-1.067
```

---

## 8. Rezultati (quick režim)

Sledeći rezultati su iz pokretanja:

```bash
python train_baseline.py --quick
```

Postavke: 500 primera, outer=3, inner=2, TF-IDF + lowercase, C ∈ {1.0, 10.0}.

### 8.1 Pregled

| Model | Ponderisanje | Lowercase | best C | Accuracy | Macro-F1 |
|-------|--------------|-----------|--------|----------|----------|
| **SVM** | TF-IDF | da | 1.0 | **0.486** | **0.450** |
| LR | TF-IDF | da | 10.0 | 0.478 | 0.440 |

Najbolji po macro-F1: **Linear SVM + TF-IDF + lowercase**.

### 8.2 F1 po klasama (SVM)

| Klasa | Precision | Recall | F1 |
|-------|-----------|--------|----|
| NEUTRAL | 0.344 | 0.280 | 0.308 |
| ZA-VLAST | 0.496 | 0.447 | 0.470 |
| PROTIV-VLASTI | 0.532 | 0.616 | 0.571 |

### 8.3 Interpretacija

- Accuracy ~49% i macro-F1 ~0.45 su **iznad slučajnog nagađanja** (~33% za 3 klase), ali daleko od jakog sistema.
- Najslabija klasa je **NEUTRAL** (najmanje primera + najmanje jasan jezički signal).
- Najjača je **PROTIV-VLASTI** (najveći support + često eksplicitna leksika).
- Ovo je očekivano za:
  - mali skup (500)
  - bučne društvene mreže (slang, greške, ironija)
  - jednostavne bag-of-words odlike bez konteksta

Puni prolaz (`python train_baseline.py` bez `--quick`) treba koristiti za **zvanične** tabele u dokumentaciji projekta (10-fold + sve kombinacije TF/IDF/TF-IDF × lowercase).

---

## 9. Veza sa PDF specifikacijom

| Zahtev iz PDF-a (Faza 3.1) | Pokriveno? | Gde |
|----------------------------|------------|-----|
| Osnovni modeli (LR / SVM / Naive Bayes) | Da | `make_classifier` |
| Ručno definisane odlike | Da | TF/IDF/TF-IDF n-grami |
| Efekti lowercasing | Da | `--lowercase` / petlja u `main` |
| Efekti TF / IDF / TF-IDF (ne kao default model) | Da | `make_vectorizer` |
| Stemovanje / lematizacija | Da | `text_preprocess.py` / `--normalize` |
| 10-slojna stratifikovana CV | Da (default) | `StratifiedKFold` + `--outer-folds 10` |
| Ugnežđena CV za regularizaciju `C` | Da | `GridSearchCV` unutar `cross_val_predict` |
| Obuka na CPU | Da | scikit-learn |
| Enkoderski LLM (BERTić / mBERT) | Da | `train_encoder.py`, `infer_encoder.py` |
| Dekoderski LLM + prompting | Da | `eval_decoder.py`, `infer_decoder.py` |
| Zero / few-shot | Da | `--shots zero few` |
| Prompt SR vs EN | Da | `--langs sr en` |
| Kratak vs detaljan prompt | Da | `--styles short detailed` |
| ChatGPT i/ili Gemini | Da | `--providers openai gemini` |

---

## 12. Enkoderski modeli (`train_encoder.py` / `infer_encoder.py`)

### 12.1 Zahtev iz PDF-a + preporuka profesora

- fine-tuning **enkoderskih** Transformer modela
- barem **jedan monolingvalni** (BERTić) i **jedan multilingvalni** (mBERT)
- **Simple Transformers** interfejs (`ClassificationModel`) — lakši za učenje;
  ispod koristi Hugging Face Transformers (kao na predavanjima za sentiment)
- **10-slojna stratifikovana CV**
- poređenje varijanti po **broju epoha**
- GPU preporučen (Colab / Azure)

### 12.2 Zašto Simple Transformers?

Umesto ručnog `Trainer` + `Dataset` + tokenizacije, dovoljno je:

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd

train_df = pd.DataFrame({"text": tekstovi, "labels": oznake})

model_args = ClassificationArgs(num_train_epochs=3, overwrite_output_dir=True)
model = ClassificationModel("bert", "classla/bcms-bertic",
                            num_labels=3, args=model_args)
model.train_model(train_df)
predikcije, skorovi = model.predict(["Pumpaj!"])
```

To je isti stil kao na predavanjima (analiza sentimenta u nekoliko linija).

### 12.3 Modeli

| Ključ CLI | HF ime | Tip |
|-----------|--------|-----|
| `bertic` | `classla/bcms-bertic` | monolingvalni BCMS (BERTić) |
| `mbert` | `bert-base-multilingual-cased` | multilingvalni BERT |

Oba se učitavaju sa `model_type="bert"` u Simple Transformers.

### 12.4 Tok treniranja

1. Učitaj `dataset_all.txt`
2. Za svaki `(model × broj_epoha)`: stratifikovani k-fold; u foldu `train_model` + `predict`
3. Rangiraj po macro-F1
4. Najbolju konfiguraciju trenira na celom skupu → `output/encoder_best/`

### 12.5 Bitne funkcije

| Funkcija | Uloga |
|----------|--------|
| `make_args` | `ClassificationArgs` (epohe, batch, LR, `labels_list`) |
| `build_model` | `ClassificationModel(...)` |
| `train_one_fold` | DataFrame → `train_model` → `predict` |
| `evaluate_encoder_config` | cela stratifikovana CV |
| `train_full_and_save` | finalni model + `stance_meta.json` |
| `load_model` / `predict_texts` (infer) | učitavanje i predikcija |

### 12.6 Pokretanje

```bash
pip install simpletransformers pandas
cd phase3_model_training

python train_encoder.py --quick
python train_encoder.py
python infer_encoder.py -t "Pumpaj!"
python infer_encoder.py -i
```

### 12.7 Izlazi

- `output/encoder_results.json` / `.txt`
- `output/encoder_best/` + `stance_meta.json` (`framework: simpletransformers`)

Inferenca vraća oznaku + verovatnoće (softmax nad logitima).

---

## 10. Ograničenja i moguća poboljšanja

1. **Mali i neuravnotežen skup** — više anotiranih primera (posebno NEUTRAL) bi podiglo stabilnost.
2. **Baseline bag-of-words** ne vidi ironiju ni dugoročni kontekst.
3. **Encoder fine-tuning** na CPU je spor; za zvanične 10-fold rezultate koristi GPU.
4. Quick rezultati nisu zamena za puni eksperiment u izveštaju.
5. Finalni sačuvani modeli (`.joblib` / `encoder_best`) fitovani su na celom skupu radi demo inferencije — za ocenu generalizacije gledati CV metrike.
6. Dekoder zavisi od **API ključa** i cene poziva; koristi keš (`decoder_cache/`) i `--quick` / `--limit` dok prototipišeš.

---

## 13. Dekoderski modeli (`eval_decoder.py` / `infer_decoder.py`)

### 13.1 Zahtev iz predloga projekta

- ChatGPT (GPT-4o / GPT-4o mini) i/ili Gemini
- **zero-shot** i **few-shot** klasifikacija (prompt engineering)
- poređenje instrukcija na **srpskom** i **engleskom**
- različiti formati (kratki vs detaljni prompt)
- evaluacija na **celom** anotiranom skupu; poređenje sa supervizovanim modelima

### 13.2 Tok rada

1. Učitaj `dataset_all.txt`
2. Za svaku kombinaciju `(provider × model × lang × style × shot)`:
   - sastavi prompt (`decoder_prompts.py`)
   - pošalji komentar API-ju (ili uzmi iz keša)
   - parsira oznaku (`parse_label`)
3. Izračunaj accuracy / macro-F1 / F1 po klasama
4. Rangiraj konfiguracije po macro-F1

### 13.3 Bitne funkcije / moduli

| Komponenta | Uloga |
|------------|--------|
| `build_messages` | system + user prompt (SR/EN, short/detailed, zero/few) |
| `FEWSHOT_EXAMPLES` | ručni primeri bez curenja iz eval skupa |
| `OpenAIClient` / `GeminiClient` | tanki API wrapperi (`temperature=0`) |
| `parse_label` | normalizacija odgovora → `LABELS` |
| `evaluate_config` | prolaz kroz skup + keš + metrike |

### 13.4 Pokretanje

```bash
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
# $env:GEMINI_API_KEY = "..."

cd phase3_model_training
python eval_decoder.py --dry-run --quick
python eval_decoder.py --quick
python eval_decoder.py --providers openai
python infer_decoder.py -t "Pumpaj!"
python infer_decoder.py -i --shot few --lang sr
```

### 13.5 Izlazi

- `output/decoder_results.json` / `.txt` — agregirane metrike
- `output/decoder_predictions.json` — po-primer predikcije
- `output/decoder_cache/` — keš sirovih API odgovora

---

## 11. Brzi „cheat sheet“

```bash
# instalacija
pip install -r phase3_model_training/requirements.txt

cd phase3_model_training

# baseline
python train_baseline.py --quick
python infer.py -t "Tvoj komentar"

# enkoder (BERTić / mBERT)
python train_encoder.py --quick
python train_encoder.py
python infer_encoder.py -t "Tvoj komentar"
python infer_encoder.py -i

# dekoder (ChatGPT / Gemini)
$env:OPENAI_API_KEY = "sk-..."
python eval_decoder.py --dry-run --quick
python eval_decoder.py --quick
python infer_decoder.py -t "Tvoj komentar"
```

Fajlovi rezultata:

- Baseline: `output/baseline_results.json`, `baseline_results.txt`, `baseline_model.joblib`
- Encoder: `output/encoder_results.json`, `encoder_results.txt`, `encoder_best/`
- Decoder: `output/decoder_results.json`, `decoder_results.txt`, `decoder_predictions.json`
