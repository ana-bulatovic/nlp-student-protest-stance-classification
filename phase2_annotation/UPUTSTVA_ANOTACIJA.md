# Uputstva za anotaciju

**Projekat:** klasifikacija stava (stance) u komentarima o studentskim protestima u Srbiji  
**Faza:** 2 — anotacija  
**Oznake:** `NEUTRAL` | `ZA-VLAST` | `PROTIV-VLASTI`

Ovaj dokument je obavezan za sve anotatore. Cilj je da svi rade **po istim pravilima**, pa da se merenjem saglasnosti (kalibracija) proveri da li oznake ima smisla koristiti za treniranje modela.

---

## 1. Šta anotiramo

Anotira se **jedan komentar** (jedna linija), uzet sa Instagrama, X-a, YouTube-a ili Facebooka, u kontekstu studentskih protesta / blokada i političke situacije u Srbiji.

**Pitanje koje oznaka odgovara:** kakav je **stav autora komentara** prema **vlasti** (i, posredno, prema studentskom protestu)?

**Format** (UTF-8, separator `|`):

```text
tekst_komentara|url_izvora|OZNAKA
```

- `url_izvora` — URL objave sa koje je komentar. Ako nije poznat, ostaje prazan.
- `OZNAKA` — tačno jedna od tri vrednosti ispod (velika slova, crtica kako piše).

Za kalibraciju ulazni fajl ima praznu treću kolonu (`tekst|url|`). Anotator upisuje oznaku u tu kolonu.

---



## 2. Definicije oznaka



### `PROTIV-VLASTI`

Komentar **kritikuje vlast** (Vučić, SNS, institucije pod kontrolom vlasti, policiju u službi režima, režimske medije…) **i/ili podržava studente / blokade / protest**.

Tipični signali:

- podrška studentima, blokadama, šetnjama, zahtevima;
- kritika korupcije, nasilja, cenzure, laži vlasti;
- pozitivan apel studentima („bravo“, „uz vas smo“, „napred“) kada je jasno da je u kontekstu protesta;
- negativan odnos prema predsedniku / vladajućoj stranci.


| Primer                                   | Zašto                   |
| ---------------------------------------- | ----------------------- |
| „Dokle će Vučić da se sprda sa narodom?“ | direktna kritika vlasti |
| „Uz studente smo, samo napred“           | podrška protestu        |
| „Suditi za izdaju“                       | napad na vlast / sistem |




### `ZA-VLAST`

Komentar **podržava vlast** **i/ili napada / omalovažava studente, blokade, protest**.

Tipični signali:

- podrška Vučiću / SNS / „Srbija pobeđuje“;
- tvrdnje da su studenti plaćeni, nasilni, „blokaderi“, sekta, strani agenti;
- poziv da se vrate na fakultet, da prestanu da „maltretiraju narod“;
- odbrana institucija u kontekstu napada na protest.


| Primer                                   | Zašto                       |
| ---------------------------------------- | --------------------------- |
| „Živeo Aleksandar Vučić!“                | podrška vlasti              |
| „Vratite se na fakultet, narod vas neće“ | napad na studente / protest |
| „Blokaderi hoće nasiljem na vlast“       | diskreditacija protesta     |




### `NEUTRAL`

Komentar **nema jasan stance** prema vlasti / protestu, ili je:

- van teme,
- pitanje / tehnička informacija,
- čista emocija bez mete (bez jasnog „za“ ili „protiv“),
- previše kratak / nejasan da bi se pouzdano klasifikovao.


| Primer                        | Zašto                               |
| ----------------------------- | ----------------------------------- |
| „U koliko sati?“              | informativno pitanje                |
| „Gde pogledati celu emisiju?“ | van stava                           |
| „Joj kako vas sve volim!!!“   | emocija bez jasnog političkog stava |


---



## 3. Sarkazam i ironija — preskačemo

**Sarkastične i ironične komentare ne anotiramo stavom.** Preskačemo ih.

Razlog: takvi komentari su **dvosmisleni**. Doslovno čitanje i namera autora često se raspadaju (npr. „Bravo, predsedniče…“ može biti i podrška i ruganje). Model uči iz kratkog teksta, bez tona glasa i šireg konteksta, pa sarkazam unosi **šum** i pogrešan signal.

Pravilo u praksi:

1. Ako je komentar **jasno sarkastičan / ironičan** → **preskoči**, ne stavljaj `ZA-VLAST` ni `PROTIV-VLASTI`.
2. Ne pokušavaj da „pogodiš“ pravi stav iza sarkazma.
3. Ako nisi siguran da li je sarkazam ili iskren stav → tretiraj kao **nejasan**, ne nagađaj ironiju.

Ovi komentari **ne ulaze** u skup za treniranje.

---



## 4. Pravila odlučivanja

Kad nisi siguran, idi ovim redom:

1. Da li je komentar sarkazam / ironija? → **preskoči**
2. Da li eksplicitno podržava vlast ili napada studente/protest? → `ZA-VLAST`
3. Da li eksplicitno kritikuje vlast ili podržava studente/protest? → `PROTIV-VLASTI`
4. Inače → `NEUTRAL`



### Dodatna pravila

1. **Samo tekst komentara.** Ne nagađaj stav sa profila, lajkova ili susednih komentara.
2. **Jedna dominantna poruka.** Ako ima i kritiku vlasti i kritiku studenata, uzmi jači / glavni smisao. Ako su ravnopravni — `NEUTRAL`.
3. **Uvreda sama po sebi nije oznaka.** „Idioti“ bez mete → `NEUTRAL`. „Vučić idiot“ → `PROTIV-VLASTI`. „Studenti idioti“ → `ZA-VLAST`.
4. **Emoji / uzvici.** „🔥👏“ uz jasan kontekst podrške studentima može biti `PROTIV-VLASTI`; sami emoji bez sadržaja → `NEUTRAL`.
5. **Mešavina pisama / greške.** Ignoriši pravopis; gledaj značenje.
6. **Ne „popravljaj“ tekst.** Ostavi komentar kako jeste; samo dodaj oznaku.

---



## 5. Granični slučajevi


| Situacija                                | Preporuka                                                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Podrška studentima + kritika opozicije   | `PROTIV-VLASTI` ako je fokus podrška protestu; ako samo „opozicija je loša“ bez stava o vlasti/protestu → `NEUTRAL` |
| „Srbija“ / patriotizam bez politike      | `NEUTRAL`, osim ako jasno glorifikuje vlast ili napada protest                                                      |
| „Nasilje nije rešenje“ bez mete          | `NEUTRAL`                                                                                                           |
| Verski / moralni apel bez politike       | `NEUTRAL`                                                                                                           |
| Link / „pogledajte ovo“ bez stava        | `NEUTRAL`                                                                                                           |
| Odgovor tipa „tačno“, „laž“, „bravo“     | gledaj **na šta** se odnosi; ako nije jasno iz samog teksta → `NEUTRAL`                                             |
| Veoma kratko: „Bravo“, „MRŠ“, „Naravno“  | često dvosmisleno → `NEUTRAL`, osim ako je meta očigledna                                                           |
| Komentar o medijima (N1, Informer, RTS…) | napad na režimske medije / odbrana kritičkih → često `PROTIV-VLASTI`; obrnuto → `ZA-VLAST`; nejasno → `NEUTRAL`     |


Ako i posle pravila nisi siguran: stavi `NEUTRAL`. Bolje konzervativno nego nagađanje.

---



## 6. Kako se anotira

1. Otvori `annotation_tool.html` i učitaj svoj fajl (`kalibracija_<ime>.txt`).
2. Za svaki komentar dodeli tačno jednu oznaku, ili `SKIP` ako je sarkazam.
3. Ne dogovaraj se sa drugima dok traje tvoj krug (kalibracija je nezavisna).
4. Na kraju eksportuj anotirani fajl.

Tastatura u alatu: **1** = `NEUTRAL`, **2** = `ZA-VLAST`, **3** = `PROTIV-VLASTI`, **⌫** = `SKIP`.

---



## 7. Kalibracija

Cilj: da članovi tima usklade razumevanje oznaka, mere saglasnost, i tek onda koriste oznake kao pouzdan skup.

### 7.1 Fajlovi

U `phase2_annotation/calibration/`:


| Fajl                       | Šta je                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------- |
| `kalibracija_ana.txt`      | istih 250 komentara, bez oznake                                                    |
| `kalibracija_natalija.txt` | isto                                                                               |
| `kalibracija_nikola.txt`   | isto                                                                               |
| `kalibracija_marija.txt`   | isto                                                                               |
| `kalibracija_gold.txt`     | iste linije **sa originalnom oznakom** iz dataseta (ne otvarati dok svi ne završe) |


Svi anotatori rade **isti uzorak**, nezavisno. `kalibracija_gold.txt` služi za poređenje **posle** anotacije.

### 7.2 Tok

1. Pročitaj ova uputstva.
2. Anotiraj svoj `kalibracija_<ime>.txt` bez gledanja tuđih fajlova i bez gold fajla.
3. Predaj anotirani izvoz.
4. Uporedi međusobno i sa `kalibracija_gold.txt`.



### 7.3 Mere saglasnosti (orijentir)


| Metrika                         | Cilj                            |
| ------------------------------- | ------------------------------- |
| Procenat slaganja (parovi)      | ≥ 80% poželjno                  |
| Cohen’s κ (2 anotatora)         | ≥ 0.6 prihvatljivo, ≥ 0.7 dobro |
| Krippendorff’s α (2+ anotatora) | ≥ 0.67 prihvatljivo             |


Ako je κ / α **< 0.6**: ne tretiraj kalibraciju kao uspešnu; uskladi pravila na sastanku i po potrebi ponovi uzorak.

---



## 8. Brzi ček-lista

- [ ] Koristim samo `NEUTRAL` / `ZA-VLAST` / `PROTIV-VLASTI` (ili `SKIP` za sarkazam)
- [ ] Gledam stav prema **vlasti / protestu**, ne lični utisak
- [ ] Sarkazam i ironiju **preskačem** — model to ne uči pouzdano
- [ ] Kratko i dvosmisleno → `NEUTRAL`
- [ ] Format: `tekst|url|OZNAKA`
- [ ] Kalibracija je nezavisna: gold fajl se ne gleda unapred

---



## 9. Posle anotacije

Finalni anotirani skup ide u **Fazu 3** (treniranje i evaluacija). Kvalitet oznaka direktno utiče na macro-F1 — zato se sarkazam izbacuje, a kalibracija radi pre nego što se oznake tretiraju kao pouzdane.