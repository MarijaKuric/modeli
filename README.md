# README

## Završni rad

Usporedba modela za preporučivanje proizvoda temeljenih na graf neuronskim mrežama

## Opis rada

Projekt implementira četiri modela za sustave preporučivanja proizvoda:

* NGCF (Neural Graph Collaborative Filtering)
* LightGCN (Light Graph Convolutional Network)
* GraphSAGE
* PinSage

Modeli su razvijeni u programskom jeziku Python koristeći biblioteku PyTorch te su evaluirani pomoću metrika Recall@20 i NDCG@20.

## Struktura projekta

* `data/` – skup podataka (`train.txt` i `test.txt`)
* `NGCF.py` – implementacija NGCF modela
* `LightGCN.py` – implementacija LightGCN modela
* `GraphSAGE.py` – implementacija GraphSAGE modela
* `PinSage.py` – implementacija PinSage modela
* `usporedbaModela.py` – prikaz rezultata i grafova

## Zahtjevi

Projekt koristi Python 3 te sljedeće biblioteke:

* torch
* numpy
* scipy
* pandas
* matplotlib
* seaborn

Instalacija potrebnih paketa:

```bash
pip install torch numpy scipy pandas matplotlib seaborn
```

## Pokretanje

Za pokretanje pojedinog modela izvršiti odgovarajuću Python datoteku, primjerice:

```bash
python NGCF.py
```

Na isti način mogu se pokrenuti i ostali modeli:

```bash
python LightGCN.py
python GraphSAGE.py
python PinSage.py
```

## Rezultati

Nakon završetka treniranja ispisuju se vrijednosti Recall@20, NDCG@20 te epoha u kojoj je model ostvario najbolji rezultat. Rezultati se mogu dodatno prikazati tablicama i grafovima pomoću skripte za vizualizaciju.

## Autor

Ime i prezime: Marija Kurić
Studijski smjer: Informatika
Kolegij: Programsko inženjerstvo
Znanstveno područje: Društvene znanosti
Znanstveno polje: Informacijske i komunikacijske znanosti
Znanstvena grana: Informacijski sustavi i informatologija
Mentor: izv. prof. dr. sc. Nikola Tanković

