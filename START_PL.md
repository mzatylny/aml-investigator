# AML Investigator — zacznij tutaj

Projekt AI/ML do wykrywania podejrzanych przepływów pieniędzy. Zawiera sieć grafową GraphSAGE, Random Forest, generator danych syntetycznych, porównanie z regułami i panel do analizy transakcji. Kod i panel są po angielsku, aby można było wykorzystać je w międzynarodowym portfolio.

Wersja 0.2 dodaje GNN analizujący powiązane transakcje. Rozwijamy tutaj projekt AML; dane nie są transakcjami autoryzacji kartowych.

## Uruchomienie

1. Rozpakuj ZIP i otwórz terminal w folderze `aml-investigator`.
2. Użyj Pythona 3.11. Sprawdzisz wersję poleceniem `python --version` (na części komputerów polecenie nazywa się `python3`).
3. Utwórz środowisko:

```bash
python -m venv .venv
```

4. Aktywuj środowisko.

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

5. Zainstaluj biblioteki i uruchom panel:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

6. Otwórz `http://localhost:8501` w przeglądarce. Pierwsze uruchomienie generuje dane i trenuje oba modele. Instalacja wymaga internetu, później panel działa lokalnie. GNN korzysta z PyTorch i działa na procesorze; nie wymaga karty graficznej. Możesz odznaczyć „Train GraphSAGE”, aby uruchomić tylko modele bazowe.

Jeżeli aktywacja środowiska w PowerShell jest zablokowana, użyj bezpośrednio `.venv\Scripts\python.exe -m pip install -r requirements.txt` oraz `.venv\Scripts\python.exe -m streamlit run app.py`.

## Co pokazać w portfolio

- **Overview:** lista alertów, liczba transakcji do sprawdzenia, precision i recall. Suwak pokazuje wpływ progu na obciążenie analityka i liczbę błędnych alarmów.
- **Investigation:** wybór transakcji i graf przepływów. Złoty kolor wskazuje wybrany przelew; strzałki pokazują kierunek. Wyświetlana historia pochodzi z poprzednich 24 godzin.
- **Graph neural network:** sieć transakcji, z których GNN pobiera informacje. Węzeł oznacza transakcję, a strzałka przepływ informacji od wcześniejszej do późniejszej transakcji. To inny graf niż graf kont i przelewów z Investigation.
- **Model & evaluation:** wyniki na odłożonym okresie, porównanie z regułami, wykres precision–recall oraz znaczenie cech.
- **Import CSV:** ocena pliku w zdefiniowanym formacie i pobranie wyników. Model jest wytrenowany na danych syntetycznych.

## Wyniki bez uruchamiania panelu

Otwórz [raport przykładowy](docs/example-results/REPORT.md). W folderze `docs/example-results` znajduje się również wykres HTML do otwarcia w przeglądarce oraz plik z dokładnymi metrykami.

Aby samodzielnie odtworzyć eksperyment:

```bash
python -m aml demo --gnn --seed 42 --output artifacts
```

Aby uruchomić testy:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Jak opowiedzieć o projekcie

„Projekt bada wykrywanie podejrzanych przepływów na danych syntetycznych. Model korzysta z historii przelewów i cech grafu dostępnych w momencie transakcji. Porównuję go z prostymi regułami i analizuję kompromis między wykrywaniem wzorców a liczbą błędnych alarmów. Panel pozwala prześledzić powiązania między kontami.”

Moduł GNN stosuje dwie warstwy GraphSAGE, które przekazują informacje między powiązanymi transakcjami. W pierwszym eksperymencie Random Forest uzyskał lepsze wyniki od GNN — oba wyniki są pokazane w raporcie. To pozwala omówić również ograniczenia złożonego modelu.

Wyniki dotyczą tego generatora. Nie oznaczają skuteczności w prawdziwym banku, a wysoki wynik modelu nie potwierdza prania pieniędzy. W dokumentacji angielskiej opisano ograniczenia i kolejne możliwe rozszerzenia.
