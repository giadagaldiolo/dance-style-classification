"""
Sustainability Tracker: strumento riutilizzabile per monitorare tempo,
energia e CO2 stimata di esperimenti di machine learning, con uno storico
persistente su cui costruire confronti nel tempo (non un singolo
risultato usa-e-getta).

Nato per confrontare il costo computazionale di due approcci diversi
all'interno di questa tesi (Random Forest per la classificazione, VAE per
la generazione), ma pensato per essere riutilizzabile su qualunque
progetto ML, non solo su questi due modelli.

Uso base:
    from sustainability_tracker import track

    with track("training_random_forest", metadata={"model": "RandomForest"}):
        pipeline.fit(X_train, y_train)

Aggiungere metriche calcolate DOPO il training (es. accuratezza sul test
set) a un run già registrato:
    from sustainability_tracker import log_metric
    log_metric("training_random_forest", accuracy=0.90)

Installazione (opzionale ma consigliata, per energia/CO2 oltre al tempo):
    pip install codecarbon
"""

import os
import csv
import json
import time
from contextlib import contextmanager
from datetime import datetime

try:
    from codecarbon import EmissionsTracker
    _HAS_CODECARBON = True
except ImportError:
    _HAS_CODECARBON = False

LOG_PATH = "outputs/sustainability/experiments_log.jsonl"
CODECARBON_DETAILS_DIR = "outputs/sustainability/codecarbon_details"


def _ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def _append_record(record):
    _ensure_log_dir()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_records():
    if not os.path.exists(LOG_PATH):
        return []
    records = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_file_size_mb(path):
    """Comodo per loggare la dimensione di un modello salvato su disco."""
    return round(os.path.getsize(path) / (1024 * 1024), 3) if os.path.exists(path) else None


def get_last_tracked_energy_kwh(label):
    """Ritorna l'energia (kWh) dell'ultimo record salvato con quella
    etichetta, o None se non trovato/non misurata. Utile per stimare
    l'energia di fasi non tracciate direttamente (vedi
    evaluation_energy_estimated_kwh in 8.train_lma_pipeline.py)."""
    records = [r for r in _load_records() if r["label"] == label]
    if not records:
        return None
    return records[-1]["energy_kwh"]


def _read_last_energy_from_csv(csv_path):
    """Legge l'energia consumata (kWh) dall'ultima riga scritta da
    codecarbon nel suo file emissions.csv. Piu' robusto di leggere
    attributi interni dell'oggetto EmissionsTracker."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        value = rows[-1].get("energy_consumed")
        return float(value) if value else None
    except (ValueError, KeyError):
        return None


def _read_hardware_breakdown_from_csv(csv_path):
    """Legge dall'ultima riga del CSV di codecarbon la scomposizione
    dell'energia per componente hardware (CPU, GPU, RAM), in kWh -- dato
    gia' presente nel file, semplicemente non ancora esposto nel log."""
    if not os.path.exists(csv_path):
        return {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last = rows[-1]
        breakdown = {}
        for key, col in [("cpu_energy_kwh", "cpu_energy"),
                          ("gpu_energy_kwh", "gpu_energy"),
                          ("ram_energy_kwh", "ram_energy")]:
            value = last.get(col)
            breakdown[key] = float(value) if value else None
        return breakdown
    except (ValueError, KeyError):
        return {}


@contextmanager
def track(label, metadata=None):
    """
    Misura un blocco di codice (tipicamente il training di un modello) e
    salva un record persistente in LOG_PATH: ogni chiamata si AGGIUNGE
    allo storico, non lo sovrascrive, cosi' col tempo si accumula un
    confronto tra piu' esperimenti, non solo l'ultimo.

    metadata: dict opzionale con informazioni note SUBITO (es. tipo di
    modello, iperparametri). Metriche note solo dopo (es. accuratezza sul
    test set) si aggiungono con log_metric().
    """
    print(f"\n[TRACKER] Inizio: {label}")

    tracker = None
    if _HAS_CODECARBON:
        os.makedirs(CODECARBON_DETAILS_DIR, exist_ok=True)
        tracker = EmissionsTracker(
            project_name=label,
            output_dir=CODECARBON_DETAILS_DIR,
            output_file="emissions.csv",
            log_level="error",
            measure_power_secs=1,  # <-- il default e' 15s: un training rapido
                                    # come un Random Forest finisce prima che
                                    # codecarbon riesca a fare anche una sola
                                    # misura, risultando in dati vuoti
        )
        tracker.start()

    t0 = time.time()

    try:
        yield
    finally:
        elapsed = time.time() - t0
        co2_kg = None
        energy_kwh = None

        if tracker is not None:
            co2_kg = tracker.stop()
            # Rilegge dal CSV che codecarbon stesso scrive, invece di
            # affidarsi ad attributi interni dell'oggetto (che possono non
            # esistere o cambiare nome tra versioni della libreria).
            csv_path = os.path.join(CODECARBON_DETAILS_DIR, "emissions.csv")
            energy_kwh = _read_last_energy_from_csv(csv_path)
            hardware_breakdown = _read_hardware_breakdown_from_csv(csv_path)

            if co2_kg is None or energy_kwh is None:
                print("[TRACKER] ATTENZIONE: codecarbon non ha misurato energia/CO2 "
                      "per questo run. Se il training e' durato meno di 1-2 secondi "
                      "potrebbe non bastare nemmeno measure_power_secs=1: prova ad "
                      "aumentare artificialmente il carico (es. cross-validation, "
                      "piu' run ripetuti) solo per la misurazione.")
        else:
            hardware_breakdown = {}

        record = {
            "label": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(elapsed, 2),
            "energy_kwh": energy_kwh,
            "co2_g": (co2_kg * 1000) if co2_kg is not None else None,
            "metadata": {**(metadata or {}), **hardware_breakdown},
        }
        _append_record(record)

        if co2_kg is not None:
            print(f"[TRACKER] Fine: {label}  ->  {elapsed:.1f}s, "
                  f"{energy_kwh * 1000:.2f} Wh, {co2_kg * 1000:.2f} g CO2eq")
        else:
            print(f"[TRACKER] Fine: {label}  ->  {elapsed:.1f}s "
                  f"(codecarbon non installato: misurato solo il tempo)")


def log_metric(label, **metrics):
    """
    Aggiunge metriche calcolate dopo il training (es. accuracy=0.9,
    model_size_mb=1.2) all'ULTIMO record salvato con quel label, cosi' nel
    report si possono confrontare non solo i costi ma anche "costo per
    risultato ottenuto".
    """
    records = _load_records()
    updated = False
    for record in reversed(records):
        if record["label"] == label:
            record["metadata"].update(metrics)
            updated = True
            break

    if not updated:
        print(f"[TRACKER] Nessun record trovato con label '{label}': "
              f"esegui prima track('{label}') almeno una volta.")
        return

    _ensure_log_dir()
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"[TRACKER] Metriche aggiunte a '{label}': {metrics}")