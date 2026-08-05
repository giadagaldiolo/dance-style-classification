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
    t0 = time.time()

    tracker = None
    if _HAS_CODECARBON:
        os.makedirs(CODECARBON_DETAILS_DIR, exist_ok=True)
        tracker = EmissionsTracker(
            project_name=label,
            output_dir=CODECARBON_DETAILS_DIR,
            output_file="emissions.csv",
            log_level="error",
        )
        tracker.start()

    try:
        yield
    finally:
        elapsed = time.time() - t0
        co2_kg = None
        energy_kwh = None

        if tracker is not None:
            co2_kg = tracker.stop()
            energy_attr = getattr(tracker, "_total_energy", None)
            energy_kwh = energy_attr.kWh if energy_attr is not None else None

        record = {
            "label": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(elapsed, 2),
            "energy_kwh": energy_kwh,
            "co2_g": (co2_kg * 1000) if co2_kg is not None else None,
            "metadata": metadata or {},
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