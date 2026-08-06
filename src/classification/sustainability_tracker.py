import os
import json
import time
from contextlib import contextmanager
from datetime import datetime
from codecarbon import EmissionsTracker


LOG_PATH = "outputs/sustainability/experiments_log.jsonl"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
CODECARBON_DETAILS_DIR = "outputs/sustainability/codecarbon_details"
os.makedirs(CODECARBON_DETAILS_DIR, exist_ok=True)


def _append_record(record):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_records():
    records = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_file_size_mb(path):
    return round(os.path.getsize(path) / (1024 * 1024), 3) if os.path.exists(path) else None


@contextmanager
def track(label, metadata=None):
    print(f"Inizio: {label}")
    t0 = time.time()

    tracker = None
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
            "metadata": metadata or {}, # dict opzionale con informazioni note SUBITO (es. tipo di modello, iperparametri).
        }
        _append_record(record)

        if co2_kg is not None:
            print(f"Fine: {label}  ->  {elapsed:.1f}s, "
                  f"{energy_kwh * 1000:.2f} Wh, {co2_kg * 1000:.2f} g CO2eq")
        else:
            print(f"Fine: {label}  ->  {elapsed:.1f}s "
                  f"(codecarbon non installato: misurato solo il tempo)")


def log_metric(label, **metrics): # Metriche note solo dopo (es. accuratezza sultest set) si aggiungono con log_metric()
    records = _load_records()
    updated = False
    for record in reversed(records):
        if record["label"] == label:
            record["metadata"].update(metrics)
            updated = True
            break

    if not updated:
        print(f"Nessun record trovato con label '{label}': ")
        return

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Metriche aggiunte a '{label}': {metrics}")