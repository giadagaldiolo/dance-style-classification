"""
Sustainability Tracker: reusable tool for monitoring the time, energy,
and estimated CO2 of machine learning experiments, with a persistent
history to build comparisons over time.

Built to compare the computational cost of different methodological
choices within this thesis (e.g. whole-video vs. segment-based
classification with a Random Forest), but designed to be reusable on any
ML project, not tied to a specific model.

Basic usage:
    from sustainability_tracker import track

    with track("training_random_forest", metadata={"model": "RandomForest"}):
        pipeline.fit(X_train, y_train)

Adding metrics computed AFTER training (e.g. test set accuracy) to an
already-recorded run:
    from sustainability_tracker import log_metric
    log_metric("training_random_forest", accuracy=0.90)

Installation:
    pip install codecarbon
"""

import os
import csv
import json
import time
from contextlib import contextmanager
from datetime import datetime
from codecarbon import EmissionsTracker

LOG_PATH = "outputs/sustainability/experiments_log.jsonl"
CODECARBON_DETAILS_DIR = "outputs/sustainability/codecarbon_details"


def _ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def _append_record(record):
    """Appends one JSON record per line (JSONL format) -- each call to
    track() adds a new line, the file is never overwritten, so history
    accumulates across every run of every script."""
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
    """Handy for logging the size of a model saved to disk."""
    return round(os.path.getsize(path) / (1024 * 1024), 3) if os.path.exists(path) else None


def get_last_tracked_energy_kwh(label):
    """Returns the energy (kWh) of the last record saved with that
    label, or None if not found / not measured. Useful for estimating
    the energy of phases that aren't tracked directly (see
    evaluation_energy_estimated_kwh in 8.train_lma_pipeline.py)."""
    records = [r for r in _load_records() if r["label"] == label]
    if not records:
        return None
    return records[-1]["energy_kwh"]


def _read_last_energy_from_csv(csv_path):
    """Reads the energy consumed (kWh) from the last row written by
    codecarbon in its emissions.csv file. More robust than relying on
    internal attributes of the EmissionsTracker object."""
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
    """Reads, from the last row of codecarbon's CSV, the energy
    breakdown by hardware component (CPU, GPU, RAM), in kWh -- data
    already present in the file, simply not yet exposed in the log. """
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
    Measures a block of code and saves a
    persistent record to LOG_PATH: each call APPENDS to the history
    instead of overwriting it, so a comparison across multiple
    experiments accumulates over time, not just the latest one.

    metadata: optional dict with information already known UP FRONT
    (e.g. model type, hyperparameters). Metrics only known afterwards
    (e.g. test set accuracy) are added with log_metric().

    If the wrapped code raises an exception, the "finally" block below
    still runs and a record is still logged (with whatever time/energy
    was measured up to that point) before the exception propagates """
    print(f"\n[TRACKER] Inizio: {label}")

    tracker = None
    os.makedirs(CODECARBON_DETAILS_DIR, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=label,
        output_dir=CODECARBON_DETAILS_DIR,
        output_file="emissions.csv",
        log_level="error",
        measure_power_secs=1,  # default is 15s: a fast training run
                                # like a Random Forest finishes before
                                # codecarbon can take even a single
                                # measurement, resulting in empty data
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
            # Re-reads from the CSV that codecarbon itself writes,
            # instead of relying on internal attributes of the object
            csv_path = os.path.join(CODECARBON_DETAILS_DIR, "emissions.csv")
            energy_kwh = _read_last_energy_from_csv(csv_path)
            hardware_breakdown = _read_hardware_breakdown_from_csv(csv_path)

            if co2_kg is None or energy_kwh is None:
                print("[TRACKER] ATTENZIONE: codecarbon non ha misurato energia/CO2 "
                      "per questo run. Se il training e' durato meno di 1-2 secondi")
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
    """Adds metrics computed after training (e.g. accuracy=0.9,
    model_size_mb=1.2) to the LAST record saved with that label, so the
    report can compare not just costs but also "cost per unit of result
    achieved"."""
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

    # Rewrites the WHOLE log file with the updated record in place
    _ensure_log_dir()
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"[TRACKER] Metriche aggiunte a '{label}': {metrics}")