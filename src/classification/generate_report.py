"""
Genera un report comparativo (tabella + grafici) a partire dallo storico
salvato da sustainability_tracker.py — pensato per confrontare più
esperimenti tra loro (es. Random Forest vs VAE), non solo per leggere un
singolo risultato isolato.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

from sustainability_tracker import _load_records, LOG_PATH


REPORT_DIR = "outputs/sustainability"


def build_dataframe():
    records = _load_records()
    if not records:
        raise RuntimeError(
            f"Nessun esperimento registrato in {LOG_PATH}. "
            "Usa track(...) attorno ad almeno un training prima di generare il report."
        )

    rows = []
    for r in records:
        row = {
            "label": r["label"],
            "timestamp": r["timestamp"],
            "duration_sec": r["duration_sec"],
            "energy_wh": (r["energy_kwh"] * 1000) if r["energy_kwh"] is not None else None,
            "co2_g": r["co2_g"],
        }
        row.update(r.get("metadata", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = build_dataframe()

    print("\n=== STORICO ESPERIMENTI ===")
    print(df.to_string(index=False))

    csv_path = os.path.join(REPORT_DIR, "report_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTabella completa salvata -> {csv_path}")

    # Un solo confronto per etichetta: se hai ri-allenato piu' volte lo
    # stesso modello, usa l'ultimo run registrato per quella label.
    latest = df.drop_duplicates(subset="label", keep="last")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("duration_sec", "Tempo (secondi)"),
        ("energy_wh", "Energia stimata (Wh)"),
        ("co2_g", "CO2 stimata (g)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        if col in latest.columns and latest[col].notna().any():
            ax.bar(latest["label"], latest[col])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=25)
        else:
            ax.set_title(f"{title}\n(dati non disponibili)")

    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "comparison_plot.png")
    plt.savefig(plot_path)
    print(f"Grafico comparativo salvato -> {plot_path}")

    # Se e' presente una colonna "accuracy" nei metadata (tipicamente solo
    # per il classificatore, non per la VAE), calcola un indice di
    # efficienza: non solo "quanto costa" ma "quanto costa PER RISULTATO
    # OTTENUTO" -- piu' informativo di un confronto grezzo dei soli costi.
    if "accuracy" in latest.columns and latest["accuracy"].notna().any():
        latest = latest.copy()
        latest["accuracy_per_wh"] = latest["accuracy"] / latest["energy_wh"]
        print("\n=== EFFICIENZA (accuratezza per Wh, solo dove disponibile) ===")
        cols = ["label", "accuracy", "energy_wh", "accuracy_per_wh"]
        print(latest[cols].to_string(index=False))


if __name__ == "__main__":
    main()