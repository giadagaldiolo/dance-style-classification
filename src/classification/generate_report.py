import os

import pandas as pd
import matplotlib.pyplot as plt

from sustainability_tracker import _load_records, LOG_PATH


REPORT_DIR = "outputs/sustainability"
os.makedirs(REPORT_DIR, exist_ok=True)

def build_dataframe():
    records = _load_records()
    if not records:
        raise RuntimeError(f"Nessun esperimento registrato in {LOG_PATH}")

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
    df = build_dataframe()

    csv_path = os.path.join(REPORT_DIR, "report_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTabella completa salvata -> {csv_path}")

    latest = df.drop_duplicates(subset="label", keep="last")


    has_accuracy = "accuracy" in latest.columns and latest["accuracy"].notna().any()
    if has_accuracy:
        latest = latest.copy()
        latest["accuracy_per_wh"] = latest["accuracy"] / latest["energy_wh"]

    metrics_row1 = [
        ("duration_sec", "Tempo (secondi)"),
        ("energy_wh", "Energia stimata (Wh)"),
        ("co2_g", "CO2 stimata (g)"),
    ]
    metrics_row2 = []
    if has_accuracy:
        metrics_row2.append(("accuracy", "Accuratezza"))
        metrics_row2.append(("accuracy_per_wh", "Accuratezza per Wh"))

    n_cols = 3
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    def plot_metric(ax, col, title):
        if col in latest.columns and latest[col].notna().any():
            ax.bar(latest["label"], latest[col])
            ax.set_title(title)
            for tick_label in ax.get_xticklabels():
                tick_label.set_rotation(25)
                tick_label.set_ha("right")
        else:
            ax.set_title(f"{title}\n(dati non disponibili)")

    for ax, (col, title) in zip(axes[0], metrics_row1):
        plot_metric(ax, col, title)

    for ax, (col, title) in zip(axes[1], metrics_row2):
        plot_metric(ax, col, title)


    for ax in axes[1][len(metrics_row2):]:
        ax.axis("off")

    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "comparison_plot.png")
    plt.savefig(plot_path)
    print(f"Grafico comparativo salvato -> {plot_path}")

    if has_accuracy:
        print("\n EFFICIENZA")
        cols = ["label", "accuracy", "energy_wh", "accuracy_per_wh"]
        print(latest[cols].to_string(index=False))


if __name__ == "__main__":
    main()