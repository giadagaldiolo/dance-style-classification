import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from sustainability_tracker import _load_records, LOG_PATH


REPORT_DIR = "outputs/sustainability"
os.makedirs(REPORT_DIR, exist_ok=True)

# Solo questi due esperimenti in questo report -- altre etichette nel log
# (es. webcam_capture, pose_estimation_mmpose, usate per il grafico di
# composizione della pipeline) non c'entrano con questo confronto.
RELEVANT_LABELS = ["whole_video", "segments"]


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


def plot_metric(ax, latest, col, title):
    if col in latest.columns and latest[col].notna().any():
        ax.bar(latest["label"], latest[col])
        ax.set_title(title)
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(25)
            tick_label.set_ha("right")
    else:
        ax.set_title(f"{title}\n(dati non disponibili)")


def plot_scatter(ax, latest):
    """Scatter Energia (x) vs Accuratezza (y). Linee di riferimento sottili
    per orientare la lettura a quadranti (alto-sx = ideale), SENZA sfondo
    colorato."""
    valid = latest.dropna(subset=["energy_wh", "accuracy"])
    if len(valid) == 0:
        ax.set_title("Energia vs accuratezza\n(dati non disponibili)")
        ax.axis("off")
        return

    xs = valid["energy_wh"].tolist()
    ys = valid["accuracy"].tolist()
    labels = valid["label"].tolist()

    x_min, x_max = min(xs) * 0.7, max(xs) * 1.3
    y_min, y_max = min(ys) - 0.03, max(ys) + 0.03

    for x, y, label in zip(xs, ys, labels):
        ax.scatter(x, y, s=120, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Energia (Wh)")
    ax.set_ylabel("Accuratezza")
    ax.set_title("Energia e accuratezza per modello")


def main():
    df = build_dataframe()

    csv_path = os.path.join(REPORT_DIR, "report_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTabella completa salvata -> {csv_path}")

    latest = df.drop_duplicates(subset="label", keep="last")
    latest = latest[latest["label"].isin(RELEVANT_LABELS)]

    has_accuracy = "accuracy" in latest.columns and latest["accuracy"].notna().any()
    if has_accuracy:
        latest = latest.copy()
        latest["accuracy_per_wh"] = latest["accuracy"] / latest["energy_wh"]

    metrics_row1 = [
        ("duration_sec", "Tempo (secondi)"),
        ("energy_wh", "Energia (Wh)"),
        ("co2_g", "CO2 (g)"),
    ]

    n_cols = 3
    fig = plt.figure(figsize=(4 * n_cols, 8))
    gs = gridspec.GridSpec(2, n_cols, figure=fig)

    # Riga 1: invariata, 3 pannelli separati
    for i, (col, title) in enumerate(metrics_row1):
        ax = fig.add_subplot(gs[0, i])
        plot_metric(ax, latest, col, title)

    # Riga 2: un solo pannello a piena larghezza -- lo scatter richiesto dal relatore
    ax_scatter = fig.add_subplot(gs[1, :])
    if has_accuracy:
        plot_scatter(ax_scatter, latest)
    else:
        ax_scatter.set_title("Energia vs accuratezza\n(dati non disponibili)")
        ax_scatter.axis("off")

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