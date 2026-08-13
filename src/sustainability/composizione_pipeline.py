"""
Visualizza l'INTERA pipeline (dal video grezzo alla classificazione),
distinguendo chiaramente quali fasi entrano nel confronto di sostenibilità
(estrazione feature + training, sullo stesso identico input per entrambe
le varianti whole-video/segmenti) e quali ne restano fuori (cattura
webcam, stima della posa, valutazione) -- pur mostrando una stima anche
per queste ultime, per trasparenza.

Perché webcam e pose estimation sono escluse dal confronto: il dataset
AIST++ fornisce i keypoint già estratti, quindi in QUESTO esperimento
(whole-video vs segmenti) queste due fasi non vengono mai eseguite. Sono
comunque mostrate perché fanno parte della pipeline completa (rilevanti
per i dati OOD da YouTube e per il sistema realtime), e perché sono
IDENTICHE per entrambe le varianti confrontate -- non differenziano la
scelta, quindi la loro esclusione dal confronto specifico è
metodologicamente corretta, non un'omissione arbitraria.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sustainability_tracker import _load_records


REPORT_DIR = "outputs/sustainability"
OUT_PATH = os.path.join(REPORT_DIR, "composizione_pipeline.png")

TRAINING_LABEL = "whole_video"
POSE_LABEL = "pose_estimation_mmpose"
WEBCAM_LABEL = "webcam_capture"

COLOR_COUNTED = "#4C72B0"
COLOR_EXCLUDED = "#B0B0B0"


def get_latest(label):
    records = [r for r in _load_records() if r["label"] == label]
    return records[-1] if records else None


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    training_record = get_latest(TRAINING_LABEL)
    pose_record = get_latest(POSE_LABEL)
    webcam_record = get_latest(WEBCAM_LABEL)

    # Ogni fase: (nome, durata_sec, energia_kwh_o_None, conteggiata_bool, stimata_bool)
    phases = []

    if webcam_record:
        phases.append(("Cattura webcam\n(non eseguita su AIST++)",
                        webcam_record["duration_sec"], webcam_record["energy_kwh"], False, False))
    if pose_record:
        phases.append(("Stima della posa\n(non eseguita su AIST++)",
                        pose_record["duration_sec"], pose_record["energy_kwh"], False, False))

    if training_record:
        meta = training_record.get("metadata", {})
        total_dur = training_record["duration_sec"]
        total_energy = training_record["energy_kwh"]
        extraction_pct = meta.get("extraction_pct")
        training_pct = meta.get("training_pct")

        if extraction_pct is not None and training_pct is not None:
            extraction_dur = total_dur * extraction_pct / 100
            training_dur = total_dur * training_pct / 100
            extraction_energy = total_energy * extraction_pct / 100 if total_energy else None
            training_energy = total_energy * training_pct / 100 if total_energy else None
            phases.append(("Estrazione feature", extraction_dur, extraction_energy, True, False))
            phases.append(("Training", training_dur, training_energy, True, False))
        else:
            phases.append(("Estrazione + Training (non ancora separati)",
                            total_dur, total_energy, True, False))

    if not phases:
        raise RuntimeError(
            "Nessuna fase trovata nel log. Esegui prima i benchmark e il training."
        )

    labels = [p[0] for p in phases]
    durations = [p[1] for p in phases]
    energies = [p[2] for p in phases]
    counted = [p[3] for p in phases]
    estimated = [p[4] for p in phases]
    colors = [COLOR_COUNTED if c else COLOR_EXCLUDED for c in counted]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, durations, color=colors)
    ax.set_xlabel("Tempo (secondi)")
    ax.set_title("Composizione pipeline")
    ax.invert_yaxis()

    for bar, energy, is_estimated in zip(bars, energies, estimated):
        if energy is not None:
            suffix = " (stimata)" if is_estimated else ""
            text = f"{energy*1000:.3f} Wh{suffix}"
        else:
            text = "solo tempo (energia non stimata)"
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                 text, va="center", fontsize=9)

    legend_handles = [
        mpatches.Patch(color=COLOR_COUNTED, label="Conteggiata nel confronto"),
        mpatches.Patch(color=COLOR_EXCLUDED, label="Esclusa dal confronto"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PATH)
    print(f"Grafico salvato -> {OUT_PATH}")

    print("\nFasi mancanti nel log (se presenti, il grafico le omette):")
    for name, record in [("webcam", webcam_record), ("posa", pose_record),
                          ("training/estrazione", training_record)]:
        if record is None:
            print(f"  - {name}: MANCA, esegui il benchmark/script corrispondente")


if __name__ == "__main__":
    main()