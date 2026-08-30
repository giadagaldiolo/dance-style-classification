"""
Due file, ciascuno con DUE pannelli affiancati che condividono lo stesso
asse y (le fasi della pipeline, etichettate una sola volta, a sinistra):

1. pipeline_live.png: pannello sinistro = tempo, pannello destro = energia,
   per la sessione live di 10 secondi.
2. pipeline_training.png: stessa struttura, per la pipeline di training
   sull'intero dataset -- qui entrambi i pannelli usano la scala
   logaritmica, perché lo squilibrio tra le fasi è troppo grande per una
   scala lineare sia in tempo sia in energia.

In entrambi i pannelli, la potenza media (W) è riportata in etichetta
accanto a ciascuna barra.

Nota sulla stima dell'estrazione keypoint per il training: AIST++
fornisce i file di keypoint già estratti, quindi questa fase non viene
mai eseguita realmente in questo lavoro. Il suo costo è stimato per
estrapolazione lineare dal benchmark su un singolo video di 10 secondi
(pose_estimation_mmpose), moltiplicato per il rapporto tra la durata
totale del dataset di training e quei 10 secondi. Presuppone quindi che
il costo cresca linearmente con la durata del video -- un'assunzione
ragionevole ma non verificata direttamente.
"""

import os

import matplotlib.pyplot as plt

from sustainability_tracker import _load_records


REPORT_DIR = "outputs/sustainability"

# Durata totale del dataset di training, da compute_training_dataset_duration.py
TRAINING_DATASET_DURATION_SEC = 18776.8
SINGLE_VIDEO_DURATION_SEC = 10.0


def get_latest(label):
    records = [r for r in _load_records() if r["label"] == label]
    return records[-1] if records else None


def avg_power_w(energy_kwh, duration_sec):
    """Potenza media (W) = energia (Wh) x 3600 / durata (s)."""
    if energy_kwh is None or duration_sec is None or duration_sec <= 0:
        return None
    return energy_kwh * 1000 * 3600 / duration_sec


def _draw_panel(ax, labels, values, energies, durations, xlabel, log_scale, show_power_label=True):
    bars = ax.barh(labels, values, color="#4C72B0")
    ax.set_xlabel(xlabel + (" -- scala log." if log_scale else ""), fontsize=13)
    ax.tick_params(axis="both", labelsize=12)

    if log_scale:
        ax.set_xscale("log")

    if show_power_label:
        for bar, energy, dur in zip(bars, energies, durations):
            power = avg_power_w(energy, dur)
            text = f"~{power:.0f} W medi" if power is not None else "n.d."
            x_pos = bar.get_width() * (1.05 if log_scale else 1.02)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                     text, va="center", fontsize=10.5)


def plot_pipeline_dual(phases, title, out_path, log_scale=False):
    """phases: lista di (nome, durata_sec, energia_kwh_o_None).

    Disegna un'unica figura con due pannelli affiancati (tempo | energia),
    che condividono lo stesso asse y -- le etichette delle fasi compaiono
    una sola volta, sul pannello di sinistra.
    """
    labels = [p[0] for p in phases]
    durations = [p[1] for p in phases]
    energies = [p[2] for p in phases]
    energies_wh = [(e * 1000 if e is not None else 0) for e in energies]

    fig, (ax_time, ax_energy) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    _draw_panel(ax_time, labels, durations, energies, durations,
                "Tempo (secondi)", log_scale, show_power_label=False)
    _draw_panel(ax_energy, labels, energies_wh, energies, durations,
                "Energia (Wh)", log_scale, show_power_label=True)

    ax_time.invert_yaxis()  # una sola volta: ax_energy condivide lo stesso asse y
    ax_energy.tick_params(labelleft=False)  # niente etichette duplicate a destra

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Grafico salvato -> {out_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --- Pipeline live (sessione di 10 secondi) ---
    webcam_record = get_latest("webcam_capture")
    pose_record = get_latest("pose_estimation_mmpose")
    single_video_record = get_latest("single_video_pipeline")

    phases_live = []
    if webcam_record:
        phases_live.append(("Cattura webcam", webcam_record["duration_sec"], webcam_record["energy_kwh"]))
    if pose_record:
        phases_live.append(("Estrazione keypoints", pose_record["duration_sec"], pose_record["energy_kwh"]))
    if single_video_record:
        phases_live.append(("Estrazione feature\n+ classificazione",
                             single_video_record["duration_sec"], single_video_record["energy_kwh"]))

    if phases_live:
        plot_pipeline_dual(phases_live, "Pipeline per un video di 10 secondi",
                            os.path.join(REPORT_DIR, "pipeline_live.png"))
    else:
        print("Nessun dato per la pipeline live -- esegui prima i benchmark "
              "(webcam_capture, pose_estimation_mmpose, single_video_pipeline).")

    # --- Pipeline di training (sull'intero dataset) ---
    training_record = get_latest("whole_video")
    phases_training = []

    if pose_record:
        scale_factor = TRAINING_DATASET_DURATION_SEC / SINGLE_VIDEO_DURATION_SEC
        estimated_duration = pose_record["duration_sec"] * scale_factor
        estimated_energy = (pose_record["energy_kwh"] * scale_factor
                             if pose_record["energy_kwh"] is not None else None)
        phases_training.append(("Estrazione keypoints", estimated_duration, estimated_energy))

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
            phases_training.append(("Estrazione feature", extraction_dur, extraction_energy))
            phases_training.append(("Training", training_dur, training_energy))
        else:
            phases_training.append(("Estrazione feature\n+ Training", total_dur, total_energy))

    if phases_training:
        plot_pipeline_dual(phases_training, "Pipeline di training",
                            os.path.join(REPORT_DIR, "pipeline_training.png"),
                            log_scale=True)
    else:
        print("Nessun dato per la pipeline di training -- esegui prima 8.train_lma_pipeline.py.")


if __name__ == "__main__":
    main()