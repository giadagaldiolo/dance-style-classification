"""
Due grafici SEPARATI, invece di uno solo che mescolava scale diverse:

1. Pipeline per UNA sessione live: cattura webcam, stima della posa,
   estrazione feature + classificazione -- tutto su un singolo video di
   circa 10 secondi.
2. Pipeline di training: estrazione feature + training sull'INTERO
   dataset di training (molte ore di contenuto, vedi
   compute_training_dataset_duration.py per il numero esatto).

Ogni barra mostra tempo, energia stimata E potenza MEDIA in Watt -- la
potenza è quella confrontabile direttamente con il caricabatterie del
laptop (controlla l'etichetta sul tuo alimentatore, tipicamente
"Output: 19.5V, X A" -- Watt = Volt x Ampere).
"""

import os

import matplotlib.pyplot as plt

from sustainability_tracker import _load_records


REPORT_DIR = "outputs/sustainability"


def get_latest(label):
    records = [r for r in _load_records() if r["label"] == label]
    return records[-1] if records else None


def avg_power_w(energy_kwh, duration_sec):
    """Potenza media (W) = energia (Wh) x 3600 / durata (s)."""
    if energy_kwh is None or duration_sec is None or duration_sec <= 0:
        return None
    return energy_kwh * 1000 * 3600 / duration_sec


def plot_pipeline(phases, title, out_path):
    """phases: lista di (nome, durata_sec, energia_kwh_o_None)."""
    labels = [p[0] for p in phases]
    durations = [p[1] for p in phases]
    energies = [p[2] for p in phases]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars = ax.barh(labels, durations, color="#4C72B0")
    ax.set_xlabel("Tempo (secondi)")
    ax.set_title(title)
    ax.invert_yaxis()

    for bar, energy, dur in zip(bars, energies, durations):
        power = avg_power_w(energy, dur)
        if energy is not None and power is not None:
            text = f"{energy*1000:.3f} Wh  (~{power:.0f} W medi)"
        else:
            text = "solo tempo (energia non stimata)"
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                 text, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Grafico salvato -> {out_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --- Grafico 1: pipeline per UNA sessione live ---
    webcam_record = get_latest("webcam_capture")
    pose_record = get_latest("pose_estimation_mmpose")
    single_video_record = get_latest("single_video_pipeline")

    phases_live = []
    if webcam_record:
        phases_live.append(("Cattura webcam", webcam_record["duration_sec"], webcam_record["energy_kwh"]))
    if pose_record:
        phases_live.append(("Stima della posa", pose_record["duration_sec"], pose_record["energy_kwh"]))
    if single_video_record:
        phases_live.append(("Estrazione feature\n+ classificazione",
                             single_video_record["duration_sec"], single_video_record["energy_kwh"]))

    if phases_live:
        plot_pipeline(phases_live, "Pipeline per un video di 10 secondi",
                      os.path.join(REPORT_DIR, "pipeline_live.png"))
    else:
        print("Nessun dato per la pipeline live -- esegui prima i benchmark "
              "(webcam_capture, pose_estimation_mmpose, single_video_pipeline).")

    # --- Grafico 2: pipeline di TRAINING (sull'intero dataset) ---
    training_record = get_latest("whole_video")
    phases_training = []

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
            phases_training.append(("Estrazione features", extraction_dur, extraction_energy))
            phases_training.append(("Training", training_dur, training_energy))
        else:
            phases_training.append(("Estrazione + Training", total_dur, total_energy))

    if phases_training:
        plot_pipeline(phases_training, "Pipeline di training",
                      os.path.join(REPORT_DIR, "pipeline_training.png"))
    else:
        print("Nessun dato per la pipeline di training -- esegui prima 8.train_lma_pipeline.py.")


if __name__ == "__main__":
    main()