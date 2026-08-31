"""
Two files, each with TWO side-by-side panels sharing the same y axis
(the pipeline phases, labelled only once, on the left):

1. pipeline_live.png: left panel = time, right panel = energy,
   for the 10-second live session.
2. pipeline_training.png: same structure, for the training pipeline on
   the whole dataset.  here both panels use a logarithmic scale

In both panels, the average power (W) is reported as a label next to
each bar.

"""

import os

import matplotlib.pyplot as plt

from sustainability_tracker import _load_records


REPORT_DIR = "outputs/sustainability"

# Total duration of the training dataset
TRAINING_DATASET_DURATION_SEC = 18776.8
SINGLE_VIDEO_DURATION_SEC = 10.0


def get_latest(label):
    records = [r for r in _load_records() if r["label"] == label]
    return records[-1] if records else None


def avg_power_w(energy_kwh, duration_sec):
    """Average power (W) = energy (Wh) x 3600 / duration (s)."""
    if energy_kwh is None or duration_sec is None or duration_sec <= 0:
        return None
    return energy_kwh * 1000 * 3600 / duration_sec


def _draw_panel(ax, labels, values, energies, durations, xlabel, log_scale, show_power_label=True):
    """Draws one panel (either the time or the energy side) of the dual
    chart. `values` is what actually determines the bar length (time or
    energy, depending on the caller); `energies`/`durations` are always
    passed so the average power label can be computed the same way
    regardless of which metric is on this particular panel's axis."""
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
    """phases: list of (name, duration_sec, energy_kwh_or_None).

    Draws a single figure with two side-by-side panels (time | energy)
    that share the same y axis -- phase labels appear only once, on the
    left panel.
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

    # Called ONCE, on ax_time only: since ax_time and ax_energy share the
    # same y axis (sharey=True), calling invert_yaxis() on BOTH would
    # cancel out (the second call flips it back), silently undoing the
    # first one.
    ax_time.invert_yaxis()
    ax_energy.tick_params(labelleft=False)  # no duplicated labels on the right

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Grafico salvato -> {out_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --- Live pipeline (10-second session) ---
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

    # --- Training pipeline (on the whole dataset) ---
    training_record = get_latest("whole_video")
    phases_training = []

    # Keypoint extraction: ESTIMATED by linear extrapolation from the
    # single-video benchmark, not measured directly.
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