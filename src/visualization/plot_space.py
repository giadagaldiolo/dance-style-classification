"""
Plot 5 — SPACE: spostamento del bacino su finestre di 1 secondo nel tempo.

Mostra la linea della MEDIANA (feature reale: space_median_vel_1sec),
coerente con la scelta di usare la mediana per le feature "derivate dal
tempo" (più robusta agli outlier rispetto alla media — vedi discussione
sulla correlazione tra rumore di tracking e feature di Effort).

NOTA: ricalcola esattamente la stessa logica presente in extract_features
(finestra di k = round(fps) frame), quindi se in lma_extractor.py passi da
np.nanmean a np.nanmedian per questa feature, aggiorna anche qui la riga
corrispondente per restare coerente (già impostato su nanmedian in questo
script, dato che è la scelta finale discussa).
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_keypoints, plot_skeleton_with_timeseries

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import normalize_keypoints, LEFT_HIP, RIGHT_HIP


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"  # <-- cambia con un tuo file
OUT_PATH = "outputs/feature_plots/plot5_space_median_vel_1sec.mp4"
MAX_FRAMES = None  # es. 150 per accorciare l'animazione, None = tutta la sequenza


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    keypoints, fps = load_keypoints(PKL_PATH)
    kp_norm = normalize_keypoints(keypoints)
    x, y = kp_norm[:, :, 0], kp_norm[:, :, 1]

    hip_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
    hip_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2

    k = int(round(fps))  # frame equivalenti a 1 secondo
    n = len(hip_x)

    # Stessa logica di extract_features: spostamento tra coppie di frame
    # distanti k frame. Allineiamo il valore al frame "finale" della
    # finestra per poterlo sincronizzare con l'animazione; i primi k frame
    # restano NaN (non c'è ancora 1s di storia).
    vel_1sec = np.full(n, np.nan)
    if n > k:
        dx = hip_x[k:] - hip_x[:-k]
        dy = hip_y[k:] - hip_y[:-k]
        vel_1sec[k:] = np.sqrt(dx**2 + dy**2)

    median_vel = np.nanmedian(vel_1sec)

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=vel_1sec,
        feature_label="spostamento bacino (finestra 1s)",
        title="Space: spostamento del bacino su finestre di 1 secondo",
        ylabel="distanza (unità normalizzate)",
        out_path=OUT_PATH,
        hlines=[{"value": median_vel, "label": "mediana (= feature reale)", "color": "green"}],
    )


if __name__ == "__main__":
    main()