"""
Script di riconoscimento stile di danza in tempo reale.

Flusso:
1. Cattura dalla webcam per BUFFER_SECONDS secondi (default 10s).
2. Appena il buffer è pieno, lancia estrazione keypoint + classificazione
   in un thread separato, cosi' la cattura video continua senza bloccarsi
   (il ballerino puo' continuare a ballare mentre il sistema elabora).
3. Appena la classificazione e' pronta, avvia la traccia musicale associata
   allo stile riconosciuto.

IMPORTANTE (da non modificare mai per "velocizzare" lo script):
la logica di estrazione feature (extract_features) e' importata da
lma_extractor.py, lo STESSO modulo usato per costruire il dataset di
training e per l'estrazione keypoint offline. Qualunque differenza, anche
minima, tra come le feature vengono calcolate offline e in tempo reale
reintroduce lo stesso tipo di "distribution shift" gia' diagnosticato e
corretto nella fase di training/OOD (rumore di tracking, normalizzazione,
fps). Se in futuro si vuole alleggerire il modello di pose estimation
(es. rtmpose-s invece di rtmpose-m) per guadagnare velocita', va prima
ri-validato con le stesse diagnostiche (correlazione con la confidenza,
tabella degli z-score) usate per il dataset offline.
"""

import os
import sys
import time
import threading
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import joblib
import torch

from mmpose.apis import MMPoseInferencer

# Import dal modulo condiviso con la pipeline offline (src/classification/).
# Aggiungiamo "src/" al path di ricerca dei moduli così l'import funziona
# indipendentemente da dove/come viene lanciato questo script, senza dover
# copiare o duplicare lma_extractor.py in questa cartella.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import extract_features


# --------------------------------------------------------------------------
# CONFIGURAZIONE
# --------------------------------------------------------------------------

MODEL_PATH = "outputs/classification/multiclass_classification.pkl"
LIVE_SESSIONS_DIR = "outputs/keypoints_live"  # stesso formato dei .pkl offline
BUFFER_SECONDS = 10
WEBCAM_INDEX = 0
BATCH_SIZE = 16

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]


MUSIC_BY_CLASS = {
    "gBR": ("music/BR.mp3", "0:04"),
    "gHO": ("music/HO.mp3", "0:15"),
    "gJB": ("music/JB.mp3", "1:22"),
    "gJS": ("music/JS.mp3", "0:11"),
    "gKR": ("music/KR.mp3", "0:23"),
    "gLH": ("music/LH.mp3", "2:16"),
    "gLO": ("music/LO.mp3", "0:00"),
    "gMH": ("music/MH.mp3", "1:33"),
    "gPO": ("music/PO.mp3", "1:15"),
    "gWA": ("music/WA.mp3", "0:17"),
}


def parse_timestamp(mmss):
    """Converte una stringa 'M:SS' (es. '1:22') in secondi (float)."""
    minutes, seconds = mmss.split(":")
    return int(minutes) * 60 + int(seconds)


# --------------------------------------------------------------------------
# ESTRAZIONE KEYPOINT
# (stessa logica dello script offline, adattata a una lista di frame
# gia' in memoria invece che letti da file video)
# --------------------------------------------------------------------------

def load_pose_inferencer():
    return MMPoseInferencer(pose2d="human", device=DEVICE)


def extract_keypoints_from_frames(frames, inferencer, batch_size=BATCH_SIZE):
    """Ritorna un array (T, 17, 3) di keypoint 2D + score, un frame per riga."""
    result_generator = inferencer(
        list(frames),
        batch_size=batch_size,
        show=False,
        return_vis=False,
    )

    keypoints_all = []
    for result in result_generator:
        preds = result.get("predictions", [])
        if len(preds) == 0 or len(preds[0]) == 0:
            kp = np.full((17, 3), np.nan, dtype=np.float32)
        else:
            person = preds[0][0]
            xy = np.array(person["keypoints"], dtype=np.float32)
            score = np.array(person["keypoint_scores"], dtype=np.float32).reshape(-1, 1)
            kp = np.concatenate([xy, score], axis=1)
        keypoints_all.append(kp)

    return np.stack(keypoints_all, axis=0)  # (T, 17, 3)


def create_output(keypoints_all, fps, width, height):
    """Stessa struttura dello script offline (extract_keypoints_mmpose.py),
    cosi' un file catturato dal vivo e' riutilizzabile ovunque un .pkl
    offline lo sarebbe (es. come nuovo campione OOD, o per debug)."""
    n_frames = keypoints_all.shape[0]
    timestamps = np.array(
        [int(i / fps * 1_000_000) for i in range(n_frames)],
        dtype=np.int64,
    )
    return {
        "keypoints2d": keypoints_all[None, ...],  # (1, T, 17, 3)
        "timestamps": timestamps,
        "fps": fps,
        "width": width,
        "height": height,
    }


def save_output(output, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------
# CLASSIFICAZIONE + AVVIO MUSICA
# (eseguito nel thread in background, non blocca la cattura webcam)
# --------------------------------------------------------------------------

def classify_and_play_music(frames_buffer, effective_fps, width, height,
                             inferencer, clf, result_holder):
    t0 = time.time()
    print(f"[BG] Buffer ricevuto: {len(frames_buffer)} frame, "
          f"fps effettivo stimato: {effective_fps:.1f}")
    print("[BG] Estrazione keypoint in corso...")

    keypoints = extract_keypoints_from_frames(frames_buffer, inferencer)

    # Salva subito la sessione grezza, PRIMA di classificare: cosi' i
    # keypoint restano disponibili anche se la classificazione fallisce
    # (es. persona non rilevata a sufficienza) o per riuso futuro.
    os.makedirs(LIVE_SESSIONS_DIR, exist_ok=True)
    session_name = datetime.now().strftime("live_%Y%m%d_%H%M%S")
    output_pkl = os.path.join(LIVE_SESSIONS_DIR, session_name + ".pkl")
    output = create_output(keypoints, effective_fps, width, height)
    save_output(output, output_pkl)
    print(f"[BG] Sessione salvata → {output_pkl}")

    print("[BG] Estrazione feature LMA...")
    features = extract_features(keypoints, effective_fps)

    if features is None:
        print("[BG] Troppi frame non validi (persona non rilevata a sufficienza). "
              "Impossibile classificare.")
        result_holder["error"] = "troppi frame invalidi"
        return

    df = pd.DataFrame([features])

    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        expected_cols = clf.named_steps["imputer"].feature_names_in_
    df = df.reindex(columns=expected_cols, fill_value=np.nan)

    pred_idx = clf.predict(df)[0]
    pred_class = CLASSES[pred_idx]

    elapsed = time.time() - t0
    print(f"[BG] Stile riconosciuto: {pred_class}  (elaborazione: {elapsed:.1f}s)")

    result_holder["style"] = pred_class
    play_music_for_style(pred_class)


def play_music_for_style(style):
    try:
        import pygame
    except ImportError:
        print("pygame non installato. Esegui: pip install pygame")
        return

    entry = MUSIC_BY_CLASS.get(style)
    if not entry:
        print(f"Nessuna traccia associata allo stile '{style}'")
        return

    path, start_str = entry
    if not os.path.exists(path):
        print(f"File non trovato per lo stile '{style}': {path}")
        return

    start_seconds = parse_timestamp(start_str)

    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(start=start_seconds)
    print(f"Musica avviata per stile: {style} (da {start_str})")


# --------------------------------------------------------------------------
# LOOP PRINCIPALE: cattura webcam + orchestrazione
# --------------------------------------------------------------------------

def main():
    print(f"Device in uso: {DEVICE}")

    print("Caricamento modello di classificazione...")
    clf = joblib.load(MODEL_PATH)

    print("Caricamento modello di pose estimation (MMPose)...")
    inferencer = load_pose_inferencer()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire la webcam (indice {WEBCAM_INDEX})")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Webcam aperta ({width}x{height}). Registrazione buffer di {BUFFER_SECONDS}s in corso...")
    print("Premi 'q' per uscire in qualsiasi momento.")

    frames_buffer = []
    start_time = time.time()
    bg_thread = None
    result_holder = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossibile leggere dalla webcam, interrompo.")
            break

        elapsed = time.time() - start_time

        if elapsed <= BUFFER_SECONDS:
            frames_buffer.append(frame)
        elif bg_thread is None:
            # Buffer completo: calcola l'fps EFFETTIVO osservato durante la
            # cattura reale (piu' affidabile del valore nominale della
            # webcam, che spesso non riflette il vero framerate ottenuto,
            # soggetto a overhead di visualizzazione/elaborazione).
            buffer_snapshot_time = time.time()
            n_frames = len(frames_buffer)
            effective_fps = n_frames / (buffer_snapshot_time - start_time)

            print(f"\nBuffer completo: {n_frames} frame in "
                  f"{buffer_snapshot_time - start_time:.2f}s -> "
                  f"fps effettivo: {effective_fps:.2f}")
            print("Avvio elaborazione in background. "
                  "Il ballerino puo' continuare a ballare...\n")

            bg_thread = threading.Thread(
                target=classify_and_play_music,
                args=(list(frames_buffer), effective_fps, width, height,
                      inferencer, clf, result_holder),
                daemon=True,
            )
            bg_thread.start()

        # Feedback visivo a schermo
        display_frame = frame.copy()
        if "style" in result_holder:
            status = f"Stile riconosciuto: {result_holder['style']}"
            color = (0, 255, 0)
        elif "error" in result_holder:
            status = f"Errore: {result_holder['error']}"
            color = (0, 0, 255)
        elif elapsed <= BUFFER_SECONDS:
            status = f"REGISTRAZIONE... {elapsed:.1f}/{BUFFER_SECONDS}s"
            color = (0, 165, 255)
        else:
            status = "Elaborazione in corso, continua a ballare..."
            color = (255, 255, 0)

        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Dance Style Recognition - Live", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if bg_thread is not None:
        bg_thread.join(timeout=5)  # attende la fine del thread prima di chiudere


if __name__ == "__main__":
    main()