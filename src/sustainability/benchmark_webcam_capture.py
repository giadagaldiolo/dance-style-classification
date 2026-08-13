"""
Benchmark ISOLATO: misura solo il costo della cattura webcam (lettura +
visualizzazione), senza nessuna stima della posa in corso — nessun
carico GPU/MMPose, così il numero misurato riguarda solo questa fase.

Cattura per una durata fissa abbastanza lunga da dare a codecarbon il
tempo di misurare in modo affidabile (vedi discussione sui blocchi troppo
brevi).
"""

import os
import sys
import time

import cv2


from sustainability_tracker import track, log_metric

WEBCAM_INDEX = 0
DURATION_SECONDS = 10  # stessa durata di BUFFER_SECONDS nel sistema realtime


def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire la webcam (indice {WEBCAM_INDEX})")

    print(f"Cattura webcam per {DURATION_SECONDS}s (nessuna elaborazione AI)...")

    n_frames = 0
    with track("webcam_capture", metadata={"duration_target_sec": DURATION_SECONDS}):
        start_time = time.time()
        while time.time() - start_time < DURATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Impossibile leggere dalla webcam, interrompo.")
                break
            n_frames += 1
            cv2.imshow("Benchmark cattura webcam (nessun AI)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    log_metric("webcam_capture", n_frames_captured=n_frames)
    print(f"Fatto: {n_frames} frame catturati.")


if __name__ == "__main__":
    main()