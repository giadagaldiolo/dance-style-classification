"""
Calcola la durata totale (in secondi) di tutte le sequenze usate per il
training, sommando n_frames/fps per ciascun file di keypoint -- serve per
confrontare in modo equo la pipeline "singolo video" (~10s) con la
pipeline di training (sull'intero dataset, molte ore di contenuto).
"""

import os
import pickle

KEYPOINT_DIR = "annotations/keypoints2d"


def main():
    total_seconds = 0.0
    n_sequences = 0

    for filename in os.listdir(KEYPOINT_DIR):
        if not filename.endswith(".pkl") or "_sMM_" in filename:
            continue
        path = os.path.join(KEYPOINT_DIR, filename)
        with open(path, "rb") as f:
            data = pickle.load(f)

        keypoints = data["keypoints2d"][0]
        fps = data.get("fps", 60)
        n_frames = len(keypoints)
        total_seconds += n_frames / fps
        n_sequences += 1

    print(f"Sequenze incluse nel training: {n_sequences}")
    print(f"Durata totale: {total_seconds:.1f} secondi "
          f"({total_seconds/60:.1f} minuti, {total_seconds/3600:.2f} ore)")
    print(f"Durata media per sequenza: {total_seconds/n_sequences:.1f} secondi")
    print(f"\nRapporto rispetto a un video singolo di 10s: "
          f"{total_seconds/10:.0f}x")


if __name__ == "__main__":
    main()