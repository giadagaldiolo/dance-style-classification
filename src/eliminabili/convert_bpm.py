"""
Converte tutti i brani a un tempo comune (115 BPM = stesso tempo del
metronomo), usando ffmpeg con il filtro atempo (cambia la velocità senza
alterare l'intonazione). Il fattore per ciascun brano è calcolato
automaticamente dal suo BPM originale.
"""

import os
import subprocess

TARGET_BPM = 115


SONGS = {
    "gBR": ("music/BR.mp3", 110),
    "gHO": ("music/HO.mp3", 123),
    "gJB": ("music/JB.mp3", 105),
    "gJS": ("music/JS.mp3", 128),
    "gKR": ("music/KR.mp3", 130),
    "gLH": ("music/LH.mp3", 104),
    "gLO": ("music/LO.mp3", 117),
    "gMH": ("music/MH.mp3", 123),
    "gPO": ("music/PO.mp3", 103),
    "gWA": ("music/WA.mp3", 125),
}

OUTPUT_DIR = "music_115bpm"


def convert_song(path, original_bpm, target_bpm, output_dir):
    factor = target_bpm / original_bpm

    if not (0.5 <= factor <= 2.0):
        print(f"ATTENZIONE: fattore {factor:.3f} fuori dal range "
              f"supportato da atempo (0.5-2.0) per {path}")
        return

    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, filename)

    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-filter:a", f"atempo={factor:.4f}",
        output_path,
    ]
    print(f"{filename}: {original_bpm} -> {target_bpm} BPM (fattore {factor:.3f})")
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for style, (path, bpm) in SONGS.items():
        if not os.path.exists(path):
            print(f"File non trovato, salto: {path}")
            continue
        convert_song(path, bpm, TARGET_BPM, OUTPUT_DIR)

    print(f"\nFatto. File convertiti in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()