"""
Genera l'animazione delle due barre (in coda / elaborati) su sfondo
BIANCO PIENO, a partire dal log (reale o sintetico) salvato in formato
CSV con colonne: elapsed_sec, phase_label, queue_size, n_processed.

Il risultato è una sequenza di immagini PNG numerate, da convertire poi
in un video con ffmpeg per l'inserimento diretto in PowerPoint (Inserisci
-> Video), senza bisogno di gestire trasparenza.
"""

import os
import csv

from PIL import Image, ImageDraw, ImageFont

BARS_LOG_CSV = "outputs/bars_log/live_sintetico.csv"
OUTPUT_DIR = "outputs/bars_overlay/live_sintetico_frames"
OUTPUT_MP4 = "outputs/bars_overlay/barre_animazione.mp4"

OUTPUT_FPS = 30  # deve corrispondere a OUTPUT_FPS usato nel generatore del log

CANVAS_W = 480
CANVAS_H = 150
BAR_X = 10
BAR_WIDTH = 300
BAR_HEIGHT = 20

EXPECTED_TOTAL_FRAMES = 296  # stesso valore usato in realtime_registrazione.py
CAPTURE_DURATION_SEC = 10    # durata della fase di cattura, per il contatore

COLOR_BG = (255, 255, 255, 255)       # sfondo bianco pieno
COLOR_QUEUE = (255, 165, 0, 255)      # arancione
COLOR_PROCESSED = (0, 200, 0, 255)    # verde, leggermente scurito per contrasto su bianco
COLOR_BORDER = (120, 120, 120, 255)
COLOR_TEXT = (0, 0, 0, 255)           # nero, visibile su sfondo bianco
COLOR_PHASE = (0, 90, 160, 255)       # blu scuro, visibile su sfondo bianco (il ciano originale non lo era)
COLOR_TIMER = (0, 0, 0, 255)          # nero, per il contatore dei secondi


def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def read_bars_log(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "elapsed": float(row["elapsed_sec"]),
                "phase": row["phase_label"],
                "queue_size": int(row["queue_size"]),
                "n_processed": int(row["n_processed"]),
            })
    return rows


def draw_frame(row, font_phase, font_label, font_timer):
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), COLOR_BG)
    draw = ImageDraw.Draw(img)

    draw.text((BAR_X, 5), row["phase"], font=font_phase, fill=COLOR_PHASE)

    # Contatore dei secondi trascorsi, in alto a destra -- si ferma a
    # 10s quando la cattura termina, non ha senso farlo proseguire oltre
    timer_value = min(row["elapsed"], CAPTURE_DURATION_SEC)
    timer_text = f"{timer_value:.1f}s / {CAPTURE_DURATION_SEC:.0f}s"
    bbox = draw.textbbox((0, 0), timer_text, font=font_timer)
    timer_w = bbox[2] - bbox[0]
    draw.text((CANVAS_W - timer_w - BAR_X, 2), timer_text, font=font_timer, fill=COLOR_TIMER)

    y1 = 40
    queue_fill = int(BAR_WIDTH * min(row["queue_size"] / EXPECTED_TOTAL_FRAMES, 1.0))
    draw.rectangle([BAR_X, y1, BAR_X + BAR_WIDTH, y1 + BAR_HEIGHT], outline=COLOR_BORDER, width=1)
    draw.rectangle([BAR_X, y1, BAR_X + queue_fill, y1 + BAR_HEIGHT], fill=COLOR_QUEUE)
    draw.text((BAR_X + BAR_WIDTH + 10, y1 + 3), f"In coda: {row['queue_size']}",
               font=font_label, fill=COLOR_TEXT)

    y2 = 75
    proc_fill = int(BAR_WIDTH * min(row["n_processed"] / EXPECTED_TOTAL_FRAMES, 1.0))
    draw.rectangle([BAR_X, y2, BAR_X + BAR_WIDTH, y2 + BAR_HEIGHT], outline=COLOR_BORDER, width=1)
    draw.rectangle([BAR_X, y2, BAR_X + proc_fill, y2 + BAR_HEIGHT], fill=COLOR_PROCESSED)
    draw.text((BAR_X + BAR_WIDTH + 10, y2 + 3), f"Elaborati: {row['n_processed']}",
               font=font_label, fill=COLOR_TEXT)

    return img


def main():
    rows = read_bars_log(BARS_LOG_CSV)
    print(f"{len(rows)} istanti letti dal log.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_phase = load_font(16)
    font_label = load_font(14)
    font_timer = load_font(20)

    for i, row in enumerate(rows):
        img = draw_frame(row, font_phase, font_label, font_timer)
        img.convert("RGB").save(os.path.join(OUTPUT_DIR, f"frame_{i:05d}.png"))

    print(f"Sequenza PNG salvata in: {OUTPUT_DIR}")
    print("Ora genera il video con:")
    print(f'  ffmpeg -framerate {OUTPUT_FPS} -i {OUTPUT_DIR}/frame_%05d.png '
          f'-c:v libx264 -pix_fmt yuv420p {OUTPUT_MP4}')


if __name__ == "__main__":
    main()