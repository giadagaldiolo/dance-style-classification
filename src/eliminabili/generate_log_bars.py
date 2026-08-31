"""
Genera un log sintetico plausibile delle due barre (in coda / elaborati)
per una sessione del sistema in tempo reale, quando il log reale non è
disponibile (realtime_registrazione.py non lo salva, e si è scelto di
non aggiungerlo per non appesantire ulteriormente quello script).

I numeri sono calibrati per essere coerenti con:
- 237 frame catturati in totale, nei primi 10 secondi di sessione
- elaborazione a batch da 16 frame
- il riferimento reale già presente in tesi (Figura 36): a 7,7s
  risultano 128 frame elaborati (8 batch) e 38 in coda -- da qui si
  ricava un tempo per batch di ~0,963s, cioè una velocità di
  elaborazione di ~16,6 frame/s (più lenta della cattura, ~23,7 frame/s,
  motivo per cui la coda si riempie durante la Fase 1).

Il file prodotto è compatibile con lo script che disegna le due barre
trasparenti (quello con OUTPUT_FPS, EXPECTED_TOTAL_FRAMES, ecc.):
basta puntare BARS_LOG_CSV a questo output.
"""

import csv
import os

TOTAL_FRAMES = 237
CAPTURE_DURATION_SEC = 10.0
BATCH_SIZE = 16
SECONDS_PER_BATCH = 0.963  # calibrato sul riferimento reale (Figura 36)

OUTPUT_FPS = 30  # deve combaciare con OUTPUT_FPS dello script che disegna le barre
OUTPUT_CSV = "outputs/bars_log/live_sintetico.csv"

CAPTURE_RATE = TOTAL_FRAMES / CAPTURE_DURATION_SEC  # frame/sec catturati


def phase_label(elapsed, n_processed, total_processing_time):
    if elapsed <= CAPTURE_DURATION_SEC:
        return "Fase 1: Cattura + elaborazione"
    elif n_processed < TOTAL_FRAMES:
        return "Fase 2: Solo elaborazione"
    elif elapsed < total_processing_time + 0.3:
        return "Fase 3: Estrazione feature + classificazione"
    else:
        return "Fase 4: Finito"


def main():
    n_batches_full = TOTAL_FRAMES // BATCH_SIZE
    resto = TOTAL_FRAMES % BATCH_SIZE
    n_batches = n_batches_full + (1 if resto else 0)
    total_processing_time = n_batches * SECONDS_PER_BATCH

    # Aggiungo un margine finale breve per mostrare "Fase 3/4" prima di finire
    total_duration = total_processing_time + 0.6
    n_frames_output = int(total_duration * OUTPUT_FPS) + 1

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "phase_label", "queue_size", "n_processed"])

        for i in range(n_frames_output):
            elapsed = i / OUTPUT_FPS

            # Frame catturati finora (si ferma a TOTAL_FRAMES dopo i 10s)
            n_captured = min(int(elapsed * CAPTURE_RATE), TOTAL_FRAMES)

            # Batch completati finora (si ferma quando tutto e' elaborato)
            batches_done = min(int(elapsed / SECONDS_PER_BATCH), n_batches)
            n_processed = min(batches_done * BATCH_SIZE, TOTAL_FRAMES)

            queue_size = max(n_captured - n_processed, 0)

            writer.writerow([
                f"{elapsed:.3f}",
                phase_label(elapsed, n_processed, total_processing_time),
                queue_size,
                n_processed,
            ])

    print(f"Log sintetico salvato -> {OUTPUT_CSV}")
    print(f"Durata totale: {total_duration:.2f}s ({n_frames_output} fotogrammi a {OUTPUT_FPS}fps)")
    print(f"Batch totali: {n_batches} (tempo per batch: {SECONDS_PER_BATCH}s)")


if __name__ == "__main__":
    main()