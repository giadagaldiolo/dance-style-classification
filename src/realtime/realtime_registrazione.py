import os
import sys
import time
import threading
import queue
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import joblib
import torch
import pygame
import subprocess

from mmpose.apis import MMPoseInferencer

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import extract_features


MODEL_PATH = "outputs/classification/multiclass_classification.pkl"
LIVE_SESSIONS_DIR = "outputs/keypoints_live"  
BUFFER_SECONDS = 10
WEBCAM_INDEX = 0
BATCH_SIZE = 16 
LIVE_VIDEOS_DIR = "outputs/videos_live" 


_SENTINEL = object()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

MUSIC_BY_CLASS = {
    "gBR": ("music_115bpm/BR.mp3", "0:04"),
    "gHO": ("music_115bpm/HO.mp3", "0:17"),
    "gJB": ("music_115bpm/JB.mp3", "0:54"),
    "gJS": ("music_115bpm/JS.mp3", "0:13"),
    "gKR": ("music_115bpm/KR.mp3", "0:35"),
    "gLH": ("music_115bpm/LH.mp3", "1:06"),
    "gLO": ("music_115bpm/LO.mp3", "0:00"),
    "gMH": ("music_115bpm/MH.mp3", "1:40"),
    "gPO": ("music_115bpm/PO.mp3", "1:07"),
    "gWA": ("music_115bpm/WA.mp3", "0:18"),
}

METRONOME_BPM = 115.0
POSE_MAX_DIM = 640  # lato lungo massimo per l'inferenza posa (velocità)

def compute_pose_size(width, height, max_dim=POSE_MAX_DIM):
    scale = max_dim / max(width, height)
    if scale >= 1.0:
        return width, height
    return int(width * scale), int(height * scale)

def resize_for_pose(frame, target_w, target_h):
    if (frame.shape[1], frame.shape[0]) == (target_w, target_h):
        return frame
    return cv2.resize(frame, (target_w, target_h))



def start_video_writer_ffmpeg(video_path, width, height, fps, crf=18, preset="fast"):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", video_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def parse_timestamp(mmss):
    minutes, seconds = mmss.split(":")
    return int(minutes) * 60 + int(seconds)



def metronome_loop(bpm, stop_event, sound):
    """Gira in un thread separato: suona un click a intervalli regolari
    (60/bpm secondi) finché `stop_event` non viene attivato dall'esterno."""
    interval = 60.0 / bpm
    while not stop_event.is_set():
        sound.play()
        stop_event.wait(timeout=interval)  # si interrompe subito se stop_event scatta



def load_pose_inferencer():
    return MMPoseInferencer(pose2d="human", device=DEVICE)


def extract_keypoints_from_frames(frames, inferencer, batch_size=BATCH_SIZE):
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



def pose_worker_and_classify(frame_queue, inferencer, clf, width, height,
                              pose_width, pose_height, shared_state, result_holder,
                              session_name, metronome_stop_event,
                              pose_batch_size=BATCH_SIZE):
    t_start = time.time()
    keypoints_list = []
    pending_frames = []
    capture_ended = False

    while True:
        try:
            item = frame_queue.get(timeout=0.5)
        except queue.Empty:
            pass  # nessun frame arrivato in questo intervallo, si riprova
        else:
            if item is _SENTINEL:
                capture_ended = True
            else:
                pending_frames.append(item)

        if len(pending_frames) >= pose_batch_size or (capture_ended and pending_frames):
            kp_batch = extract_keypoints_from_frames(
                pending_frames, inferencer, batch_size=len(pending_frames)
            )
            kp_batch = rescale_keypoints(kp_batch, pose_width, pose_height, width, height) 
            keypoints_list.extend(list(kp_batch))
            pending_frames = []

        if capture_ended and frame_queue.empty() and not pending_frames:
            break


    n_frames = len(keypoints_list)
    print(f"{n_frames} frame")

    if n_frames == 0:
        print("Nessun frame valido ricevuto")
        result_holder["error"] = "nessun frame ricevuto"
        metronome_stop_event.set()
        return

    keypoints = np.stack(keypoints_list, axis=0)  # (T, 17, 3)
    effective_fps = shared_state["effective_fps"]

    os.makedirs(LIVE_SESSIONS_DIR, exist_ok=True)
    output_pkl = os.path.join(LIVE_SESSIONS_DIR, session_name + ".pkl")
    output = create_output(keypoints, effective_fps, width, height)
    save_output(output, output_pkl)
    print(f"Keypoints salvati: {output_pkl}")

    features = extract_features(keypoints, effective_fps)

    if features is None:
        print("Troppi frame non validi")
        result_holder["error"] = "troppi frame invalidi"
        metronome_stop_event.set()
        return

    df = pd.DataFrame([features])

    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        expected_cols = clf.named_steps["imputer"].feature_names_in_
    df = df.reindex(columns=expected_cols, fill_value=np.nan)

    pred_idx = clf.predict(df)[0]
    pred_class = CLASSES[pred_idx]

    total_elapsed = time.time() - t_start
    print(f"Stile riconosciuto: {pred_class}")

    result_holder["style"] = pred_class
    metronome_stop_event.set()
    play_music_for_style(pred_class)


def play_music_for_style(style):
    entry = MUSIC_BY_CLASS.get(style)
    if not entry:
        print(f"Nessuna traccia associata allo stile '{style}'")
        return

    path, start_str = entry
    if not os.path.exists(path):
        print(f"File non trovato per lo stile '{style}': {path}")
        return

    start_seconds = parse_timestamp(start_str)

    pygame.mixer.music.load(path)
    pygame.mixer.music.play(start=start_seconds)
    print(f"Musica avviata per stile: {style} (da {start_str})")


def rescale_keypoints(keypoints, pose_width, pose_height, video_width, video_height):
    """Riporta le coordinate dei keypoint dallo spazio 'ridotto' usato per
    l'inferenza (più rapida) allo spazio del video a piena risoluzione,
    così il .pkl salvato resta coerente con il video per l'overlay."""
    scale_x = video_width / pose_width
    scale_y = video_height / pose_height
    kp = keypoints.copy()
    kp[..., 0] *= scale_x  # coordinate x
    kp[..., 1] *= scale_y  # coordinate y
    return kp

def main():
    print(f"Device in uso: {DEVICE}")

    clf = joblib.load(MODEL_PATH)
    inferencer = load_pose_inferencer()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire la webcam (indice {WEBCAM_INDEX})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_width, pose_height = compute_pose_size(width, height)
    session_name = datetime.now().strftime("live_%Y%m%d_%H%M%S")

    os.makedirs(LIVE_VIDEOS_DIR, exist_ok=True)
    video_path = os.path.join(LIVE_VIDEOS_DIR, session_name + ".mp4")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # fallback se iVCam non riporta un fps valido

    video_process = start_video_writer_ffmpeg(video_path, width, height, video_fps)

    frame_queue = queue.Queue()
    shared_state = {"effective_fps": None}
    result_holder = {}

    pygame.mixer.init()
    click_sound = pygame.mixer.Sound("music/strong_beat.wav")
    metronome_stop_event = threading.Event()
    metronome_thread = threading.Thread(
        target=metronome_loop,
        args=(METRONOME_BPM, metronome_stop_event, click_sound),
        daemon=True,
    )
    metronome_thread.start()
    print(f"Metronomo avviato a {METRONOME_BPM:.1f} BPM (media dei brani).")


    worker_thread = threading.Thread(
        target=pose_worker_and_classify,
        args=(frame_queue, inferencer, clf, width, height, pose_width, pose_height,
            shared_state, result_holder, session_name, metronome_stop_event),
        daemon=True,
    )
    worker_thread.start()

    n_captured = 0
    start_time = time.time()
    sentinel_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossibile leggere dalla webcam, interrompo.")
            break

        elapsed = time.time() - start_time

        video_process.stdin.write(frame.tobytes())

        if elapsed <= BUFFER_SECONDS:
            pose_frame = resize_for_pose(frame, pose_width, pose_height)
            frame_queue.put(pose_frame)          # piccolo → veloce per la posa
            n_captured += 1
        elif not sentinel_sent:
            buffer_snapshot_time = time.time()
            shared_state["effective_fps"] = n_captured / (buffer_snapshot_time - start_time)
            frame_queue.put(_SENTINEL)
            sentinel_sent = True

            print(f"\nCattura completata")


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
            status = "Elaborazione in corso..."
            color = (255, 255, 0)

        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Dance Style Recognition", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_process.stdin.close()
    video_process.wait()
    print(f"Video salvato: {video_path}")
    cv2.destroyAllWindows()


    if not sentinel_sent:
        buffer_snapshot_time = time.time()
        shared_state["effective_fps"] = max(n_captured, 1) / max(buffer_snapshot_time - start_time, 1e-6)
        frame_queue.put(_SENTINEL)

    worker_thread.join(timeout=10)  
    metronome_stop_event.set()
    metronome_thread.join(timeout=2)


if __name__ == "__main__":
    main()