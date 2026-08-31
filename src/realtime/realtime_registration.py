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


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = "outputs/classification/multiclass_classification.pkl"
LIVE_SESSIONS_DIR = "outputs/keypoints_live"
BUFFER_SECONDS = 10          # length of the capture window (Phase 1)
WEBCAM_INDEX = 0
BATCH_SIZE = 16              # pose estimation mini-batch size
LIVE_VIDEOS_DIR = "outputs/videos_live"

# Fixed reference value for the two progress bars, NOT the actual 
# number of frames captured in a given session. keeping this fixed 
# means the bars have a consistent visualscale across different runs
EXPECTED_TOTAL_FRAMES = 296


_SENTINEL = object()  # marks "capture has ended" on the frame queue

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

# Each recognized style triggers a matching track, starting from a
# specific point (skipping the intro) to reach an energetic section immediately.
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

# All 10 tracks above were converted to this same BPM, so the metronome
# (which plays before the real track starts) keeps a consistent tempo
# with whichever track ends up playing.
METRONOME_BPM = 115.0
POSE_MAX_DIM = 640  # longest side used for pose inference, for speed


# =============================================================================
# Video / frame utilities
# =============================================================================

def compute_pose_size(width, height, max_dim=POSE_MAX_DIM):
    """Computes a downscaled (width, height) for pose estimation input,
    capping the longest side at max_dim to speed up inference -- the
    full-resolution frame is still used for display and recording."""
    scale = max_dim / max(width, height)
    if scale >= 1.0:
        return width, height
    return int(width * scale), int(height * scale)


def resize_for_pose(frame, target_w, target_h):
    if (frame.shape[1], frame.shape[0]) == (target_w, target_h):
        return frame
    return cv2.resize(frame, (target_w, target_h))


def start_video_writer_ffmpeg(video_path, width, height, fps, crf=18, preset="fast"):
    """Spawns an ffmpeg subprocess that reads raw BGR frames from stdin
    and encodes them into an H.264 .mp4 file -- used to record the full
    session at full resolution, independently of the pose estimation
    pipeline (which works on downscaled frames)."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", video_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def rescale_keypoints(keypoints, pose_width, pose_height, video_width, video_height):
    """Maps keypoint coordinates from the downscaled space used for pose
    inference (faster) back to the full-resolution video space, so the
    saved .pkl file stays consistent with the video for the overlay."""
    scale_x = video_width / pose_width
    scale_y = video_height / pose_height
    kp = keypoints.copy()
    kp[..., 0] *= scale_x  # coordinate x
    kp[..., 1] *= scale_y  # coordinate y
    return kp


# =============================================================================
# Metronome
# =============================================================================

def parse_timestamp(mmss):
    """Converts a "M:SS" string (e.g. "1:06") into total seconds."""
    minutes, seconds = mmss.split(":")
    return int(minutes) * 60 + int(seconds)


def metronome_loop(bpm, stop_event, sound):
    """Runs in a separate thread: plays a click at regular intervals
    (60/bpm seconds) until `stop_event` is triggered from outside."""
    interval = 60.0 / bpm
    while not stop_event.is_set():
        sound.play()
        stop_event.wait(timeout=interval)  


# =============================================================================
# Pose estimation
# =============================================================================

def load_pose_inferencer():
    return MMPoseInferencer(pose2d="human", device=DEVICE)


def extract_keypoints_from_frames(frames, inferencer, batch_size=BATCH_SIZE):
    """Runs MMPose on a list of (already downscaled) frames and returns
    a (T, 17, 3) array of keypoints -- NaN-filled rows for frames where
    no person was detected """
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
    """Packs keypoints and metadata into the same dictionary structure
    used throughout the project (matching the AIST++ keypoints2d format)."""
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


# =============================================================================
# Worker thread: pose estimation + classification (Phases 1-3)
# =============================================================================

def pose_worker_and_classify(frame_queue, inferencer, clf, width, height,
                              pose_width, pose_height, shared_state, result_holder,
                              session_name, metronome_stop_event,
                              pose_batch_size=BATCH_SIZE):
    """Runs in a dedicated thread, in parallel with frame capture in
    main(). Continuously pulls frames from `frame_queue` in mini-batches
    and runs pose estimation on them as soon as enough are available.

    Once every frame has been processed (queue fully drained), extracts
    LMA features from the full sequence, classifies the style, and
    triggers the matching music track (Phase 3).
    """

    keypoints_list = []
    pending_frames = []
    capture_ended = False

    while True:
        try:
            # Waits up to 0.5s for a new frame; if none arrives, loops
            # back around (this timeout also lets the loop periodically
            # re-check the exit condition below even if the queue stays
            # empty for a while).
            item = frame_queue.get(timeout=0.5)
        except queue.Empty:
            pass  
        else:
            if item is _SENTINEL:
                # Placed on the queue by main() once the 10-second
                # capture window has ended 
                capture_ended = True
            else:
                pending_frames.append(item)

        # Processes a mini-batch either when enough frames have piled up
        # (normal case, during Phase 1), or -- once capture has ended --
        # processes whatever is left, even if smaller than a full batch
        # (this is what drains the final, smaller-than-usual batch during
        # Phase 2).
        if len(pending_frames) >= pose_batch_size or (capture_ended and pending_frames):
            kp_batch = extract_keypoints_from_frames(
                pending_frames, inferencer, batch_size=len(pending_frames)
            )
            kp_batch = rescale_keypoints(kp_batch, pose_width, pose_height, width, height)
            keypoints_list.extend(list(kp_batch))
            shared_state["n_processed"] = len(keypoints_list)
            pending_frames = []

        # Exit condition: capture has ended AND there is nothing left to
        # process, neither in the queue nor pending.
        if capture_ended and frame_queue.empty() and not pending_frames:
            shared_state["queue_drained"] = True
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

    # --- Phase 3: feature extraction + classification ---
    features = extract_features(keypoints, effective_fps)

    if features is None:
        print("Troppi frame non validi")
        result_holder["error"] = "troppi frame invalidi"
        metronome_stop_event.set()
        return

    df = pd.DataFrame([features])

    # Aligns the feature columns to the exact order/names the model was
    # trained on.
    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        expected_cols = clf.named_steps["imputer"].feature_names_in_
    df = df.reindex(columns=expected_cols, fill_value=np.nan)

    pred_idx = clf.predict(df)[0]
    pred_class = CLASSES[pred_idx]

    print(f"Stile riconosciuto: {pred_class}")

    result_holder["style"] = pred_class
    metronome_stop_event.set()
    play_music_for_style(pred_class)


def play_music_for_style(style):
    """Loads and plays the track associated with the recognized style,
    starting from a pre-chosen timestamp (skipping the intro)."""
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


# =============================================================================
# On-screen display
# =============================================================================

def draw_progress_bars(display_frame, queue_size, n_processed, max_queue, max_processed, phase_label):
    h, w = display_frame.shape[:2]
    bar_x = 20
    bar_width = 300
    bar_height = 20

    cv2.putText(display_frame, phase_label, (bar_x, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Bar 1: frames IN QUEUE (fills up during capture, empties afterwards)
    y1 = h - 75
    queue_fill = int(bar_width * min(queue_size / max_queue, 1.0))
    cv2.rectangle(display_frame, (bar_x, y1), (bar_x + bar_width, y1 + bar_height), (80, 80, 80), 1)
    cv2.rectangle(display_frame, (bar_x, y1), (bar_x + queue_fill, y1 + bar_height), (0, 165, 255), -1)
    cv2.putText(display_frame, f"In coda: {queue_size}", (bar_x + bar_width + 10, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Bar 2: frames PROCESSED (cumulative, always increasing)
    y2 = h - 40
    processed_fill = int(bar_width * min(n_processed / max_processed, 1.0))
    cv2.rectangle(display_frame, (bar_x, y2), (bar_x + bar_width, y2 + bar_height), (80, 80, 80), 1)
    cv2.rectangle(display_frame, (bar_x, y2), (bar_x + processed_fill, y2 + bar_height), (0, 255, 0), -1)
    cv2.putText(display_frame, f"Elaborati: {n_processed}", (bar_x + bar_width + 10, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# =============================================================================
# Main: capture loop (Phase 1) + orchestration
# =============================================================================

def main():
    print(f"Using device: {DEVICE}")

    clf = joblib.load(MODEL_PATH)
    inferencer = load_pose_inferencer()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam (index {WEBCAM_INDEX})")

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
        print("Warning: Invalid FPS from webcam, using default 30 FPS.")
        video_fps = 30  

    # Full-resolution video recording, independent of the (downscaled)
    # pose estimation pipeline -- this is the raw footage later used for
    # the keypoint overlay added in post-production.
    video_process = start_video_writer_ffmpeg(video_path, width, height, video_fps)

    frame_queue = queue.Queue()
    shared_state = {"effective_fps": None, "n_processed": 0, "queue_drained": False}
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

    # Starts the pose estimation + classification worker thread, running
    # in parallel with the capture loop below.
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
            print("Cannot read from webcam, stopping.")
            break

        elapsed = time.time() - start_time

        # Full-resolution frame goes straight to the video file
        video_process.stdin.write(frame.tobytes())

        if elapsed <= BUFFER_SECONDS:
            #  a downscaled copy is queued for pose estimation
            # (Phase 1: capture and pose estimation run in parallel).
            pose_frame = resize_for_pose(frame, pose_width, pose_height)
            frame_queue.put(pose_frame)          
            n_captured += 1
        elif not sentinel_sent:
            # Capture window has just ended: computes the REAL 
            # capture rate and signals the
            # worker thread that no more frames are coming.
            buffer_snapshot_time = time.time()
            shared_state["effective_fps"] = n_captured / (buffer_snapshot_time - start_time)
            frame_queue.put(_SENTINEL)
            sentinel_sent = True

            print(f"\nCapture completed")

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


        if elapsed <= BUFFER_SECONDS:
            phase_label = "Fase 1: Cattura + elaborazione"
        elif not shared_state["queue_drained"]:
            phase_label = "Fase 2: Solo elaborazione"
        elif "style" not in result_holder and "error" not in result_holder:
            phase_label = "Fase 3: Estrazione feature + classificazione"
        else:
            phase_label = "Fase 4: Finito"

        draw_progress_bars(
            display_frame,
            queue_size=frame_queue.qsize(),
            n_processed=shared_state["n_processed"],
            max_queue=EXPECTED_TOTAL_FRAMES,
            max_processed=EXPECTED_TOTAL_FRAMES,
            phase_label=phase_label,
        )

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

    # Safety net: if the capture loop exited early (e.g. webcam
    # disconnected) before ever reaching the normal sentinel-sending
    # branch above, make sure the worker thread still gets a sentinel
    # and a reasonable effective_fps estimate, instead of waiting forever.
    if not sentinel_sent:
        buffer_snapshot_time = time.time()
        shared_state["effective_fps"] = max(n_captured, 1) / max(buffer_snapshot_time - start_time, 1e-6)
        frame_queue.put(_SENTINEL)

    worker_thread.join(timeout=10)
    metronome_stop_event.set()
    metronome_thread.join(timeout=2)


if __name__ == "__main__":
    main()