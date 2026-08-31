"""
ISOLATED benchmark: measures only the cost of webcam capture (reading +
display + video encoding), with no pose estimation running at the same
time -- no GPU/MMPose load, so the measured figure reflects only this
phase.

Matches the real system's Phase 1 (see realtime_registrazione.py) as
closely as possible: same explicit resolution (1280x720), and each
captured frame is also piped to an ffmpeg subprocess for encoding, just
like the real capture loop does -- so the encoding overhead is included
here too, not just reading and displaying frames.

Captures for a fixed duration long enough to give codecarbon time to
measure reliably.
"""

import os
import time
import subprocess

import cv2


from sustainability_tracker import track, log_metric

WEBCAM_INDEX = 0
DURATION_SECONDS = 10  # same duration as BUFFER_SECONDS in the real-time system

# Throwaway output path -- content is not meant to be kept, only the
# encoding workload matters for this benchmark.
BENCHMARK_VIDEO_PATH = "outputs/sustainability/benchmark_webcam_capture_output.mp4"


def start_video_writer_ffmpeg(video_path, width, height, fps, crf=18, preset="fast"):
    """Same ffmpeg settings (codec, crf, preset) as
    realtime_registration.py's start_video_writer_ffmpeg(), so the
    encoding workload replicated here matches the real system's."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", video_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    # Opening the webcam is a one-time setup cost, kept OUTSIDE the
    # tracked block 
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire la webcam (indice {WEBCAM_INDEX})")

    # Same explicit resolution request as the real system.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  

    os.makedirs(os.path.dirname(BENCHMARK_VIDEO_PATH), exist_ok=True)
    video_process = start_video_writer_ffmpeg(BENCHMARK_VIDEO_PATH, width, height, video_fps)

    print(f"Cattura webcam per {DURATION_SECONDS}s (nessuna elaborazione AI)...")

    n_frames = 0
    # includes the same three things the real system's Phase 1 does
    # at once: reading, encoding to video, and displaying each frame.
    with track("webcam_capture", metadata={"duration_target_sec": DURATION_SECONDS}):
        start_time = time.time()
        while time.time() - start_time < DURATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Impossible to read from the webcam, stopping.")
                break
            n_frames += 1
            video_process.stdin.write(frame.tobytes())
            cv2.imshow("Benchmark cattura webcam (nessun AI)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    video_process.stdin.close()
    video_process.wait()
    cv2.destroyAllWindows()

    log_metric("webcam_capture", n_frames_captured=n_frames)
    print(f"Done: {n_frames} frame captured.")


if __name__ == "__main__":
    main()