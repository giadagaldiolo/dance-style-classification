import os
import sys
import cv2
import pickle
import numpy as np
import torch

from mmpose.apis import MMPoseInferencer

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from sustainability.sustainability_tracker import track, log_metric

# Hardcoded path to a single video -- this script processes ONE video at a time 
VIDEO_PATH = "outputs/videos_live/live_20260811_012422.mp4"

OUTPUT_DIR = "outputs/keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PKL = os.path.join(OUTPUT_DIR, BASE_NAME + ".pkl")
# Cache of the raw decoded video frames (as a .npy array), so re-running
# this script on the same video doesn't require re-reading/decoding it
# with OpenCV every time.
CACHE_PATH = os.path.join("outputs/videos_live", BASE_NAME + ".npy")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device in uso: {DEVICE}")


def load_model():
    # Loads the MMPose 2D human pose inferencer (RTMPose + RTMDet under
    # the hood) onto the selected device. Done once, OUTSIDE the tracked
    # block in main(), since model loading is a one-time setup cost, not
    # part of the per-video inference cost being measured
    return MMPoseInferencer(
        pose2d="human",
        device=DEVICE
    )


def load_or_cache_video(video_path):
    """Reads all frames of a video into memory via OpenCV, or loads them
    from a cached .npy file if this video has already been read once
    before. Also done OUTSIDE the tracked block in main(), so video
    decoding I/O is not counted as part of the pose estimation cost."""
    if os.path.exists(CACHE_PATH):
        print("Loading cached video")
        return np.load(CACHE_PATH, allow_pickle=True)

    print("Reading video")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    frames = np.array(frames)

    np.save(CACHE_PATH, frames)

    return frames


def extract_person_keypoints(preds):
    """Extracts the 17 COCO keypoints (x, y, confidence) for a single
    person from one frame's MMPose predictions"""
    if len(preds) == 0 or len(preds[0]) == 0:
        # No person detected at all in this frame -- return an
        # all-NaN row so the sequence keeps one row per frame.
        return np.full((17, 3), np.nan, dtype=np.float32)

    person = preds[0][0]

    xy = np.array(person["keypoints"], dtype=np.float32)

    score = np.array(
        person["keypoint_scores"],
        dtype=np.float32
    ).reshape(-1, 1)

    return np.concatenate([xy, score], axis=1)


def process_video(frames, inferencer, fps=60, batch_size=32):
    """Runs MMPose inference on every frame of the video (in batches of
    `batch_size`), and builds a list of per-frame keypoints plus their
    timestamps (in microseconds, based on frame index / fps) """
    keypoints_all = []
    timestamps = []

    result_generator = inferencer(
        list(frames),
        batch_size=batch_size,
        show=False,
        return_vis=False,
    )

    for frame_id, result in enumerate(result_generator):
        preds = result.get("predictions", [])
        kp = extract_person_keypoints(preds)
        timestamp = int(frame_id / fps * 1_000_000)
        keypoints_all.append(kp)
        timestamps.append(timestamp)

    return keypoints_all, timestamps


def create_output(keypoints_all, timestamps, fps, width, height):
    """Packs the extracted keypoints and metadata into the same
    dictionary structure used throughout the project (matching the
    AIST++ keypoints2d format), ready to be pickled."""
    keypoints_all = np.stack(keypoints_all, axis=0)
    timestamps = np.array(timestamps, dtype=np.int64)

    return {
        "keypoints2d": keypoints_all[None, ...],  # (1, T, 17, 3)
        "timestamps": timestamps,
        "fps": fps,
        "width": width,
        "height": height
    }


def save_output(output, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    inferencer = load_model()

    frames = load_or_cache_video(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Frames: {len(frames)} | FPS: {fps}")

    # This is the source of the "pose estimation" energy/time figures
    # used in the sustainability chapter 
    with track("pose_estimation_mmpose", metadata={
        "n_frames": len(frames),
        "device": DEVICE,
    }):
        keypoints_all, timestamps = process_video(
            frames,
            inferencer,
            fps=fps
        )

    log_metric("pose_estimation_mmpose", n_frames_processed=len(frames))

    output = create_output(
        keypoints_all,
        timestamps,
        fps,
        width,
        height
    )

    save_output(output, OUTPUT_PKL)

    print(f"Saved keypoints → {OUTPUT_PKL}")


if __name__ == "__main__":
    main()