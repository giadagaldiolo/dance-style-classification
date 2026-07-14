import os
import cv2
import pickle
import numpy as np
import torch

from mmpose.apis import MMPoseInferencer

VIDEO_PATH = "downloaded_videos/gWA_yt_03.mp4"

OUTPUT_DIR = "outputs/keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PKL = os.path.join(OUTPUT_DIR, BASE_NAME + ".pkl")
CACHE_PATH = os.path.join("downloaded_videos", BASE_NAME + ".npy")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device in uso: {DEVICE}")


def load_model():
    return MMPoseInferencer(
        pose2d="human",
        device=DEVICE
    )

def load_or_cache_video(video_path):
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
    if len(preds) == 0 or len(preds[0]) == 0:
        return np.full((17, 3), np.nan, dtype=np.float32)

    person = preds[0][0]

    xy = np.array(person["keypoints"], dtype=np.float32)

    score = np.array(
        person["keypoint_scores"],
        dtype=np.float32
    ).reshape(-1, 1)

    return np.concatenate([xy, score], axis=1)

def process_video(frames, inferencer, fps=60, batch_size=32):
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

    keypoints_all, timestamps = process_video(
        frames,
        inferencer,
        fps=fps
    )

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