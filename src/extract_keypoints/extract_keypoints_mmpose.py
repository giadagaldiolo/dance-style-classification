import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

from mmpose.apis import MMPoseInferencer


# =========================
# CONFIG
# =========================
VIDEO_PATH = "videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"

OUTPUT_DIR = "outputs/keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PKL = os.path.join(OUTPUT_DIR, BASE_NAME + ".pkl")

DEVICE = "cpu"   # 👈 cambia in "cuda" solo se sei sicura


# =========================
# LOAD MODEL (STABLE)
# =========================
print("Loading MMPose model...")

inferencer = MMPoseInferencer(
    pose2d='human',
    device=DEVICE
)


# =========================
# LOAD VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

print(f"FPS: {fps}")


# =========================
# STORAGE
# =========================
keypoints_all = []
timestamps = []

frame_id = 0


# =========================
# PROCESS VIDEO
# =========================
print("Extracting keypoints...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    result = next(inferencer(frame, show=False))

    preds = result.get("predictions", [])

    # default skeleton COCO (17 keypoints)
    if len(preds) == 0 or len(preds[0]) == 0:
        kp = np.full((17, 3), np.nan, dtype=np.float32)

    else:
        # persona più confidente
        person = preds[0][0]

        xy = np.array(person["keypoints"], dtype=np.float32)
        score = np.array(person["keypoint_scores"], dtype=np.float32).reshape(-1, 1)

        kp = np.concatenate([xy, score], axis=1)

    ts = int(frame_id / fps * 1_000_000)

    keypoints_all.append(kp)
    timestamps.append(ts)

    frame_id += 1


cap.release()

# =========================
# FORMAT OUTPUT (AIST++)
# =========================
keypoints_all = np.stack(keypoints_all, axis=0)  # (T,17,3)
timestamps = np.array(timestamps, dtype=np.int64)

output = {
    "keypoints2d": keypoints_all[None, ...],  # (1,T,17,3)
    "timestamps": timestamps
}

# =========================
# SAVE
# =========================
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nSaved:", OUTPUT_PKL)
print("Shape:", output["keypoints2d"].shape)