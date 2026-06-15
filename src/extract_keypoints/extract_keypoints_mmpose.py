import os
import cv2
import pickle
import numpy as np

from mmpose.apis import MMPoseInferencer


# VIDEO_PATH = "videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
VIDEO_PATH = "downloaded_videos/hiphop_2.mp4"

OUTPUT_DIR = "outputs/keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PKL = os.path.join(OUTPUT_DIR, BASE_NAME + ".pkl")

DEVICE = "cpu"



def load_model():
    inferencer = MMPoseInferencer(
        pose2d="human", # MMPose carica rtmpose-m (addestrato su COCO)
        device=DEVICE
    )

    return inferencer



def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, width, height



def extract_person_keypoints(preds):
    """
    Restituisce array (17,3)
    [x, y, confidence]
    """

    if len(preds) == 0 or len(preds[0]) == 0:
        return np.full((17, 3), np.nan, dtype=np.float32)

    person = preds[0][0]

    xy = np.array(person["keypoints"], dtype=np.float32)

    score = np.array( # confidence
        person["keypoint_scores"],
        dtype=np.float32
    ).reshape(-1, 1)

    return np.concatenate([xy, score], axis=1)


def process_video(cap, fps, inferencer):
    keypoints_all = []
    timestamps = []

    frame_id = 0

   
    while True:
        ret, frame = cap.read()

        if not ret: # ret = False = fine video
            break

        result = next(inferencer(frame, show=False))

        preds = result.get("predictions", [])

        kp = extract_person_keypoints(preds)

        timestamp = int(frame_id / fps * 1_000_000)

        keypoints_all.append(kp)
        timestamps.append(timestamp)

        frame_id += 1

    cap.release()

    return keypoints_all, timestamps



def create_output(keypoints_all, timestamps, fps=60, width=1920, height=1080):
    keypoints_all = np.stack(
        keypoints_all,
        axis=0
    )  # (T,17,3)

    timestamps = np.array(
        timestamps,
        dtype=np.int64
    )

    return {
        "keypoints2d": keypoints_all[None, ...],  # (1,T,17,3)
        "timestamps": timestamps,
        "fps": fps,
        "width": width,
        "height": height
    }


def save_output(output, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(
            output,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

  

def main():
    inferencer = load_model()

    cap, fps, width, height = load_video(VIDEO_PATH)

    keypoints_all, timestamps = process_video(
        cap,
        fps,
        inferencer
    )

    output = create_output(
        keypoints_all,
        timestamps,
        fps,
        width,
        height
    )

    save_output(
        output,
        OUTPUT_PKL
    )


if __name__ == "__main__":
    main()