import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FILE = "annotations/keypoints2d/gWA_sMM_cAll_d27_mWA5_ch10.pkl"
# FILE = "outputs/keypoints/gJS_yt_02.pkl"
CAMERA = 0
OUTPUT_DIR = "outputs/animations/normalized"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_name = os.path.splitext(os.path.basename(FILE))[0]
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, base_name + ".mp4")

_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85]
]

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

NOSE = 0
LEFT_HIP = 11
RIGHT_HIP = 12
RIGHT_ANKLE = 16


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_keypoints(keypoints):
    kp = keypoints.copy()

    initial_center = (
        kp[0, LEFT_HIP, :2] +
        kp[0, RIGHT_HIP, :2]
    ) / 2

    kp[:, :, :2] -= initial_center[None, None, :]


    nose = kp[:, NOSE, :2]
    ankle = kp[:, RIGHT_ANKLE, :2]

    distances = np.linalg.norm(nose - ankle, axis=1)

    max_dist = np.nanmax(distances)

    if max_dist > 0:
        kp[:, :, :2] /= max_dist

    return kp


def main():
    data = load_data(FILE)

    keypoints = data["keypoints2d"][CAMERA]
    keypoints = normalize_keypoints(keypoints)

    num_frames = len(keypoints)
    fps = data.get("fps", 60)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.5, -1.5)
    ax.set_aspect("equal")

    def update(frame):
        ax.clear()

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(1.5, -1.5)
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
        ax.set_yticks(np.arange(-1.5, 1.6, 0.5))

        kp = keypoints[frame]

        x = kp[:, 0]
        y = kp[:, 1]

        for i in range(17):
            if not np.isnan(x[i]) and not np.isnan(y[i]):
                ax.scatter(
                    x[i],
                    y[i],
                    s=18,
                    color=np.array(_COLORS[i]) / 255.0
                )

        for a, b in SKELETON:
            if np.isnan(x[a]) or np.isnan(x[b]):
                continue

            ax.plot(
                [x[a], x[b]],
                [y[a], y[b]],
                color="black",
                linewidth=1
            )

        ax.set_title(f"Normalized Keypoints - Frame {frame}")

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps
    )

    anim.save(
        OUTPUT_VIDEO,
        writer="ffmpeg",
        fps=fps
    )


if __name__ == "__main__":
    main()