import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FILE = "annotations/keypoints2d/gWA_sBM_cAll_d25_mWA0_ch02.pkl"
# FILE = "outputs/keypoints/gJB_yt_02.pkl"

CAMERA = 0

OUTPUT_DIR = "outputs/animations"
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


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    data = load_data(FILE)

    keypoints = data["keypoints2d"][CAMERA]  # (T, 17, 3)
    num_frames = len(keypoints)

    W = data.get("width", 1920)
    H = data.get("height", 1080)
    fps = data.get("fps", 60)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, W + 1, 200))
    ax.set_yticks(np.arange(0, H + 1, 200))

    points = []

    for i in range(17):
        point = ax.scatter(
            [],
            [],
            s=12,
            color=np.array(_COLORS[i]) / 255.0
        )
        points.append(point)


    lines = []

    for _ in SKELETON:
        line, = ax.plot(
            [],
            [],
            color="black",
            linewidth=1
        )
        lines.append(line)

    title = ax.set_title("Frame 0")

    def init():
        for point in points:
            point.set_offsets(np.empty((0, 2)))

        for line in lines:
            line.set_data([], [])

        title.set_text("Frame 0")

        return points + lines + [title]

    def update(frame):
        kp = keypoints[frame]

        x = kp[:, 0]
        y = kp[:, 1]

        for i, point in enumerate(points):
            if np.isnan(x[i]) or np.isnan(y[i]):
                point.set_offsets(np.empty((0, 2)))
            else:
                point.set_offsets([[x[i], y[i]]])

        for line, (a, b) in zip(lines, SKELETON):
            if (
                np.isnan(x[a]) or np.isnan(y[a]) or
                np.isnan(x[b]) or np.isnan(y[b])
            ):
                line.set_data([], [])
            else:
                line.set_data(
                    [x[a], x[b]],
                    [y[a], y[b]]
                )

        title.set_text(f"Frame {frame}")

        return points + lines + [title]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        interval=1000 / fps,
        blit=True
    )

    anim.save(
        OUTPUT_VIDEO,
        writer="ffmpeg",
        fps=fps
    )


if __name__ == "__main__":
    main()