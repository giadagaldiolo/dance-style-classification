import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# FILE = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
FILE = "outputs/keypoints/gJB_yt_02.pkl"

OUTPUT_DIR = "outputs/trajectories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_name = os.path.splitext(os.path.basename(FILE))[0]
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, base_name + ".mp4")

CAMERA = 0
ID = 10
TRAIL_LENGTH = 30

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

    keypoints = data["keypoints2d"][CAMERA]
    num_frames = len(keypoints)

    W = data["width"]
    H = data["height"]
    fps = data["fps"]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(0, W + 1, 200))
    ax.set_yticks(np.arange(0, H + 1, 200))

    points = []

    for i in range(17):
        p = ax.scatter(
            [],
            [],
            s=12,
            color=np.array(_COLORS[i]) / 255
        )
        points.append(p)

    lines = []

    for _ in SKELETON:
        line, = ax.plot(
            [],
            [],
            color="black",
            linewidth=1
        )
        lines.append(line)


    trajectory, = ax.plot(
        [],
        [],
        color="red",
        linewidth=2,
        label="Trajectory"
    )

    current_artist = ax.scatter(
        [],
        [],
        s=30,
        color="red"
    )

    title = ax.set_title("Frame 0")

    traj_x = []
    traj_y = []

    def init():

        for point in points:
            point.set_offsets(np.empty((0, 2)))

        for line in lines:
            line.set_data([], [])

        trajectory.set_data([], [])
        current_artist.set_offsets(np.empty((0, 2)))

        title.set_text("Frame 0")

        return (
            points
            + lines
            + [trajectory, current_artist, title]
        )

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

        if not np.isnan(x[ID]) and not np.isnan(y[ID]):

            traj_x.append(x[ID])
            traj_y.append(y[ID])

            if len(traj_x) > TRAIL_LENGTH:
                traj_x.pop(0)
                traj_y.pop(0)

        trajectory.set_data(traj_x, traj_y)

        if len(traj_x) > 0:
            current_artist.set_offsets([[traj_x[-1], traj_y[-1]]])
        else:
            current_artist.set_offsets(np.empty((0, 2)))

        title.set_text(f"Frame {frame}")

        return (
            points
            + lines
            + [trajectory, current_artist, title]
        )

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