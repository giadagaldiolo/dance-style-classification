import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FILE = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
CAMERA = 0
OUTPUT_DIR = "outputs/trajectories"
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
    (5,6),
    (5,7), (7,9),
    (6,8), (8,10),
    (5,11), (6,12),
    (11,12),
    (11,13), (13,15),
    (12,14), (14,16)
]

def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    data = load_data(FILE)
    keypoints = data["keypoints2d"][CAMERA]  # (T, 17, 3)
    num_frames = len(keypoints)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_title("Hand trajectory + keypoints")


    wrist_id = 10
    traj_x, traj_y = [], []

    def update(frame):
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)

        kp = keypoints[frame]

        x = kp[:, 0]
        y = kp[:, 1]

        for i in range(17):
            ax.scatter(
                x[i],
                y[i],
                s=12, 
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

        traj_x.append(x[wrist_id])
        traj_y.append(y[wrist_id])

        ax.plot(traj_x, traj_y, linewidth=2, color="red")

        ax.set_title(f"Frame {frame}")

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000/60,
        BLIT=True
    )

    anim.save(
        OUTPUT_VIDEO,
        writer="ffmpeg",
        fps=60
    )



if __name__ == "__main__":
    main()