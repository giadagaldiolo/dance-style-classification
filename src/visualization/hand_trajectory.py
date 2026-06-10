import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# FILE = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
FILE = "outputs/keypoints/break_1.pkl"
OUTPUT_DIR = "outputs/trajectories"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_name = os.path.splitext(os.path.basename(FILE))[0]
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, base_name + ".mp4")

CAMERA = 0
ID = 10 # punto da tracciare
TRAIL_LENGTH = 50 

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

    W = data["width"]
    H = data["height"]
    fps = data["fps"]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title("Hand trajectory + keypoints")

    traj_x, traj_y = [], []

    def update(frame):
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(0, W + 1, 200))
        ax.set_yticks(np.arange(0, H + 1, 200))  

        kp = keypoints[frame]

        x = kp[:, 0]
        y = kp[:, 1]

        for i in range(17):
            if not np.isnan(x[i]) and not np.isnan(y[i]):
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
            
        if not np.isnan(x[ID]) and not np.isnan(y[ID]):
            traj_x.append(x[ID])
            traj_y.append(y[ID])

            if len(traj_x) > TRAIL_LENGTH:
                traj_x.pop(0)
                traj_y.pop(0)

        if len(traj_x) > 1:
            ax.plot(
                traj_x,
                traj_y,
                color="red",
                linewidth=2
            )

        ax.set_title(f"Frame {frame}")

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000/fps
    )

    anim.save(
        OUTPUT_VIDEO,
        writer="ffmpeg",
        fps=fps
    )



if __name__ == "__main__":
    main()