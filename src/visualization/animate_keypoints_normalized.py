import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#FILE = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch10.pkl"
FILE = "outputs/keypoints/gMH_yt_03.pkl"

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
LEFT_ANKLE = 15


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_keypoints(keypoints):
    kp = keypoints.copy()
    hips_center = (kp[:, LEFT_HIP, :2] + kp[:, RIGHT_HIP, :2]) / 2
    valid_center = ~(np.isnan(hips_center[:, 0]) | np.isnan(hips_center[:, 1]))

    if np.sum(valid_center) == 0:
        return kp

    first_valid_idx = np.where(valid_center)[0][0]
    initial_center = hips_center[first_valid_idx]
    kp[:, :, :2] -= initial_center[None, None, :]

    nose = kp[:, NOSE, :2]

    l_ankle = kp[:, LEFT_ANKLE, :2]
    r_ankle = kp[:, RIGHT_ANKLE, :2]
    
    dist_l = np.linalg.norm(nose - l_ankle, axis=1)
    dist_r = np.linalg.norm(nose - r_ankle, axis=1)
    
    valid_distances = np.concatenate([dist_l[~np.isnan(dist_l)], dist_r[~np.isnan(dist_r)]])

    if len(valid_distances) == 0:
        return kp

    max_dist = np.max(valid_distances)
    if max_dist > 0:
        kp[:, :, :2] /= max_dist

    return kp

def main():
    data = load_data(FILE)

    keypoints = data["keypoints2d"][CAMERA]
    keypoints = normalize_keypoints(keypoints)

    num_frames = len(keypoints)
    fps = data.get("fps", 60)

    x_all = keypoints[:, :, 0]
    y_all = keypoints[:, :, 1]

    xmin = np.nanmin(x_all) - 0.2
    xmax = np.nanmax(x_all) + 0.2
    ymin = np.nanmin(y_all) - 0.2
    ymax = np.nanmax(y_all) + 0.2

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 0.5, 0.5))
    ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 0.5, 0.5))

    points = []
    for i in range(17):
        point = ax.scatter(
            [],
            [],
            s=18,
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

    title = ax.set_title("Normalized Keypoints - Frame 0")

    def init():
        for point in points:
            point.set_offsets(np.empty((0, 2)))

        for line in lines:
            line.set_data([], [])

        title.set_text("Normalized Keypoints - Frame 0")

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

        title.set_text(f"Normalized Keypoints - Frame {frame}")

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