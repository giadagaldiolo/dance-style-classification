import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# FILE = "annotations/keypoints2d/gWA_sBM_cAll_d25_mWA1_ch01.pkl"
FILE = "outputs/keypoints/gJB_yt_02.pkl"
CAMERA = 0
OUTPUT_DIR = "outputs/animations/angles"
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
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
RIGHT_ANKLE = 16


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_keypoints(keypoints):
    kp = keypoints.copy()

    hips_center = (
        kp[:, LEFT_HIP, :2] +
        kp[:, RIGHT_HIP, :2]
    ) / 2

    valid_center = ~(
        np.isnan(hips_center[:, 0]) |
        np.isnan(hips_center[:, 1])
    )

    if np.sum(valid_center) == 0:
        return kp

    first_valid_idx = np.where(valid_center)[0][0]
    initial_center = hips_center[first_valid_idx]

    kp[:, :, :2] -= initial_center[None, None, :]

    nose = kp[:, NOSE, :2]
    ankle = kp[:, RIGHT_ANKLE, :2]

    distances = np.linalg.norm(nose - ankle, axis=1)
    valid_distances = distances[~np.isnan(distances)]

    if len(valid_distances) == 0:
        return kp

    max_dist = np.max(valid_distances)

    if max_dist > 0:
        kp[:, :, :2] /= max_dist

    return kp


def forearm_angle(keypoints, shoulder_id, elbow_id, wrist_id, side):
    shoulder = keypoints[:, shoulder_id, :2]
    elbow = keypoints[:, elbow_id, :2]
    wrist = keypoints[:, wrist_id, :2]

    upper_arm = shoulder - elbow
    forearm = wrist - elbow

    upper_angle = np.degrees(
        np.arctan2(upper_arm[:, 1], upper_arm[:, 0])
    )

    forearm_angle = np.degrees(
        np.arctan2(forearm[:, 1], forearm[:, 0])
    )

    if side == "left":
        # braccio sinistro: senso orario
        angle = (forearm_angle - upper_angle + 360) % 360

    elif side == "right":
        # braccio destro: senso antiorario
        angle = (upper_angle - forearm_angle + 360) % 360

   
    invalid = (
        np.isnan(upper_arm[:, 0]) |
        np.isnan(upper_arm[:, 1]) |
        np.isnan(forearm[:, 0]) |
        np.isnan(forearm[:, 1])
    )

    angle[invalid] = np.nan

    return angle


def draw_angle_arc(ax, elbow, shoulder, angle_value, side, radius):
    if np.isnan(angle_value):
        return

    upper_arm = shoulder - elbow

    upper_angle = np.arctan2(
        upper_arm[1],
        upper_arm[0]
    )

    if side == "right":
        # destro: senso antiorario
        theta = upper_angle - np.linspace(
            0,
            np.deg2rad(angle_value),
            50
        )

    elif side == "left":
        # sinistro: senso orario
        theta = upper_angle + np.linspace(
            0,
            np.deg2rad(angle_value),
            50
        )

    arc_x = elbow[0] + radius * np.cos(theta)
    arc_y = elbow[1] + radius * np.sin(theta)

    ax.plot(arc_x, arc_y, linewidth=2)

def main():
    data = load_data(FILE)

    keypoints = data["keypoints2d"][CAMERA]
    keypoints = normalize_keypoints(keypoints)

    left_angle = forearm_angle(
        keypoints,
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST,
        side="left"
    )

    right_angle = forearm_angle(
        keypoints,
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST,
        side="right"
    )

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
                    s=20,
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

        if not np.isnan(kp[[LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], :2]).any():
            draw_angle_arc(
                ax,
                kp[LEFT_ELBOW, :2],
                kp[LEFT_SHOULDER, :2],
                left_angle[frame],
                "left",
                0.15
            )

        # evidenzia braccio destro
        if not np.isnan(kp[[RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], :2]).any():

            draw_angle_arc(
                ax,
                kp[RIGHT_ELBOW, :2],
                kp[RIGHT_SHOULDER, :2],
                right_angle[frame],
                "right",
                0.20
            )

        ax.set_title(
            f"Frame {frame} | "
            f"L={left_angle[frame]:.1f}° | "
            f"R={right_angle[frame]:.1f}°"
        )

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

    print("Saved:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()