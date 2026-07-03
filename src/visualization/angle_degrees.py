import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FILE = "annotations/keypoints2d/gWA_sBM_cAll_d25_mWA0_ch02.pkl"
# FILE = "outputs/keypoints/gBR_yt_02.pkl"

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
        angle = (forearm_angle - upper_angle + 360) % 360

    elif side == "right":
        angle = (upper_angle - forearm_angle + 360) % 360

   
    invalid = (
        np.isnan(upper_arm[:, 0]) |
        np.isnan(upper_arm[:, 1]) |
        np.isnan(forearm[:, 0]) |
        np.isnan(forearm[:, 1])
    )

    angle[invalid] = np.nan

    return angle


def compute_angle_arc(elbow, shoulder, angle_value, side, radius):
    if np.isnan(angle_value):
        return [], []

    upper_arm = shoulder - elbow

    if np.isnan(upper_arm[0]) or np.isnan(upper_arm[1]):
        return [], []

    upper_angle = np.arctan2(
        upper_arm[1],
        upper_arm[0]
    )

    if side == "right":
        theta = upper_angle - np.linspace(
            0,
            np.deg2rad(angle_value),
            50
        )
    elif side == "left":
        theta = upper_angle + np.linspace(
            0,
            np.deg2rad(angle_value),
            50
        )
    else:
        raise ValueError("side must be 'left' or 'right'")

    arc_x = elbow[0] + radius * np.cos(theta)
    arc_y = elbow[1] + radius * np.sin(theta)

    return arc_x, arc_y


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
            s=20,
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

    left_arc, = ax.plot(
        [],
        [],
        linewidth=2
    )

    right_arc, = ax.plot(
        [],
        [],
        linewidth=2,
    )

    title = ax.set_title("Frame 0")

    def init():
        for point in points:
            point.set_offsets(np.empty((0, 2)))

        for line in lines:
            line.set_data([], [])

        left_arc.set_data([], [])
        right_arc.set_data([], [])
        title.set_text("Frame 0")

        return (
            points
            + lines
            + [
                left_arc,
                right_arc,
                title
            ]
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

        if not np.isnan(kp[[LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], :2]).any():
            lx, ly = compute_angle_arc(
                elbow=kp[LEFT_ELBOW, :2],
                shoulder=kp[LEFT_SHOULDER, :2],
                angle_value=left_angle[frame],
                side="left",
                radius=0.12
            )
            left_arc.set_data(lx, ly)
        else:
            left_arc.set_data([], [])

        if not np.isnan(kp[[RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], :2]).any():
            rx, ry = compute_angle_arc(
                elbow=kp[RIGHT_ELBOW, :2],
                shoulder=kp[RIGHT_SHOULDER, :2],
                angle_value=right_angle[frame],
                side="right",
                radius=0.12
            )
            right_arc.set_data(rx, ry)
        else:
            right_arc.set_data([], [])

        title.set_text(
            f"Frame {frame} | "
            f"L={left_angle[frame]:.1f}° | "
            f"R={right_angle[frame]:.1f}°"
        )

        return (
            points
            + lines
            + [
                left_arc,
                right_arc,
                title
            ]
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