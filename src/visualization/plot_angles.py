import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYPOINT_DIR = "annotations/keypoints2d"
OUTPUT_DIR = "outputs/angle_speed_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERA = 0
N_VIDEOS = 10

STYLE_A = "gJB"  # Ballet Jazz
STYLE_B = "gWA"  # Waacking

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

    hips_center = (kp[:, LEFT_HIP, :2] + kp[:, RIGHT_HIP, :2]) / 2

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

    upper_angle = np.degrees(np.arctan2(upper_arm[:, 1], upper_arm[:, 0]))
    forearm_angle = np.degrees(np.arctan2(forearm[:, 1], forearm[:, 0]))

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


def angular_speed(angles, fps):
    valid = ~np.isnan(angles)
    angles = angles[valid]

    if len(angles) < 2:
        return np.array([])

    diff = np.diff(angles)
    diff = (diff + 180) % 360 - 180

    return np.abs(diff) * fps


def get_angle_speeds_from_file(path):
    data = load_data(path)

    keypoints = data["keypoints2d"][CAMERA]
    fps = data.get("fps", 60)

    keypoints = normalize_keypoints(keypoints)

    left_angle = forearm_angle(
        keypoints,
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST,
        "left"
    )

    right_angle = forearm_angle(
        keypoints,
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST,
        "right"
    )

    left_speed = angular_speed(left_angle, fps)
    right_speed = angular_speed(right_angle, fps)

    return left_speed, right_speed, fps


def get_files_for_style(style, n):
    files = []

    for filename in sorted(os.listdir(KEYPOINT_DIR)):
        if not filename.endswith(".pkl"):
            continue

        if "_sMM_" in filename:
            continue

        if filename.startswith(style):
            files.append(filename)

    return files[:n]


def plot_speed_over_time(filename, style_name):
    path = os.path.join(KEYPOINT_DIR, filename)

    left_speed, right_speed, fps = get_angle_speeds_from_file(path)

    t_left = np.arange(len(left_speed)) / fps
    t_right = np.arange(len(right_speed)) / fps

    plt.figure(figsize=(10, 5))
    plt.plot(t_left, left_speed, label="Left forearm")
    plt.plot(t_right, right_speed, label="Right forearm")

    plt.xlabel("Time (s)")
    plt.ylabel("Angular speed (deg/s)")
    plt.title(f"Angular speed over time - {style_name}\n{filename}")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(
        OUTPUT_DIR,
        filename.replace(".pkl", "_angular_speed_time.png")
    )

    plt.savefig(out)
    plt.close()

    print("Saved:", out)


def compute_median_speeds(files, style_label):
    rows = []

    for filename in files:
        path = os.path.join(KEYPOINT_DIR, filename)

        left_speed, right_speed, fps = get_angle_speeds_from_file(path)

        if len(left_speed) == 0 and len(right_speed) == 0:
            continue

        rows.append({
            "style": style_label,
            "sequence": filename.replace(".pkl", ""),
            "left_forearm_angular_speed_median": (
                np.median(left_speed) if len(left_speed) > 0 else np.nan
            ),
            "right_forearm_angular_speed_median": (
                np.median(right_speed) if len(right_speed) > 0 else np.nan
            )
        })

    return rows


def plot_boxplot(df):
    bj_left = df[df["style"] == "Ballet Jazz"]["left_forearm_angular_speed_median"].dropna()
    bj_right = df[df["style"] == "Ballet Jazz"]["right_forearm_angular_speed_median"].dropna()

    wa_left = df[df["style"] == "Waacking"]["left_forearm_angular_speed_median"].dropna()
    wa_right = df[df["style"] == "Waacking"]["right_forearm_angular_speed_median"].dropna()

    plt.figure(figsize=(9, 5))

    plt.boxplot(
        [bj_left, bj_right, wa_left, wa_right],
        labels=[
            "Ballet Jazz\nLeft",
            "Ballet Jazz\nRight",
            "Waacking\nLeft",
            "Waacking\nRight"
        ]
    )

    plt.ylabel("Median angular speed (deg/s)")
    plt.title("Median forearm angular speed by style and arm")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "boxplot_left_right_median_angular_speed.png")
    plt.savefig(out)
    plt.close()

    print("Saved:", out)


def main():
    bj_files = get_files_for_style(STYLE_A, N_VIDEOS)
    wa_files = get_files_for_style(STYLE_B, N_VIDEOS)

    print("Ballet Jazz files:", len(bj_files))
    print("Waacking files:", len(wa_files))

    if len(bj_files) > 0:
        plot_speed_over_time(bj_files[1], "Ballet Jazz")

    if len(wa_files) > 0:
        plot_speed_over_time(wa_files[1], "Waacking")

    rows = []
    rows += compute_median_speeds(bj_files, "Ballet Jazz")
    rows += compute_median_speeds(wa_files, "Waacking")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, "left_right_angular_speed_summary.csv")
    df.to_csv(csv_path, index=False)

    print("Saved:", csv_path)
    print(df)

    plot_boxplot(df)


if __name__ == "__main__":
    main()