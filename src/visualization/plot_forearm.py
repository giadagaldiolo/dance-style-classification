import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYPOINT_DIR = "annotations/keypoints2d"
OUTPUT_DIR = "outputs/forearm_orientation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERA = 0
N_VIDEOS = 10

BALLET = "gJB"
WAACK = "gWA"

LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def forearm_orientation(keypoints, elbow_id, wrist_id):
    elbow = keypoints[:, elbow_id, :2]
    wrist = keypoints[:, wrist_id, :2]

    dx = wrist[:, 0] - elbow[:, 0]
    dy = wrist[:, 1] - elbow[:, 1]

    orientation = np.degrees(np.arctan2(dy, dx))

    invalid = np.isnan(dx) | np.isnan(dy)
    orientation[invalid] = np.nan

    return orientation


def angular_speed(orientation, fps):
    valid = ~np.isnan(orientation)
    orientation = orientation[valid]

    if len(orientation) < 2:
        return np.array([])

    diff = np.diff(orientation)

    # corregge salti tipo 179 -> -179
    diff = (diff + 180) % 360 - 180

    return np.abs(diff) * fps


def get_orientation_and_speed(filename):
    path = os.path.join(KEYPOINT_DIR, filename)
    data = load_data(path)

    keypoints = data["keypoints2d"][CAMERA]
    fps = data.get("fps", 60)

    left_orientation = forearm_orientation(keypoints, LEFT_ELBOW, LEFT_WRIST)
    right_orientation = forearm_orientation(keypoints, RIGHT_ELBOW, RIGHT_WRIST)

    left_speed = angular_speed(left_orientation, fps)
    right_speed = angular_speed(right_orientation, fps)

    return left_orientation, right_orientation, left_speed, right_speed, fps


def get_files(style):
    files = []

    for filename in sorted(os.listdir(KEYPOINT_DIR)):
        if not filename.endswith(".pkl"):
            continue

        if "_sMM_" in filename:
            continue

        if filename.startswith(style):
            files.append(filename)

    return files[:N_VIDEOS]


def plot_one_video(filename, style_name):
    left_orientation, right_orientation, left_speed, right_speed, fps = (
        get_orientation_and_speed(filename)
    )

    t_orientation = np.arange(len(left_orientation)) / fps
    t_left_speed = np.arange(len(left_speed)) / fps
    t_right_speed = np.arange(len(right_speed)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(t_orientation, left_orientation, label="Left forearm")
    axes[0].plot(t_orientation, right_orientation, label="Right forearm")
    axes[0].set_ylabel("Orientation (deg)")
    axes[0].set_title(f"{style_name} - Forearm orientation\n{filename}")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_left_speed, left_speed, label="Left forearm")
    axes[1].plot(t_right_speed, right_speed, label="Right forearm")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular speed (deg/s)")
    axes[1].set_title("Forearm orientation angular speed")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        filename.replace(".pkl", "_orientation_speed.png")
    )

    plt.savefig(output)
    plt.close()

    print("Saved:", output)


def compute_medians(files, style_name):
    rows = []

    for filename in files:
        _, _, left_speed, right_speed, fps = get_orientation_and_speed(filename)

        rows.append({
            "style": style_name,
            "sequence": filename.replace(".pkl", ""),
            "left_median": np.median(left_speed) if len(left_speed) > 0 else np.nan,
            "right_median": np.median(right_speed) if len(right_speed) > 0 else np.nan
        })

    return rows


def plot_boxplot(df):
    ballet_left = df[df["style"] == "Ballet Jazz"]["left_median"].dropna()
    ballet_right = df[df["style"] == "Ballet Jazz"]["right_median"].dropna()
    waack_left = df[df["style"] == "Waacking"]["left_median"].dropna()
    waack_right = df[df["style"] == "Waacking"]["right_median"].dropna()

    plt.figure(figsize=(9, 5))

    plt.boxplot(
        [ballet_left, ballet_right, waack_left, waack_right],
        labels=[
            "Ballet Jazz\nLeft",
            "Ballet Jazz\nRight",
            "Waacking\nLeft",
            "Waacking\nRight"
        ]
    )

    plt.ylabel("Median angular speed (deg/s)")
    plt.title("Median forearm orientation angular speed")
    plt.tight_layout()

    output = os.path.join(OUTPUT_DIR, "boxplot_median_orientation_speed.png")
    plt.savefig(output)
    plt.close()

    print("Saved:", output)


def main():
    ballet_files = get_files(BALLET)
    waack_files = get_files(WAACK)

    print("Ballet files:", ballet_files)
    print("Waack files:", waack_files)

    if len(ballet_files) > 0:
        plot_one_video(ballet_files[0], "Ballet Jazz")

    if len(waack_files) > 0:
        plot_one_video(waack_files[0], "Waacking")

    rows = []
    rows += compute_medians(ballet_files, "Ballet Jazz")
    rows += compute_medians(waack_files, "Waacking")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, "median_orientation_speed.csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    plot_boxplot(df)


if __name__ == "__main__":
    main()