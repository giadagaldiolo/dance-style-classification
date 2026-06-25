import os
import pickle
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODEL_PATH = "outputs/classification/rf_model.pkl"
PKL_DIR = "outputs/keypoints"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

JOINTS = [
    5, 6,   # shoulders
    7, 8,   # elbows
    9, 10,  # wrists
    11, 12  # hips
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
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def load_model():
    return joblib.load(MODEL_PATH)


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


def joint_speed_features(keypoints, joint_id, name, fps):
    x = keypoints[:, joint_id, 0]
    y = keypoints[:, joint_id, 1]

    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return {f"{name}_speed_median": 0.0}

    vx = np.diff(x) * fps
    vy = np.diff(y) * fps
    speed = np.sqrt(vx**2 + vy**2)

    return {f"{name}_speed_median": np.median(speed)}


def forearm_angle(keypoints, shoulder_id, elbow_id, wrist_id):
    shoulder = keypoints[:, shoulder_id, :2]
    elbow = keypoints[:, elbow_id, :2]
    wrist = keypoints[:, wrist_id, :2]

    upper_arm = elbow - shoulder
    forearm = wrist - elbow

    angle_upper = np.degrees(
        np.arctan2(upper_arm[:, 1], upper_arm[:, 0])
    )

    angle_forearm = np.degrees(
        np.arctan2(forearm[:, 1], forearm[:, 0])
    )

    angle = (angle_forearm - angle_upper + 360) % 360

    return angle


def angle_stats(angles, name, fps):
    valid = ~np.isnan(angles)
    angles = angles[valid]

    if len(angles) < 2:
        return {f"{name}_angular_speed_median": 0.0}

    diff = np.diff(angles)

    # correzione salto 359 -> 1 gradi
    diff = (diff + 180) % 360 - 180

    angular_speed = np.abs(diff) * fps

    return {f"{name}_angular_speed_median": np.median(angular_speed)}


def angle_histogram(angles, name):
    valid = ~np.isnan(angles)
    angles = angles[valid]

    if len(angles) == 0:
        hist = np.zeros(8)
    else:
        hist, _ = np.histogram(
            angles,
            bins=8,
            range=(0, 360)
        )
        hist = hist / np.sum(hist)

    return {
        f"{name}_hist_{i}": hist[i]
        for i in range(8)
    }


def extract_features(keypoints, fps):
    keypoints = normalize_keypoints(keypoints)
    selected = keypoints[:, JOINTS, :2]

    x = selected[:, :, 0]
    y = selected[:, :, 1]

    valid = ~(np.isnan(x) | np.isnan(y))

    if np.sum(valid) == 0:
        return None

    features = {}

    for j in range(len(JOINTS)):
        xj = x[:, j][valid[:, j]]
        yj = y[:, j][valid[:, j]]

        if len(xj) == 0:
            continue

        features[f"mean_x_{j}"] = np.mean(xj)
        features[f"mean_y_{j}"] = np.mean(yj)

        features[f"std_x_{j}"] = np.std(xj)
        features[f"std_y_{j}"] = np.std(yj)

        features[f"min_x_{j}"] = np.min(xj)
        features[f"max_x_{j}"] = np.max(xj)

        features[f"range_x_{j}"] = np.max(xj) - np.min(xj)
        features[f"range_y_{j}"] = np.max(yj) - np.min(yj)

    features.update(joint_speed_features(keypoints, LEFT_WRIST, "left_hand", fps))
    features.update(joint_speed_features(keypoints, RIGHT_WRIST, "right_hand", fps))
    features.update(joint_speed_features(keypoints, LEFT_ANKLE, "left_foot", fps))
    features.update(joint_speed_features(keypoints, RIGHT_ANKLE, "right_foot", fps))

    left_angle = forearm_angle(
        keypoints,
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST
    )

    right_angle = forearm_angle(
        keypoints,
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST
    )

    features.update(angle_stats(left_angle, "left_forearm", fps))
    features.update(angle_stats(right_angle, "right_forearm", fps))

    features.update(angle_histogram(left_angle, "left_forearm"))
    features.update(angle_histogram(right_angle, "right_forearm"))

    return features


def get_label(filename):
    if filename.startswith("gBR"):
        return 0
    if filename.startswith("gHO"):
        return 1
    if filename.startswith("gJB"):
        return 2
    if filename.startswith("gJS"):
        return 3
    if filename.startswith("gKR"):
        return 4
    if filename.startswith("gLH"):
        return 5
    if filename.startswith("gLO"):
        return 6
    if filename.startswith("gMH"):
        return 7
    if filename.startswith("gPO"):
        return 8
    if filename.startswith("gWA"):
        return 9
    return None


def main():
    clf = load_model()

    X = []
    y = []
    names = []

    for f in os.listdir(PKL_DIR):
        if not f.endswith(".pkl"):
            continue

        label = get_label(f)
        if label is None:
            continue

        with open(os.path.join(PKL_DIR, f), "rb") as file:
            data = pickle.load(file)

        kp = data["keypoints2d"][0]
        fps = data.get("fps", 60)

        feat = extract_features(kp, fps)

        if feat is None:
            print("Skipped:", f)
            continue

        X.append(feat)
        y.append(label)
        names.append(f)

    df = pd.DataFrame(X)
    y = np.array(y)

    # forza stesso ordine colonne del modello sklearn
    df = df.reindex(columns=clf.feature_names_in_, fill_value=0.0)

    print("\nYOUTUBE TEST RESULTS")
    pred = clf.predict(df)

    for name, true, p in zip(names, y, pred):
        print(f"{name}: true={CLASSES[true]}, pred={CLASSES[p]}")

    print()
    print(classification_report(y, pred, target_names=CLASSES))

    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("YouTube Test — Confusion Matrix")
    plt.tight_layout()
    plt.show()

    proba = clf.predict_proba(df)
    top3 = top_k_accuracy_score(
        y,
        proba,
        k=3
    )

    print("Top-3 Accuracy:", top3)
    print("Macro F1:", f1_score(y, pred, average="macro"))


if __name__ == "__main__":
    main()