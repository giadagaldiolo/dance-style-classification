import numpy as np


# COCO 17-keypoint indices, as returned by MMPose (see extract_keypoints_mmpose.py).
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# The 12 joints actually used for feature computation 
JOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
          LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
          LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]


def normalize_keypoints(keypoints):
    """Normalizes a (T, 17, 3) keypoint sequence in two steps, so that
    features are comparable across sequences filmed under very different
    conditions (camera distance, framing, resolution):

    1. Translation: shifts all coordinates so that the hip center (the
       midpoint between the two hips) in the first valid frame sits at
       the origin.
    2. Scaling: divides all coordinates by the largest nose-to-ankle
       distance observed anywhere in the sequence, so distances end up
       expressed as a fraction of the dancer's own body size, not in
       absolute pixels.
    """
    kp = keypoints.copy()
    hips_center = (kp[:, LEFT_HIP, :2] + kp[:, RIGHT_HIP, :2]) / 2
    valid_center = ~(np.isnan(hips_center[:, 0]) | np.isnan(hips_center[:, 1]))

    if np.sum(valid_center) == 0:
        # No frame has a valid hip center, nothing to normalize against.
        return kp

    first_valid_idx = np.where(valid_center)[0][0]
    initial_center = hips_center[first_valid_idx]
    kp[:, :, :2] -= initial_center[None, None, :]

    nose = kp[:, NOSE, :2]

    l_ankle = kp[:, LEFT_ANKLE, :2]
    r_ankle = kp[:, RIGHT_ANKLE, :2]

    # Per-frame nose-to-ankle distance, for both ankles.
    dist_l = np.linalg.norm(nose - l_ankle, axis=1)
    dist_r = np.linalg.norm(nose - r_ankle, axis=1)

    valid_distances = np.concatenate([dist_l[~np.isnan(dist_l)], dist_r[~np.isnan(dist_r)]])

    if len(valid_distances) == 0:
        return kp

    # Scale by the single largest distance observed across the WHOLE
    # sequence, so the body's maximum extension is used as the reference "unit" of body size.
    max_dist = np.max(valid_distances)
    if max_dist > 0:
        kp[:, :, :2] /= max_dist

    return kp


def get_speed_accel(x_arr, y_arr, fps):
    """Computes per-frame speed and acceleration of a single 2D point
    over time, from arrays that may contain NaN (missing/undetected)
    values. Speed/acceleration are computed only between consecutive
    VALID frames (gaps caused by missing detections are skipped, not
    treated as zero motion), using the actual time elapsed between them.

    Returns two arrays the same length as the input, with NaN wherever
    speed/acceleration could not be computed.
    """
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 3:
        # Not enough valid points to compute even a single acceleration value.
        return np.full(len(x_arr), np.nan), np.full(len(x_arr), np.nan)

    vx = np.full_like(x_arr, np.nan)
    vy = np.full_like(y_arr, np.nan)

    # Time elapsed between consecutive VALID frames (not necessarily
    # adjacent frame indices, if some frames in between were invalid).
    # valid_idx holds strictly increasing unique integer indices, so
    # np.diff(valid_idx) is always >= 1 -- dt is therefore always > 0.
    dt = np.diff(valid_idx) / fps

    vx[valid_idx[:-1]] = np.diff(x_arr[valid_idx]) / dt
    vy[valid_idx[:-1]] = np.diff(y_arr[valid_idx]) / dt
    speed = np.sqrt(vx**2 + vy**2)

    ax = np.full_like(x_arr, np.nan)
    ay = np.full_like(y_arr, np.nan)

    # Acceleration: rate of change of velocity, using the same
    # valid-frame-aware approach as above (same reasoning: always > 0).
    dt_accel = np.diff(valid_idx[:-1]) / fps

    ax[valid_idx[:-2]] = np.diff(vx[valid_idx[:-1]]) / dt_accel
    ay[valid_idx[:-2]] = np.diff(vy[valid_idx[:-1]]) / dt_accel
    accel = np.sqrt(ax**2 + ay**2)

    return speed, accel


def joint_angle(keypoints, joint_a, vertex, joint_b, side):
    """Computes, for every frame, the angle (in degrees, 0-360) at
    `vertex` between the two segments vertex->joint_a and vertex->joint_b
    -- e.g. the elbow angle between the upper arm (shoulder-elbow) and
    forearm (elbow-wrist).

    `side` ("left" or "right") controls the direction in which the angle
    is measured, so that left/right limbs produce comparable, consistently
    oriented angle values instead of mirrored ones.

    Raises ValueError if `side` is anything other than "left" or "right".
    """
    seg_a = keypoints[:, joint_a, :2] - keypoints[:, vertex, :2]
    seg_b = keypoints[:, joint_b, :2] - keypoints[:, vertex, :2]

    angle_a = np.degrees(np.arctan2(seg_a[:, 1], seg_a[:, 0]))
    angle_b = np.degrees(np.arctan2(seg_b[:, 1], seg_b[:, 0]))

    if side == "left":
        angle = (angle_b - angle_a + 360) % 360
    elif side == "right":
        angle = (angle_a - angle_b + 360) % 360
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    invalid = (np.isnan(seg_a[:, 0]) | np.isnan(seg_a[:, 1]) |
               np.isnan(seg_b[:, 0]) | np.isnan(seg_b[:, 1]))
    angle[invalid] = np.nan
    return angle


def angle_stats(angles, name, fps):
    """Computes the MEDIAN ANGULAR SPEED (degrees/second) of a joint
    angle over time

    The (diff + 180) % 360 - 180 step correctly handles angle wraparound:
    without it, a change from e.g. 350 degrees to 10 degrees would be
    computed as a 340-degree jump, when the real angular change is only
    20 degrees (the short way around the circle).
    """
    valid = ~np.isnan(angles)
    valid_idx = np.where(valid)[0]
    valid_angles = angles[valid_idx]

    if len(valid_angles) < 2:
        return {f"{name}_angular_speed_median": 0.0}

    diff = np.diff(valid_angles)
    diff = (diff + 180) % 360 - 180

    dt = np.diff(valid_idx) / fps
    # Same reasoning as in get_speed_accel: valid_idx is strictly
    # increasing, so dt is always > 0, no zero-division guard needed.

    angular_speed = np.abs(diff) / dt
    return {f"{name}_angular_speed_median": np.median(angular_speed)}


def angle_histogram(angles, name):
    """Computes an 8-bin histogram of a joint angle over the whole
    sequence, normalized to proportions rather than raw counts, so the histogram
    is comparable across sequences of different lengths."""
    valid = ~np.isnan(angles)
    angles = angles[valid]

    if len(angles) == 0:
        hist = np.zeros(8)
    else:
        hist, _ = np.histogram(angles, bins=8, range=(0, 360))
        hist = hist / np.sum(hist)

    return {f"{name}_hist_{i}": hist[i] for i in range(8)}


def extract_features(keypoints, fps):
    """Computes the full set of 73 LMA-inspired features for one
    sequence (or one segment of a sequence), organized into the four
    LMA categories: Body (24 features), Shape (40), Effort (8), and
    Space (1). Returns None if the sequence has too few valid frames to
    produce reliable features.
    """
    kp = normalize_keypoints(keypoints)
    features = {}

    x = kp[:, :, 0]
    y = kp[:, :, 1]

    valid_frames = ~(np.all(np.isnan(x), axis=1) | np.all(np.isnan(y), axis=1))
    if np.sum(valid_frames) < 10:
        return None

    hip_center_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
    hip_center_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2

    # ANGLE COMPUTATION -- shared by both Shape (histograms) and Effort
    # (angular speed) below. Elbow angles are labelled "forearm" and knee
    # angles "calf", after the limb segment whose orientation they track.
    left_elbow = joint_angle(kp, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, "left")
    right_elbow = joint_angle(kp, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, "right")
    left_knee = joint_angle(kp, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, "left")
    right_knee = joint_angle(kp, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, "right")

    # --- BODY (24 features: 12 joints x mean + std) ---
    # Distance of each of the 12 joints from the hip center, aggregated
    # over the sequence: the mean describes how far limbs typically
    # extend from the body's center, the std how much that varies over time.
    for j in JOINTS:
        dist_to_hip = np.sqrt((x[:, j] - hip_center_x)**2 + (y[:, j] - hip_center_y)**2)
        features[f'body_dist_hip_mean_{j}'] = np.nanmean(dist_to_hip)
        features[f'body_dist_hip_std_{j}'] = np.nanstd(dist_to_hip)

    # --- SHAPE (40 features) ---
    # Bounding box area of the 12 joints, per frame. Aggregated with mean AND max 
    # to capture the maximum extension reached.
    with np.errstate(invalid='ignore'):
        min_x, max_x = np.nanmin(x[:, JOINTS], axis=1), np.nanmax(x[:, JOINTS], axis=1)
        min_y, max_y = np.nanmin(y[:, JOINTS], axis=1), np.nanmax(y[:, JOINTS], axis=1)

    body_area = (max_x - min_x) * (max_y - min_y)
    features['shape_body_area_mean'] = np.nanmean(body_area)
    features['shape_body_area_max'] = np.nanmax(body_area)

    # Wrist-to-wrist and ankle-to-ankle distances
    w2w = np.sqrt((x[:, LEFT_WRIST] - x[:, RIGHT_WRIST])**2 + (y[:, LEFT_WRIST] - y[:, RIGHT_WRIST])**2)
    a2a = np.sqrt((x[:, LEFT_ANKLE] - x[:, RIGHT_ANKLE])**2 + (y[:, LEFT_ANKLE] - y[:, RIGHT_ANKLE])**2)
    features['shape_w2w_mean'] = np.nanmean(w2w)
    features['shape_w2w_std'] = np.nanstd(w2w)

    features['shape_a2a_mean'] = np.nanmean(a2a)
    features['shape_a2a_std'] = np.nanstd(a2a)

    # Cross-body distance (left wrist to right ankle, and right wrist to
    # left ankle, averaged), intended to capture torso twists/rotations.
    cross1 = np.sqrt((x[:, LEFT_WRIST] - x[:, RIGHT_ANKLE])**2 + (y[:, LEFT_WRIST] - y[:, RIGHT_ANKLE])**2)
    cross2 = np.sqrt((x[:, RIGHT_WRIST] - x[:, LEFT_ANKLE])**2 + (y[:, RIGHT_WRIST] - y[:, LEFT_ANKLE])**2)
    cross_mean = (cross1 + cross2) / 2
    features['shape_cross_distance_mean'] = np.nanmean(cross_mean)
    features['shape_cross_distance_std'] = np.nanstd(cross_mean)

    # Angle histograms for elbows and knees (8 bins x 4 joints = 32
    # features): which angular configurations the dancer assumes most
    # often over the sequence -- already a frequency distribution, so no
    # further mean/std aggregation is needed.
    features.update(angle_histogram(left_elbow, "left_forearm"))
    features.update(angle_histogram(right_elbow, "right_forearm"))
    features.update(angle_histogram(left_knee, "left_calf"))
    features.update(angle_histogram(right_knee, "right_calf"))

    # --- EFFORT (8 features) ---
    # Speed and acceleration of wrists and ankles. Left and right sides
    # are POOLED TOGETHER (concatenated) before taking the median, rather
    # than averaged per frame first
    lw_s, lw_a = get_speed_accel(x[:, LEFT_WRIST], y[:, LEFT_WRIST], fps)
    rw_s, rw_a = get_speed_accel(x[:, RIGHT_WRIST], y[:, RIGHT_WRIST], fps)
    la_s, la_a = get_speed_accel(x[:, LEFT_ANKLE], y[:, LEFT_ANKLE], fps)
    ra_s, ra_a = get_speed_accel(x[:, RIGHT_ANKLE], y[:, RIGHT_ANKLE], fps)

    features['effort_wrist_speed_median'] = np.nanmedian(np.concatenate([lw_s, rw_s]))
    features['effort_wrist_accel_median'] = np.nanmedian(np.concatenate([lw_a, rw_a]))
    features['effort_ankle_speed_median'] = np.nanmedian(np.concatenate([la_s, ra_s]))
    features['effort_ankle_accel_median'] = np.nanmedian(np.concatenate([la_a, ra_a]))

    # Angular speed of elbows/knees -- median chosen over mean/std 
    # because it is less sensitive to outlier values, 
    # which are common in speed/acceleration-like measures.
    features.update(angle_stats(left_elbow, "left_forearm", fps))
    features.update(angle_stats(right_elbow, "right_forearm", fps))
    features.update(angle_stats(left_knee, "left_calf", fps))
    features.update(angle_stats(right_knee, "right_calf", fps))

    # --- SPACE (1 feature) ---
    # Median displacement of the hip center over 1-second-apart frame
    # pairs -- how much the dancer travels through space over time,
    # independent of the more local movements already captured by the
    # other three categories (a dancer can move a lot on the spot,
    # with very active limbs, while this feature stays low).
    k = int(fps)  # Number of frame equivalent to 1 second

    if len(hip_center_x) > k:
        dx = hip_center_x[k:] - hip_center_x[:-k]
        dy = hip_center_y[k:] - hip_center_y[:-k]

        dist_1sec = np.sqrt(dx**2 + dy**2)

        with np.errstate(invalid='ignore'):
            median_vel_1sec = np.nanmedian(dist_1sec)
            if np.isnan(median_vel_1sec):
                median_vel_1sec = 0.0
    else:
        # Sequence shorter than 1 second's worth of frames -- feature
        # cannot be computed, falls back to 0.
        median_vel_1sec = 0.0

    features['space_median_vel_1sec'] = median_vel_1sec

    return features