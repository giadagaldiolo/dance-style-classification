import numpy as np


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

JOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, 
          LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, 
          LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

UPPER_JOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
LOWER_JOINTS = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]


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


def get_speed_accel(x_arr, y_arr, fps):
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    valid_idx = np.where(valid)[0]
    
    if len(valid_idx) < 3:
        return np.full(len(x_arr), np.nan), np.full(len(x_arr), np.nan)
    
    vx = np.full_like(x_arr, np.nan)
    vy = np.full_like(y_arr, np.nan)
    
    # Calcolo tempo reale trascorso tra frame validi
    dt = np.diff(valid_idx) / fps
    dt[dt == 0] = 1/fps # Sicurezza contro divisioni per 0
    
    vx[valid_idx[:-1]] = np.diff(x_arr[valid_idx]) / dt
    vy[valid_idx[:-1]] = np.diff(y_arr[valid_idx]) / dt
    speed = np.sqrt(vx**2 + vy**2)
    
    ax = np.full_like(x_arr, np.nan)
    ay = np.full_like(y_arr, np.nan)
    
    # Il gap temporale per l'accelerazione
    dt_accel = np.diff(valid_idx[:-1]) / fps
    dt_accel[dt_accel == 0] = 1/fps
    
    ax[valid_idx[:-2]] = np.diff(vx[valid_idx[:-1]]) / dt_accel
    ay[valid_idx[:-2]] = np.diff(vy[valid_idx[:-1]]) / dt_accel
    accel = np.sqrt(ax**2 + ay**2)
    
    return speed, accel


def joint_angle(keypoints, joint_a, vertex, joint_b, side):
    seg_a = keypoints[:, joint_a, :2] - keypoints[:, vertex, :2]
    seg_b = keypoints[:, joint_b, :2] - keypoints[:, vertex, :2]

    angle_a = np.degrees(np.arctan2(seg_a[:, 1], seg_a[:, 0]))
    angle_b = np.degrees(np.arctan2(seg_b[:, 1], seg_b[:, 0]))

    if side == "left":
        angle = (angle_b - angle_a + 360) % 360
    elif side == "right":
        angle = (angle_a - angle_b + 360) % 360

    invalid = (np.isnan(seg_a[:, 0]) | np.isnan(seg_a[:, 1]) | 
               np.isnan(seg_b[:, 0]) | np.isnan(seg_b[:, 1]))
    angle[invalid] = np.nan
    return angle

def angle_stats(angles, name, fps):
    valid = ~np.isnan(angles)
    valid_idx = np.where(valid)[0]
    valid_angles = angles[valid_idx]

    if len(valid_angles) < 2:
        return {f"{name}_angular_speed_median": 0.0 }

    diff = np.diff(valid_angles)
    diff = (diff + 180) % 360 - 180
    
    # Tempo reale trascorso tra gli angoli calcolati
    dt = np.diff(valid_idx) / fps
    dt[dt == 0] = 1/fps

    angular_speed = np.abs(diff) / dt
    return {f"{name}_angular_speed_median": np.median(angular_speed)}

def angle_histogram(angles, name):
    valid = ~np.isnan(angles)
    angles = angles[valid]

    if len(angles) == 0:
        hist = np.zeros(8)
    else:
        hist, _ = np.histogram(angles, bins=8, range=(0, 360))
        hist = hist / np.sum(hist)

    return {f"{name}_hist_{i}": hist[i] for i in range(8)}

def extract_features(keypoints, fps):
    kp = normalize_keypoints(keypoints)
    features = {}
    
    x = kp[:, :, 0]
    y = kp[:, :, 1]
    
    valid_frames = ~(np.all(np.isnan(x), axis=1) | np.all(np.isnan(y), axis=1))
    if np.sum(valid_frames) < 10: 
        return None

    hip_center_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
    hip_center_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2

    # CALCOLO ANGOLI
    left_elbow = joint_angle(kp, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, "left")
    right_elbow = joint_angle(kp, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, "right")
    left_knee = joint_angle(kp, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, "left")
    right_knee = joint_angle(kp, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, "right")


    # BODY
    for j in JOINTS:
        dist_to_hip = np.sqrt((x[:, j] - hip_center_x)**2 + (y[:, j] - hip_center_y)**2)
        features[f'body_dist_hip_mean_{j}'] = np.nanmean(dist_to_hip)
        features[f'body_dist_hip_std_{j}'] = np.nanstd(dist_to_hip)


    # SHAPE 
    with np.errstate(invalid='ignore'):
        min_x, max_x = np.nanmin(x[:, JOINTS], axis=1), np.nanmax(x[:, JOINTS], axis=1)
        min_y, max_y = np.nanmin(y[:, JOINTS], axis=1), np.nanmax(y[:, JOINTS], axis=1)
    
    body_area = (max_x - min_x) * (max_y - min_y)
    features['shape_body_area_mean'] = np.nanmean(body_area)
    features['shape_body_area_max'] = np.nanmax(body_area)

    w2w = np.sqrt((x[:, LEFT_WRIST] - x[:, RIGHT_WRIST])**2 + (y[:, LEFT_WRIST] - y[:, RIGHT_WRIST])**2)
    a2a = np.sqrt((x[:, LEFT_ANKLE] - x[:, RIGHT_ANKLE])**2 + (y[:, LEFT_ANKLE] - y[:, RIGHT_ANKLE])**2)
    features['shape_w2w_mean'] = np.nanmean(w2w)
    features['shape_a2a_mean'] = np.nanmean(a2a)

    cross1 = np.sqrt((x[:, LEFT_WRIST] - x[:, RIGHT_ANKLE])**2 + (y[:, LEFT_WRIST] - y[:, RIGHT_ANKLE])**2)
    cross2 = np.sqrt((x[:, RIGHT_WRIST] - x[:, LEFT_ANKLE])**2 + (y[:, RIGHT_WRIST] - y[:, LEFT_ANKLE])**2)
    cross_mean = (cross1 + cross2) / 2
    features['shape_cross_distance_mean'] = np.nanmean(cross_mean)
    features['shape_cross_distance_std'] = np.nanstd(cross_mean)

    uc_x = np.nanmean(x[:, UPPER_JOINTS], axis=1)
    uc_y = np.nanmean(y[:, UPPER_JOINTS], axis=1)
    upper_dispersion = np.nanmean([np.sqrt((x[:, j] - uc_x)**2 + (y[:, j] - uc_y)**2) for j in UPPER_JOINTS], axis=0)
    features['shape_upper_dispersion_mean'] = np.nanmean(upper_dispersion)

    lc_x = np.nanmean(x[:, LOWER_JOINTS], axis=1)
    lc_y = np.nanmean(y[:, LOWER_JOINTS], axis=1)
    lower_dispersion = np.nanmean([np.sqrt((x[:, j] - lc_x)**2 + (y[:, j] - lc_y)**2) for j in LOWER_JOINTS], axis=0)
    features['shape_lower_dispersion_mean'] = np.nanmean(lower_dispersion)

    features.update(angle_histogram(left_elbow, "left_forearm"))
    features.update(angle_histogram(right_elbow, "right_forearm"))
    features.update(angle_histogram(left_knee, "left_calf"))
    features.update(angle_histogram(right_knee, "right_calf"))

    # EFFORT
    lw_s, lw_a = get_speed_accel(x[:, LEFT_WRIST], y[:, LEFT_WRIST], fps)
    rw_s, rw_a = get_speed_accel(x[:, RIGHT_WRIST], y[:, RIGHT_WRIST], fps)
    la_s, la_a = get_speed_accel(x[:, LEFT_ANKLE], y[:, LEFT_ANKLE], fps)
    ra_s, ra_a = get_speed_accel(x[:, RIGHT_ANKLE], y[:, RIGHT_ANKLE], fps)

    features['effort_wrist_speed_median'] = np.nanmedian(np.concatenate([lw_s, rw_s]))
    features['effort_wrist_accel_median'] = np.nanmedian(np.concatenate([lw_a, rw_a]))
    features['effort_ankle_speed_median'] = np.nanmedian(np.concatenate([la_s, ra_s]))
    features['effort_ankle_accel_median'] = np.nanmedian(np.concatenate([la_a, ra_a]))

    features.update(angle_stats(left_elbow, "left_forearm", fps))
    features.update(angle_stats(right_elbow, "right_forearm", fps))
    features.update(angle_stats(left_knee, "left_calf", fps))
    features.update(angle_stats(right_knee, "right_calf", fps))

    # SPACE 
    valid_hips = ~(np.isnan(hip_center_x) | np.isnan(hip_center_y))
    hc_x_valid = hip_center_x[valid_hips]
    hc_y_valid = hip_center_y[valid_hips]

    if len(hc_x_valid) > 1:
        path_length = np.sum(np.sqrt(np.diff(hc_x_valid)**2 + np.diff(hc_y_valid)**2))
        displacement = np.sqrt((hc_x_valid[-1] - hc_x_valid[0])**2 + (hc_y_valid[-1] - hc_y_valid[0])**2)
        directness = displacement / path_length if path_length > 0 else 0.0
    else:
        path_length = 0.0
        displacement = 0.0
        directness = 0.0

    features['space_path_length'] = path_length
    features['space_displacement'] = displacement
    features['space_directness'] = directness

    return features