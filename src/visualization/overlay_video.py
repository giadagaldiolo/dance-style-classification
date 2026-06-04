import os
import pickle
import numpy as np
import ffmpeg


VIDEO_FILE = "videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
PKL_FILE = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"

OUTPUT_DIR = "outputs/overlays"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_name = os.path.splitext(os.path.basename(VIDEO_FILE))[0]
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, base_name + ".mp4")

CACHE_VIDEO = VIDEO_FILE.replace(".mp4", "_60fps.npy")

FPS = 60
CAMERA = 0
ID = 10 # punto da tracciare (0-16, vedi keypoints2d)
TRAIL_LENGTH = 50
VIDEO_START_TIME = 15.3  # tuning

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

def ffmpeg_video_read(video_path, fps=None):
    """Video reader based on FFMPEG.

    This function supports setting fps for video reading. It is critical
    as AIST++ Dataset are constructed under exact 60 fps, while some of
    the AIST dance videos are not percisely 60 fps.

    Args:
        video_path: A video file.
        fps: Use specific fps for video reading. (optional)
    Returns:
        A `np.array` with the shape of [seq_len, height, width, 3]
    """
    assert os.path.exists(video_path), f'{video_path} does not exist!'
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    video_info = next(stream for stream in probe['streams']
                        if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    stream = ffmpeg.input(video_path)
    if fps:
        stream = ffmpeg.filter(stream, 'fps', fps=fps, round='down')
    stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
    out, _ = ffmpeg.run(stream, capture_stdout=True)
    out = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return out.copy()


def ffmpeg_video_write(data, video_path, fps=25):
    """Video writer based on FFMPEG.

    Args:
        data: A `np.array` with the shape of [seq_len, height, width, 3]
        video_path: A video file.
        fps: Use specific fps for video writing. (optional)
    """
    assert len(data.shape) == 4, f'input shape is not valid! Got {data.shape}!'
    _, height, width, _ = data.shape
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = (
        ffmpeg
        .input('pipe:', framerate=fps, format='rawvideo',
                pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(video_path, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in data:
        writer.stdin.write(frame.astype(np.uint8).tobytes())
    writer.stdin.close()


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_video():
    if os.path.exists(CACHE_VIDEO):
        print("Loading cached video...")
        return np.load(CACHE_VIDEO)

    print("Reading video with ffmpeg (first time only)...")
    video = ffmpeg_video_read(VIDEO_FILE, fps=FPS)

    np.save(CACHE_VIDEO, video)

    return video


def draw_line(img, x1, y1, x2, y2, color, thickness=2):
    import cv2
    cv2.line(img,
             (int(x1), int(y1)),
             (int(x2), int(y2)),
             color, thickness)
    return img


def draw_circle(img, x, y, color, r=3):
    import cv2
    cv2.circle(img,
               (int(x), int(y)),
               r,
               color,
               -1)
    return img


def draw_skeleton(frame, kp):
    x = kp[:, 0]
    y = kp[:, 1]

    for a, b in SKELETON:
        if np.isnan(x[a]) or np.isnan(x[b]):
            continue
        frame = draw_line(frame, x[a], y[a], x[b], y[b], (0, 0, 0), 2)

    for i in range(17):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        frame = draw_circle(frame, x[i], y[i], _COLORS[i], 12)

    return frame


def main():
    data = load_data(PKL_FILE)

    keypoints = data["keypoints2d"][CAMERA]
    timestamps = data["timestamps"]

    video = load_video()

    print("video frames:", len(video))
    print("keypoints frames:", len(keypoints))

    trajectory = []
    out_frames = []

    t0 = timestamps[0] / 1_000_000

    for i in range(len(keypoints)):

        t = (timestamps[i] / 1_000_000) - t0
        vf = int((t + VIDEO_START_TIME) * FPS)

        if vf < 0 or vf >= len(video):
            continue

        frame = video[vf].copy()

        kp = keypoints[i]

        frame = draw_skeleton(frame, kp)

        x, y = kp[ID][:2]

        if not np.isnan(x) and not np.isnan(y):
            trajectory.append((int(x), int(y)))

        if len(trajectory) > TRAIL_LENGTH:
            trajectory.pop(0)

        for j in range(1, len(trajectory)):
            frame = draw_line(
                frame,
                trajectory[j-1][0], trajectory[j-1][1],
                trajectory[j][0], trajectory[j][1],
                (255, 0, 0), # red
                2
            )

        out_frames.append(frame)


    ffmpeg_video_write(
        np.stack(out_frames),
        OUTPUT_VIDEO,
        fps=FPS
    )


if __name__ == "__main__":
    main()