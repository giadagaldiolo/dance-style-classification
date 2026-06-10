import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ffmpeg


VIDEO_FILE = "videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
PKL_FILE = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"

OUTPUT_DIR = "outputs/overlays"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_name = os.path.splitext(os.path.basename(VIDEO_FILE))[0]
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, base_name + ".mp4")

CACHE_VIDEO = VIDEO_FILE.replace(".mp4", ".npy")

FPS = 60
CAMERA = 0
ID = 10
TRAIL_LENGTH = 50

VIDEO_START_TIME = 15.3  # shift temporale

_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85]
]

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def ffmpeg_video_read(video_path, fps=60):
    probe = ffmpeg.probe(video_path)
    info = next(s for s in probe["streams"] if s["codec_type"] == "video")

    w = int(info["width"])
    h = int(info["height"])

    stream = ffmpeg.input(video_path)
    stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")

    out, _ = ffmpeg.run(stream, capture_stdout=True)

    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    return video, w, h

def load_video():
    if os.path.exists(CACHE_VIDEO):
        print("Loading cached video...")
        video = np.load(CACHE_VIDEO)
        return video, video.shape[2], video.shape[1]

    print("Reading video with ffmpeg (first time only)...")
    video, w, h = ffmpeg_video_read(VIDEO_FILE, fps=FPS)

    np.save(CACHE_VIDEO, video)

    return video, w, h

def main():

    data = load_data(PKL_FILE)

    keypoints = data["keypoints2d"][CAMERA]
    timestamps = data["timestamps"]

    video, W, H = load_video()

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")

    traj_x = []
    traj_y = []

    def update(i):
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        t = (timestamps[i] / 1_000_000) - (timestamps[0] / 1_000_000)
        vf = int((t + VIDEO_START_TIME) * FPS)

        if vf < 0 or vf >= len(video):
            return []

        frame = video[vf]

        ax.imshow(frame) # mostra il video come background
        

        kp = keypoints[i]
        x = kp[:, 0]
        y = kp[:, 1]

        for j in range(17):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                ax.scatter(
                    x[j], 
                    y[j],
                    s=12,
                    color=np.array(_COLORS[j]) / 255.0
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

        ax.set_title(f"Frame {i}")
        ax.axis("off")

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(keypoints),
        interval=1000/60
    )

    anim.save(
        OUTPUT_VIDEO,
        writer="ffmpeg",
        fps=FPS
    )


if __name__ == "__main__":
    main()