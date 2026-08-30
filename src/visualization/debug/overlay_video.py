import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ffmpeg


VIDEO_FILE = "downloaded_videos/gJB_yt_02.mp4"
PKL_FILE = "outputs/keypoints/gJB_yt_02.pkl"

OUTPUT_DIR = "outputs/overlays"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_name = os.path.splitext(os.path.basename(VIDEO_FILE))[0]
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, base_name + ".mp4")

CACHE_VIDEO = VIDEO_FILE.replace(".mp4", ".npy")

CAMERA = 0
ID = 10

_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85]
]

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ffmpeg_video_read(video_path, fps, w, h):
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")

    out, _ = ffmpeg.run(stream, capture_stdout=True)

    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    return video


def load_video(fps, w, h):
    if os.path.exists(CACHE_VIDEO):
        print("Loading cached video")
        return np.load(CACHE_VIDEO)

    print("Reading video with ffmpeg")
    video = ffmpeg_video_read(VIDEO_FILE, fps=fps, w=w, h=h)

    np.save(CACHE_VIDEO, video)

    return video


def main():
    data = load_data(PKL_FILE)

    keypoints = data["keypoints2d"][CAMERA]
    timestamps = data["timestamps"]

    W = data["width"]
    H = data["height"]
    FPS = data["fps"]

    video = load_video(FPS, W, H)

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    image_artist = ax.imshow(
        np.zeros((H, W, 3), dtype=np.uint8)
    )

    points = []

    for j in range(17):
        point = ax.scatter(
            [],
            [],
            s=100,
            color=np.array(_COLORS[j]) / 255.0
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

    current_artist = ax.scatter(
        [],
        [],
        s=180,
        color="red"
    )

    title = ax.set_title("Frame 0")

    def init():
        image_artist.set_data(np.zeros((H, W, 3), dtype=np.uint8))

        for point in points:
            point.set_offsets(np.empty((0, 2)))

        for line in lines:
            line.set_data([], [])

        current_artist.set_offsets(np.empty((0, 2)))
        title.set_text("Frame 0")

        return (
            [image_artist]
            + points
            + lines
        )

    def update(i):
        t = timestamps[i] / 1_000_000
        vf = int(t * FPS)

        if vf < 0 or vf >= len(video):
            return init()

        frame = video[vf]
        image_artist.set_data(frame)

        kp = keypoints[i]

        x = kp[:, 0]
        y = kp[:, 1]

        for j, point in enumerate(points):
            if np.isnan(x[j]) or np.isnan(y[j]):
                point.set_offsets(np.empty((0, 2)))
            else:
                point.set_offsets([[x[j], y[j]]])

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

       

        title.set_text(f"Frame {i}")

        return (
            [image_artist]
            + points
            + lines
        )

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(keypoints),
        interval=1000 / FPS,
        blit=True
    )

    anim.save(
        OUTPUT_VIDEO,
        writer="ffmpeg",
        fps=FPS
    )


if __name__ == "__main__":
    main()