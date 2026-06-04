import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

FILE = "annotations/keypoints2d/gJB_sBM_cAll_d07_mJB0_ch01.pkl"
CAMERA = 0
ID = 10  # mano

OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_name = os.path.splitext(os.path.basename(FILE))[0]
OUTPUT_PATH_X = os.path.join(OUTPUT_DIR, base_name + "_hand_x.png")
OUTPUT_PATH_Y = os.path.join(OUTPUT_DIR, base_name + "_hand_y.png")

def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    data = load_data(FILE)
    keypoints = data["keypoints2d"][CAMERA]  

    x_vals = []
    y_vals = []

    for frame in keypoints:
        kp = frame[ID]
        x, y = kp[0], kp[1]

        x_vals.append(x)
        y_vals.append(y)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    t = np.arange(len(x_vals))

    plt.figure()
    plt.plot(t, x_vals)
    plt.title("Hand X position over time")
    plt.xlabel("Frame")
    plt.ylabel("X position")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH_X)
    plt.close()

    plt.figure()
    plt.plot(t, y_vals)
    plt.title("Hand Y position over time")
    plt.xlabel("Frame")
    plt.ylabel("Y position")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH_Y)
    plt.close()

if __name__ == "__main__":
    main()