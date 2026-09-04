# Dance Style Classification

A system for classifying dance styles from video, developed as a Bachelor's thesis project in Computer  Engineering (SUPSI DTI). The system extracts 2D body keypoints via pose estimation, computes a set of interpretable features inspired by Laban Movement Analysis (LMA), and classifies the style with a Random Forest model. It also includes a real-time demonstration application and an analysis of the pipeline's computational and environmental cost.

## Pipeline overview

```
Video → Keypoint extraction (MMPose, 17 COCO points) → Normalization
      → LMA features (Body, Effort, Shape, Space) → Random Forest → Recognized style
```

The model is trained and evaluated on the [AIST++](https://google.github.io/aistplusplus_dataset/) dataset (10 dance styles), and also tested on a set of YouTube videos to assess its generalization to uncontrolled recording conditions.

## Repository structure

```
src/
├── extract_keypoints/       # 2D keypoint extraction from videos (MMPose)
├── classification/          # LMA feature extraction, model training and evaluation
├── realtime/                # Real-time demo application (webcam)
├── sustainability/          # Energy cost measurement (CodeCarbon)
├── visualization/           # Feature plots and animations, used in the thesis
└── utils/                   # Dataset exploration

music_115bpm/                # Music tracks associated with each style (real-time system)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is required for keypoint extraction and for the real-time system (MMPose).

## Data

The AIST++ dataset (keypoints, videos, annotations) **is not included** in this repository due to its size — it must be downloaded separately from the [official website](https://google.github.io/aistplusplus_dataset/) and placed in `annotations/keypoints2d/`.

The YouTube videos used for the generalization test are not redistributed in this repository (copyright).

## Usage

**Extracting keypoints from a video:**
```bash
python src/extract_keypoints/extract_keypoints_mmpose.py
```

**Training the model (on the full AIST++ dataset):**
```bash
python src/classification/train_whole_video.py
```

**Evaluating on a dataset (e.g. the OOD YouTube videos):**
```bash
python src/classification/test_whole_video.py
```

**Real-time recognition system (requires a webcam):**
```bash
python src/realtime/realtime_registration.py
```

**Computational sustainability analysis:**
```bash
python src/sustainability/generate_report.py
python src/sustainability/plot_two_pipelines.py
```

## Main results

| Test set | Accuracy | Top-3 accuracy |
|---|---|---|
| AIST++ (internal test set) | 89.72% | 100.00% |
| YouTube (out-of-domain) | 42.50% | 75.00% |

Full details, methodology, and discussion of the results are available in the thesis report.

## Thesis report

The full report, including methodology, results, and discussion, is available at [`docs/DOC_GALDIOLO.pdf`](docs/DOC_GALDIOLO.pdf).

## Author

- Giada Galdiolo 
- Supervisor: Alessandro Giusti 
