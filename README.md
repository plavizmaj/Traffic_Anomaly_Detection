# Traffic Anomaly Detection (TAD)

This project detects abnormal events (such as traffic accidents and other irregular road behaviour) in traffic surveillance footage that has been stored as sequences of `.jpg` frames. Each video folder is converted into a sequence of 512-dimensional ResNet18 embeddings, split into overlapping 16-frame windows, and classified as normal or abnormal by an LSTM-based binary classifier. The goal is to remove the need for a human operator to watch hours of camera footage: the model produces an anomaly probability per time window, so only the flagged segments need to be reviewed. It is intended for computer vision and machine learning engineers, students, and traffic monitoring teams who want a compact, reproducible baseline for video-level anomaly classification in PyTorch.

## Features

- Automatic dataset download from Kaggle via `kagglehub` (`nikanvasei/traffic-anomaly-dataset-tad`), with a skip check if raw data already exists.
- Exploratory data analysis: file extension counts and discovery of video folders by walking the raw data tree.
- Label inference from folder names using `BINARY_LABEL_MAP`, which maps `normal`, `train_normal` and `test_normal` to `0` and `abnormal`, `anomaly`, `anomalous`, `test_anomaly` to `1`.
- Frozen ResNet18 (ImageNet weights, `fc` replaced by `nn.Identity()`) used as a per-frame feature extractor in `eval()` mode, with batches of 64 frames.
- Per-video feature caching to `.npy` files, so re-running the notebook skips already processed folders.
- Sliding-window sequence construction (`WINDOW_SIZE = 16`, `WINDOW_STRIDE = 8`, i.e. 50% overlap).
- Video-level stratified train/validation/test split (70/15/15) so that windows from the same video never appear in two different splits, which prevents leakage.
- `LSTMBinaryClassifier`: single-layer LSTM plus a two-layer MLP head with ReLU and dropout, producing one logit per window.
- Class-imbalance handling through `BCEWithLogitsLoss(pos_weight=...)`, computed from the training set anomaly rate.
- Best-checkpoint saving based on validation loss.
- Evaluation with accuracy, precision, recall, F1, ROC AUC, a saved ROC curve, an annotated confusion matrix, and a `test_metrics.json` dump.
- Inference block that scores a new folder of `.jpg` frames and plots the anomaly score over time with a 90th percentile threshold line.

## Project Structure

The project is a single Jupyter notebook that creates its own directory tree at runtime under `PROJECT_ROOT`.

```
Traffic_Anomaly_Detection.ipynb   Full pipeline: config, download, EDA, features, training, evaluation, inference
tad_project/
├── data/
│   ├── raw/          Downloaded dataset (video folders containing .jpg frames)
│   ├── frames/       Created by the setup cell, reserved for extracted frames
│   ├── features/     One .npy per video folder, shape (n_frames, 512)
│   └── processed/    train.npz, val.npz, test.npz with X, y and video_ids arrays
├── models/
│   └── lstm_binary_classifier.pt    Best checkpoint by validation loss
└── reports/
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── test_metrics.json
```

### Notebook sections

| Section | Purpose |
|---|---|
| 1. Install dependencies and imports | pip install line (commented out) and all imports |
| 2. Configuration | Paths, image size, window settings, split ratios, hyperparameters, seeds, device selection |
| 3. Download the dataset | `kagglehub.dataset_download` plus copy into `data/raw` |
| 4. Exploratory data analysis | Extension counts, `video_folders` dictionary with path, label and frame count |
| 5. CNN feature extraction | ResNet18 embeddings written to `data/features` |
| 6. Window preparation and split | Stratified video-level split, sliding windows, `.npz` export |
| 7. Model definition | `LSTMBinaryClassifier` class |
| 8. Training | Adam optimiser, weighted BCE loss, manual mini-batch loop |
| 9. Evaluation | Metrics, ROC curve, confusion matrix, JSON report |
| 10. Inference on a new image folder | Scores an unseen folder of frames |

## Requirements

### GPU is required

The pipeline is built for CUDA and will not be practical on CPU. ResNet18 runs a forward pass over every single frame of every video folder in the dataset, and the LSTM is trained for 25 epochs over all windows. Without a GPU, feature extraction alone takes hours.

The notebook selects the device automatically:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
```

Before running anything else, confirm the printed output is `Using device: cuda`. If it prints `cpu`, stop and fix the environment (enable the GPU runtime, or install a CUDA-enabled PyTorch build). The notebook metadata records a T4 GPU accelerator, which is sufficient.

Additional notes on GPU setup:

- Install a CUDA build of PyTorch, not the CPU wheel. Verify with `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- The whole training set is held in host memory as `X_train_t` and moved to the GPU batch by batch, but `X_val_t`, `y_val_t` and `X_test_t` are moved to the GPU in full. On a small-memory card, lower `BATCH_SIZE` or evaluate in chunks.
- Feature extraction uses fixed batches of 64 preprocessed frames. Reduce that number if you hit out-of-memory errors.

### Software

- Python 3
- Kaggle credentials available to `kagglehub` for the dataset download

## Installation

```bash
pip install kagglehub opencv-python torch torchvision
pip install numpy pandas scikit-learn matplotlib pillow
```

For an explicit CUDA build of PyTorch (adjust the CUDA version to match your driver):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then start the notebook:

```bash
jupyter notebook Traffic_Anomaly_Detection.ipynb
```

## Usage

Run the cells in order, from section 1 to section 9. Each section depends on the artifacts produced by the previous one.

1. **Sections 1 and 2**: imports and configuration. Confirm the device print shows `cuda`.
2. **Section 3**: downloads the dataset. If `data/raw` is not empty the download is skipped.
3. **Section 4**: builds `video_folders`. Check the printed counts of normal and abnormal folders. If fewer than 10 folders are found, the notebook prints a warning, because that means the frames sit directly inside `normal/` and `abnormal/` without per-video subfolders and file-name grouping logic has to be added.
4. **Section 5**: extracts and caches features. Already existing `.npy` files are skipped.
5. **Section 6**: creates the splits and writes `train.npz`, `val.npz`, `test.npz`.
6. **Sections 7 and 8**: define and train the model. The checkpoint with the lowest validation loss is written to `models/lstm_binary_classifier.pt`.
7. **Section 9**: reloads the best checkpoint and writes metrics and plots to `reports/`.

### Inference on a new folder of frames

Set `INFERENCE_FOLDER` in the last cell to a directory containing `.jpg` frames and run it:

```python
INFERENCE_FOLDER = "/content/tad_project/data/raw/normal/video_xyz"
```

The cell extracts ResNet18 features, builds windows with the same `WINDOW_SIZE` and `WINDOW_STRIDE`, scores each window with `torch.sigmoid(model(X_inf))`, marks every window above the 90th percentile of the scores, and plots the score curve. It raises a `RuntimeError` if the folder contains fewer than `WINDOW_SIZE` (16) images. If `INFERENCE_FOLDER` stays `None`, the cell just prints a reminder to set it.

## Configuration

There are no environment variables or config files. All settings live in section 2 of the notebook.

| Constant | Value | Meaning |
|---|---|---|
| `KAGGLE_DATASET_SLUG` | `nikanvasei/traffic-anomaly-dataset-tad` | Source dataset |
| `PROJECT_ROOT` | `/content/tad_project` | Root of all generated folders, change it for a local run |
| `FRAME_SIZE` | `(224, 224)` | Resize target before ImageNet normalisation |
| `WINDOW_SIZE` | `16` | Frames per sequence window |
| `WINDOW_STRIDE` | `8` | Step between windows (50% overlap) |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | `0.7` / `0.15` / `0.15` | Video-level split |
| `RANDOM_SEED` | `42` | Seeds `random`, `numpy` and `torch` |
| `FEATURE_DIM` | `512` | ResNet18 penultimate-layer embedding size |
| `BATCH_SIZE` | `32` | Training mini-batch size |
| `NUM_EPOCHS` | `25` | Training epochs |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `WEIGHT_DECAY` | `1e-5` | Adam weight decay |
| `LSTM_HIDDEN_SIZE` | `128` | LSTM hidden units |
| `DROPOUT` | `0.3` | Dropout in the classifier head |
| `BINARY_LABEL_MAP` | see notebook | Folder name to binary label mapping |

## Class Weighting, and Why Recall Is Preferred Over Precision

The dataset is imbalanced: normal traffic windows heavily outnumber abnormal ones. Two design choices in the notebook deliberately push the model towards catching anomalies rather than towards being conservative.

**1. Positive class weighting in the loss.**

```python
pos_rate = max(y_train.mean(), 1e-6)
pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

`pos_weight` is the ratio of negatives to positives in the training set. Without it, an unweighted BCE loss can be minimised almost perfectly by predicting "normal" for every window, since that is already correct most of the time. The model would show high accuracy and near-zero recall, which is useless. Weighting the positive class by `(1 - pos_rate) / pos_rate` makes each missed anomaly cost as much in aggregate as all the normal windows, so the gradient actually pushes the network to learn what an abnormal sequence looks like.

**2. A very low decision threshold.**

```python
preds = (probs >= 0.01).astype(int)
```

Instead of the default 0.5, a window is flagged as abnormal at a sigmoid probability of just 0.01. This trades precision for recall on purpose.

The reason is the asymmetry of the two error types in this application. A false negative is a real accident that no one is alerted to, which can mean a delayed emergency response and, potentially, a life. A false positive is a clip of normal traffic that a human reviews for a few seconds and dismisses. The costs are not remotely comparable, so the operating point is chosen where the model rarely misses an event, and the price is a larger number of harmless alarms. In other words, the system is designed as a screening filter that shrinks the footage a human has to look at, not as a final authority that decides what is or is not an accident.

The same logic drives the inference cell, where the threshold is the 90th percentile of the scores in the analysed sequence, so the top 10% of the most suspicious windows are always surfaced for review. ROC AUC is also reported, since it summarises performance across all thresholds and is therefore more informative here than accuracy at a single cut-off.

## Technologies Used

| Technology | Role |
|---|---|
| Python 3 | Implementation language |
| PyTorch (`torch`, `torch.nn`) | LSTM model, training loop, checkpointing |
| torchvision | ResNet18 with `IMAGENET1K_V1` weights, `transforms` preprocessing pipeline |
| NumPy | Feature arrays, windowing, `.npy` and `.npz` storage |
| pandas | Extension counts during EDA |
| scikit-learn | `train_test_split` with stratification, accuracy, precision, recall, F1, ROC AUC, ROC curve, confusion matrix |
| Pillow (PIL) | Frame loading and RGB conversion |
| Matplotlib | ROC curve, confusion matrix, anomaly score plot |
| kagglehub | Dataset download |
| OpenCV (`cv2`) | Imported for image and video utilities |

## License

The frames used in this project come from the Traffic Anomaly Dataset (TAD), downloaded through `kagglehub` with the slug `nikanvasei/traffic-anomaly-dataset-tad`. The dataset was introduced in the work below, and any use of this project's data or results should cite it:

> Hui Lv, Chuanwei Zhou, Zhen Cui, Chunyan Xu, Yong Li, Jian Yang.
> "Localizing Anomalies from Weakly-Labeled Videos."
> IEEE Transactions on Image Processing (TIP), 2021.

BibTeX:

```bibtex
@article{wsal_tip21,
  author    = {Hui Lv and
               Chuanwei Zhou and
               Zhen Cui and
               Chunyan Xu and
               Yong Li and
               Jian Yang},
  title     = {Localizing Anomalies from Weakly-Labeled Videos},
  journal   = {IEEE Transactions on Image Processing (TIP)},
  year      = {2021}
}
```

Note on labels: the original dataset is weakly labeled, meaning the annotation states whether a video contains an anomaly, not which frames the anomaly occupies. This pipeline follows that constraint directly. In section 6 every sliding window inherits the label of its parent video folder, as the comment in the code states ("Binary label: every frame in this video has the same label"). Windows drawn from an abnormal video that in fact show only normal traffic are therefore labeled as abnormal during training. This is another reason the evaluation leans on recall and ROC AUC instead of precision: part of the measured false positive rate is an artifact of the weak labels, not necessarily a model error.

## Author

*M. E. Petra Bošković*
