# FeXT AutoEncoder: Extraction of Images Features

## 1. Introduction
FeXT AutoEncoder is a desktop-first pipeline for inspecting image datasets, training convolutional autoencoders, exporting latent representations, and inspecting results without leaving a single UI. The application combines a PySide6 + Qt Material interface with a PyTorch/Triton-powered backend to deliver GPU-accelerated training, resumable sessions, and rich logging. Inspired by the VGG16 family (https://keras.io/api/applications/vgg/), FeXT focuses on compact representations that can be reused for anomaly detection, similarity search, or any downstream feature-driven task.

Key capabilities include automated dataset validation, configurable training and inference flows, background workers that keep the UI responsive, and built-in viewers for both intermediate plots and reconstructed images. Everything runs locally, so datasets never leave your machine.

![VGG16 encoder](FEXT/app/assets/figures/VGG16_encoder.png)
Architecture of the VGG16 encoder

## 2. FeXT AutoEncoder model
The FeXT AutoEncoder is a deep convolutional autoencoder that compresses images into expressive latent vectors and reconstructs them with minimal loss. Core architectural traits:

- Residual convolutional blocks with layer normalization provide stable training while keeping kernels small and efficient.
- Downsampling through stride-1 convolutions and max pooling mirrors VGG16 and ensures that spatial resolution shrinks while the channel dimension grows, letting the encoder focus on higher-order abstractions.
- A Compression Layer forms the bottleneck. Dropout may be enabled here to improve generalization when working with noisy datasets or small sample sizes.
- The decoder mirrors the encoder with transposed convolutions and nearest-neighbor upsampling, plus residual links that preserve fine detail.
- All data pipelines rely on `tf.data` datasets with prefetching, which keeps the GPUs/CPUs busy while the UI orchestrates long-running tasks through worker threads or processes.

FeXT ships with three model families so you can balance fidelity and throughput:

- **FextAE Redux** — minimal channels, fast experimentation, ideal for laptops.
- **FextAE Medium** — the recommended preset for most datasets; includes richer residual stacks and deeper compression.
- **FextAE Large** — maximizes capacity with wider convolutions and deeper decoder stages; best suited for high-resolution input and powerful GPUs.

Every training run creates versioned checkpoints, captures the training configuration, and stores reconstructed examples plus metrics so you can resume later, compare models, or export embeddings at any point.

## 3. Training dataset
The reference model was trained and evaluated on the Flickr 30K dataset (https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). The pipeline is dataset-agnostic: drop any JPEG or PNG set into `FEXT/resources/database/images` and FeXT will resize, normalize, and split it according to the selected configuration. The validation service computes descriptive statistics (mean, standard deviation, noise ratios), pixel distributions, and train/validation drift checks so you can spot outliers before training.

## 4. Installation
The project targets Windows 10/11 and requires roughly 10 GB of free disk space for the embedded Python runtime, dependencies, checkpoints, and datasets. A CUDA-capable NVIDIA GPU is recommended but not mandatory. Ensure you have the latest GPU drivers installed when enabling TorchInductor + Triton acceleration.

1. **Download the project**: clone the repository or extract the release archive into a writable location (avoid paths that require admin privileges).
2. **Configure environment variables**: copy `FEXT/resources/templates/.env` into `FEXT/app/.env` and adjust values (e.g., backend selection).
3. **Run `start_on_windows.bat`**: the bootstrapper installs a portable Python 3.12 build, downloads Astral’s `uv`, syncs dependencies from `pyproject.toml`, prunes caches, then launches the UI through `uv run`. The script is idempotent—rerun it any time to repair the environment or re-open the app.

Running the script the first time can take several minutes depending on bandwidth. Subsequent runs reuse the cached Python runtime and only re-sync packages when `pyproject.toml` changes.

### 4.1 Just-In-Time (JIT) Compiler
`torch.compile` is enabled throughout the training and inference pipelines. TorchInductor optimizes the computation graph, performs kernel fusion, and lowers operations to Triton-generated kernels on NVIDIA GPUs or to optimized CPU kernels otherwise. Triton is bundled automatically so no separate CUDA toolkit installation is required.

### 4.2 Manual or developer installation
If you prefer managing Python yourself (for debugging or CI):

1. Install Python 3.12.x and `uv` (https://github.com/astral-sh/uv).
2. From the repository root run `uv sync` to create a virtual environment with the versions pinned in `pyproject.toml`.
3. Copy `.env` as described earlier and ensure the `KERAS_BACKEND` is set to `torch`.
4. Launch the UI with `uv run python FEXT/app/app.py`.
5. Developers can edit configurations under `FEXT/resources/configurations` to define dataset paths, batch sizes, augmentation policies, and JIT preferences. The UI exposes these settings through dedicated dialogs so you can save/load presets without editing JSON manually.

## 5. How to use
Launch the application by double-clicking `start_on_windows.bat` (or via `uv run python FEXT/app/app.py` after a manual install). On startup the UI loads the last-used configuration, scans the resources folder, and initializes worker pools so long-running jobs (training, inference, validation) do not block the interface.

1. **Prepare data**: verify that `resources/database/images` (training) and `resources/database/inference` (inference) contain the expected files. Large datasets can be sub-sampled through the configuration dialog (`train_sample_size`).
2. **Adjust configuration**: use the toolbar to load/save configuration templates or modify each parameter manually from the UI.
3. **Run a pipeline**: pick an action under the Data, Model, or Viewer tabs. Progress bars, log panes, and popup notifications keep you informed. Background workers can be interrupted at any time without crashing the UI.

**Data tab:** dataset analysis and validation.

- Calculate pixel statistics (mean, std, min/max) and noise ratios at dataset or subset level.
- Compare train vs. validation distributions, verify class balance, and export the generated plots to `resources/database/validation`.
- Build SQLite-based summaries so you can filter runs later or inspect metadata with DB Browser for SQLite.

**Model tab:** training, evaluation, and encoding.

- Train any FeXT variant from scratch with on-the-fly data loaders (`tf.data` with caching, prefetching, and parallel decoding).
- Resume training from any checkpoint; the corresponding configuration is reloaded automatically and you can extend training for `n` additional epochs.
- Evaluate checkpoints with reconstruction metrics (MSE, MAE) and qualitative visualizations (random reconstructions, embedding plots) saved under `resources/checkpoints/<run>/evaluation`.
- Run inference to encode arbitrary folders of images; latent vectors are exported as `.npy` files under `resources/database/inference` together with a manifest so they can be reused in downstream tasks.

**Viewer tab:** visualization hub.

- Browse raw training images, inference inputs, reconstructed samples, and any plots generated during dataset or model evaluation.
- Leverages Qt graphics views for panning/zooming plus Matplotlib figure embedding for plot inspection.
- Useful for quick sanity checks without leaving the application.

## 5.1 Setup and Maintenance
`setup_and_maintenance.bat` launches a lightweight maintenance console with these options:

- **Update project**: performs a `git pull` (or fetches release artifacts) so the local checkout stays in sync.
- **Remove logs**: clears `resources/logs` to save disk space or to reset diagnostics before a new run.
- **Open tools**: quick shortcuts to DB Browser for SQLite or other external utilities defined in the script.

Run this script periodically to stay current or whenever you want to reset artifacts without touching datasets/checkpoints.

### 5.2 Resources
The `FEXT/resources` tree keeps all mutable assets, making backups and migrations straightforward:

- **checkpoints** — versioned folders containing saved models, training history, evaluation reports, reconstructed samples, and the JSON configuration that produced them. These folders are what you load when resuming training or running inference.
- **configurations** — reusable JSON presets surfaced by the UI dialogs. Store both stock and custom configurations here to share setups with teammates.
- **database** — includes sub-folders for `images` (training data), `inference` (raw inputs and exported `.npy` embeddings), `metadata` (SQLite records), and `validation` (plots + stats reports).
- **logs** — rotating application logs for troubleshooting. Attach these when reporting issues.
- **templates** — contains `.env` and other templates that need to be copied into write-protected directories (`FEXT/app`).

Environmental variables reside in `FEXT/app/.env` (never committed). Copy the template from `resources/templates/.env` and adjust as needed:

| Variable              | Description                                                               |
|-----------------------|---------------------------------------------------------------------------|
| KERAS_BACKEND         | Backend for Keras 3; keep `torch` unless you explicitly need TensorFlow.  |
| TF_CPP_MIN_LOG_LEVEL  | Controls TensorFlow logging verbosity (set to `2` to suppress INFO logs). |
| MPLBACKEND            | Matplotlib backend; `Agg` keeps plotting headless for worker threads.     |

## 6. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
