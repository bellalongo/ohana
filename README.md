# ohana

**Observational H2RG Anomaly Noise Analyzer**

A deep learning-based toolkit for detecting and classifying transient anomalies in astronomical detector data.

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)

## About

Ohana (Observational H2RG Anomaly Noise Analyzer) is a deep learning-based toolkit for detecting and classifying transient anomalies in astronomical detector data. It focuses on anomalies common in H2RG infrared detectors (2048×2048 pixel arrays used by JWST, Euclid, and others), such as cosmic ray hits, random telegraph noise (RTN), and "snowballs." Identifying these artifacts is crucial for improving data quality and instrument calibration, ensuring that real astrophysical signals are not confused or obscured by detector noise.

### How It Works

The ohana pipeline combines synthetic data simulation, a 3D convolutional neural network, and classical algorithms to detect anomalies across an exposure's time series:

1. **Synthetic Data Generation**: Simulated up-the-ramp exposure cubes are created with injected anomalies. Realistic cosmic ray hits (sudden jumps in signal), telegraph noise events (pixels toggling between states), and snowballs (large, diffuse charge deposits) are added to generate labeled training data.

2. **3D U-Net Model Training**: A 3D U-Net deep neural network is trained on patches from these simulated cubes. By analyzing temporal stacks of frames, the model learns to recognize anomalies based on both their spatial shape and their time-domain signature (e.g. a cosmic ray appears in one frame and persists, RTN shows repetitive switching, etc.).

3. **Anomaly Detection & Classification**: For a new exposure, ohana applies a hybrid approach. The trained 3D U-Net (if provided) predicts anomaly masks frame by frame. In parallel, built-in rule-based detectors (inspired by JWST pipeline algorithms) analyze temporal features to flag cosmic ray candidates, RTN pixels, and snowball events. These results are combined into a consolidated list of detected anomalies with classifications.

4. **Results & Outputs**: Ohana produces an output mask and a catalog of anomalies. Each detected event is recorded with its type (cosmic ray, telegraph noise, snowball), location, intensity, and size. Visualization tools are included to overlay the anomaly mask on the image and plot time-series of affected pixels, helping users verify and interpret the detections.

### Dependencies

- astropy
- pytorch
- scipy
- scikit-image
- numpy
- pandas

## Installation

Ohana is available as a Python package. Install it via pip:

```bash
pip install ohana
```

**Requirements**: Python 3.8+ and a PyTorch-compatible environment (GPU recommended for training).

After installation, command-line utilities will be available for each stage of the workflow (dataset creation, model training, and prediction).

## Configuration

Ohana uses configuration files to control simulation and detection parameters. Key configurations include:

### Training Data Configuration (YAML)

Synthetic data generation is configured via a YAML file (e.g. `creator_config.yaml`). This file specifies the detector and simulation parameters:

```yaml
# Example creator_config.yaml
output_dir: "./data/processed"              # Directory to save generated dataset
image_shape: [2048, 2048]                   # Full frame dimensions of the detector
num_frames: 450                             # Number of time frames in each exposure ramp
patch_size: [256, 256]                      # Size of image patches to extract from each exposure
overlap: 32                                 # Overlap (pixels) between adjacent patches
num_exposure_events: 1                      # Number of exposures *with* anomalies to simulate (per detector gain)
num_baseline_exposures: 1                   # Number of baseline (no anomaly) exposures to simulate (per gain)
injection_type: "all"                       # Type of anomalies to inject: "cosmic_ray", "snowball", "rtn", or "all"
num_classes: 4                              # Total anomaly classes (e.g. 4 = [background, CR, RTN, snowball])

cosmic_ray_per_exp_range: [100, 4000]       # Random range for how many cosmic rays to inject per exposure
snowballs_per_exp_range: [0, 3]             # Range for number of snowballs per exposure
rtn_per_exp_range: [100, 1500]              # Range for number of RTN pixels per exposure

gain_library:                                # Dictionary of detector IDs and their gains (e-/count)
  '18220_SCA': 1.062
  '18248_SCA': 1.021
  # ... (additional detector IDs)

exp_time: 45000                             # Exposure time in seconds
saturation_dn: 65535                        # Saturation level in DN (detector counts)
dark_current_range: [0.01, 0.02]            # Range of dark current (electrons)
read_noise_range: [15.0, 25.0]              # Read noise range (electrons)
gaussian_noise_range: [0.5, 10]             # Additional Gaussian noise range (electrons)

cosmic_ray_charge_range: [2.0e2, 6.0e3]     # Charge deposited by cosmic rays (electrons)
rtn_offset_range: [30, 300]                 # Amplitude range for RTN step (electrons)
rtn_period_range: [20, 200]                 # Period (frames) of RTN toggling
rtn_duty_fraction_range: [0.1, 0.9]         # Duty cycle of RTN (fraction of time in high state)

snowball_radius_range: [5.0, 200.0]         # Range of snowball radius (pixels)
snowball_halo_amplitude_ratio_range: [0.02, 0.15]  # Brightness ratio of snowball halo to core
snowball_halo_decay_scale_range: [3.0, 30.0]       # Exponential decay length of snowball halo (pixels)
```

This configuration controls how synthetic exposures are generated. For example, the above settings will simulate 1 exposure with anomalies and 1 without, for each detector listed in `gain_library`. Each anomalous exposure will have a random number of cosmic rays (100–4000), snowballs (0–3), and RTN pixels (100–1500) injected. The resulting up-the-ramp data cubes are then broken into overlapping 256×256 patches (with 32-pixel overlap) for model training.

### Detection Configuration (DetectorConfig)

Ohana's detection algorithms use a `DetectorConfig` dataclass (with default values) to set thresholds for identifying events in real data:

```python
# Key DetectorConfig parameters (default values):
sigma_threshold = 5.0                       # Sigma multiplier for initial noise thresholding
min_anomaly_pixels = 1                      # Minimum connected pixels for a valid anomaly event
min_confidence = 0.5                         # Minimum confidence (if ML model outputs probabilities)

# Cosmic Ray detection thresholds:
cosmic_ray_min_intensity = 40.0             # Minimum per-pixel intensity jump to consider (DN)
cosmic_ray_max_spatial_extent = 15          # Max size (pixels) for a single cosmic ray event
cosmic_ray_min_step = 25.0                  # Minimum step amplitude (DN) for a valid cosmic ray (post-pre jump)

# Random Telegraph Noise (RTN) thresholds:
rtn_min_transitions = 2                     # Min number of state changes (high/low) to detect RTN
rtn_max_transitions = 50                    # Max allowed transitions (to filter excessive flicker)
rtn_amplitude_range = (10.0, 300.0)         # Allowed signal amplitude range for RTN events (DN)
rtn_frequency_range = (0.001, 0.5)          # Allowed toggling frequency range (Hz or per frame fraction)
rtn_fit_quality_threshold = 0.3             # Minimum R² fit for two-state (high/low) model
rtn_min_confidence = 0.5                    # Minimum confidence for classifying an RTN event

# Snowball detection thresholds:
snowball_min_confidence = 0.2               # Confidence threshold for identifying snowball events
snowball_min_intensity = 30.0               # Min intensity for snowball core (DN)
snowball_max_intensity = 500.0              # Max intensity (to exclude saturated events if needed)
snowball_min_radius = 3                     # Min radius (pixels) for snowball core
snowball_max_radius = 15                    # Max radius for snowball core
snowball_circularity_threshold = 0.7        # How circular the event must be (0–1)
snowball_expansion_rate = 0.1               # Max expansion rate in radius per frame (pixels/frame)
snowball_min_area = 75                      # Min area (pixels) for snowball event (core + halo)
snowball_max_area = 120000                  # Max area for snowball event
```

These parameters govern the rule-based detection stage. For instance, `cosmic_ray_min_intensity = 40.0` (DN) means a pixel must jump by at least 40 DN to be flagged as a cosmic ray candidate. The RTN settings ensure that only pixels toggling at least twice with a decent fit to a two-level model are considered. Snowball criteria require a minimum size and a roughly circular shape. You can adjust these thresholds in code or via a custom DetectorConfig to fine-tune sensitivity.

## Usage

Once installed, ohana provides command-line tools to run each part of the anomaly detection workflow. Below is a typical usage sequence from data generation to anomaly analysis:

### 1. Generate a Synthetic Training Dataset (optional)

If you plan to train a new model (or have no real labeled data), use `ohana-create-training` to simulate exposures with anomalies:

```bash
ohana-create-training --config configs/creator_config.yaml --start_index 0 --end_index 0
```

This command will read the specified YAML config and produce synthetic exposures (with anomalies) and baseline exposures for the index range 0 to 0 (i.e., one set). The output (patches and metadata) will be saved under the configured `output_dir` (e.g. `data/processed/`).

### 2. Train the 3D U-Net Model

Once synthetic data is prepared (or if you have your own training dataset in the same format), train the model using `ohana-train`:

```bash
ohana-train --config configs/creator_config.yaml --epochs 20 --batch_size 2 --output_dir ./trained_models
```

This will load the patch dataset from the `output_dir` specified in the YAML (here `data/processed/`), then train a 3D U-Net for 20 epochs (with batch size 2 and default learning rate 1e-4). Model weights and a training history log will be saved in `./trained_models` (or your specified `--output_dir`). By default, the best model (highest validation accuracy) is saved as `best_model_unet3d.pth` along with a `training_history_unet3d.json` containing loss/accuracy per epoch.

### 3. Run Anomaly Detection on New Data

With a trained model available, you can analyze new exposure data for anomalies using `ohana-predict`:

```bash
ohana-predict --model trained_models/best_model_unet3d.pth --input /path/to/exposure.fits --output prediction_outputs
```

Replace `/path/to/exposure.fits` with the FITS file of the exposure you want to analyze. This command will load your model and run the full detection pipeline. If no model is provided, ohana will default to using only the built-in rule-based detectors. The results (detailed below) will be saved in a new directory (default `prediction_outputs/`).

**Note**: The command-line examples above assume you have the config YAML and any input files available at the given paths. The `ohana-predict` step requires an exposure file (FITS format expected) as input. This should be a data cube or up-the-ramp dataset from an H2RG detector. For example, you can try an exposure from JWST NIRCam or a Euclid test dataset. If you generated synthetic data in step 1, you could convert one of those simulated exposures to FITS for a test run (alternatively, adjust the code to accept `.npy` arrays). GPU acceleration (if available) will be used for model inference to speed up the analysis.

## Output Files

After running the detection pipeline, ohana will produce a set of output files summarizing the anomalies found, as well as intermediate data. Below is an overview of the outputs:

- **Processed Data Cube** – A NumPy array of the exposure after basic preprocessing, saved as an `.npy` file. (During prediction, the raw up-the-ramp data is read from the FITS, reference pixel corrections are applied, and the result is cached as `[exposure_name]_processed.npy` for reuse.)

- **Temporal Features** – A compressed NumPy `.npz` file containing computed per-pixel temporal metrics (e.g. first frame of signal appearance, persistence length, etc.). This is used by the rule-based detectors and saved as `[exposure_name]_temporal.npz`.

- **Anomaly Mask** – A 2D mask of the same size as one frame of the exposure, saved as `prediction_mask.npy`. Each pixel in this mask is coded as 0 for background or an integer ID for the anomaly type (e.g. 1 = cosmic ray, 2 = telegraph noise, etc.). This mask combines the results of the ML model (if used) and rule-based detectors.

- **Detections List (JSON)** – A JSON file `detections.json` listing all detected anomalies. Each entry in this list contains detailed information about one anomaly:
  - `type`: the anomaly classification (e.g. "cosmic_ray", "telegraph_noise", or "snowball").
  - `centroid` or `position`: the coordinates of the anomaly in the image (for an extended cosmic ray or snowball, a centroid of the affected pixels; for a single-pixel RTN, the pixel position).
  - `first_frame`: (for transient events) the frame index at which the anomaly first appears.
  - `mean_intensity` / `max_intensity`: brightness metrics of the anomaly (e.g. the average and peak DN values in the affected pixels).
  - `spatial_extent`: the size (in pixels) of the anomaly (e.g. number of connected pixels for a cosmic ray cluster).
  - Additional fields: method-specific metrics such as a step-function fit quality or amplitude for cosmic rays, number of state transitions for RTN, etc., which help assess the detection confidence.

These outputs enable both quantitative analysis and visualization of the anomalies. For instance, you can load `prediction_mask.npy` and overplot it on the image to see where anomalies occur, or examine `detections.json` to filter events by type or intensity. Ohana's `ResultVisualizer` class can take the processed data and detections JSON to create quick-look plots (e.g. an image with all anomaly locations marked, and zoomed time-series plots for individual events).

### Example

After installing ohana, you might run:

```bash
pip install ohana

# Generate synthetic data (optional, for training)
ohana-create-training --config configs/creator_config.yaml --start_index 0 --end_index 0

# Train the model on the synthetic dataset
ohana-train --config configs/creator_config.yaml --epochs 5 --output_dir ./trained_models

# Detect anomalies in a new exposure using the trained model
ohana-predict --model trained_models/best_model_unet3d.pth --input my_exposure.fits --output prediction_outputs
```

Upon completion, check the `prediction_outputs/` folder for results. For example, open `detections.json` to see the list of anomalies and their details, and use the visualization utilities or your preferred plotting library to inspect `prediction_mask.npy`.
