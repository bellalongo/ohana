# ohana Developer Guide

This guide demonstrates how to programmatically use the `ohana` package for anomaly detection in astronomical detector data. Rather than using command-line tools, this approach gives you full control over the detection pipeline through Python code.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start Tutorial](#quick-start-tutorial)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

The `ohana` package provides a high-level `Predictor` class that handles all the complex steps of anomaly detection internally. The standard workflow consists of:

1. Configure paths and settings
2. Initialize the `Predictor`
3. Run the `predict` method
4. Visualize and analyze results

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (CPU or GPU version)
- NumPy, SciPy, scikit-image, astropy, pandas

### Install from pip

```bash
pip install ohana
```

### Install from source (for development)

```bash
git clone https://github.com/bellalongo/ohana.git
cd ohana
pip install -e .
```

## Quick Start Tutorial

This tutorial demonstrates the standard, high-level way to run anomaly detection using the `ohana` package.

### Step 1: Imports and Configuration

First, import the necessary classes and configure PyTorch for optimal performance:

```python
import torch
import os
import json
import numpy as np
import sys

# Configure PyTorch threading for optimal performance
print(f"Setting PyTorch to use {os.cpu_count() or 8} threads.")
torch.set_num_threads(os.cpu_count() or 8)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)

# Add the ohana source to path if running from repository
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import ohana classes
from ohana import (
    UNet3D,
    Predictor,
    DataLoader,
    Preprocessor,
    ResultVisualizer,
    DetectorConfig
)
```

### Step 2: Set Up File Paths

Configure the paths to your model, data, and output directories:

```python
# Path to the trained model (optional - will use rule-based only if not provided)
MODEL_PATH = "../trained_models/best_model_unet3d.pth"

# Path to the config file used for model training
CONFIG_PATH = "../configs/creator_config.yaml"

# Path to the exposure FITS file you want to analyze
EXPOSURE_PATH = "/path/to/your/exposure.fits"

# Output directory for results
OUTPUT_DIR = "prediction_outputs"

# Specific output file paths
PROCESSED_EXPOSURE_FILE = 'processed/exposure_processed.npy'
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, PROCESSED_EXPOSURE_FILE)
MASK_PATH = os.path.join(OUTPUT_DIR, 'prediction_mask.npy')
DETECTIONS_PATH = os.path.join(OUTPUT_DIR, 'detections.json')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Step 3: Configure Detection Parameters

The `DetectorConfig` class allows you to customize detection thresholds:

```python
# Use default configuration
config = DetectorConfig()

# Or customize specific parameters
config = DetectorConfig(
    sigma_threshold=4.0,  # Lower for more sensitive detection
    cosmic_ray_min_intensity=30.0,  # Lower threshold for cosmic rays
    min_anomaly_pixels=2,  # Require at least 2 connected pixels
    rtn_min_transitions=2,  # Minimum state changes for RTN detection
)
```

### Step 4: Run the Prediction

Initialize the predictor and run anomaly detection:

```python
# Initialize the predictor
print("Initializing the predictor...")
predictor = Predictor(
    model_path=MODEL_PATH,  # Optional - omit to use only rule-based detection
    config=config
)

# Run prediction
print(f"Running prediction on {EXPOSURE_PATH}...")
anomalies = predictor.predict(
    exposure_path=EXPOSURE_PATH,
    output_dir=OUTPUT_DIR
)

print(f"Found {len(anomalies)} total anomalies.")

# Save detections to JSON
def to_builtin(o):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o)} is not JSON serializable")

with open(DETECTIONS_PATH, "w") as f:
    json.dump(anomalies, f, indent=4, default=to_builtin)
```

### Step 5: Visualize the Results

Use the `ResultVisualizer` class to create plots and analyze the detections:

```python
# Initialize the visualizer
print("Generating visualization...")
visualizer = ResultVisualizer(
    processed_data_path=PROCESSED_DATA_PATH
)

# Load the detection results
visualizer.load_detection_list(results_path=DETECTIONS_PATH)

# Generate overview plot showing all anomalies
visualizer.plot_full_mask_overlay()

# Plot detailed views of individual detections (limit to avoid too many plots)
visualizer.plot_all_detections(max_plots=5)

# Create an animated movie for a specific event
# Find the first cosmic ray detection
first_cr = None
for detection in visualizer.detections:
    if detection['type'] == 'cosmic_ray_candidate':
        first_cr = detection
        break

if first_cr:
    print(f"Creating a movie for the cosmic ray at {first_cr['centroid']}...")
    y, x = first_cr['centroid']
    visualizer.create_event_movie(
        y=int(round(y)), 
        x=int(round(x)), 
        output_path='cosmic_ray_event.gif'  # Can also save as .mp4
    )
```

## API Reference

### Core Classes

#### `Predictor`

The main class for running anomaly detection.

```python
predictor = Predictor(
    model_path: str = None,  # Path to trained model (optional)
    config: DetectorConfig = None  # Detection configuration
)

anomalies = predictor.predict(
    exposure_path: str,  # Path to FITS file
    output_dir: str = "prediction_outputs"  # Output directory
)
```

#### `DetectorConfig`

Configuration dataclass for detection parameters.

```python
config = DetectorConfig(
    # General thresholds
    sigma_threshold: float = 5.0,
    min_anomaly_pixels: int = 1,
    min_confidence: float = 0.5,
    
    # Cosmic ray parameters
    cosmic_ray_min_intensity: float = 40.0,
    cosmic_ray_max_spatial_extent: int = 15,
    cosmic_ray_min_step: float = 25.0,
    
    # RTN parameters
    rtn_min_transitions: int = 2,
    rtn_max_transitions: int = 50,
    rtn_amplitude_range: tuple = (10.0, 300.0),
    rtn_fit_quality_threshold: float = 0.3,
    
    # Snowball parameters
    snowball_min_confidence: float = 0.2,
    snowball_min_intensity: float = 30.0,
    snowball_min_radius: int = 3,
    snowball_circularity_threshold: float = 0.7
)
```

#### `ResultVisualizer`

Class for visualizing detection results.

```python
visualizer = ResultVisualizer(
    processed_data_path: str  # Path to processed .npy file
)

# Load detections
visualizer.load_detection_list(results_path: str)

# Visualization methods
visualizer.plot_full_mask_overlay()  # Overview plot
visualizer.plot_all_detections(max_plots: int = None)  # Individual events
visualizer.create_event_movie(
    y: int, x: int,  # Pixel coordinates
    output_path: str,  # Output file (.gif or .mp4)
    window_size: int = 50,  # Size of region to show
    fps: int = 10  # Frames per second
)
```

### Detection Output Format

The `predict()` method returns a list of anomaly dictionaries with the following structure:

```python
{
    "type": "cosmic_ray_candidate",  # or "telegraph_noise", "snowball"
    "centroid": [y, x],  # Pixel coordinates
    "first_frame": 10,  # Frame where anomaly first appears
    "mean_intensity": 150.5,  # Average DN value
    "max_intensity": 200.0,  # Peak DN value
    "spatial_extent": 5,  # Number of affected pixels
    "confidence": 0.85,  # Detection confidence (if using ML model)
    # Additional type-specific fields...
}
```

## Advanced Usage

### Custom Preprocessing

For advanced users who need custom preprocessing:

```python
from ohana import Preprocessor, ReferencePixelCorrector

# Load and preprocess data manually
preprocessor = Preprocessor()
corrector = ReferencePixelCorrector(x_opt=64, y_opt=4)

# Load raw data
raw_data = preprocessor.load_exposure(EXPOSURE_PATH)

# Apply reference pixel correction
corrected_data = corrector.correct(raw_data)

# Save processed data
np.save(PROCESSED_DATA_PATH, corrected_data)
```

### Batch Processing

Process multiple exposures in a loop:

```python
import glob

# Find all FITS files in a directory
fits_files = glob.glob("/path/to/data/*.fits")

# Initialize predictor once
predictor = Predictor(model_path=MODEL_PATH, config=config)

# Process each file
results = {}
for fits_file in fits_files:
    print(f"Processing {fits_file}...")
    anomalies = predictor.predict(
        exposure_path=fits_file,
        output_dir=f"outputs/{os.path.basename(fits_file).replace('.fits', '')}"
    )
    results[fits_file] = anomalies
    print(f"  Found {len(anomalies)} anomalies")

# Save summary
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=4, default=to_builtin)
```

### Training a New Model

If you need to train a custom model on your data:

```python
from ohana import DatasetCreator, ModelTrainer

# Create synthetic training data
creator = DatasetCreator(config_path="configs/creator_config.yaml")
creator.create_dataset(start_index=0, end_index=10)  # Create 10 exposures

# Train the model
trainer = ModelTrainer(
    config_path="configs/creator_config.yaml",
    output_dir="./trained_models"
)
trainer.train(
    epochs=20,
    batch_size=2,
    learning_rate=1e-4,
    val_split=0.2
)
```

### Analyzing Detection Statistics

Analyze the types and properties of detected anomalies:

```python
import pandas as pd

# Load detections
with open(DETECTIONS_PATH, 'r') as f:
    detections = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(detections)

# Summary statistics
print("Detection Summary:")
print(df['type'].value_counts())
print("\nIntensity Statistics:")
print(df.groupby('type')['max_intensity'].describe())

# Filter high-confidence detections
if 'confidence' in df.columns:
    high_conf = df[df['confidence'] > 0.8]
    print(f"\nHigh confidence detections: {len(high_conf)}/{len(df)}")

# Spatial distribution
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
for anomaly_type in df['type'].unique():
    subset = df[df['type'] == anomaly_type]
    centroids = np.array(subset['centroid'].tolist())
    ax.scatter(centroids[:, 1], centroids[:, 0], 
               label=anomaly_type, alpha=0.6, s=20)
ax.set_xlabel('X pixel')
ax.set_ylabel('Y pixel')
ax.set_title('Spatial Distribution of Anomalies')
ax.legend()
ax.invert_yaxis()  # Match image coordinates
plt.show()
```

## Troubleshooting

### Common Issues and Solutions

#### Memory Issues

If you encounter out-of-memory errors:

```python
# Reduce batch size for model inference
predictor = Predictor(model_path=MODEL_PATH, config=config)
predictor.batch_size = 1  # Process one patch at a time

# For very large exposures, process in chunks
# This requires modifying the predictor code or processing sub-regions
```

#### No Detections Found

If the predictor finds no anomalies:

```python
# Lower detection thresholds
config = DetectorConfig(
    sigma_threshold=3.0,  # More sensitive
    cosmic_ray_min_intensity=20.0,  # Lower threshold
    min_anomaly_pixels=1  # Allow single-pixel events
)

# Check if data is being loaded correctly
data = np.load(PROCESSED_DATA_PATH)
print(f"Data shape: {data.shape}")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
print(f"Median value: {np.median(data):.2f}")
```

#### Too Many False Positives

If detecting too many spurious events:

```python
# Increase thresholds
config = DetectorConfig(
    sigma_threshold=6.0,  # Less sensitive
    cosmic_ray_min_intensity=50.0,  # Higher threshold
    min_anomaly_pixels=3,  # Require larger events
    snowball_circularity_threshold=0.8  # Stricter shape requirement
)
```

### Debugging Tips

Enable verbose logging to diagnose issues:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# The predictor will now output detailed logs
predictor = Predictor(model_path=MODEL_PATH, config=config)
```

### Performance Optimization

For faster processing:

```python
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable mixed precision (if using GPU)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Process multiple files in parallel
from concurrent.futures import ProcessPoolExecutor

def process_file(fits_file):
    predictor = Predictor(model_path=MODEL_PATH, config=config)
    return predictor.predict(exposure_path=fits_file)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, fits_files))
```
