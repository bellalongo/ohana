## Developer Guide

This guide is for developers or advanced users who want to run ohana directly from the source (cloned repository) and modify settings or code for custom use cases. It outlines how to set up the project, run the included tutorial pipeline, and adjust parameters for troubleshooting.

### Getting Started

1. **Clone the Repository**: Begin by cloning the ohana repository from GitHub.

```bash
git clone https://github.com/bellalongo/ohana.git
```

2. **Change into the project directory**:

```bash
cd ohana
```

3. **Navigate to the docs Directory**: The repository may include example configurations and tutorial scripts in the `docs/` folder.

```bash
cd docs
```

(Ensure you have the necessary dependencies installed. If you haven't installed the package via pip in editable mode, you might want to run `pip install -r ../requirements.txt` to set up the environment. A GPU-enabled PyTorch installation is recommended for training.)

4. **Prepare an Exposure File**: If you plan to test the anomaly detection on a real exposure, copy that FITS file into the `docs/` directory (or note its path for later). If you don't have real data, you can use ohana's simulation capabilities:
   - You can generate a small synthetic exposure by running the data creation step on a limited range (e.g., one exposure as in the example below). 
   - Alternatively, use one of the simulated patch files to reconstruct a full image (this requires additional code, so using a real or provided sample FITS is simpler).

5. **Open the Tutorial Script**: The docs directory contains a tutorial script (e.g. `tutorial.py` or a Jupyter notebook) that demonstrates the end-to-end usage of ohana. Open `tutorial.py` in a text editor. Inside, you will find placeholders for configuration and file paths. Update the following:

   - **Config Path**: Ensure the script is pointing to the correct YAML config file for data generation (e.g. `configs/creator_config.yaml`). If your config file is elsewhere or named differently, update the path.
   
   - **Exposure Path**: Set the path to the FITS exposure you want to analyze. For example:
   ```python
   EXPOSURE_PATH = "my_exposure.fits"
   ```
   If you generated a synthetic exposure and saved it as a FITS or .npy file, provide that path.
   
   - **Model Path**: If you have a pre-trained model you want to use for detection, update the path to point to the .pth file. If you plan to train a model as part of the tutorial, you can set this after training or leave it as a default until the model is trained.

6. **Set Run Parameters**: The tutorial may include flags to control whether to perform each step. For example, you might see a variable like:

```python
fresh_start = True
```

Set this to `True` for the first run (to generate data and train the model). The script will then create the synthetic dataset and train a new model. On subsequent runs, you can set it to `False` to reuse existing data and models, skipping directly to the detection phase.

Similarly, ensure any other parameters in the script are appropriately set:
- If there's an `apply_smoothing` or similar option (for preprocessing frames), decide based on your data. Generally, you can leave this `False` for clarity unless your data is very noisy and you want to smooth it.
- If the script requires specifying device (CPU/GPU), it typically will auto-detect GPU. Just ensure your environment is set up for CUDA if available.

7. **Run the Tutorial Pipeline**: Execute the tutorial script to run the full pipeline:

```bash
python tutorial.py
```

This will perform, in sequence, any required data simulation, model training, and anomaly detection on the target exposure, depending on how `tutorial.py` is set up. Monitor the console output for progress:
- During data creation, you should see logs about simulating exposures and saving patch files.
- During training, you'll see epoch-by-epoch loss and accuracy printed (with progress bars via `tqdm`).
- During prediction, the script will log the steps of loading data, preprocessing, and running detection algorithms.

After the script finishes, you should have new output directories (for dataset, model, and predictions) in the `docs/` folder.

### Adjustments for Troubleshooting

If you encounter issues (e.g., missed detections, too many false positives, or training not converging), you can adjust several parameters in the configuration or code. Below are some common adjustments:

#### In tutorial.py (or the Tutorial Notebook)

- **File Paths and Modes**: Ensure that the paths for config, model, and exposure are correct as described above. If you want to skip time-consuming steps, you can modify flags:
  - For example, if the tutorial script has separate steps for data creation, training, and prediction, you can comment out or toggle off the data creation and training steps once you have `data/processed/` and `trained_models/best_model_unet3d.pth` ready. This saves time when iterating on detection tweaks.
  - Conversely, if you need to regenerate data with new settings, set the script to do so (e.g., `fresh_start = True` as mentioned).

- **Data/Model Reuse**: The script might be designed to pick up existing artifacts. Check if it loads existing patch datasets or models when `fresh_start` is `False`. If you change the configuration significantly (say, add more anomaly injections or change image size), it's safest to delete or move old outputs (in `data/processed/` or `trained_models/`) and run fresh to avoid mismatches.

#### In creator_config.yaml

The YAML file governs the synthetic dataset generation. Adjusting these values can help create a training set that better represents your use case:

- **Image Shape** (`image_shape`): Can be reduced if memory is a concern or if you want to simulate a smaller section of the detector. For quick tests, you might use a smaller image (e.g. 512×512) to speed up simulation and training.

- **Number of Frames** (`num_frames`): If your real exposures have fewer frames than the default (450), you can lower this to simulate shorter ramps. However, ensure this still covers the typical duration where anomalies would appear.

- **Number of Exposures**: Increase `num_exposure_events` and `num_baseline_exposures` if you need a larger training dataset. By default it might simulate 1 of each; generating more (e.g. 5–10 exposures with anomalies) can improve model training at the cost of longer simulation time and more disk space.

- **Injection Density**: The ranges for `cosmic_ray_per_exp_range`, `snowballs_per_exp_range`, and `rtn_per_exp_range` determine how many anomalies are injected per exposure. If, for example, you find the model isn't catching very crowded cosmic ray scenarios, you could raise the upper bound of `cosmic_ray_per_exp_range` to expose the model to more crowded conditions. Conversely, if the synthetic data seems unrealistically crowded with events, you can lower these numbers.

- **Anomaly Types**: If you are only interested in, say, cosmic rays and not RTN or snowballs, you can set `injection_type` to `"cosmic_ray"` to generate exposures with only that anomaly. Make sure `num_classes` matches the number of classes you expect (e.g. 2 classes if only cosmic rays vs background).

- **Patch Parameters**: `patch_size` and `overlap` affect how the detector images are broken into patches. The defaults (256 px with 32 px overlap) are a balance between training efficiency and capturing enough context. If your GPU memory is very limited, you might reduce `patch_size` (e.g. 128) to make training patches smaller (you'll also need to adjust the model if you expect a different input shape). Generally, stick with the default unless you encounter memory issues.

- **Physical Parameters**: The various `*_range` fields (charge, intensity, etc.) influence how realistic the injections are. For instance, `cosmic_ray_charge_range` could be widened if you suspect your cosmic rays have higher energy deposition than simulated. The same goes for `snowball_halo_amplitude_ratio_range`, etc. Tuning these requires domain knowledge; the provided defaults are reasonable for H2RG detectors, but feel free to experiment if needed.

#### In config.py (DetectorConfig thresholds)

If the anomaly detection (especially the rule-based part) is not performing to your liking, you can tweak the thresholds in the DetectorConfig dataclass:

- **Cosmic Ray Thresholds**: Lower `cosmic_ray_min_intensity` to catch fainter cosmic rays (at the risk of more false positives from noise). For example, changing from 40 DN to 30 DN will flag smaller jumps as candidates. Also consider `min_anomaly_pixels`: if single-pixel events are often noise in your data, set this to 2 so that only events affecting at least 2 adjacent pixels count as a cosmic ray hit.

- **RTN Criteria**: If random telegraph noise events are not being detected, you might reduce `rtn_min_transitions` to 1 (to catch pixels that toggle only once, though be cautious as this can also catch cosmic-ray-like jumps). Adjust `rtn_fit_quality_threshold` upward if you want a stricter fit requirement (or downward to be more lenient in classifying something as RTN).

- **Snowball Criteria**: The default requires a snowball to have area ≥ 75 pixels and fairly round shape. If your data shows smaller "snowball" events, lower `snowball_min_area` or `snowball_min_intensity`. If too many false snowballs are detected, you could increase these or the circularity threshold.

- **Sigma Threshold**: The `sigma_threshold` (5.0 by default) is used in initial noise thresholding. Reducing this to 4.0 or 3.5 will make the algorithm more sensitive to small deviations at the cost of potentially more false positives. This can help if anomalies are barely above the noise.

Most of these settings are exposed in the DetectorConfig for convenience. You can create a custom DetectorConfig in your own code and pass it to the Predictor if running detection manually. If using the CLI, you might need to modify `config.py` defaults directly or adjust the code in `predictor.py` to load thresholds from a file if desired.

#### In train_model.py

Model training can sometimes require tuning of hyperparameters:

- **Epochs**: The default (20 epochs, for example) might not fully converge, or might be overkill depending on dataset size. Monitor the training and validation loss/accuracy in the console. If the model hasn't plateaued by the last epoch, consider increasing `--epochs` (e.g., to 30 or 40). If it converges early or you're just trying a quick run, fewer epochs (10) might suffice.

- **Learning Rate**: The default learning rate is 1e-4 for the Adam optimizer. If you see training loss oscillating or not decreasing at all, you might try a lower rate (e.g. 5e-5). Conversely, if training is very slow to learn anything, a slightly higher rate (2e-4) could be tried, but be cautious as too high can destabilize training.

- **Batch Size**: If you have a powerful GPU with more memory, increasing `--batch_size` can speed up training by utilizing more patches per iteration. If you run out of memory or are on CPU, reduce batch size to 1 (at the cost of slower training and noisier gradients).

- **Validation Split**: The `--val_split` (default 0.2 or 20%) determines how much of the dataset is held out for validation. If you have very limited data, you might use a smaller split (10%) to train on more patches, but ensure you still have enough to validate model performance.

- **Normalization**: In the training loop, the code normalizes each patch's pixel values between its min and max. This is usually fine. If you have issues with the dynamic range, ensure that extremely bright outliers (e.g. saturated pixels) are handled – they could be affecting training. You could implement clipping or a different normalization if needed.

#### In predict_cosmic_rays.py (Detection Algorithm Tweaks)

For advanced users comfortable editing the code, the rule-based algorithms have a few hard-coded factors you might adjust:

- In the cosmic ray detector (`src/ohana/predict/predict_cosmic_rays.py`), the initial intensity screen uses `0.7 * cosmic_ray_min_intensity` as a threshold (i.e., it's a bit generous initially, then relies on step fitting to confirm). If you want to be stricter right away, you could change this factor to 1.0 to require the full threshold at screening, or lower it further to 0.5 to catch even weaker candidates for fitting.

- The persistence criterion for cosmic rays currently requires an anomaly to persist at least 3 frames (`persistence >= 3`). If your exposures are short or some cosmic rays appear in fewer frames, you might reduce this to 2 – but be aware this may include single-frame noise fluctuations.

- Similar files exist for `predict_rtn.py` and `predict_snowball.py` with their own logic. For instance, the telegraph noise detector might have thresholds on allowed amplitude or uses a certain window of frames to fit states. You can adjust those if you find the need (refer to the comments in those files for guidance).

These code-level changes shouldn't be needed in most cases, but they're available if fine-grained control is required beyond the config parameters.

### Output Directory Structure

After running the tutorial pipeline (with `docs/tutorial.py`), your `docs/` directory will be populated with several output folders containing the results:

```
docs/
├── data/
│   └── processed/
│       ├── E000_18220_patches.h5           # Example patch file for an exposure
│       ├── E000_18220CASE_patches.h5       # Additional patches (if multiple detectors or gain settings)
│       └── metadata/
│           ├── 18220_SCA/
│           │   ├── E000.json               # Metadata for exposure E000 on detector 18220_SCA
│           │   └── E001.json               # Metadata for exposure E001 on detector 18220_SCA (if generated)
│           └── ... (other detector IDs subfolders with their JSON files)
├── trained_models/
│   ├── best_model_unet3d.pth               # Best model weights saved during training
│   └── training_history_unet3d.json        # Training history (loss and accuracy per epoch)
├── prediction_outputs/
│   ├── raw/
│   │   └── my_exposure_raw.npy             # Cached raw data cube (after loading from FITS)
│   ├── processed/
│   │   └── my_exposure_processed.npy       # Processed data cube (after reference pixel correction)
│   ├── temporal/
│   │   └── my_exposure_temporal.npz        # Temporal features for each pixel (numpy archive)
│   ├── prediction_mask.npy                 # 2D array mask of detected anomalies
│   └── detections.json                     # List of detected anomalies with details
```

(In the above, "my_exposure" would be replaced by the actual base name of your FITS file. For synthetic data, it might use a naming scheme combining an index and detector ID as shown.)

**Understanding the Output Structure**:
- The `data/processed/` directory contains the training dataset patches and metadata generated by `ohana-create-training`. Each HDF5 file (with `_patches.h5` suffix) holds the image patches for one exposure. The metadata JSONs describe the parameters and patch locations for each exposure.
- The `trained_models/` folder holds your trained model. If you trained multiple times, you may have multiple files (consider renaming or moving old ones to avoid confusion).
- The `prediction_outputs/` directory is created when you run the predictor. It organizes intermediate and final results. The subfolders `raw/`, `processed/`, and `temporal/` are caching mechanisms: if you run detection on the same exposure again, ohana will load these rather than recompute everything. The final outputs of interest are `prediction_mask.npy` and `detections.json` as described in the main README.

With this structure, you can inspect and utilize the results. For example, open the JSON to see how many cosmic rays were found and their properties, or load the `prediction_mask.npy` in Python to create a quick overlay on the median image. The metadata and intermediate files can usually be ignored unless you are debugging or extending the pipeline.

By adjusting configurations and parameters as above, you should be able to improve ohana's performance on your specific data and needs. Happy anomaly hunting!
