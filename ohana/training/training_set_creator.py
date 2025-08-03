import os
import h5py
import yaml
import json
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import sys

from ohana.training.injections import (
    generate_baseline_ramp,
    inject_cosmic_ray,
    inject_rtn,
    inject_snowball,
)

# --- Multiprocessing Worker Function ---
def worker(args):
    """
    Worker function for the multiprocessing pool. It initializes a DataSetCreator
    instance and calls the processing method for a single exposure.
    """
    config_path, detector_id, index, inject_events = args
    creator = DataSetCreator(config_path)
    # Each worker returns the metadata for the exposure it processed.
    return creator._process_single_exposure(detector_id, index, inject_events)


class DataSetCreator:
    """Generate synthetic detector data, inject events, and create patched datasets."""

    def __init__(self, config_path: str):
        """Initializes the creator with configuration from a YAML file."""
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.output_dir = self.cfg["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # --- Set up logging ---
        self.logger = logging.getLogger(f"DataSetCreator_{os.getpid()}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            log_file = os.path.join(self.output_dir, "simulation.log")
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(message)s'))
            self.logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(ch)

    def _extract_patches(self, volume: np.ndarray):
        """
        Extracts overlapping patches from a 3D data volume (T, H, W).
        """
        _, H, W = volume.shape
        ph, pw = self.cfg["patch_size"]
        overlap = self.cfg["overlap"]
        step_h, step_w = ph - overlap, pw - overlap

        patches = []
        for i in range(0, H - ph + 1, step_h):
            for j in range(0, W - pw + 1, step_w):
                patch = volume[:, i : i + ph, j : j + pw]
                patches.append((patch, (i, j)))
        return patches

    def _validate_and_convert_range(self, key):
        """
        Validates a range from the config and converts its values to floats.
        """
        val = self.cfg.get(key)
        if not isinstance(val, list) or len(val) != 2:
            raise ValueError(f"Configuration error in '{key}': Must be a list of two numbers [min, max]. Found: {val}")
        try:
            return [float(v) for v in val]
        except (ValueError, TypeError):
            raise ValueError(f"Configuration error in '{key}': Values must be convertible to numbers. Found: {val}")


    def _process_single_exposure(self, detector_id: str, index: int, inject_events: bool = True):
        """
        Simulates one full exposure, creates difference images, extracts patches,
        and saves them to an HDF5 file. Returns metadata for this exposure.
        """
        event_type = "events" if inject_events else "baseline"
        exposure_id = f"{detector_id}_{event_type}_{index:04d}"
        self.logger.info(f"Starting simulation for {exposure_id}")

        # --- 1. Set up random parameters ---
        gain = self.cfg["gain_library"][detector_id]
        shape = tuple(self.cfg["image_shape"])
        num_frames = self.cfg["num_frames"]
        sat_dn = self.cfg["saturation_dn"]

        dark_current_range = self._validate_and_convert_range("dark_current_range")
        read_noise_range = self._validate_and_convert_range("read_noise_range")
        gaussian_noise_range = self._validate_and_convert_range("gaussian_noise_range")

        dark_current = np.random.uniform(*dark_current_range)
        read_noise = np.random.uniform(*read_noise_range)
        gaussian_noise = np.random.uniform(*gaussian_noise_range)
        
        params = {
            "gain": float(gain), "dark_current": float(dark_current), 
            "read_noise": float(read_noise), "gaussian_noise": float(gaussian_noise), 
            "injected_events": {"counts": {}, "details": []}
        }

        # --- 2. Generate baseline ramp ---
        ramps = generate_baseline_ramp(
            shape, num_frames, gain, sat_dn,
            dark_current, read_noise, gaussian_noise
        )

        # --- 3. Inject events if requested ---
        if inject_events:
            cr_range = self._validate_and_convert_range("cosmic_ray_per_exp_range")
            num_crs = np.random.randint(low=cr_range[0], high=cr_range[1] + 1)
            
            sb_range = self._validate_and_convert_range("snowballs_per_exp_range")
            num_snowballs = np.random.randint(low=sb_range[0], high=sb_range[1] + 1)
            
            rtn_range = self._validate_and_convert_range("rtn_per_exp_range")
            num_rtn = np.random.randint(low=rtn_range[0], high=rtn_range[1] + 1)
            
            cosmic_ray_charge_range = self._validate_and_convert_range("cosmic_ray_charge_range")
            snowball_radius_range = self._validate_and_convert_range("snowball_radius_range")
            snowball_core_charge_range = self._validate_and_convert_range("snowball_core_charge_range")
            rtn_offset_range = self._validate_and_convert_range("rtn_offset_range")
            
            # --- Load new snowball halo parameter ranges ---
            snowball_halo_amp_range = self._validate_and_convert_range("snowball_halo_amplitude_ratio_range")
            snowball_halo_decay_range = self._validate_and_convert_range("snowball_halo_decay_scale_range")

            params["injected_events"]["counts"]["cosmic_rays"] = int(num_crs)
            for _ in range(num_crs):
                position = tuple(int(p) for p in (np.random.randint(0, s) for s in shape))
                frame = int(np.random.randint(1, num_frames))
                charge_e = float(np.random.uniform(*cosmic_ray_charge_range))
                inject_cosmic_ray(ramps, position, frame, charge_e, gain, sat_dn)
                params["injected_events"]["details"].append({
                    "type": "cosmic_ray", "position": position, "frame": frame, "charge_e": charge_e
                })

            params["injected_events"]["counts"]["snowballs"] = int(num_snowballs)
            for _ in range(num_snowballs):
                center = tuple(int(c) for c in (np.random.randint(0, s) for s in shape))
                radius = float(np.random.uniform(*snowball_radius_range))
                core_charge = float(np.random.uniform(*snowball_core_charge_range))
                impact_frame = int(np.random.randint(1, num_frames))
                
                # Randomize halo parameters for each snowball ---
                halo_amplitude_ratio = float(np.random.uniform(*snowball_halo_amp_range))
                halo_decay_scale = float(np.random.uniform(*snowball_halo_decay_range))

                # Create a unique halo profile function for this specific snowball
                def halo_profile(d):
                    amplitude = core_charge * halo_amplitude_ratio
                    return amplitude * np.exp(-d / halo_decay_scale)

                inject_snowball(ramps, center, radius, core_charge, halo_profile, gain, sat_dn, impact_frame)
                
                # Save the new halo parameters in the metadata
                params["injected_events"]["details"].append({
                    "type": "snowball", "center": center, "radius": radius, 
                    "core_charge_e": core_charge, "impact_frame": impact_frame,
                    "halo_amplitude_ratio": halo_amplitude_ratio,
                    "halo_decay_scale": halo_decay_scale
                })

            rtn_offset_range = self._validate_and_convert_range("rtn_offset_range")
            # Add these new ranges to your config
            rtn_period_range = self._validate_and_convert_range("rtn_period_range")
            rtn_duty_fraction_range = self._validate_and_convert_range("rtn_duty_fraction_range")

            params["injected_events"]["counts"]["rtn"] = int(num_rtn)
            for _ in range(num_rtn):
                position = tuple(int(p) for p in (np.random.randint(0, s) for s in shape))
                offset_e = float(np.random.uniform(*rtn_offset_range))
                
                # Draw period (T) and duty fraction (f) for each RTN event
                period = int(np.random.uniform(*rtn_period_range))
                duty_fraction = float(np.random.uniform(*rtn_duty_fraction_range))

                inject_rtn(ramps, position, offset_e, period, duty_fraction, gain, sat_dn)
                
                params["injected_events"]["details"].append({
                    "type": "rtn", 
                    "position": position, 
                    "offset_e": offset_e, 
                    "period_frames": period,
                    "duty_fraction": duty_fraction
                })
            
            self.logger.info(f"  Injecting: {num_crs} CRs, {num_snowballs} Snowballs, {num_rtn} RTN pixels.")

        diff_ramps = ramps[1:] - ramps[0]
        patches = self._extract_patches(diff_ramps)

        h5_filename = f"{exposure_id}_patches.h5"
        h5_path = os.path.join(self.output_dir, h5_filename)
        
        patch_metadata = []
        with h5py.File(h5_path, "w") as hf:
            for patch_idx, (patch_data, (r, c)) in enumerate(patches):
                dset_name = f"patch_{patch_idx:04d}"
                hf.create_dataset(dset_name, data=patch_data, compression="gzip")
                patch_metadata.append({"id": dset_name, "coords": [int(r), int(c)]})

        self.logger.info(f"  Saved {len(patches)} patches to {h5_filename}")

        exposure_meta = {
            "exposure_id": exposure_id,
            "parameters": params,
            "h5_file": h5_filename,
            "patch_info": {
                "patch_size": self.cfg["patch_size"],
                "overlap": self.cfg["overlap"],
                "patches": patch_metadata,
            }
        }
        return exposure_meta

    def create_dataset(self):
        """
        Main method to create the full dataset in parallel using a multiprocessing pool.
        """
        self.logger.info("="*50)
        self.logger.info("Starting dataset creation...")
        self.logger.info("="*50)

        num_event_exp = self.cfg["num_exposure_events"]
        num_baseline_exp = self.cfg["num_baseline_exposures"]
        num_workers = self.cfg["num_workers"]

        jobs = []
        for detector_id in self.cfg["gain_library"].keys():
            if detector_id == '18220_SCA':
                for i in range(num_event_exp):
                    jobs.append((self.config_path, detector_id, i, True))
                for i in range(num_baseline_exp):
                    jobs.append((self.config_path, detector_id, i, False))

        self.logger.info(f"Total jobs to process: {len(jobs)} with {num_workers} workers.")

        all_metadata = {}
        try:
            with mp.Pool(processes=num_workers) as pool:
                for meta in tqdm(pool.imap_unordered(worker, jobs), total=len(jobs), desc="Overall Progress"):
                    if meta:
                        all_metadata[meta["exposure_id"]] = meta
        except Exception as e:
            self.logger.error(f"A critical error occurred during multiprocessing: {e}")
            self.logger.error("Aborting dataset creation.")
            sys.exit(1)

        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(all_metadata, f, indent=4)
        
        self.logger.info(f"Saved aggregated metadata to {metadata_path}")
        self.logger.info("="*50)
        self.logger.info("Dataset creation complete.")
        self.logger.info("="*50)
