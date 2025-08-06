import os
import h5py
import yaml
import json
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import sys

from ohana.training.injections import (
    generate_baseline_ramp,
    inject_cosmic_ray,
    inject_rtn,
    inject_snowball,
)


def worker(args):
    """
        Intializes a fresh DataSetCreator so the process has its own config and logger
        then reruns the per exposure pipeline
        Arguments:
            args (tuple[str, str, int, bool]): packed tuple containing:
                config_path (str): path to the yaml configuration
                detector_id (str): key into cfg['gain_library']
                index (int): exposure index (used in filenames/metadata)
                inject_events (bool): if True: inject anomalies; False: baseline only
        Returns:
            None
    """
    # Unpack the argument tuple from Pool.imap or map
    config_path, detector_id, index, inject_events = args

    # Create a new creator per process
    creator = DataSetCreator(config_path)

    # Run the single exposure sim
    creator._process_single_exposure(detector_id, index, inject_events)


class DataSetCreator:
    """
        Generate synthetic detector data, inject events, tile into patches, and save by
        reading the config params, creating the up the ramp data cubes per exposure,
        injecting the events, and saving the datacube to a hdf5
    """

    def __init__(self, config_path: str):
        """
            Arguments:
                config_path (str): path to a YAML file with keys including
                    output_dir (str)
                    image_shape (List[int, int])
                    num_frames (int)
                    saturation_dn (float)
                    patch_size (List[int, int])
                    overlap (int)
                    num_exposure_events (int)
                    num_baseline_exposures (int)
                    num_workers (int)
                    gain_library (Dict[str, float])
                    dark_current_range (List[float, float])
                    read_noise_range (List[float, float])
                    gaussian_noise_range (List[float, float])
                    cosmic_ray_per_exp_range (List[int, int])
                    cosmic_ray_charge_range (List[float, float])
                    snowballs_per_exp_range (List[int, int])
                    snowball_radius_range (List[float, float])
                    snowball_halo_amplitude_ratio_range (List[float, float])
                    snowball_halo_decay_scale_range (List[float, float])
                    rtn_per_exp_range (List[int, int])
                    rtn_offset_range (List[float, float])
                    rtn_period_range (List[float, float])  # frames
                    rtn_duty_fraction_range (List[float, float])
            Attributes:
                config_path (str): stored config file path
                cfg (dict[str, Any]): parsed YAML configuration dictionary
                output_dir (str): base folder for patches and metadata
                logger (logging.Logger): per-process logger writing to file + console
        """
        # Grab the config path
        self.config_path = config_path

        # Open the config file
        with open(config_path, "r") as f:
            # Save its contents
            self.cfg = yaml.safe_load(f)

        # Have outpit directory be the one from the config
        self.output_dir = self.cfg["output_dir"]

        # Make the output directory if it doesnt exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up the logger (nothing import here, not my code)
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
            Create a 2D data cube (T, H, W) with overlapping patches across H x W by
            using a sliding window with a stride of the overlap, and then extracting 
            the sub-volumes of the patch of each top-left coord of the patch
            Arguments:
                volume (np.ndarray): input cube with shape (T, H, W)
            Returns:
                list[tuple[np.ndarray, tuple[int, int]]]: list of (patch, (row, col))
        """
        # Unpack the spatial dimensions 
        _, H, W = volume.shape

        # Grab the patch width windoew size from the config
        patch_height, patch_width = self.cfg["patch_size"]

        # Grab the patch overlap from the config
        overlap = self.cfg["overlap"]

        # Creat the strides from the patch details and the overlap
        step_h, step_w = patch_height - overlap, patch_width - overlap

        # Initialize patches lists
        patches = []

        # Iterate through all heights (sliding window pt 1)
        for i in range(0, H - patch_height + 1, step_h):
            # Iterate through all widths (sliding window pt 2)
            for j in range(0, W - patch_width + 1, step_w):
                # Create the patch from the sliding window
                patch = volume[:, i : i + patch_height, j : j + patch_width]

                # Save the patches details to the patches list
                patches.append((patch, (i, j)))

        return patches

    def _validate_and_convert_range(self, key):
        """
            Validate the two-element numeric range in the config and return it's floats
            Arguments:
                key (str): config file key to read
            Returns:
                list(float): two-element list as floats
            Raises:
                ValueError: if the entry is not a two-element list of the vals are not nums
        """
        # Read from the config
        val = self.cfg.get(key)

        # Check if it is a two-element list
        if not isinstance(val, list) or len(val) != 2:
            raise ValueError(f"Configuration error in '{key}': Must be a list of two numbers [min, max]. Found: {val}")
        
        # Try to make the vals into floats
        try:
            return [float(v) for v in val]
        
        # Raise error if not
        except (ValueError, TypeError):
            raise ValueError(f"Configuration error in '{key}': Values must be convertible to numbers. Found: {val}")


    def _process_single_exposure(self, detector_id: str, index: int, inject_events: bool = True):
        """
            Simulate one exposure, and optinally injecting events by creating
            an up-the-ramp baseline, injecting anomalies using anomaly
            functions, subtracting the 0th read the up ramp, and saving 
            the exposure as hdf5 and its metadata as a json file
            Arguments:
                detector_id (str): key into gain library 
                index (int): exposure index to include in filenames/metadata
                inject_events (bool, optional): if True, simulate anomalies; if False,
                    produce baseline-only exposure (default is True)
            Returns:
                None      
        """
        # Exposure naming, include wheather an injection occured
        event_type = "events" if inject_events else "baseline"
        exposure_id = f"{detector_id}_{event_type}_{index:04d}"
        self.logger.info(f"Starting simulation for {exposure_id}")

        # Grab the random sim params from the config
        gain = self.cfg["gain_library"][detector_id]
        shape = tuple(self.cfg["image_shape"])
        num_frames = self.cfg["num_frames"]
        sat_dn = self.cfg["saturation_dn"]

        # Grab the ranges and convert them to the correct formatting
        dark_current_range = self._validate_and_convert_range("dark_current_range")
        read_noise_range = self._validate_and_convert_range("read_noise_range")
        gaussian_noise_range = self._validate_and_convert_range("gaussian_noise_range")

        # Sample specific values for this exposure with a uniform distribution
        dark_current = np.random.uniform(*dark_current_range)
        read_noise = np.random.uniform(*read_noise_range)
        gaussian_noise = np.random.uniform(*gaussian_noise_range)
        
        # Collect the metadata parameters
        params = {
            "gain": float(gain), "dark_current": float(dark_current), 
            "read_noise": float(read_noise), "gaussian_noise": float(gaussian_noise), 
            "injected_events": {"counts": {}, "details": []}
        }

        # Generate a baseline ramp
        ramps = generate_baseline_ramp(
            shape, num_frames, gain, sat_dn,
            dark_current, read_noise, gaussian_noise
        )

        # Inject the events if opted
        if inject_events:
            # Grab the cr range per exposure and create a random count
            cr_range = self._validate_and_convert_range("cosmic_ray_per_exp_range")
            num_crs = np.random.randint(low=cr_range[0], high=cr_range[1] + 1)
            
            # Grab the snowball range and gen a random num from the range
            sb_range = self._validate_and_convert_range("snowballs_per_exp_range")
            num_snowballs = np.random.randint(low=sb_range[0], high=sb_range[1] + 1)
            
            # Grab the rtn range and gen a random num from the range
            rtn_range = self._validate_and_convert_range("rtn_per_exp_range")
            num_rtn = np.random.randint(low=rtn_range[0], high=rtn_range[1] + 1)
            
            # Grab the cr charge range 
            cosmic_ray_charge_range = self._validate_and_convert_range("cosmic_ray_charge_range")

            # Grab the snowball radius range
            snowball_radius_range = self._validate_and_convert_range("snowball_radius_range")

            # Grab the rtn height range
            rtn_offset_range = self._validate_and_convert_range("rtn_offset_range")

            # Grab the rtn period range
            rtn_period_range = self._validate_and_convert_range("rtn_period_range")

            # Grab rtn duty fraction range
            rtn_duty_fraction_range = self._validate_and_convert_range("rtn_duty_fraction_range")
            
            # Grab the snowball amplitude range
            snowball_halo_amp_range = self._validate_and_convert_range("snowball_halo_amplitude_ratio_range")

            # Grab the snowball radial decay range
            snowball_halo_decay_range = self._validate_and_convert_range("snowball_halo_decay_scale_range")

            """Inject cosmic rays"""
            # Save the number of crs to be injected
            params["injected_events"]["counts"]["cosmic_rays"] = int(num_crs)
            
            # Iterate through every cr event
            for _ in range(num_crs):
                # Generate a random coord 
                position = tuple(int(p) for p in (np.random.randint(0, s) for s in shape))

                # Generate a random frame
                frame = int(np.random.randint(1, num_frames))

                # Generate a random charge
                charge_e = float(np.random.uniform(*cosmic_ray_charge_range))

                # Inject the event
                inject_cosmic_ray(ramps, position, frame, charge_e, gain, sat_dn)

                # Save the cr's metadata
                params["injected_events"]["details"].append({
                    "type": "cosmic_ray", 
                    "position": position, 
                    "frame": frame, 
                    "charge_e": charge_e
                })

            """Inject snowball"""
            # Save the total number of snowballs to be injected
            params["injected_events"]["counts"]["snowballs"] = int(num_snowballs)

            # Iterate through every snowball
            for _ in range(num_snowballs):
                # Generate a random center for the event
                center = tuple(int(c) for c in (np.random.randint(0, s) for s in shape))

                # Generate a random radius for the event
                radius = float(np.random.uniform(*snowball_radius_range))

                # Core charge has to be the saturation
                core_charge = sat_dn * gain

                # Generate a random impact frame
                impact_frame = int(np.random.randint(1, num_frames))
                
                # Generate a random amplitude radtio
                halo_amplitude_ratio = float(np.random.uniform(*snowball_halo_amp_range))

                # Generate a random rate of halo decay
                halo_decay_scale = float(np.random.uniform(*snowball_halo_decay_range))

                # Define a halo profile for the snowball !NOTE you can change me!!
                def halo_profile(d):
                    amplitude = core_charge * halo_amplitude_ratio
                    return amplitude * np.exp(-d / halo_decay_scale)

                # Inject the event
                inject_snowball(ramps, center, radius, core_charge, halo_profile, gain, sat_dn, impact_frame)
                
                # Save the snowballs metadata
                params["injected_events"]["details"].append({
                    "type": "snowball", 
                    "center": center, 
                    "radius": radius, 
                    "core_charge_e": core_charge, 
                    "impact_frame": impact_frame,
                    "halo_amplitude_ratio": halo_amplitude_ratio,
                    "halo_decay_scale": halo_decay_scale
                })

            """Inject Random Telegraph Noise"""
            # Save the total number of rtns
            params["injected_events"]["counts"]["rtn"] = int(num_rtn)

            # Iterate through all of the rtn
            for _ in range(num_rtn):
                # Genreate a random position
                position = tuple(int(p) for p in (np.random.randint(0, s) for s in shape))

                # Genreate a random height for the rtn
                offset_e = float(np.random.uniform(*rtn_offset_range))
                
                # Draw period (T) and duty fraction (f) for each RTN event
                period = int(np.random.uniform(*rtn_period_range))
                duty_fraction = float(np.random.uniform(*rtn_duty_fraction_range))

                # Inject the event
                inject_rtn(ramps, position, offset_e, period, duty_fraction, gain, sat_dn)
                
                # Save the rtn metadata
                params["injected_events"]["details"].append({
                    "type": "rtn", 
                    "position": position, 
                    "offset_e": offset_e, 
                    "period_frames": period,
                    "duty_fraction": duty_fraction
                })
            
            # Log girlie's success!
            self.logger.info(f"  Injecting: {num_crs} CRs, {num_snowballs} Snowballs, {num_rtn} RTN pixels.")

        # Subtract the 0th read
        diff_ramps = ramps[1:] - ramps[0]

        # Create the patches
        patches = self._extract_patches(diff_ramps)

        # Create a filename for the current exp
        h5_filename = f"{exposure_id}_patches.h5"
        h5_path = os.path.join(self.output_dir, h5_filename)
        
        # Save all of the patches metadata
        patch_metadata = []

        # Open the patches filepath
        with h5py.File(h5_path, "w") as hf:
            # Iterate through all of the patch data
            for patch_idx, (patch_data, (r, c)) in enumerate(patches):
                # Create a name for the patch
                dset_name = f"patch_{patch_idx:04d}"

                # Save the patches pixel details
                hf.create_dataset(dset_name, data=patch_data, compression="gzip")
                
                # Save the patches metadata
                patch_metadata.append({"id": dset_name, "coords": [int(r), int(c)]})

        # Log success
        self.logger.info(f"Saved {len(patches)} patches to {h5_filename}")

        # Generate metadata for the current exposure
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

        # Create the metadata directory structure: output_dir/metadata/detector_id/
        meta_dir = os.path.join(self.output_dir, "metadata", detector_id)

        # Make the metadata directory
        os.makedirs(meta_dir, exist_ok=True)
        
        # Create a filename using the exposure id
        meta_filename = f"{exposure_id}.json"

        # Create the metadata path
        meta_path = os.path.join(meta_dir, meta_filename)

        # Open the metadata file and save the current exposures data
        with open(meta_path, 'w') as f:
            json.dump(exposure_meta, f, indent=4)
        
        # Log girlie
        self.logger.info(f"  Saved metadata to {os.path.join('metadata', detector_id, meta_filename)}")

        return

    def create_dataset(self, start, end):
        """
            Create exposures for a certain index range in a slurm array script, 
            where for each index, two jobs are queued per detectors, one 
            baseline and one with events injected
            Arguments:
                start (int): first exposure index 
                end (int): last exposure index
            Returns:
                None
            Note:
                * contigent on YOU USING A SLURM ARRAY!!!
        """
        # Logger header
        self.logger.info("="*50)
        self.logger.info(f"Starting dataset creation for index range: {start} to {end}")
        self.logger.info("="*50)

        # Grab the wanted number of workers
        num_workers = self.cfg["num_workers"]
        
        # Jobs list which will hold of the tasks for each worker (cute!!)
        jobs = []

        # Iterate through the range provided by miss slurm
        for i in range(start, end + 1):
            # Iterate through all of the detector ids
            for detector_id in self.cfg["gain_library"].keys():
                # Generate a baseline exposure
                jobs.append((self.config_path, detector_id, i, True)) 
                
                # Generate an exposure with events
                jobs.append((self.config_path, detector_id, i, False))

        # Log that you gave the workers jobs!! thank god
        self.logger.info(f"Created {len(jobs)} total jobs for this worker to process.")

        # Check to see if there are multiple works
        if num_workers > 1:
            # Using pooling to spread the tasks
            with Pool(processes=num_workers) as pool:
                # Call the process single exposure wrapper so the workers dont fight
                pool.starmap(self._process_single_exposure_wrapper, jobs)
        else:
            # Serial procesing (just for debugging, ignore)
            for job_args in jobs:
                self._process_single_exposure_wrapper(*job_args)
        
        # Log girlie
        self.logger.info(f"Finished processing all jobs for index range: {start} to {end}")

    def _process_single_exposure_wrapper(self, detector_id, index, is_baseline):
        """
            Adapter for Pool.starmap to call the method with NO FIGHTING
            Arguments:
                detector_id (str): detector key as in the gain library
                index (int): exposure index
                is_baseline (bool): if True, create baseline; if False, inject events
            Returns:
                None
            Notes:
                * multiprocessing spawns fresh processes; ensure any required state is 
                  available from self or the reloaded method
        """
        self._process_single_exposure(detector_id, index, is_baseline)
