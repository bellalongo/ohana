import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from ohana.training.injections import (
    generate_baseline_ramp,
    inject_cosmic_ray,
    inject_rtn,
    inject_snowball,
)


class TrainingSetCreator:
    """Generate synthetic spatio‑temporal detector data, patch it, and save to HDF5."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.output_dir = self.cfg["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.patch_size = tuple(self.cfg["patch_size"])  # (H, W)
        self.overlap = self.cfg["overlap"]
        self.num_workers = self.cfg.get("num_workers", mp.cpu_count())
        self.num_samples = self.cfg["num_samples"]

    # Patch extraction
    def _extract_patches(self, volume: np.ndarray):
        """Return list of (patch, (row_start, col_start))."""
        T, H, W = volume.shape
        ph, pw = self.patch_size
        step_h, step_w = ph - self.overlap, pw - self.overlap

        patches = []
        for i in range(0, H - ph + 1, step_h):
            for j in range(0, W - pw + 1, step_w):
                patches.append((volume[:, i : i + ph, j : j + pw], (i, j)))
        return patches

    # Synthetic event simulation
    def _simulate_event(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        shape = self.cfg["image_shape"]
        num_frames = self.cfg["num_frames"]
        gain = self.cfg["gain"]
        sat_dn = self.cfg["saturation_dn"]

        dark_current = np.random.uniform(*self.cfg["dark_current_range"])
        read_noise = np.random.uniform(*self.cfg["read_noise_range"])
        extra_noise = np.random.uniform(*self.cfg["extra_noise_range"])

        ramps = generate_baseline_ramp(
            shape,
            num_frames,
            gain,
            sat_dn,
            dark_current,
            read_noise,
            extra_noise,
        )

        inj_type = self.cfg["injection_type"]
        if inj_type == "cosmic_ray":
            position = tuple(np.random.randint(0, s) for s in shape)
            frame = np.random.randint(0, num_frames)
            charge_e = np.random.uniform(*self.cfg["cosmic_ray_charge_range"])
            inject_cosmic_ray(ramps, position, frame, charge_e, gain, sat_dn)

        elif inj_type == "rtn":
            position = tuple(np.random.randint(0, s) for s in shape)
            offset_e = np.random.uniform(*self.cfg["rtn_offset_range"])
            switch_frames = sorted(
                np.random.choice(range(num_frames), size=5, replace=False)
            )
            inject_rtn(ramps, position, offset_e, switch_frames, gain, sat_dn)

        elif inj_type == "snowball":
            center = tuple(np.random.randint(0, s) for s in shape)
            radius = np.random.uniform(*self.cfg["snowball_radius_range"])
            core_charge = np.random.uniform(*self.cfg["snowball_core_charge_range"])
            impact_frame = np.random.randint(0, num_frames)

            def halo_profile(d):
                return np.exp(-d)

            inject_snowball(
                ramps,
                center,
                radius,
                core_charge,
                halo_profile,
                gain,
                sat_dn,
                impact_frame,
            )

        return ramps

    # Per‑sample workflow
    def _process_single(self, idx: int):
        event_id = f"event_{idx:06d}"
        volume = self._simulate_event(seed=None)
        patches = self._extract_patches(volume)

        meta = {"event_id": event_id, "shape": volume.shape, "patches": []}
        h5_path = os.path.join(self.output_dir, f"{event_id}.h5")

        with h5py.File(h5_path, "w") as h5f:
            for patch, (i, j) in patches:
                dset = f"patch_{i}_{j}"
                h5f.create_dataset(dset, data=patch, compression="gzip")
                meta["patches"].append({"id": dset, "coords": [i, j]})

        return meta

    # Public API
    def create_training_set(self):
        indices = list(range(self.num_samples))
        metadata = {}

        with mp.Pool(processes=self.num_workers) as pool:
            for meta in tqdm(pool.imap(self._process_single, indices), total=self.num_samples):
                metadata[meta["event_id"]] = meta

        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# ----------------------------------------------------------------------
# CLI entry‑point
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic training‑set generator")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    TrainingSetCreator(args.config).create_training_set()


if __name__ == "__main__":
    main()
