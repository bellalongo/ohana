import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

# Configure logging
logger = logging.getLogger(__name__)

class OhanaDataset(Dataset):
    """
    PyTorch Dataset for loading ohana simulation patches.

    This class scans a directory for metadata files, links them to HDF5 patch
    files, and prepares labeled training samples. A patch is labeled based on
    the type of anomaly injected within its spatial boundaries.
    """

    def __init__(self, data_dir, patch_size, class_map=None, transform=None):
        """
        Args:
            data_dir (str): The root directory containing the processed data
                            (e.g., './data/processed').
            patch_size (list or tuple): The [height, width] of the patches.
            class_map (dict, optional): A mapping from event type string to
                                        integer label. Defaults to a standard map.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.data_dir = data_dir
        self.patch_size = tuple(patch_size)
        self.transform = transform
        
        if class_map is None:
            self.class_map = {
                "background": 0,
                "cosmic_ray": 1,
                "snowball": 2,
                "rtn": 3,
            }
        else:
            self.class_map = class_map
            
        self.samples = []
        self._create_sample_list()

    def _create_sample_list(self):
        """
        Scans the data directory, parses metadata, and creates a master list
        of all patches with their corresponding labels.
        """
        logger.info("Creating sample list from metadata...")
        metadata_root = os.path.join(self.data_dir, "metadata")
        if not os.path.isdir(metadata_root):
            raise FileNotFoundError(f"Metadata directory not found at {metadata_root}")

        for detector_id in sorted(os.listdir(metadata_root)):
            detector_path = os.path.join(metadata_root, detector_id)
            if not os.path.isdir(detector_path):
                continue
            
            for meta_file in sorted(os.listdir(detector_path)):
                if not meta_file.endswith(".json"):
                    continue
                
                meta_path = os.path.join(detector_path, meta_file)
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                h5_path = os.path.join(self.data_dir, metadata["h5_file"])
                if not os.path.exists(h5_path):
                    logger.warning(f"HDF5 file {h5_path} not found for metadata {meta_file}. Skipping.")
                    continue

                # Create a quick lookup for anomaly locations
                anomalies = []
                if metadata["parameters"]["injected_events"]["details"]:
                    for event in metadata["parameters"]["injected_events"]["details"]:
                        pos_key = "position" if "position" in event else "center"
                        anomalies.append({
                            "type": event["type"],
                            "pos": tuple(event[pos_key])
                        })

                # For each patch, determine its label
                for patch_info in metadata["patch_info"]["patches"]:
                    patch_id = patch_info["id"]
                    patch_coords = tuple(patch_info["coords"]) # (top, left)
                    
                    label = "background" # Default label
                    
                    # Check if any anomaly falls within this patch
                    for anomaly in anomalies:
                        ay, ax = anomaly["pos"]
                        py, px = patch_coords
                        ph, pw = self.patch_size
                        
                        if (py <= ay < py + ph) and (px <= ax < px + pw):
                            label = anomaly["type"]
                            break # Label with the first anomaly found
                    
                    self.samples.append({
                        "h5_path": h5_path,
                        "patch_id": patch_id,
                        "label": self.class_map[label]
                    })
        
        logger.info(f"Found {len(self.samples)} total patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches a single labeled patch from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_info = self.samples[idx]
        h5_path = sample_info["h5_path"]
        patch_id = sample_info["patch_id"]
        label = sample_info["label"]

        with h5py.File(h5_path, 'r') as hf:
            patch_data = hf[patch_id][:] # Reads the data into memory

        # Ensure data is in (T, H, W) format and float32
        patch_tensor = torch.from_numpy(patch_data.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)

        sample = {"patch": patch_tensor, "label": label_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_class_weights(self):
        """
        Computes class weights for handling class imbalance.
        Should be used with a weighted loss function.
        
        Returns:
            torch.Tensor: A tensor of weights for each class.
        """
        class_counts = np.zeros(len(self.class_map))
        for sample in self.samples:
            class_counts[sample["label"]] += 1
        
        total_samples = len(self.samples)
        
        # weight = total_samples / (num_classes * count)
        weights = total_samples / (len(self.class_map) * class_counts)
        
        # Handle cases where a class might not be present
        weights[np.isinf(weights)] = 0
        
        return torch.from_numpy(weights.astype(np.float32))

