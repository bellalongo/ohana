
import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class OhanaDataset(Dataset):
    # ... (the rest of your OhanaDataset class from before is fine) ...
    # ... (__init__, _find_exposures, _get_patch_info, __len__, _create_target_mask) ...
    # ONLY the __getitem__ method needs this small change.

    def __init__(self, data_dir, patch_size, class_map=None, transform=None):
        self.data_dir = data_dir
        self.patch_size = tuple(patch_size)
        self.transform = transform
        
        if class_map is None:
            self.class_map = {"background": 0, "cosmic_ray": 1, "snowball": 2, "rtn": 3}
        else:
            self.class_map = class_map
            
        self.exposures = self._find_exposures()
        self.patch_info_cache = {}

    def _find_exposures(self):
        logger.info("Scanning for exposure metadata files...")
        exposures = []
        metadata_root = os.path.join(self.data_dir, "metadata")
        if not os.path.isdir(metadata_root):
            raise FileNotFoundError(f"Metadata directory not found at {metadata_root}")

        for detector_id in sorted(os.listdir(metadata_root)):
            detector_path = os.path.join(metadata_root, detector_id)
            if not os.path.isdir(detector_path): continue
            for meta_file in sorted(os.listdir(detector_path)):
                if meta_file.endswith(".json"):
                    exposures.append(os.path.join(detector_path, meta_file))
        
        logger.info(f"Found {len(exposures)} exposure metadata files.")
        return exposures

    def _get_patch_info(self, meta_path):
        if meta_path in self.patch_info_cache:
            return self.patch_info_cache[meta_path]
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        self.patch_info_cache[meta_path] = metadata
        return metadata

    def __len__(self):
        if not self.exposures: return 0
        try:
            metadata = self._get_patch_info(self.exposures[0])
            return len(self.exposures) * len(metadata['patch_info']['patches'])
        except (KeyError, IndexError):
            return 0 # Return 0 if metadata is malformed

    def _create_target_mask(self, patch_coords, all_events):
        ph, pw = self.patch_size
        py, px = patch_coords
        target_mask = np.zeros((ph, pw), dtype=np.uint8)
        event_priority = {'snowball': 3, 'rtn': 2, 'cosmic_ray': 1}
        for event in all_events:
            event_type = event['type']
            pos_key = "position" if "position" in event else "center"
            ey, ex = tuple(event[pos_key])
            if (py <= ey < py + ph) and (px <= ex < px + pw):
                rel_y, rel_x = ey - py, ex - px
                if event_priority.get(event_type, 0) > target_mask[rel_y, rel_x]:
                    target_mask[rel_y, rel_x] = self.class_map[event_type]
        return target_mask

    def __getitem__(self, idx):
        """
        Fetches a patch and adds a channel dimension for the 3D U-Net.
        """
        # Figure out which exposure and which patch within that exposure to load
        metadata = self._get_patch_info(self.exposures[0])
        num_patches_per_exp = len(metadata['patch_info']['patches'])
        
        exp_idx = idx // num_patches_per_exp
        patch_idx_in_exp = idx % num_patches_per_exp

        meta_path = self.exposures[exp_idx]
        metadata = self._get_patch_info(meta_path)
        
        patch_info = metadata['patch_info']['patches'][patch_idx_in_exp]
        h5_path = os.path.join(self.data_dir, metadata['h5_file'])
        patch_id = patch_info['id']
        
        with h5py.File(h5_path, 'r') as hf:
            patch_data = hf[patch_id][:].astype(np.float32)

        target_mask = self._create_target_mask(tuple(patch_info['coords']), metadata['parameters']['injected_events']['details'])
        
        # Original patch_data shape: (T, H, W)
        # Add a channel dimension for the 3D model: (C, T, H, W) where C=1
        patch_tensor = torch.from_numpy(patch_data).unsqueeze(0) 
        mask_tensor = torch.from_numpy(target_mask.astype(np.int64))

        sample = {"patch": patch_tensor, "mask": mask_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample