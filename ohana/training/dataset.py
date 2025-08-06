
import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class OhanaDataset(Dataset):
    """
        Dataset for patch-wise training for the 3DU-Net on the datacubes by scanning a directory
        for an exposure metadata json file, reads the patch coordinates and hdf5 tensors, makes
        the target mask for each patch from event annotations and returns the tensors
        to be used by the model!
    """
    def __init__(self, data_dir, patch_size, class_map=None, transform=None):
        """
            Arguments:
                data_dir (str): root directory containing:
                    'metadata/<detector_id>/*.json' files with patch metadata
                    'metadata['h5_file']' hdf5 files referenced by the metadata 
                patch_size (tuple[int, int] or list[int]): spatial size (H, W) per patch
                class_map (optional[dict[str, int]]): mapping from class name to index
                    defaults to '{"background": 0, "cosmic_ray": 1, "snowball": 2, "rtn": 3}'
                transform (optional[callable]): optional callable applied to the sample
                    dict '{"patch": torch.Tensor, "mask": torch.Tensor}' used for 
                    augmentation/normalization, should return the same keys
            Attributes:
                data_dir (str): stored dataset root
                patch_size (tuple[int, int]): (H, W) patch size as a tuple
                transform (optional[callable]): optional transform applied per sample
                class_map (dict[str, int]): class-to-index mapping
                exposures (list[str]): list of metadata file paths discovered
                patch_info_cache (dict[str, dict[str, Any]]): cache from meta path to parsed JSON
        """
        # Store the init arguments
        self.data_dir = data_dir
        self.patch_size = tuple(patch_size)
        self.transform = transform
        
        # Use the defualt class map if none was provided
        if class_map is None:
            self.class_map = {"background": 0, "cosmic_ray": 1, "snowball": 2, "rtn": 3}
        # Have option to use a different class map
        else:
            self.class_map = class_map
            
        # Find the wanted exposure
        self.exposures = self._find_exposures()

        # Create a cache to store the patch information into
        self.patch_info_cache = {}

    def _find_exposures(self):
        """
            Find all of the exposure metadata JSON files under the 'metadata' folder
            Arguments:
                None
            Returns:
                list(str): sorted list of json file paths
        """
        # Log girlie
        logger.info("Scanning for exposure metadata files...")

        # Initialize where the exposure outputs will be saved to
        exposures = []

        # Grab the metadata root
        metadata_root = os.path.join(self.data_dir, "metadata")

        # Check if the metadata directory exists
        if not os.path.isdir(metadata_root):
            raise FileNotFoundError(f"Metadata directory not found at {metadata_root}")

        # Iterate through every detector id in the metadata root
        for detector_id in sorted(os.listdir(metadata_root)):
            # Create a path from the detector id
            detector_path = os.path.join(metadata_root, detector_id)

            # Check if the path exists
            if not os.path.isdir(detector_path): continue

            # Iterate through all of the metadata files in directory
            for meta_file in sorted(os.listdir(detector_path)):
                # Make sure that the metadata file is a json
                if meta_file.endswith(".json"):
                    # Add the exposure path to exposures
                    exposures.append(os.path.join(detector_path, meta_file))
        
        # Log girlie finished
        logger.info(f"Found {len(exposures)} exposure metadata files.")

        return exposures

    def _get_patch_info(self, meta_path):
        """
            Read and cache the json file
            Arguments:
                meta_path (str): path to an exposure metadata json file
            Returns:
                dict([str, Any]): parsed json as a python dictionary
        """
        # Check if the current patch has already been cached
        if meta_path in self.patch_info_cache:
            return self.patch_info_cache[meta_path]
        
        # Open and load the file 
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Save the patch info as the current files metadata
        self.patch_info_cache[meta_path] = metadata

        return metadata

    def __len__(self):
        """
            Get the total number of samples in the dataset
            Arguments:
                None
            Returns:
                int: total number of patches across all exposures
        """
        # Check if the files even exists
        if not self.exposures: return 0

        # Try to grab the metadata for an exposure
        try:
            # Grab the patch info for the exposure
            metadata = self._get_patch_info(self.exposures[0])

            return len(self.exposures) * len(metadata['patch_info']['patches'])
        
        # Don't grab the metadata if the file is incomplete
        except (KeyError, IndexError):
            return 0

    def _create_target_mask(self, patch_coords, all_events):
        """
            Make a per-pixel class mask for a given patch from the event labels by checking 
            to see if the event's center is in the patch, mark the pixel with the class
            index. If multiple eventsd are in one pixel, use snowball as priority becasue
            girlie is HUGE
            Arguments:
                patch_coords (tuple([int, int])): top-left (y, x) of the patch
                all_events (list(dicr[(str, Any)])): list of event dictionaries
                    type (str): class name in the class map
                    position or center (list(int, int)): (y, x) cords of the event
            Returns:
                np.ndarray: integer mask of the shape (H, W) for the patch
            Notes:
                * priority can change, but currently it is 
                  snowball > rtn > cosmic ray > background
        """
        # Grab the patch dimensions
        patch_height, patch_width = self.patch_size

        # Grab the patch coordinates
        patch_y, patch_x = patch_coords

        # Initialize to be the background
        target_mask = np.zeros((patch_height, patch_width), dtype=np.uint8)

        # Manual priority !NOTE change me if dif wanted order or dif training data
        event_priority = {'snowball': 3, 'rtn': 2, 'cosmic_ray': 1}

        # Iterate over all events in the patch
        for event in all_events:
            # Grab the event type
            event_type = event['type']

            # Grab either the event postion (rtn, cr) or center (snowball)
            pos_key = "position" if "position" in event else "center"

            # Make the position as a tuple
            event_y, event_x = tuple(event[pos_key])

            # Check if the event is within the current patche's bounds
            if (patch_y <= event_y < patch_y + patch_height) and (patch_x <= event_x < patch_x + patch_width):
                # Convert the patch to the relevant coordinates
                rel_y, rel_x = event_y - patch_y, event_x - patch_x

                # Apply the event priority to the patch
                if event_priority.get(event_type, 0) > target_mask[rel_y, rel_x]:
                    target_mask[rel_y, rel_x] = self.class_map[event_type]

        return target_mask

    def __getitem__(self, idx):
        """
            Grab one sample from the model (C=1, T, H, W) patch tensor and (H, W) mask tensor by
            loading the patch tensor from a hdf5 ref by the exposure metadata and the channel
            dim is added to match the model's input layout
            Arguments:
                idx (int): global sample index
            Returns:
                dict(str, torch.Tensor): a dictionary with
                    patch: float tensor of shape (1, T, H, W)
                    mask: long tensor of shape (H, W) with the class indicies
        """
        # Figure out the per exposure patch count using the very first exp
        metadata = self._get_patch_info(self.exposures[0])
        num_patches_per_exp = len(metadata['patch_info']['patches'])
        
        # Map the global index
        exp_idx = idx // num_patches_per_exp
        patch_idx_in_exp = idx % num_patches_per_exp

        # Retrieve the metadata from the target exposure
        meta_path = self.exposures[exp_idx]
        metadata = self._get_patch_info(meta_path)
        
        # Select the patch record and the source hdf5 path
        patch_info = metadata['patch_info']['patches'][patch_idx_in_exp]
        h5_path = os.path.join(self.data_dir, metadata['h5_file'])
        patch_id = patch_info['id']
        
        # Read the patch volume from the file as floast32
        with h5py.File(h5_path, 'r') as hf:
            patch_data = hf[patch_id][:].astype(np.float32)

        # Make the per-patch target mask (H, W) from the event annotations
        target_mask = self._create_target_mask(tuple(patch_info['coords']), metadata['parameters']['injected_events']['details'])
        
        # Add a channel dimension for the 3D model with (C, T, H, W) where C=1
        patch_tensor = torch.from_numpy(patch_data).unsqueeze(0) 

        # Target mask should be a torch.long for the loss function like CEL
        mask_tensor = torch.from_numpy(target_mask.astype(np.int64))

        # Create the sample dictionary
        sample = {"patch": patch_tensor, "mask": mask_tensor}

        # Check if the sample needs to be transformed (optional)
        if self.transform:
            sample = self.transform(sample)

        return sample