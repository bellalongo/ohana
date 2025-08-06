import torch
import numpy as np
import yaml
from tqdm import tqdm
from scipy.ndimage import label, center_of_mass
from torch.cuda.amp import autocast
from collections import OrderedDict

from ohana.preprocessing.data_loader import DataLoader
from ohana.preprocessing.preprocessor import Preprocessor
from ohana.models.unet_3d import UNet3D


class Predictor:
    """
        Patch-wise U 3D U-Net infereence and post-processing for anomally detection by loading the 
        trained model, preprocessing an exposure (either fits or tif), aggregating per-patch 
        predictions into a full-frame mask, and extracting obejct centroids with connected
        components
    """
    def __init__(self, model_path: str, config_path: str):
        """
            Arguments:
                model_path (str): Path to the .pt or .pth file with the model weights
                config_path (str): Path to a YAML config with keys:
                    class_map (dict): "background":0, "cosmic_ray": 1, etc
                    patch_size (list[int, int]): [height, width] of 2D crop / patch
                    overlap (int): pixels of overlap between adjacant patches
            Attributes:
                model_path (str): stored path to model weights
                cfg (dict[str, Any]): parsed yaml configuration
                device (torch.device): cuda if available, else cpu
                data_loader (DataLoader): handles raw exposure loading
                preprocessor (Preprocessor): preprocessing pipeline before model inference
                model (UNet3D): loaded 3D U-Net model in evaluation mode
                processed_cube (np.ndarray or None): preprocessed input volume (T, H, W)
                prediction_mask (np.ndarray or None): final 2D per-pixel class mask (H, W)
        """
        # Save the model path
        self.model_path = model_path

        # Load the yaml config file
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set up device to be either cpu or cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initailize the helper functions for the preprocessing
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()

        # Load the model and set it to be in evaluation mode
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize the processed data cube, as well as prediction mask
        self.processed_cube = None
        self.prediction_mask = None

    def _load_model(self):
        """
            Load the 3DUnet and her weights, and data parallel checkpoints
            Arguments:
                None
            Returns:
                UNet3D: model moves to the determined device with the loaded weights
        """
        # Log whihc file is being loaded for the model
        print(f"Loading 3D U-Net model from: {self.model_path}")

        # Determine the length of the classes based on the config (should be these) !NOTE CHANGE IF DOING A DIF MODEL
        class_map_default = {"background": 0, "cosmic_ray": 1, "snowball": 2, "rtn": 3}
        num_classes = len(self.cfg.get('class_map', class_map_default))

        # Build the model with the input being just a single channel !NOTE CHANGE IF DIF TRAINING DATA
        model = UNet3D(n_channels=1, n_classes=num_classes)

        # Load the state dictionary on the correct device
        state_dict = torch.load(self.model_path, map_location=self.device)

        # If saved with DataParallel (2 GPU) the keys are module!
        if next(iter(state_dict)).startswith('module.'):
            # Store the keys without module prefix
            new_state_dict = OrderedDict()

            # Iterate through the key and value pairs in state dict
            for k, v in state_dict.items():
                # Strip module (7 chars)
                new_state_dict[k[7:]] = v

            # Load the fixed state dictionary
            model.load_state_dict(new_state_dict)

        # If not saved with DataParallel (1 GPU)
        else:
            # The the state dictionary as is
            model.load_state_dict(state_dict)

        # Move the model to the predetermined device
        model.to(self.device)

        return model

    def _extract_patches(self, volume: np.ndarray):
        """
            Create the overlapping 2D patches across (H x W) for each non destructive read by sliding a 
            (patch_width x patch_height) window over (H x W) with a stride of (patch_height_overlap, 
            patch_width_overlap) extracting 
            Arguments:
                volumes (np.ndarray): preprocessed data (T, H, W)
            Returns:
                list(tuple[np.ndarray, tuple(int, int)]): list of (patch_volume, (row, col)) where 
                    patch_volume is (T, patch_height, patch_width) and (row, col) is top-left cord
        """
        # Grab the height and width of the datacube
        _, H, W = volume.shape

        # Grab patch height and width 
        patch_height, patch_width = self.cfg["patch_size"]

        # Grab the patch overlap
        overlap = self.cfg["overlap"]

        # Compute the strides using the overlap and patch dims
        step_h, step_w = patch_height - overlap, patch_width - overlap

        # Initailize an array to store the patches in
        patches = []

        # Iterate through all patch heights (first param of sliding window)
        for i in range(0, H - patch_height + 1, step_h):
            # Itetate through all patch widths (second param of sliding window)
            for j in range(0, W - patch_width + 1, step_w):
                # Append the current patch to the patches list
                patches.append((volume[:, i:i+patch_height, j:j+patch_width], (i, j)))

        return patches

    def _find_objects_from_mask(self, mask: np.ndarray) -> list:
        """
            Run connected components analysis per class (not background), for each event find
            the connected components, compute their centroids and return a list of detections
            Arguments:
                mask (np.ndarray): 2D class index mask (H, W)
            Returns:
                list(dict[str, Any]): each detection has:
                    type (str): class label
                    location_px (list[int, int]): [row, col] of the centroid pixel
        """ 
        # Initialize a list to store the detections in
        detections = []

        # Build inverse mapping for {class_index: class_name}
        class_map_default = {"background": 0, "cosmic_ray": 1, "snowball": 2, "rtn": 3}
        class_map_inv = {v: k for k, v in self.cfg.get('class_map', class_map_default).items()}

        # Iterate though classes by their index and name
        for class_idx, class_name in class_map_inv.items():
            # Check if the current event is just background
            if class_idx == 0: continue

            # Create a binary mask for the current class
            class_mask = (mask == class_idx).astype(int)

            # Perform connected components on the mask, returning the labels and features
            labeled_array, num_features = label(class_mask)

            # Check if an object was even present
            if num_features > 0:
                # Grab the centers of the event
                centers = center_of_mass(class_mask, labeled_array, range(1, num_features + 1))

                # Iterate through the centroids of each event
                for center in centers:
                    # Append the event details to detections
                    detections.append({'type': class_name, 'location_px': [int(round(center[0])), int(round(center[1]))]})

        return detections

    def predict(self, exposure_path: str, processed_exposure_file: str) -> list:
        """
            Run the prediction on a single exposure
            Arguments:
                exposure_path (str): file path or id understood by DataLoader
            Returns:
                list(dict[str, Any]): list of detections (class label and centroids)
        """
        # Log which exposure is being loaded (predicted on)
        print(f"\n--- Analyzing exposure: {exposure_path} ---")

        # Load the raw exposure
        raw_exposure = self.data_loader.load_exposure(exposure_path)

        # Preprorcess the exposure into a numpy array (T, H, W)
        self.processed_cube = self.preprocessor.process_exposure(raw_exposure, processed_exposure_file)

        # Extract the overlapping patches of the exposure
        patches = self._extract_patches(self.processed_cube)
        
        # Prepare an empty 2D mask to accumlate the predictions on
        _, H, W = self.processed_cube.shape
        self.prediction_mask = np.zeros((H, W), dtype=np.uint8)

        # Iterate over patches with a progress bar
        for patch_data, (r, c) in tqdm(patches, desc="Predicting Patches"):
            # Convert the patch dimensions to be a torch tensor
            input_tensor = torch.from_numpy(patch_data).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Perform the min max scaling (0, 1)
            b, c_in, t, h, w = input_tensor.shape

            # Flatten the tensor without the batch
            tensor_flat = input_tensor.reshape(b, -1)

            # Grab the per sample min
            min_val = tensor_flat.min(dim=1, keepdim=True)[0]

            # Grab the per sample max
            max_val = tensor_flat.max(dim=1, keepdim=True)[0]

            # Correct for a divide by 0 by adding epsilon (e)
            tensor_flat = (tensor_flat - min_val) / (max_val - min_val + 1e-6)

            # Reshape the tensor to be unflattened
            normalized_tensor = tensor_flat.view(b, c_in, t, h, w)

            # Perform the inference on the patches
            with torch.no_grad():
                # (1, C, T', patch_height, patch_width)
                with autocast():
                    logits = self.model(normalized_tensor)

                # Use the center time slice of the model output as the final prediction
                T_out = logits.shape[2]
                central_logits = logits[:, :, T_out // 2, :, :]

                # Convert the logits to the class indices by using argmax
                pred_mask = torch.argmax(central_logits, dim=1).squeeze(0).cpu().numpy()
            
            # Paste the per patch predictions into the global mask 
            ph, pw = pred_mask.shape

            # Use the per pixel max to combine the overlapping predictions 
            self.prediction_mask[r:r+ph, c:c+pw] = np.maximum(self.prediction_mask[r:r+ph, c:c+pw], pred_mask)

        # Convert the 2D mask to a list of object detections by using connected componetnts
        detections = self._find_objects_from_mask(self.prediction_mask)

        # Log miss girl
        print(f"Found a total of {len(detections)} individual anomalies.")

        return detections