import torch
import numpy as np
import yaml
from tqdm import tqdm

# Import the new, separated classes
from ohana.preprocessing.data_loader import DataLoader
from ohana.preprocessing.preprocessor import Preprocessor
# Import your model's class definition
from ohana.models.crnn_attention import CRNNAttention

class Predictor:
    """
    Handles loading a trained model and running predictions on new H2RG data.
    """
    def __init__(self, model_path: str, config_path: str):
        """
        Initializes the Predictor.
        """
        self.model_path = model_path
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Predictor will use device: {self.device}")

        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.model = self._load_model()
        self.model.eval()
        
        # This will hold the processed data after predict is called
        self.processed_cube = None

    def _load_model(self):
        """Loads the PyTorch model from the specified path."""
        print(f"Loading model from: {self.model_path}")
        num_classes = self.cfg.get('num_classes')
        if num_classes is None:
            raise ValueError("Your config file must contain a 'num_classes' parameter.")
        model = CRNNAttention(num_classes=num_classes)
        state_dict = torch.load(self.model_path, map_location=self.device)
        if next(iter(state_dict)).startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def _extract_patches(self, volume: np.ndarray):
        """Extracts overlapping patches from a 3D data volume."""
        _, H, W = volume.shape
        ph, pw = self.cfg["patch_size"]
        overlap = self.cfg["overlap"]
        step_h, step_w = ph - overlap, pw - overlap
        patches = []
        for i in range(0, H - ph + 1, step_h):
            for j in range(0, W - pw + 1, step_w):
                patches.append((volume[:, i:i+ph, j:j+pw], (i, j)))
        return patches

    def _preprocess_patch(self, patch_data: np.ndarray) -> torch.Tensor:
        """Preprocesses a single data patch for the model."""
        return torch.from_numpy(patch_data.copy()).float().unsqueeze(0).to(self.device)

    def _postprocess_patch_output(self, model_output: tuple, patch_coords: tuple) -> list:
        """Converts model output for a single patch into a list of detections."""
        logits, _ = model_output
        patch_y, patch_x = patch_coords
        detections = []
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        class_map = {0: 'background', 1: 'cosmic_ray', 2: 'snowball', 3: 'rtn'}
        predicted_class = class_map.get(predicted_idx.item())

        if predicted_class != 'background':
            ph, pw = self.cfg["patch_size"]
            center_y, center_x = patch_y + ph // 2, patch_x + pw // 2
            detections.append({
                'type': predicted_class,
                'location_px': [int(center_y), int(center_x)],
                'confidence': float(confidence.item())
            })
        return detections

    def predict(self, exposure_path: str) -> list:
        """
        Runs the full prediction pipeline on a single exposure file or directory.
        """
        print(f"\n--- Analyzing exposure source: {exposure_path} ---")
        raw_exposure = self.data_loader.load_exposure(exposure_path)
        
        # Store the processed cube so it can be saved later
        self.processed_cube = self.preprocessor.process_exposure(raw_exposure)
        
        patches = self._extract_patches(self.processed_cube)
        if not patches:
            return []
        
        print(f"Extracted {len(patches)} patches for processing.")
        all_detections = []
        for patch_data, patch_coords in tqdm(patches, desc="Analyzing Patches"):
            input_tensor = self._preprocess_patch(patch_data)
            with torch.no_grad():
                model_output = self.model(input_tensor)
            patch_detections = self._postprocess_patch_output(model_output, patch_coords)
            all_detections.extend(patch_detections)
            
        print(f"Found a total of {len(all_detections)} potential detections.")
        return all_detections
