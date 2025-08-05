
import torch
import numpy as np
import yaml
from tqdm import tqdm
from scipy.ndimage import label, center_of_mass

from ohana.preprocessing.data_loader import DataLoader
from ohana.preprocessing.preprocessor import Preprocessor
from ohana.models.unet_3d import UNet3D

class Predictor:
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.model = self._load_model()
        self.model.eval()
        self.processed_cube = None
        self.prediction_mask = None

    def _load_model(self):
        print(f"Loading 3D U-Net model from: {self.model_path}")
        num_classes = len(self.cfg.get('class_map', {"background": 0, "cosmic_ray": 1, "snowball": 2, "rtn": 3}))
        model = UNet3D(n_channels=1, n_classes=num_classes)
        state_dict = torch.load(self.model_path, map_location=self.device)
        if next(iter(state_dict)).startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model.to(device)
        return model

    def _extract_patches(self, volume: np.ndarray):
        _, H, W = volume.shape
        ph, pw = self.cfg["patch_size"]
        overlap = self.cfg["overlap"]
        step_h, step_w = ph - overlap, pw - overlap
        patches = []
        for i in range(0, H - ph + 1, step_h):
            for j in range(0, W - pw + 1, step_w):
                patches.append((volume[:, i:i+ph, j:j+pw], (i, j)))
        return patches

    def _find_objects_from_mask(self, mask: np.ndarray) -> list:
        detections = []
        class_map_inv = {0: 'background', 1: 'cosmic_ray', 2: 'snowball', 3: 'rtn'}
        for class_idx, class_name in class_map_inv.items():
            if class_idx == 0: continue
            class_mask = (mask == class_idx).astype(int)
            labeled_array, num_features = label(class_mask)
            if num_features > 0:
                centers = center_of_mass(class_mask, labeled_array, range(1, num_features + 1))
                for i, center in enumerate(centers):
                    detections.append({'type': class_name, 'location_px': [int(round(center[0])), int(round(center[1]))]})
        return detections

    def predict(self, exposure_path: str) -> list:
        print(f"\n--- Analyzing exposure: {exposure_path} ---")
        raw_exposure = self.data_loader.load_exposure(exposure_path)
        self.processed_cube = self.preprocessor.process_exposure(raw_exposure)
        patches = self._extract_patches(self.processed_cube)
        
        _, H, W = self.processed_cube.shape
        self.prediction_mask = np.zeros((H, W), dtype=np.uint8)

        for patch_data, (r, c) in tqdm(patches, desc="Predicting Patches"):
            input_tensor = torch.from_numpy(patch_data).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(input_tensor)
                T_out = logits.shape[2]
                central_logits = logits[:, :, T_out // 2, :, :]
                pred_mask = torch.argmax(central_logits, dim=1).squeeze(0).cpu().numpy()
            
            ph, pw = pred_mask.shape
            self.prediction_mask[r:r+ph, c:c+pw] = np.maximum(self.prediction_mask[r:r+ph, c:c+pw], pred_mask)

        detections = self._find_objects_from_mask(self.prediction_mask)
        print(f"Found a total of {len(detections)} individual anomalies.")
        return detections