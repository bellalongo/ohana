
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ResultVisualizer:
    def __init__(self, processed_data_path: str, prediction_mask_path: str):
        print(f"Loading pre-processed data from: {processed_data_path}")
        self.diff_image_cube = np.load(processed_data_path)
        
        print(f"Loading prediction mask from: {prediction_mask_path}")
        self.prediction_mask = np.load(prediction_mask_path)
        
        self.detections = []

    def load_detection_list(self, results_path: str):
        with open(results_path, 'r') as f:
            self.detections = json.load(f)
        print(f"Loaded {len(self.detections)} individual detections from {results_path}")

    def plot_full_mask_overlay(self):
        data_image = np.median(self.diff_image_cube, axis=0)
        fig, ax = plt.subplots(figsize=(12, 12))
        
        vmin, vmax = np.percentile(data_image, [1, 99])
        ax.imshow(data_image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')

        cmap = mcolors.ListedColormap(['none', 'cyan', 'yellow', 'magenta'])
        ax.imshow(self.prediction_mask, cmap=cmap, alpha=0.5, origin='lower', interpolation='none')

        if self.detections:
            types = {d['type'] for d in self.detections}
            for t in types:
                coords = np.array([d['location_px'] for d in self.detections if d['type'] == t])
                ax.scatter(coords[:, 1], coords[:, 0], s=10, marker='+', label=t)

        ax.set_title("Full Exposure with Predicted Anomaly Mask")
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        if self.detections:
            ax.legend()
        
        plt.tight_layout()
        plt.show()