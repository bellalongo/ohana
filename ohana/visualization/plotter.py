import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ResultVisualizer:
    """
    Handles loading detection results and pre-processed data to create
    visualizations for each detected anomaly.
    """
    def __init__(self, processed_data_path: str):
        """
        Initializes the ResultVisualizer.

        Args:
            processed_data_path (str): The path to the pre-processed
                                       (e.g., differenced) data cube (.npy file).
        """
        print(f"Loading pre-processed data from: {processed_data_path}")
        self.diff_image_cube = np.load(processed_data_path)
        self.detections = []
        print("Data ready for plotting.")

    def load_results(self, results_path: str):
        """Loads the detection results from a JSON file."""
        with open(results_path, 'r') as f:
            self.detections = json.load(f)
        if not self.detections:
            print("No detections found in the results file.")
        else:
            print(f"Loaded {len(self.detections)} detections from {results_path}")

    def show_individual_detections(self, radius: int = 20):
        """
        Iterates through each detection, creating and showing a detailed
        diagnostic plot for each one interactively in a 2,1 subplot format.
        """
        if not self.detections:
            return

        print("\nStarting interactive plot session. Close each plot window to proceed to the next.")
        for i, detection in enumerate(self.detections):
            self._create_and_show_individual_plot(detection, i, radius)

    def _create_and_show_individual_plot(self, detection: dict, detection_index: int, radius: int):
        """
        Creates and shows a single diagnostic plot for one anomaly,
        matching the user's preferred vertical layout.
        """
        try:
            y, x = detection['location_px']
            event_type = detection['type']
            confidence = detection.get('confidence', 0) * 100

            # --- 1. Prepare Data for Plotting ---
            pixel_time_series = self.diff_image_cube[:, y, x]
            time_axis = np.arange(len(pixel_time_series))
            peak_frame_index = np.argmax(np.abs(pixel_time_series))
            
            y_start, y_end = max(0, y - radius), min(self.diff_image_cube.shape[1], y + radius + 1)
            x_start, x_end = max(0, x - radius), min(self.diff_image_cube.shape[2], x + radius + 1)
            spatial_region = self.diff_image_cube[peak_frame_index, y_start:y_end, x_start:x_end]

            # --- 2. Create the Plot (Vertical Layout) ---
            sns.set_theme(style="white", palette="Blues")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
            fig.suptitle(f"Detection #{detection_index+1}: {event_type.capitalize()}", fontsize=16)

            # Subplot 1: Spatial View
            im = ax1.imshow(spatial_region, cmap='Blues', aspect='auto', origin='lower',
                            extent=[x_start, x_end, y_start, y_end])
            ax1.set_title(f"Spatial View at (y={y}, x={x}) | Frame: {peak_frame_index} | Confidence: {confidence:.1f}%")
            ax1.set_xlabel("X Pixel")
            ax1.set_ylabel("Y Pixel")
            fig.colorbar(im, ax=ax1, label='Signal (DN)')

            # Subplot 2: Temporal View (Time Series)
            ax2.plot(time_axis, pixel_time_series, marker='.', linestyle='-', color=sns.color_palette('Blues', 6)[4])
            ax2.set_title(f"Time Series for Pixel ({y},{x})")
            ax2.set_xlabel("Frame Number (Difference)")
            ax2.set_ylabel("Signal (DN)")
            ax2.grid(True, alpha=0.3)
            ax2.axvline(peak_frame_index, color='red', linestyle='--', label=f'Peak Frame: {peak_frame_index}')
            ax2.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # --- 3. Show the Plot Interactively ---
            plt.show() # This will pause the script and wait for you to close the window.
            plt.close(fig) # Close the figure to free up memory before the next one.

        except Exception as e:
            print(f"\nWarning: Could not create plot for detection #{detection_index+1}. Reason: {e}")

