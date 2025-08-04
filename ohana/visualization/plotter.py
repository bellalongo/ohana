import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import the data loading and preprocessing classes from your package
from ohana.preprocessing.data_loader import DataLoader
from ohana.preprocessing.preprocessor import Preprocessor

class ResultVisualizer:
    """
    Handles loading prediction results and the original data to create
    visualizations for each detected anomaly.
    """
    def __init__(self, exposure_path: str):
        """
        Initializes the ResultVisualizer.

        Args:
            exposure_path (str): The path to the original exposure data source
                                 (e.g., a FITS file or a directory of TIFFs).
        """
        self.exposure_path = exposure_path
        
        # Instantiate the helper classes to load and process the data
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor(perform_ref_correction=False) # Assuming no ref correction for visualization

        # Load and preprocess the data once to be used for all plots
        print("Loading and preprocessing exposure data for visualization...")
        raw_cube = self.data_loader.load_exposure(self.exposure_path)
        self.diff_image_cube = self.preprocessor.process_exposure(raw_cube)
        print("Data ready for plotting.")

    def plot_all_detections(self, results_path: str, output_dir: str, radius: int = 10):
        """
        Loads a JSON result file and creates a plot for every detection.

        Args:
            results_path (str): Path to the JSON file containing detection results.
            output_dir (str): Directory where the output plot images will be saved.
            radius (int): The radius (in pixels) for the spatial view around the anomaly.
        """
        # Load the detection results
        with open(results_path, 'r') as f:
            detections = json.load(f)

        if not detections:
            print("No detections found in the results file. Nothing to plot.")
            return

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to: {output_dir}")

        # Create a plot for each detection
        for i, detection in enumerate(tqdm(detections, desc="Generating Plots")):
            self.create_detection_plot(detection, i, output_dir, radius)

    def create_detection_plot(self, detection: dict, detection_index: int, output_dir: str, radius: int):
        """
        Creates and saves a single diagnostic plot for one detected anomaly.
        The plot contains a spatial view and a temporal (time-series) view.
        """
        try:
            y, x = detection['location_px']
            event_type = detection['type']
            confidence = detection.get('confidence', 0) * 100

            # --- 1. Prepare Data for Plotting ---
            
            # Extract the time series for the exact pixel location
            pixel_time_series = self.diff_image_cube[:, y, x]
            time_axis = np.arange(len(pixel_time_series))

            # Extract the 2D spatial region for a snapshot in time
            # We'll find the frame with the maximum deviation for the snapshot
            peak_frame_index = np.argmax(np.abs(pixel_time_series))
            
            y_start = max(0, y - radius)
            y_end = min(self.diff_image_cube.shape[1], y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(self.diff_image_cube.shape[2], x + radius + 1)
            
            spatial_region = self.diff_image_cube[peak_frame_index, y_start:y_end, x_start:x_end]

            # --- 2. Create the Plot ---
            sns.set_theme(style="whitegrid")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"Detection #{detection_index+1}: {event_type.capitalize()} at (y={y}, x={x}) | Confidence: {confidence:.2f}%", fontsize=16)

            # Subplot 1: Spatial View
            im = ax1.imshow(spatial_region, cmap='viridis', aspect='equal', origin='lower',
                            extent=[x_start, x_end, y_start, y_end])
            ax1.set_title(f"Spatial View (Frame {peak_frame_index})")
            ax1.set_xlabel("X Pixel")
            ax1.set_ylabel("Y Pixel")
            # Highlight the center pixel
            ax1.plot(x, y, 'r+', markersize=12)
            fig.colorbar(im, ax=ax1, label='Signal (DN)')

            # Subplot 2: Temporal View (Time Series)
            ax2.plot(time_axis, pixel_time_series, marker='.', linestyle='-', color='teal')
            ax2.set_title("Pixel Value vs. Time")
            ax2.set_xlabel("Frame Number (Difference)")
            ax2.set_ylabel("Signal (DN)")
            # Mark the peak frame
            ax2.axvline(peak_frame_index, color='red', linestyle='--', label=f'Peak Frame: {peak_frame_index}')
            ax2.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # --- 3. Save the Figure ---
            output_filename = f"detection_{detection_index+1}_{event_type}_({y},{x}).png"
            output_path = os.path.join(output_dir, output_filename)
            plt.show()
            plt.savefig(output_path)
            plt.close(fig) # Close the figure to free up memory

        except Exception as e:
            print(f"\nWarning: Could not create plot for detection #{detection_index+1}. Reason: {e}")

