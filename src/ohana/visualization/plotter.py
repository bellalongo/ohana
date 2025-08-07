import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import logging
from matplotlib import animation

class ResultVisualizer:
    """
    Visualizes anomaly detection results by loading a processed data cube
    and a list of classified detections.
    """
    def __init__(self, processed_data_path):
        """
        Initializes the ResultVisualizer.
        Args:
            processed_data_path (str): Path to the processed .npy data cube (T-1, H, W).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Loading processed data from: {processed_data_path}")
        self.diff_image_cube = np.load(processed_data_path)
        self.detections = []
        self.prediction_mask = None
        # Define a mapping from anomaly type string to an integer index for the mask
        self.class_map = {'cosmic_ray': 1, 'telegraph_noise': 2}

    def load_detection_list(self, results_path):
        """
        Loads a JSON file with the detection entries.
        Args:
            results_path (str): Path to the JSON file containing the detections.
        """
        with open(results_path, 'r') as f:
            self.detections = json.load(f)
        self.logger.info(f"Loaded {len(self.detections)} detections from {results_path}")

    def _create_mask_from_detections(self):
        """
        Creates a 2D integer mask from the loaded detection list. This internal
        method populates self.prediction_mask.
        """
        if not self.detections:
            self.logger.warning("No detections loaded, cannot create mask.")
            return

        _, height, width = self.diff_image_cube.shape
        self.prediction_mask = np.zeros((height, width), dtype=np.int32)

        self.logger.info("Creating prediction mask from detection list...")
        for detection in self.detections:
            anomaly_type = detection.get('type')
            class_idx = self.class_map.get(anomaly_type)

            if class_idx is None:
                continue

            # For cosmic rays, use the 'pixel_coords' to fill the mask
            if anomaly_type == 'cosmic_ray' and 'pixel_coords' in detection:
                coords = np.array(detection['pixel_coords'], dtype=int)
                if coords.size > 0:
                    self.prediction_mask[coords[:, 0], coords[:, 1]] = class_idx
            
            # For telegraph noise, use the single 'position'
            elif anomaly_type == 'telegraph_noise' and 'position' in detection:
                y, x = detection['position']
                self.prediction_mask[int(y), int(x)] = class_idx
        
        self.logger.info(f"Mask created with {len(np.unique(self.prediction_mask))-1} classes.")

    def plot_full_mask_overlay(self, alpha=0.5, palette_name='Blues_r'):
        """
        Renders the median image with the prediction mask and detection markers.
        Args:
            alpha (float): Transparency for the mask overlay.
            palette_name (str): Matplotlib colormap name for the mask.
        """
        # Create the mask from detections if it doesn't exist yet
        if self.prediction_mask is None:
            self._create_mask_from_detections()

        if self.prediction_mask is None:
            self.logger.error("Cannot plot, prediction mask is not available.")
            return

        sns.set_theme(style="white")
        data_image = np.median(self.diff_image_cube, axis=0)
        fig, ax = plt.subplots(figsize=(12, 12))

        vmin, vmax = np.percentile(data_image, [1, 99])
        ax.imshow(data_image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')

        # Create a discrete colormap where class 0 (background) is transparent
        num_classes = len(self.class_map)
        colors = plt.get_cmap(palette_name, num_classes)
        
        cmap_colors = [(0,0,0,0)] + [colors(i) for i in range(num_classes)]
        cmap = mcolors.ListedColormap(cmap_colors)

        # Overlay the prediction mask
        ax.imshow(self.prediction_mask, cmap=cmap, alpha=alpha, origin='lower', interpolation='none')

        # Plot markers for each detection type
        if self.detections:
            marker_palette = sns.color_palette('bright', num_classes)
            
            for i, (anomaly_type, class_idx) in enumerate(self.class_map.items()):
                coords = []
                for d in self.detections:
                    if d.get('type') == anomaly_type:
                        # Use centroid for CRs, position for RTN as the representative point
                        if 'centroid' in d:
                            coords.append(d['centroid'])
                        elif 'position' in d:
                            coords.append(d['position'])
                
                if coords:
                    coords = np.array(coords)
                    ax.scatter(coords[:, 1], coords[:, 0], s=20, marker='x', label=anomaly_type, c=[marker_palette[i]])

        ax.set_title("Full Exposure with Predicted Anomaly Mask")
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        if self.detections:
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_event_region_and_timeseries(self, y, x, frame_idx=None, radius=20, palette_name='Blues',
                                         show_state_lines=False, high_state_value=None, low_state_value=None):
        """
        Renders a two-panel figure showing a spatial cutout and the pixel time series.
        This method is independent of the detection list format and requires no changes.
        """
        sns.set_theme(style="white", palette=palette_name)
        palette = sns.color_palette(palette_name, 6)

        T, H, W = self.diff_image_cube.shape
        y_start, y_end = max(0, y - radius), min(H, y + radius)
        x_start, x_end = max(0, x - radius), min(W, x + radius)

        if frame_idx is None:
            frame_idx = int(np.argmax(self.diff_image_cube[:, y, x]))

        region = self.diff_image_cube[frame_idx, y_start:y_end, x_start:x_end]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        im = ax1.imshow(region, cmap=palette_name, aspect='auto', origin='lower')
        ax1.set_xlabel('X pixel')
        ax1.set_ylabel('Y pixel')
        plt.colorbar(im, ax=ax1, label='DN')

        time_series = self.diff_image_cube[:, y, x]
        frames = np.arange(T)

        ax2.plot(frames, time_series, linewidth=1, alpha=0.8, label='Pixel Signal', c=palette[3])

        if show_state_lines:
            if high_state_value is not None:
                ax2.axhline(y=high_state_value, linestyle='--', linewidth=2, alpha=0.7,
                            label=f"High: {high_state_value:.1f} DN")
            if low_state_value is not None:
                ax2.axhline(y=low_state_value, linestyle='--', linewidth=2, alpha=0.7,
                            label=f"Low: {low_state_value:.1f} DN")

        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Pixel Value (DN)')
        ax2.set_title(f'Time Series - Pixel ({y},{x})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_all_detections(self, max_plots=None, radius=20, palette_name='Blues'):
        """
        Iterates through all loaded detections and plots the region and time series for each one.
        Args:
            max_plots (int, optional): The maximum number of plots to generate. Defaults to all.
            radius (int): The radius for the spatial cutout around each event.
            palette_name (str): The color palette to use for the plots.
        """
        if not self.detections:
            self.logger.warning("No detections loaded. Call load_detection_list() first.")
            return

        num_to_plot = len(self.detections)
        if max_plots is not None:
            self.logger.info(f"Limiting plots to the first {max_plots} detections.")
            num_to_plot = min(len(self.detections), max_plots)

        self.logger.info(f"Generating detailed plots for {num_to_plot} detections...")

        for i, detection in enumerate(self.detections[:num_to_plot]):
            anomaly_type = detection.get('type')
            self.logger.info(f"Plotting detection {i+1}/{num_to_plot}: Type = {anomaly_type}")

            # Extract coordinates and other relevant info
            if anomaly_type == 'cosmic_ray_candidate':
                # Use the centroid as the representative point for the plot
                y, x = detection.get('centroid')
                y, x = int(round(y)), int(round(x))
                self.plot_event_region_and_timeseries(y, x, radius=radius, palette_name=palette_name)

            elif anomaly_type == 'rtn_candidate':
                y, x = detection.get('position')
                y, x = int(y), int(x)
                high_state = detection.get('high_state_value')
                low_state = detection.get('low_state_value')
                self.plot_event_region_and_timeseries(
                    y, x,
                    radius=radius,
                    palette_name=palette_name,
                    show_state_lines=True,
                    high_state_value=high_state,
                    low_state_value=low_state
                )
            else:
                self.logger.warning(f"Unknown detection type '{anomaly_type}' for detection {i+1}. Skipping plot.")

    def create_event_movie(self, y, x, output_path, radius=20, fps=5, palette_name='Blues_r', vmin=None, vmax=None):
        """
        Creates and saves a movie of a spatial region around an event.
        Now accepts optional vmin and vmax for a consistent color scale.
        """
        self.logger.info(f"Creating event movie for pixel ({y},{x}) at {output_path}...")

        # --- 1. Extract the spatial region across all frames ---
        T, H, W = self.diff_image_cube.shape
        y_start, y_end = max(0, y - radius), min(H, y + radius + 1)
        x_start, x_end = max(0, x - radius), min(W, x + radius + 1)
        
        region_cube = self.diff_image_cube[:, y_start:y_end, x_start:x_end]
        
        # --- Use provided vmin/vmax or calculate locally if not given ---
        if vmin is None or vmax is None:
            self.logger.info("vmin/vmax not provided, calculating from local region.")
            vmin, vmax = np.percentile(region_cube, [1, 99])

        # --- 2. Set up the plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            region_cube[0], cmap=palette_name, vmin=vmin, vmax=vmax,
            origin='lower', interpolation='none'
        )
        plt.colorbar(im, ax=ax, label='Pixel Value (DN)')
        ax.set_xlabel('X Pixel (Relative)')
        ax.set_ylabel('Y Pixel (Relative)')
        title = ax.set_title(f"Event at ({y},{x}) - Frame 0")

        # --- 3. Define the animation update function ---
        def update(frame_idx):
            im.set_data(region_cube[frame_idx])
            title.set_text(f"Event at ({y},{x}) - Frame {frame_idx}")
            return [im, title]

        # --- 4. Create and save the animation ---
        ani = animation.FuncAnimation(
            fig, update, frames=T, interval=1000 / fps, blit=True
        )
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps)
            self.logger.info(f"Animation saved successfully to {output_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save animation. Make sure 'ffmpeg' is installed. Error: {e}")
        plt.close(fig)
