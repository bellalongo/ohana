import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class ResultVisualizer:
    """
        Visualize the prediction masks and detections points over a processed exposure 
        by loading a preprocessed diff image cube, a 2D prediction mask of the class
        indicies, and a list of detections
    """
    def __init__(self, processed_data_path: str, prediction_mask_path: str):
        """
            Arguments:
                processed_data_path (str): path to .npy array with shape (T, H, W)
                prediction_mask_path (str): path to .npy array with shape (H, W)
            Attributes:
                diff_image_cube (np.ndarray): loaded difference-image cube, shape (T, H, W)
                prediction_mask (np.ndarray): 2D class-index mask, shape (H, W)
                detections (List[Dict[str, Any]]): optional detections with keys:
                    type (str): class label
                    location_px (list[int, int]): [row, col] pixel location
        """
        # Log which files are getting loaded
        print(f"Loading pre-processed data from: {processed_data_path}")

        # Load the difference image cube
        self.diff_image_cube = np.load(processed_data_path)
        
        # Log that you are loading the prediction mask
        print(f"Loading prediction mask from: {prediction_mask_path}")

        # Load the prediction mask
        self.prediction_mask = np.load(prediction_mask_path)
        
        # Initialzie where detections will be stored
        self.detections = []

    def load_detection_list(self, results_path: str):
        """
            Load a JSON file with the detection entries of the type and location
            of the detections
            Arguments: 
                results_path (str): path to the json file containing the detections
            Returns:
                None
        """
        # Read where the detections are
        with open(results_path, 'r') as f:
            # Save the detections
            self.detections = json.load(f)
        
        # Print a summmary of the loaded detections
        print(f"Loaded {len(self.detections)} individual detections from {results_path}")

    def plot_full_mask_overlay(self):
        """
            Renfer the median image with the prediction mask and detection markers
            by collapsing the time axis using the median, displaying the image and
            overlaying the class index mask with a colormap
        """
        # Grab the median image
        data_image = np.median(self.diff_image_cube, axis=0)

        # Create a subplot
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Calculate the min and max for an accurate cbar
        vmin, vmax = np.percentile(data_image, [1, 99])

        # Plot the median image
        ax.imshow(data_image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')

        # Create a cmap for the prediction mask
        cmap = mcolors.ListedColormap(['none', 'cyan', 'yellow', 'magenta'])

        # Show the prediction mask
        ax.imshow(self.prediction_mask, cmap=cmap, alpha=0.5, origin='lower', interpolation='none')
        
        # Check if there are detections in the image
        if self.detections:
            # Iterate through the types of events
            types = {d['type'] for d in self.detections}

            # Mark the events
            for t in types:
                coords = np.array([d['location_px'] for d in self.detections if d['type'] == t])
                ax.scatter(coords[:, 1], coords[:, 0], s=10, marker='+', label=t)

        # Plot basic stuff here
        ax.set_title("Full Exposure with Predicted Anomaly Mask")
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        
        if self.detections:
            ax.legend()
    
        plt.tight_layout()
        plt.show()