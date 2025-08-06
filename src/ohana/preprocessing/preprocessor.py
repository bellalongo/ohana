import numpy as np
from os.path import exists

# Import the new, dedicated corrector class
from .reference_pixel_corrector import ReferencePixelCorrector
from .temporal_analyzer import TemporalAnalyzer

class Preprocessor:
    """
    Handles the cleaning and preprocessing of raw H2RG data ramps.
    This class orchestrates the sequence of cleaning steps.
    """
    def __init__(self, perform_ref_correction: bool = True):
        """
        Initializes the Preprocessor.

        Args:
            perform_ref_correction (bool): If True, reference pixel correction
                                           will be applied.
        """
        self.perform_ref_correction = perform_ref_correction
        if self.perform_ref_correction:
            # Initialize the corrector class. It's now a component of the Preprocessor.
            self.ref_corrector = ReferencePixelCorrector()
        
        print(f"Preprocessor initialized. Reference pixel correction: {'Enabled' if perform_ref_correction else 'Disabled'}.")

    def process_exposure(self, raw_exposure_cube: np.ndarray,
                         save_path: str) -> np.ndarray:
        """
        Applies all preprocessing steps to a raw data cube.

        Args:
            raw_exposure_cube (np.ndarray): The input data cube 
                                            (frames, height, width).
            save_path (str): place where the processed exposure will be saved to 
                as a npy

        Returns:
            np.ndarray: The processed data, ready for patching and prediction.
                        Typically this is the difference image cube.
        """
        # Check if file exists already
        if exists(save_path):
            print(f"Found existing processed file. Loading from: {save_path}")
            return np.load(save_path)

        if raw_exposure_cube.ndim != 3 or raw_exposure_cube.shape[0] < 2:
            raise ValueError("Input for preprocessing must be a 3D cube with at least 2 frames.")

        print(f"Preprocessing data cube of shape: {raw_exposure_cube.shape}")
        
        current_cube = raw_exposure_cube

        # --- Step 1: Reference Pixel Correction (Conditional) ---
        if self.perform_ref_correction:
            print("Applying reference pixel correction...")
            current_cube = self.ref_corrector.batch_correct(current_cube)
            print("Reference pixel correction complete.")
        
        # --- Step 2: Frame Differencing ---
        # Create the difference image cube by subtracting the first frame from all subsequent frames.
        print("Performing frame differencing...")
        reference_frame = current_cube[0]
        diff_image_cube = current_cube[1:] - reference_frame
        print(f"Created difference image cube with shape: {diff_image_cube.shape}")

        np.save(save_path, diff_image_cube)

        return diff_image_cube
