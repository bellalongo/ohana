import numpy as np
import logging

# Import the necessary components
from .temporal_analyzer import TemporalAnalyzer
from .reference_pixel_corrector import ReferencePixelCorrector

class Preprocessor:
    """
    Handles preprocessing steps by orchestrating specialized components
    such as the ReferencePixelCorrector and TemporalAnalyzer.
    """
    def __init__(self, config):
        """
        Initializes the Preprocessor and its components.
        Args:
            config: Dataclass object holding all parameters.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- CORRECTLY INITIALIZE ALL COMPONENTS ---
        # Initialize the temporal analyzer
        self.temporal_analyzer = TemporalAnalyzer(sigma_threshold=self.config.sigma_threshold)
        
        # Initialize the reference pixel corrector with parameters from the config
        self.reference_pixel_corrector = ReferencePixelCorrector(
            x_opt=getattr(self.config, 'ref_pixel_x_opt', 64),
            y_opt=getattr(self.config, 'ref_pixel_y_opt', 4)
        )

    def correct_reference_pixels(self, raw_stack: np.ndarray) -> np.ndarray:
        """
        Performs full reference pixel correction and then creates a difference stack
        by subtracting the first frame from all subsequent frames.
        Args:
            raw_stack: Raw exposure stack of shape (T, H, W).
        Returns:
            A corrected difference stack of shape (T-1, H, W).
        """
        self.logger.info("Starting reference pixel correction workflow...")

        # --- STEP 1: Apply the detailed reference pixel correction to each frame ---
        self.logger.info("Applying reference pixel subtraction...")
        corrected_stack = self.reference_pixel_corrector.batch_correct(raw_stack)
        self.logger.info(f"Reference pixel correction complete. Corrected stack shape: {corrected_stack.shape}")

        # --- STEP 2: Create the difference stack by subtracting the 0th frame ---
        self.logger.info("Creating difference stack by subtracting the 0th frame...")
        
        if corrected_stack.shape[0] < 2:
            self.logger.warning("Cannot create difference stack, only one frame exists. Returning empty array.")
            return np.array([], dtype=np.float32)

        # Isolate the first frame as the reference
        first_frame = corrected_stack[0].astype(np.float32)
        
        # Subtract the first frame from all subsequent frames
        # Broadcasting handles the subtraction for each frame in the slice
        diff_stack = corrected_stack[1:].astype(np.float32) - first_frame
        
        self.logger.info(f"Created final difference stack with shape: {diff_stack.shape}")

        return diff_stack

    def analyze_temporal(self, diff_stack: np.ndarray) -> dict:
        """
        Analyzes the difference stack by delegating to the TemporalAnalyzer.
        Args:
            diff_stack: The final, corrected difference stack.
        Returns:
            A dictionary of temporal feature maps.
        """
        self.logger.info("Delegating to TemporalAnalyzer for feature extraction...")
        temporal_features = self.temporal_analyzer.analyze_temporal_patterns(diff_stack)
        self.logger.info("Temporal analysis complete.")
        return temporal_features
