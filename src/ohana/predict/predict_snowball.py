import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage.measure import regionprops
import logging

# Import the base class
from .base_predictor import BaseDetector

class SnowballDetector(BaseDetector):
    """
    Detects snowball events in H2RG near-infrared data following JWST methodology.
    """
    def __init__(self, config):
        """
        Initializes the SnowballDetector.
        Args:
            config: Configuration object with detector parameters.
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect(self, temporal_data: Dict, diff_stack: np.ndarray) -> Dict:
        """
        Detects snowball candidates using a combination of methods.
        Args:
            temporal_data: Dictionary of temporal features.
            diff_stack: Difference image stack.
        Returns:
            A dictionary containing the list of snowball candidates.
        """
        self.logger.info("Starting snowball detection...")
        
        # For this implementation, we will use the simple and robust detection method.
        # The complex JWST method requires specific inputs (saturation_map, jump_map)
        # that might not be available or configured in all pipelines.
        
        candidates = self._simple_snowball_detection(temporal_data, diff_stack)
        
        self.logger.info(f"Found {len(candidates)} total snowball candidates.")
        return {
            'candidates': candidates,
            'num_candidates': len(candidates)
        }

    def _simple_snowball_detection(self, temporal_data: Dict, diff_stack: np.ndarray) -> List[Dict]:
        """
        A simple yet effective snowball detection method based on intensity and size.
        """
        first_appearance = temporal_data.get('first_appearance', np.full(diff_stack.shape[1:], -1))
        max_intensity = temporal_data.get('max_intensity', np.zeros(diff_stack.shape[1:]))

        # Define criteria for a snowball candidate
        # Appears after the first frame and has a very high intensity
        sudden_appearance = first_appearance > 0
        high_intensity = max_intensity > self.config.snowball_min_intensity
        
        candidate_mask = sudden_appearance & high_intensity
        
        if not np.any(candidate_mask):
            return []

        labeled, num_features = ndimage.label(candidate_mask)
        self.logger.info(f"Simple snowball detection found {num_features} connected components.")
        
        candidates = []
        props_list = regionprops(labeled, intensity_image=max_intensity)

        for props in props_list:
            # Filter by area
            if not (self.config.snowball_min_area <= props.area <= self.config.snowball_max_area):
                continue
            
            # Filter by shape (circularity)
            if props.perimeter > 0:
                circularity = 4 * np.pi * props.area / (props.perimeter ** 2)
                if circularity < self.config.snowball_min_circularity:
                    continue
            
            # If all checks pass, create the candidate dictionary
            y_cen, x_cen = props.centroid
            candidate = {
                'type': 'snowball_candidate',
                'centroid': (y_cen, x_cen),
                'position': (y_cen, x_cen), # for compatibility
                'first_frame': int(np.min(first_appearance[props.coords[:, 0], props.coords[:, 1]])),
                'max_intensity': float(props.max_intensity),
                'mean_intensity': float(props.mean_intensity),
                'area': int(props.area),
                'pixel_coords': props.coords.tolist() # for mask creation
            }
            candidates.append(candidate)
        
        return candidates

    def classify(self, candidates: Dict) -> List[Dict]:
        """
        Classifies snowball candidates and assigns a confidence score.
        Args:
            candidates: A dictionary from the detect method.
        Returns:
            A list of classified snowball events.
        """
        classified = []
        for candidate in candidates.get('candidates', []):
            confidence = self._calculate_snowball_confidence(candidate)
            if confidence >= self.config.snowball_min_confidence:
                classified_event = {
                    'type': 'snowball',
                    'confidence': confidence,
                    **candidate
                }
                classified.append(classified_event)
        return classified

    def _calculate_snowball_confidence(self, candidate: Dict) -> float:
        """
        Calculates a confidence score for a snowball candidate.
        """
        # Confidence starts high for things that pass the simple detector
        confidence = 0.6 
        
        # Add bonus for larger, very intense snowballs
        if candidate['area'] > 50:
            confidence += 0.1
        if candidate['max_intensity'] > 30000: # Near saturation
            confidence += 0.2
            
        return min(1.0, confidence)
