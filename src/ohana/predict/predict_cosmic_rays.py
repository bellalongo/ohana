import numpy as np
from typing import Dict, List, Tuple
from scipy import ndimage
from tqdm import tqdm
import logging

# Assuming you have a base_predictor.py in the same directory
from .base_predictor import BaseDetector

class CosmicRayDetector(BaseDetector):
    """
        Detects cosmic ray hits by fitting step functions to the per-pixel time series
        (inspired by the jump detection used for the James Webb Space Telescope, JWST)
    """
    def __init__(self, config):
        """
            Arguments:
                config (Any): configuration object with detector parameters
            Attributes:
                config: stored configuration reference
                logger: module logger for this detector
        """
        # Init initialization
        super().__init__(config)

        # Create a logger scoped to the subclass name
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect(self, temporal_data, diff_stack):
        """
            Detect cosmic ray candidates by screening for pixels that exhibit a sudden,
            persistent positive jump in intensity, then confirm via step-function fitting
            Arguments:
                temporal_data (Dict): dictionary of temporal features from TemporalAnalyzer with keys:
                    * 'first_appearance' (np.ndarray): first frame index when signal appears
                    * 'persistence_count' (np.ndarray): number of frames signal persists post-jump
                    * 'max_intensity' (np.ndarray): peak intensity observed
                    * 'transition_count' (np.ndarray, optional): number of transitions
                diff_stack (np.ndarray): difference image stack of shape (T, H, W)
            Returns:
                Dict: dictionary with key 'candidates' holding a list of candidate detections
            Notes:
                * initial screening is permissive; the step fit does the heavy lifting
                * connected components group neighboring pixels into single events
        """
        # Extract relevant temporal features
        first_appearance = temporal_data['first_appearance']
        persistence = temporal_data['persistence_count']
        max_intensity = temporal_data['max_intensity']
        transition_count = temporal_data.get('transition_count', np.zeros_like(max_intensity))

        # Number of frames in the stack (time dimension)
        num_frames = diff_stack.shape[0]

        """Initial Screening"""
        # Must appear early enough to have post-jump frames
        appears_in_sequence = (first_appearance >= 0) & (first_appearance < num_frames - 3)

        # Must be reasonably strong (be a bit generous here)
        sufficient_intensity = max_intensity >= self.config.cosmic_ray_min_intensity * 0.7

        # Must persist for a few frames (no one-frame wonders)
        reasonable_persistence = persistence >= 3

        # Combine screening criteria
        initial_mask = appears_in_sequence & sufficient_intensity & reasonable_persistence

        # Log count of initial positives
        self.logger.info(f"Initial screening found {np.sum(initial_mask)} potential cosmic ray pixels.")

        # Group connected pixels into candidate regions
        labeled, num_features = ndimage.label(initial_mask)

        # Log the number of connected components
        self.logger.info(f"Grouped into {num_features} connected components (candidates).")

        # Prepare container for candidate dictionaries
        candidates = []

        # Analyze each connected component
        for i in tqdm(range(1, num_features + 1), desc="Analyzing CR candidates"):
            # Extract mask for this component
            component_mask = labeled == i

            # Count pixels in the component
            component_size = np.sum(component_mask)

            # Skip tiny specks (likely noise)
            if component_size < self.config.min_anomaly_pixels:
                continue

            # Coordinates for pixels in this component
            y_coords, x_coords = np.where(component_mask)

            # Pick the pixel with the strongest signal as representative
            component_intensities = max_intensity[component_mask]
            max_intensity_idx = np.argmax(component_intensities)
            center_y, center_x = y_coords[max_intensity_idx], x_coords[max_intensity_idx]

            # Time series for the representative pixel
            time_series = diff_stack[:, center_y, center_x]

            # Fit a step function to the representative time series
            step_fit_result = self._fit_step_function(time_series)

            # If the representative pixel is not step-like, discard component
            if not step_fit_result['is_good_fit']:
                continue

            # Validate by checking a handful of neighbors for similar step behavior
            valid_neighbors = 0
            total_neighbors_to_check = min(5, len(y_coords))
            for j in range(total_neighbors_to_check):
                # Neighbor coordinates
                y, x = y_coords[j], x_coords[j]

                # Fit step for the neighbor pixel
                neighbor_fit = self._fit_step_function(diff_stack[:, y, x])

                # Count neighbors that pass the fit test
                if neighbor_fit['is_good_fit']:
                    valid_neighbors += 1

            # Fraction of neighbors that agree
            neighbor_fraction = valid_neighbors / total_neighbors_to_check

            # Package the candidate dictionary
            candidate = {
                'type': 'cosmic_ray_candidate',
                'centroid': (np.mean(y_coords), np.mean(x_coords)),
                'first_frame': step_fit_result['step_location'],
                'mean_intensity': float(np.mean(max_intensity[component_mask])),
                'max_intensity': float(np.max(max_intensity[component_mask])),
                'spatial_extent': int(component_size),
                'pixel_coords': list(zip(y_coords, x_coords)),
                'step_fit_quality': float(step_fit_result['fit_quality']),
                'step_amplitude': float(step_fit_result['amplitude']),
                'neighbor_fraction': float(neighbor_fraction)
            }

            # Append candidate to the list
            candidates.append(candidate)

        # Log final candidate count
        self.logger.info(f"Found {len(candidates)} candidates after step function fitting.")

        # Return candidates in a dictionary wrapper
        return {'candidates': candidates}

    def _fit_step_function(self, time_series):
        """
            Fit an ideal step function to a 1D time series and evaluate the fit
            Arguments:
                time_series (np.ndarray): vector of length T (single pixel over time)
            Returns:
                Dict: best-fit parameters with keys:
                    * 'is_good_fit' (bool): whether step criteria are satisfied
                    * 'fit_quality' (float): R^2 of the step model
                    * 'step_location' (int): index of the step (where the jump happens)
                    * 'amplitude' (float): post_mean - pre_mean
                    * 'pre_level' (float): mean before the step
                    * 'post_level' (float): mean after the step
            Notes:
                * searches step locations away from the ends to avoid edge weirdness
                * positive steps only (cosmic rays brighten, they don't darken)
        """
        # Guardrail: too-short series cannot support a robust step search
        if len(time_series) < 10:
            return {'is_good_fit': False}

        # Track the best R^2 and its parameters
        best_r_squared = -1.0
        best_params = None

        # Search step locations with a margin on both ends
        for step_loc in range(3, len(time_series) - 3):
            # Split into pre-step and post-step segments
            pre_step = time_series[:step_loc]
            post_step = time_series[step_loc:]

            # Compute means for each segment
            pre_mean = float(np.mean(pre_step))
            post_mean = float(np.mean(post_step))

            # Build the idealized step model
            step_model = np.concatenate([
                np.full_like(pre_step, pre_mean, dtype=float),
                np.full_like(post_step, post_mean, dtype=float)
            ])

            # Compute residual and total sums of squares
            ss_res = float(np.sum((time_series - step_model) ** 2))
            ss_tot = float(np.sum((time_series - np.mean(time_series)) ** 2))

            # Compute R^2 (protect against zero variance)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Keep track of the best model so far
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_params = {
                    'fit_quality': r_squared,
                    'step_location': int(step_loc),
                    'amplitude': post_mean - pre_mean,
                    'pre_level': pre_mean,
                    'post_level': post_mean
                }

        # If nothing improved, return a reject
        if best_params is None:
            return {'is_good_fit': False}

        """Decision: does this look like a cosmic ray step?"""
        # Must have decent fit quality
        good_quality = best_params['fit_quality'] > 0.6

        # Must be a positive step with sufficient amplitude
        positive_and_big = best_params['amplitude'] >= self.config.cosmic_ray_min_step

        # Pre-event level should be lower than post-event level
        positive_direction = best_params['pre_level'] < best_params['post_level']

        # Pre-event baseline should be near zero
        baseline_ok = abs(best_params['pre_level']) < 30.0

        # Combine the decision logic
        is_good_fit = good_quality and positive_and_big and positive_direction and baseline_ok

        # Attach decision to the parameter dict
        best_params['is_good_fit'] = bool(is_good_fit)

        # Return the best-fit parameter bundle
        return best_params

    def classify(self, candidates):
        """
            Classify cosmic ray candidates based on fit quality and physical properties
            Arguments:
                candidates (Dict): dictionary with key 'candidates' from detect()
            Returns:
                List[Dict]: final list of classified cosmic ray events
        """
        # Container for classified events
        classified = []

        # Iterate through the proposed candidates
        for candidate in candidates.get('candidates', []):
            # Apply thresholds from the config
            too_large = candidate['spatial_extent'] > self.config.cosmic_ray_max_spatial_extent
            poor_fit = candidate['step_fit_quality'] < 0.6
            too_small_step = candidate['step_amplitude'] < self.config.cosmic_ray_min_step
            weak_neighbors = candidate['neighbor_fraction'] < 0.4

            # Skip candidates that fail any threshold
            if too_large or poor_fit or too_small_step or weak_neighbors:
                continue

            # Compute confidence score
            confidence = self._calculate_confidence(candidate)

            # Keep only sufficiently confident events
            if confidence >= self.config.min_confidence:
                classified_event = {
                    'type': 'cosmic_ray',
                    'confidence': float(confidence),
                    **candidate
                }
                classified.append(classified_event)

        # Return the final set of events
        return classified

    def _calculate_confidence(self, c):
        """
            Calculate a confidence score based on multiple candidate features
            Arguments:
                c (Dict): single candidate dictionary
            Returns:
                float: confidence score in [0, 1]
            Notes:
                * mostly driven by step-fit quality with small boosts for neighbor agreement
                * tiny, compact hits get a bonus (typical for CRs)
        """
        # Base confidence on fit quality
        conf = c['step_fit_quality'] * 0.7

        # Add bonus for neighbor agreement
        conf += c['neighbor_fraction'] * 0.2

        # Add bonus for being small and compact
        if c['spatial_extent'] <= 5:
            conf += 0.1

        # Clamp to [0, 1]
        return min(1.0, float(conf))
