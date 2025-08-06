import numpy as np
from typing import Dict, List, Tuple
from scipy import ndimage, signal, optimize
from .base_detector import BaseDetector
from tqdm import tqdm

class CosmicRayDetector(BaseDetector):
    """
    Detects cosmic ray hits by fitting step functions to time series data.
    Based on JWST jump detection methodology.
    """
    def detect(self, temporal_data: Dict, diff_stack: np.ndarray) -> Dict:
        """
        Detect cosmic ray candidates using step function fitting.
        """
        # Grab temporal data
        first_appearance = temporal_data['first_appearance']
        persistence = temporal_data['persistence_count']
        max_intensity = temporal_data['max_intensity']
        transition_count = temporal_data.get('transition_count', np.zeros_like(max_intensity))
        
        # Get the total number of frames
        num_frames = diff_stack.shape[0]
        
        # Initial screening - be permissive, let step fitting do the heavy lifting
        appears_in_sequence = (first_appearance >= 0) & (first_appearance < num_frames - 3)
        sufficient_intensity = max_intensity >= self.config.cosmic_ray_min_intensity * 0.7  # Lower threshold
        reasonable_persistence = persistence >= 3  # Very minimal requirement
        
        # Initial mask
        initial_mask = appears_in_sequence & sufficient_intensity & reasonable_persistence
        
        self.logger.info(f"Initial screening found {np.sum(initial_mask)} pixels")
        
        # Find connected components
        labeled, num_features = ndimage.label(initial_mask)
        
        self.logger.info(f"Found {num_features} connected components")
        
        # Iterate through all candidates
        candidates = []
        for i in tqdm(range(1, num_features + 1)):
            component_mask = labeled == i
            component_size = np.sum(component_mask)
            
            if component_size < self.config.min_anomaly_pixels:
                continue
            
            # Get component properties
            y_coords, x_coords = np.where(component_mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            
            # Find the pixel with the strongest signal for analysis
            component_intensities = max_intensity[component_mask]
            max_intensity_idx = np.argmax(component_intensities)
            center_y = y_coords[max_intensity_idx]
            center_x = x_coords[max_intensity_idx]
            
            # Fit step function to the central pixel
            time_series = diff_stack[:, center_y, center_x]
            step_fit_result = self._fit_step_function(time_series)
            
            if not step_fit_result['is_good_fit']:
                self.logger.debug(f"Component {i} failed step function fit")
                continue
            
            # Additional validation: check a few neighboring pixels
            valid_neighbors = 0
            total_neighbors = min(5, len(y_coords))  # Check up to 5 pixels
            
            for j in range(total_neighbors):
                y, x = y_coords[j], x_coords[j]
                neighbor_series = diff_stack[:, y, x]
                neighbor_fit = self._fit_step_function(neighbor_series)
                if neighbor_fit['is_good_fit']:
                    valid_neighbors += 1
            
            neighbor_fraction = valid_neighbors / total_neighbors
            if neighbor_fraction < 0.4:  # At least 40% of pixels should fit step function
                self.logger.debug(f"Component {i} failed neighbor validation")
                continue
            
            # Extract temporal profile
            first_frame = step_fit_result['step_location']
            mean_intensity = np.mean(max_intensity[component_mask])
            max_component_intensity = np.max(max_intensity[component_mask])
            spatial_extent = len(y_coords)
            mean_transitions = np.mean(transition_count[component_mask])
            
            candidate = {
                'type': 'cosmic_ray_candidate',
                'centroid': (centroid_y, centroid_x),
                'center_pixel': (center_y, center_x),
                'first_frame': first_frame,
                'mean_intensity': mean_intensity,
                'max_intensity': max_component_intensity,
                'spatial_extent': spatial_extent,
                'pixel_coords': list(zip(y_coords, x_coords)),
                'mask': component_mask,
                'mean_transitions': mean_transitions,
                'step_fit_quality': step_fit_result['fit_quality'],
                'step_amplitude': step_fit_result['amplitude'],
                'neighbor_fraction': neighbor_fraction
            }
            
            candidates.append(candidate)
            self.logger.info(f"Added cosmic ray candidate {i} with step fit quality {step_fit_result['fit_quality']:.3f}")
        
        self.logger.info(f"Found {len(candidates)} cosmic ray candidates after step fitting")
        
        return {
            'candidates': candidates,
            'mask': initial_mask,
            'num_candidates': len(candidates)
        }
    
    def _fit_step_function(self, time_series: np.ndarray) -> Dict:
        """
        Fit a step function to the time series data.
        Returns fit quality metrics and parameters.
        """
        if len(time_series) < 10:
            return {'is_good_fit': False, 'fit_quality': 0.0}
        
        # Try different step locations
        best_fit = None
        best_r_squared = -1
        best_location = -1
        
        # Search for step location (exclude first and last few points)
        start_search = max(1, len(time_series) // 10)
        end_search = min(len(time_series) - 3, len(time_series) - len(time_series) // 10)
        
        for step_location in range(start_search, end_search):
            try:
                r_squared, params = self._evaluate_step_fit(time_series, step_location)
                
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_location = step_location
                    best_fit = params
            except:
                continue
        
        if best_fit is None or best_r_squared < 0.5:  # Minimum R² threshold
            return {'is_good_fit': False, 'fit_quality': best_r_squared}
        
        # Additional validation criteria
        amplitude = best_fit['post_level'] - best_fit['pre_level']
        min_amplitude = getattr(self.config, 'cosmic_ray_min_step', 20.0)
        
        # Check if it looks like a cosmic ray step function
        is_cosmic_ray_like = (
            amplitude >= min_amplitude and  # Sufficient amplitude
            best_fit['pre_level'] < best_fit['post_level'] and  # Positive step
            abs(best_fit['pre_level']) < 30.0 and  # Pre-event near zero
            best_fit['post_level'] > min_amplitude * 0.8  # Post-event elevated
        )
        
        return {
            'is_good_fit': is_cosmic_ray_like and best_r_squared > 0.6,  # Higher threshold for cosmic rays
            'fit_quality': best_r_squared,
            'step_location': best_location,
            'amplitude': amplitude,
            'pre_level': best_fit['pre_level'],
            'post_level': best_fit['post_level']
        }
    
    def _evaluate_step_fit(self, time_series: np.ndarray, step_location: int) -> Tuple[float, Dict]:
        """
        Evaluate how well a step function fits at a given location.
        Returns R-squared and fit parameters.
        """
        # Split the time series
        pre_step = time_series[:step_location]
        post_step = time_series[step_location:]
        
        if len(pre_step) < 2 or len(post_step) < 2:
            return -1, {}
        
        # Calculate means for each segment
        pre_mean = np.mean(pre_step)
        post_mean = np.mean(post_step)
        
        # Create step function model
        step_model = np.concatenate([
            np.full(len(pre_step), pre_mean),
            np.full(len(post_step), post_mean)
        ])
        
        # Calculate R-squared
        ss_res = np.sum((time_series - step_model) ** 2)
        ss_tot = np.sum((time_series - np.mean(time_series)) ** 2)
        
        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Additional penalty for noisy fits
        pre_std = np.std(pre_step)
        post_std = np.std(post_step)
        amplitude = abs(post_mean - pre_mean)
        
        # Penalize if segments are too noisy relative to the step
        if amplitude > 0:
            noise_penalty = (pre_std + post_std) / (2 * amplitude)
            r_squared = r_squared * (1 - min(0.5, noise_penalty))
        
        params = {
            'pre_level': pre_mean,
            'post_level': post_mean,
            'pre_std': pre_std,
            'post_std': post_std
        }
        
        return r_squared, params
    
    def classify(self, candidates: Dict) -> List[Dict]:
        """Classify cosmic ray candidates based on step function fits."""
        classified = []
        
        for candidate in candidates['candidates']:
            # Check spatial extent
            if candidate['spatial_extent'] > self.config.cosmic_ray_max_spatial_extent:
                self.logger.debug(f"Candidate rejected: spatial extent {candidate['spatial_extent']} too large")
                continue
            
            # Check step fit quality
            if candidate['step_fit_quality'] < 0.6:
                self.logger.debug(f"Candidate rejected: step fit quality {candidate['step_fit_quality']:.3f} too low")
                continue
            
            # Check step amplitude
            if candidate['step_amplitude'] < self.config.cosmic_ray_min_intensity * 0.8:
                self.logger.debug(f"Candidate rejected: step amplitude {candidate['step_amplitude']:.1f} too low")
                continue
            
            # Check neighbor fraction
            if candidate['neighbor_fraction'] < 0.4:
                self.logger.debug(f"Candidate rejected: neighbor fraction {candidate['neighbor_fraction']:.2f} too low")
                continue
            
            # Calculate confidence based on step fit quality
            confidence = self._calculate_step_fit_confidence(candidate)
            
            classified_event = {
                'type': 'cosmic_ray',
                'confidence': confidence,
                **candidate
            }
            
            classified.append(classified_event)
        
        return classified
    
    def _calculate_step_fit_confidence(self, candidate: Dict) -> float:
        """Calculate confidence based on step function fit quality."""
        # Base confidence from fit quality
        confidence = candidate['step_fit_quality'] * 0.8
        
        # Bonus for good neighbor agreement
        confidence += candidate['neighbor_fraction'] * 0.2
        
        # Bonus for strong amplitude
        if candidate['step_amplitude'] >= 50:
            confidence += 0.1
        
        # Bonus for compact spatial extent
        if candidate['spatial_extent'] <= 5:
            confidence += 0.1
        
        # Small penalty for very large features (might be snowballs)
        if candidate['spatial_extent'] > 15:
            confidence -= 0.05
        
        return min(1.0, max(0.0, confidence))