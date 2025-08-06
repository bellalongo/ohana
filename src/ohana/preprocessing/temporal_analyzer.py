import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import logging
from scipy import stats

class TemporalAnalyzer:
    """Analyzes temporal patterns in difference images."""
    
    def __init__(self, sigma_threshold: float = 5.0):
        self.sigma_threshold = sigma_threshold
        self.logger = logging.getLogger(__name__)
    
    def analyze_temporal_patterns(self, diff_stack: np.ndarray) -> Dict:
        """Analyze when anomalies appear and their persistence."""
        # Calculate robust background statistics
        background_stats = self._calculate_background_statistics(diff_stack)
        threshold = background_stats['threshold']
        
        num_frames, height, width = diff_stack.shape
        
        # Initialize tracking arrays
        first_appearance = np.full((height, width), -1, dtype=np.int32)
        persistence_count = np.zeros((height, width), dtype=np.int32)
        max_intensity = np.zeros((height, width), dtype=np.float32)
        intensity_variance = np.zeros((height, width), dtype=np.float32)
        transition_count = np.zeros((height, width), dtype=np.int32)
        
        # Track states for telegraph noise detection
        prev_state = np.zeros((height, width), dtype=bool)
        
        temporal_evolution = []
        
        self.logger.info(f"Analyzing {num_frames} frames with {self.sigma_threshold}σ threshold")
        
        for frame_idx in tqdm(range(num_frames), desc='Temporal analysis'):
            diff_frame = diff_stack[frame_idx]
            
            # Find anomalies
            anomaly_mask = diff_frame > threshold
            
            # Update first appearance
            new_anomalies = anomaly_mask & (first_appearance == -1)
            first_appearance[new_anomalies] = frame_idx
            
            # Update persistence
            persistence_count[anomaly_mask] += 1
            
            # Update max intensity
            max_intensity = np.maximum(max_intensity, diff_frame)
            
            # Track state transitions for telegraph noise
            transitions = anomaly_mask != prev_state
            transition_count[transitions] += 1
            prev_state = anomaly_mask.copy()
            
            # Frame statistics
            if np.any(anomaly_mask):
                frame_stats = {
                    'frame': frame_idx,
                    'n_anomalies': np.sum(anomaly_mask),
                    'mean_intensity': np.mean(diff_frame[anomaly_mask]),
                    'max_intensity': np.max(diff_frame),
                    'anomaly_fraction': np.sum(anomaly_mask) / (height * width)
                }
            else:
                frame_stats = {
                    'frame': frame_idx,
                    'n_anomalies': 0,
                    'mean_intensity': 0,
                    'max_intensity': np.max(diff_frame),
                    'anomaly_fraction': 0
                }
            
            temporal_evolution.append(frame_stats)
        
        # Calculate intensity variance for persistent anomalies
        persistent_mask = persistence_count > num_frames * 0.1
        if np.any(persistent_mask):
            for y, x in np.argwhere(persistent_mask):
                values = [diff_stack[i, y, x] for i in range(num_frames)]
                intensity_variance[y, x] = np.var(values)
        
        return {
            'first_appearance': first_appearance,
            'persistence_count': persistence_count,
            'max_intensity': max_intensity,
            'intensity_variance': intensity_variance,
            'transition_count': transition_count,
            'temporal_evolution': temporal_evolution,
            'threshold_used': threshold,
            'background_stats': background_stats
        }
    
    def _calculate_background_statistics(self, diff_stack: np.ndarray) -> Dict:
        """Calculate robust background statistics."""
        # Use median absolute deviation for robust statistics
        flat_data = diff_stack.flatten()
        
        # Remove extreme outliers for background estimation
        percentile_low = np.percentile(flat_data, 1)
        percentile_high = np.percentile(flat_data, 99)
        mask = (flat_data > percentile_low) & (flat_data < percentile_high)
        background_data = flat_data[mask]
        
        median = np.median(background_data)
        mad = stats.median_abs_deviation(background_data)
        robust_std = 1.4826 * mad  # Convert MAD to std equivalent
        
        threshold = median + self.sigma_threshold * robust_std
        
        return {
            'median': median,
            'mad': mad,
            'robust_std': robust_std,
            'threshold': threshold,
            'sigma_threshold': self.sigma_threshold
        }