import numpy as np
from tqdm import tqdm
import logging
from scipy import stats

class TemporalAnalyzer:
    """
        Analyzes temporal patterns in difference images
    """
    def __init__(self, sigma_threshold=5.0):
        """
            Arguments:
                sigma_threshold (float): sigma value above the robust standard deviation
                    used to set the anomaly detection threshold
            Attributes:
                sigma_threshold (float): stored threshold multiplier
                logger (logging.Logger): module logger for this analyzer
        """
        # Store the sigma threshold for anomaly detection
        self.sigma_threshold = sigma_threshold

        # Create a logger scoped to the class
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_temporal_patterns(self, diff_stack):
        """
            Analyze when anomalies appear, their persistence, and other temporal features
            Arguments:
                diff_stack (np.ndarray): difference image stack of shape (T, H, W)
            Returns:
                dict: dictionary of temporal feature maps and run metadata containing:
                    * 'first_appearance' (np.ndarray): first frame index per pixel (H, W)
                    * 'persistence_count' (np.ndarray): number of anomalous frames per pixel (H, W)
                    * 'max_intensity' (np.ndarray): max intensity observed per pixel (H, W)
                    * 'intensity_variance' (np.ndarray): variance over time per pixel (H, W)
                    * 'transition_count' (np.ndarray): anomaly state transitions per pixel (H, W)
                    * 'temporal_evolution' (list[dict]): per-frame stats across the sequence
                    * 'threshold_used' (float): numeric threshold used for anomaly decisions
                    * 'background_stats' (dict): robust background statistics used to compute threshold
            Notes:
                * threshold is computed via robust statistics to resist outliers
                * transition_count helps identify telegraph-like switching behavior
        """
        # Calculate robust background statistics to define an anomaly threshold
        background_stats = self._calculate_background_statistics(diff_stack)

        # Unpack the scalar threshold from the statistics bundle
        threshold = background_stats['threshold']

        # Unpack the temporal and spatial dimensions
        num_frames, height, width = diff_stack.shape

        # Initialize arrays to store temporal features for each pixel
        first_appearance = np.full((height, width), -1, dtype=np.int32)
        persistence_count = np.zeros((height, width), dtype=np.int32)
        max_intensity = np.zeros((height, width), dtype=np.float32)
        intensity_variance = np.zeros((height, width), dtype=np.float32)
        transition_count = np.zeros((height, width), dtype=np.int32)

        # Track pixel anomaly state between frames for transition counting
        prev_state = np.zeros((height, width), dtype=bool)

        # Container for per-frame statistics across the sequence
        temporal_evolution = []

        # Log the analysis configuration
        self.logger.info(f"Analyzing {num_frames} frames with a {self.sigma_threshold}σ threshold ({threshold:.2f})")

        # Iterate through frames and compute temporal metrics
        for frame_idx in tqdm(range(num_frames), desc='Temporal analysis'):
            # Current difference frame
            diff_frame = diff_stack[frame_idx]

            # Identify pixels above the anomaly threshold
            anomaly_mask = diff_frame > threshold

            # Record the first frame where an anomaly appears
            new_anomalies = anomaly_mask & (first_appearance == -1)
            first_appearance[new_anomalies] = frame_idx

            # Increment persistence for currently anomalous pixels
            persistence_count[anomaly_mask] += 1

            # Update the maximum intensity seen so far
            max_intensity = np.maximum(max_intensity, diff_frame)

            # Count state transitions by comparing to previous frame state
            transitions = anomaly_mask != prev_state
            transition_count[transitions] += 1
            prev_state = anomaly_mask.copy()

            # Collect per-frame statistics for monitoring
            if np.any(anomaly_mask):
                frame_stats = {
                    'frame': int(frame_idx),
                    'n_anomalies': int(np.sum(anomaly_mask)),
                    'mean_intensity': float(np.mean(diff_frame[anomaly_mask])),
                    'max_intensity': float(np.max(diff_frame)),
                    'anomaly_fraction': float(np.sum(anomaly_mask) / (height * width))
                }
            else:
                frame_stats = {
                    'frame': int(frame_idx),
                    'n_anomalies': 0,
                    'mean_intensity': 0.0,
                    'max_intensity': float(np.max(diff_frame)),
                    'anomaly_fraction': 0.0
                }

            # Append the frame statistics to the evolution log
            temporal_evolution.append(frame_stats)

        # Create a mask for pixels that were anomalous at least once
        persistent_mask = persistence_count > 0

        # If we have any persistent pixels, compute their temporal variance
        if np.any(persistent_mask):
            # Get indices where persistence is positive
            persistent_coords = np.where(persistent_mask)

            # Gather time series for those coordinates
            pixel_time_series = diff_stack[:, persistent_coords[0], persistent_coords[1]]

            # Compute variance across time for each selected pixel
            intensity_variance[persistent_coords] = np.var(pixel_time_series, axis=0)

        # Return the full set of temporal products and metadata
        return {
            'first_appearance': first_appearance,
            'persistence_count': persistence_count,
            'max_intensity': max_intensity,
            'intensity_variance': intensity_variance,
            'transition_count': transition_count,
            'temporal_evolution': temporal_evolution,
            'threshold_used': float(threshold),
            'background_stats': background_stats
        }

    def _calculate_background_statistics(self, diff_stack):
        """
            Calculate robust background statistics using median absolute deviation (MAD)
            Arguments:
                diff_stack (np.ndarray): difference image stack of shape (T, H, W)
            Returns:
                dict: background statistics with keys:
                    * 'median' (float): median of clipped background distribution
                    * 'mad' (float): median absolute deviation (scaled-to-normal)
                    * 'robust_std' (float): robust standard deviation estimate
                    * 'threshold' (float): median + sigma_threshold * robust_std
                    * 'sigma_threshold' (float): sigma multiplier used to compute threshold
            Notes:
                * excludes extreme tails via percentile clipping to stabilize estimates
                * MAD with scale='normal' gives a std-equivalent under normality
        """
        # Flatten the entire stack to analyze the global intensity distribution
        flat_data = diff_stack.flatten()

        # Compute low/high percentiles for tail clipping
        percentile_low, percentile_high = np.percentile(flat_data, [1, 99])

        # Keep only values within the clipped percentile range
        background_data = flat_data[(flat_data > percentile_low) & (flat_data < percentile_high)]

        # Compute the median of the clipped background
        median = float(np.median(background_data))

        # Compute MAD scaled to std-equivalent units
        mad = float(stats.median_abs_deviation(background_data, scale='normal'))

        # Use MAD as a robust standard deviation
        robust_std = float(mad)

        # Compute the anomaly threshold from the robust stats
        threshold = float(median + self.sigma_threshold * robust_std)

        # Package and return the background statistics
        return {
            'median': median,
            'mad': mad,
            'robust_std': robust_std,
            'threshold': threshold,
            'sigma_threshold': float(self.sigma_threshold)
        }
