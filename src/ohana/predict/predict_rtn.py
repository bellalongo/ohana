import numpy as np
from typing import Dict, List
from scipy import signal, stats
from scipy.optimize import curve_fit
import logging

from .base_predictor import BaseDetector

class TelegraphNoiseDetector(BaseDetector):
    """
        Detects Random Telegraph Noise (RTN) by analyzing the distribution of pixel values
        over time, based on methodologies from SPIE papers. RTN is characterized by pixels
        switching between two or more discrete states
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
            Detect RTN candidates by looking for pixels with a high number of state transitions
            and bimodal intensity distributions
            Arguments:
                temporal_data (Dict): dictionary of temporal features from TemporalAnalyzer with keys:
                    * 'transition_count' (np.ndarray): number of state transitions observed
                    * 'max_intensity' (np.ndarray): peak intensity over time
                    * 'intensity_variance' (np.ndarray): variance of intensity over time
                diff_stack (np.ndarray): difference image stack of shape (T, H, W)
            Returns:
                Dict: dictionary with key 'candidates' holding a list of RTN candidate detections
            Notes:
                * RTN is commonly a single-pixel phenomenon, so we analyze pixels individually
                * initial screening is permissive; detailed histogram fitting refines the set
        """
        # Announce start of RTN detection
        self.logger.info("Starting RTN detection...")

        # Extract relevant temporal features
        transition_count = temporal_data['transition_count']
        max_intensity = temporal_data['max_intensity']
        intensity_variance = temporal_data['intensity_variance']

        """Initial Screening"""
        # Has a characteristic number of transitions within configured bounds
        has_transitions = ((transition_count >= self.config.rtn_min_transitions) &
                           (transition_count <= self.config.rtn_max_transitions))

        # Amplitude falls within a known range
        intensity_in_range = ((max_intensity >= self.config.rtn_amplitude_range[0]) &
                              (max_intensity <= self.config.rtn_amplitude_range[1]))

        # Variance should be moderate, not zero and not purely noisy
        positive_variances = intensity_variance[intensity_variance > 0]
        variance_threshold = np.percentile(positive_variances, 10) if positive_variances.size > 0 else 0.0
        moderate_variance = intensity_variance > variance_threshold

        # Combine screening criteria
        rtn_mask = has_transitions & intensity_in_range & moderate_variance

        # Extract coordinates of potential RTN pixels
        y_coords, x_coords = np.where(rtn_mask)

        # Log how many pixels we will analyze in detail
        self.logger.info(f"Analyzing {len(y_coords)} potential RTN pixels...")

        # Prepare container for candidate dictionaries
        candidates = []

        # Analyze each pixel individually
        for y, x in zip(y_coords, x_coords):
            # Extract time series for this pixel
            time_series = diff_stack[:, y, x]

            # Analyze the signal for telegraph-like characteristics
            analysis = self._analyze_telegraph_signal(time_series)

            # If analysis suggests telegraph behavior, record as candidate
            if analysis['is_telegraph']:
                candidate = {
                    'type': 'rtn_candidate',
                    'position': (int(y), int(x)),
                    'num_transitions': int(transition_count[y, x]),
                    'amplitude': float(analysis['amplitude']),
                    'high_state_value': float(analysis['high_state']),
                    'low_state_value': float(analysis['low_state']),
                    'fit_quality': float(analysis['fit_quality']),
                    'max_intensity': float(max_intensity[y, x])
                }

                # Append candidate to the list
                candidates.append(candidate)

        # Log final candidate count
        self.logger.info(f"Found {len(candidates)} RTN candidates after detailed analysis.")

        # Return candidates in a dictionary wrapper
        return {'candidates': candidates}

    def _analyze_telegraph_signal(self, time_series):
        """
            Analyze a time series by attempting to fit a bimodal (two-Gaussian) distribution
            to its histogram of values
            Arguments:
                time_series (np.ndarray): vector of length T (single pixel over time)
            Returns:
                Dict: analysis result with keys:
                    * 'is_telegraph' (bool): whether telegraph criteria are satisfied
                    * 'amplitude' (float): difference between high and low state means
                    * 'low_state' (float): estimated low state mean
                    * 'high_state' (float): estimated high state mean
                    * 'fit_quality' (float): R^2 of the two-Gaussian fit
            Notes:
                * detrending removes slow drifts that can blur the histogram modes
                * amplitude must fall within the configured RTN range
        """
        # Detrend the data to remove linear drifts
        detrended = signal.detrend(time_series)

        # Create a histogram of the signal values
        hist, bin_edges = np.histogram(detrended, bins='auto', density=True)

        # Compute bin centers for fitting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Fit two Gaussians to the histogram
        fit_result = self._fit_two_gaussians(bin_centers, hist)

        # Early exit if the fit did not converge properly
        if not fit_result['success']:
            return {'is_telegraph': False}

        # Order the states as low and high
        low_state, high_state = sorted([fit_result['mean1'], fit_result['mean2']])

        # Compute the amplitude between the two states
        amplitude = high_state - low_state

        # Validate that the fit meets RTN criteria
        is_telegraph = ((fit_result['fit_quality'] > self.config.rtn_fit_quality_threshold) &
                        (self.config.rtn_amplitude_range[0] <= amplitude <= self.config.rtn_amplitude_range[1]))

        # Package the analysis result
        return {
            'is_telegraph': bool(is_telegraph),
            'amplitude': float(amplitude),
            'low_state': float(low_state),
            'high_state': float(high_state),
            'fit_quality': float(fit_result['fit_quality'])
        }

    def _fit_two_gaussians(self, x, y):
        """
            Fit a two-Gaussian model to provided histogram data
            Arguments:
                x (np.ndarray): histogram bin centers
                y (np.ndarray): histogram densities at bin centers
            Returns:
                Dict: fit result with keys:
                    * 'success' (bool): whether the fit succeeded
                    * 'mean1' (float): mean of the first Gaussian
                    * 'mean2' (float): mean of the second Gaussian
                    * 'fit_quality' (float): R^2 of the two-Gaussian model
            Notes:
                * peak detection seeds the initial parameters for stable optimization
                * bounds keep parameters physical and avoid degenerate solutions
        """
        # Define a sum of two Gaussians
        def two_gaussians(xv, a1, mu1, sig1, a2, mu2, sig2):
            g1 = a1 * np.exp(-0.5 * ((xv - mu1) / sig1) ** 2)
            g2 = a2 * np.exp(-0.5 * ((xv - mu2) / sig2) ** 2)
            return g1 + g2

        # Attempt parameter estimation with robust initialization
        try:
            # Find peaks in the histogram to initialize the fit
            peak_height = np.max(y) * 0.1 if np.max(y) > 0 else 0.0
            peaks, _ = signal.find_peaks(y, height=peak_height, distance=3)

            # Require at least two peaks to proceed
            if len(peaks) < 2:
                return {'success': False}

            # Choose the two highest peaks as initial modes
            peak_indices = sorted(peaks, key=lambda i: y[i], reverse=True)[:2]

            # Initial guess for the parameters
            p0 = [
                y[peak_indices[0]], x[peak_indices[0]], max(1e-6, np.std(x) / 4.0),
                y[peak_indices[1]], x[peak_indices[1]], max(1e-6, np.std(x) / 4.0)
            ]

            # Constrain parameters to positive amplitudes and widths
            popt, _ = curve_fit(two_gaussians, x, y, p0=p0, maxfev=5000, bounds=(0, np.inf))

            # Predicted histogram from the fitted model
            y_pred = two_gaussians(x, *popt)

            # Compute R^2 as fit quality
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2) if np.any(y != y.mean()) else 0.0
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Return the fit summary
            return {
                'success': True,
                'mean1': float(popt[1]),
                'mean2': float(popt[4]),
                'fit_quality': float(r_squared)
            }

        # On failure, indicate unsuccessful fit
        except (RuntimeError, ValueError, FloatingPointError):
            return {'success': False}

    def classify(self, candidates):
        """
            Classify RTN candidates based on confidence
            Arguments:
                candidates (Dict): dictionary with key 'candidates' from detect()
            Returns:
                List[Dict]: final list of classified RTN events
        """
        # Container for classified events
        classified = []

        # Iterate through the proposed candidates
        for candidate in candidates.get('candidates', []):
            # Compute confidence score for this candidate
            confidence = self._calculate_confidence(candidate)

            # Keep only sufficiently confident events
            if confidence >= self.config.rtn_min_confidence:
                classified_event = {
                    'type': 'telegraph_noise',
                    'confidence': float(confidence),
                    **candidate
                }

                # Append to final set
                classified.append(classified_event)

        # Return classified events
        return classified

    def _calculate_confidence(self, c):
        """
            Calculate a confidence score for an RTN candidate
            Arguments:
                c (Dict): single candidate dictionary
            Returns:
                float: confidence score in [0, 1]
            Notes:
                * primarily driven by bimodal fit quality, with a boost for transition count
                * transitions are capped to avoid over-rewarding very noisy toggling
        """
        # Contribution from fit quality
        fit_score = c['fit_quality'] * 0.7

        # Contribution from number of transitions, capped at the configured maximum
        transition_ratio = c['num_transitions'] / max(1, self.config.rtn_max_transitions)
        transition_score = min(0.3, transition_ratio * 0.3)

        # Combine and clamp to [0, 1]
        return min(1.0, float(fit_score + transition_score))
