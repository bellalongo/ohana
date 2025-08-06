import numpy as np
from typing import Dict, List, Tuple
from scipy import signal, stats
from scipy.ndimage import label
import logging

class TelegraphNoiseDetector:
    """Detects Random Telegraph Noise in H2RG data based on SPIE paper methodology."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect(self, temporal_data: Dict, diff_stack: np.ndarray) -> Dict:
        """Detect RTN candidates using improved algorithm based on SPIE paper."""
        self.logger.info("Starting RTN detection...")
        
        transition_count = temporal_data['transition_count']
        persistence = temporal_data['persistence_count']
        max_intensity = temporal_data['max_intensity']
        intensity_variance = temporal_data['intensity_variance']
        
        # RTN should have some transitions but not too many (distinguishes from noise)
        min_transitions = self.config.rtn_min_transitions
        max_transitions = getattr(self.config, 'rtn_max_transitions', 50)  # Avoid pure noise
        
        has_transitions = (transition_count >= min_transitions) & (transition_count <= max_transitions)
        
        # RTN amplitude should be in reasonable range (from paper: mostly <300e-)
        intensity_in_range = (max_intensity >= self.config.rtn_amplitude_range[0]) & \
                           (max_intensity <= self.config.rtn_amplitude_range[1])
        
        # RTN should have moderate variance (too high = noise, too low = no signal)
        moderate_variance = intensity_variance > np.percentile(intensity_variance, 10)
        
        # Initial mask - pixels that might have RTN
        rtn_mask = has_transitions & intensity_in_range & moderate_variance
        
        # Find individual pixels (RTN is typically single-pixel phenomena)
        y_coords, x_coords = np.where(rtn_mask)
        
        candidates = []
        
        self.logger.info(f"Analyzing {len(y_coords)} potential RTN pixels...")
        
        for y, x in zip(y_coords, x_coords):
            # Extract time series for this pixel
            time_series = diff_stack[:, y, x]
            
            # Analyze telegraph characteristics using paper's methodology
            analysis = self._analyze_telegraph_signal_paper_method(time_series)
            
            if analysis['is_telegraph']:
                candidate = {
                    'type': 'rtn_candidate',
                    'position': (y, x),
                    'num_transitions': int(transition_count[y, x]),
                    'amplitude': analysis['amplitude'],
                    'high_state_value': analysis['high_state'],
                    'low_state_value': analysis['low_state'],
                    'frequency': analysis['frequency'],
                    'duty_cycle': analysis['duty_cycle'],
                    'period_frames': analysis['period_frames'],
                    'rtn_type': analysis['rtn_type'],  # 'high_freq_spike' or 'low_freq_square'
                    'gaussian_fit_quality': analysis['fit_quality'],
                    'max_intensity': max_intensity[y, x],
                    'time_series': time_series
                }
                
                candidates.append(candidate)
        
        self.logger.info(f"Found {len(candidates)} RTN candidates")
        
        return {
            'candidates': candidates,
            'mask': rtn_mask,
            'num_candidates': len(candidates)
        }
    
    def _analyze_telegraph_signal_paper_method(self, time_series: np.ndarray) -> Dict:
        """
        Analyze time series for RTN using methodology from SPIE paper.
        
        Paper method:
        1. Try to fit 2 Gaussian distributions to histogram of CDS values
        2. If fit succeeds, extract 3 RTN parameters: amplitude, period, time in high state
        3. Classify as high-frequency spike-like or low-frequency square-wave
        """
        # Remove any linear trend (detrend the data)
        detrended = signal.detrend(time_series)
        
        # Create histogram of signal values
        hist, bin_edges = np.histogram(detrended, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Try to fit two Gaussians (bimodal distribution)
        fit_result = self._fit_two_gaussians(bin_centers, hist)
        
        if not fit_result['success']:
            return self._failed_telegraph_analysis()
        
        # Extract the two states from Gaussian fit
        low_state = fit_result['mean1']
        high_state = fit_result['mean2']
        
        # Ensure proper ordering
        if low_state > high_state:
            low_state, high_state = high_state, low_state
        
        amplitude = high_state - low_state
        
        # Determine state assignment for each frame
        threshold = (low_state + high_state) / 2
        high_state_mask = detrended > threshold
        
        # Count transitions between states
        transitions = np.diff(high_state_mask.astype(int))
        num_transitions = np.sum(np.abs(transitions))
        
        if num_transitions == 0:
            return self._failed_telegraph_analysis()
        
        # Calculate RTN parameters as defined in paper
        duty_cycle = np.mean(high_state_mask)  # Time in high state
        
        # Estimate period (frames between transitions)
        if num_transitions >= 2:
            # Find transition points
            transition_points = np.where(np.abs(transitions) > 0)[0]
            if len(transition_points) >= 2:
                # Average period between transitions
                periods = np.diff(transition_points)
                avg_period = np.mean(periods) * 2  # Full cycle = 2 transitions
                period_frames = avg_period
            else:
                period_frames = len(time_series) / (num_transitions / 2)
        else:
            period_frames = len(time_series)
        
        # Calculate frequency
        frequency = num_transitions / (2 * len(time_series))  # Hz equivalent
        
        # Classify RTN type based on characteristics from paper
        rtn_type = self._classify_rtn_type(frequency, duty_cycle, amplitude, num_transitions, detrended)
        
        # Check if this meets RTN criteria
        is_telegraph = self._validate_rtn_characteristics(
            amplitude, frequency, num_transitions, fit_result['fit_quality'], detrended
        )
        
        return {
            'is_telegraph': is_telegraph,
            'amplitude': amplitude,
            'low_state': low_state,
            'high_state': high_state,
            'frequency': frequency,
            'duty_cycle': duty_cycle,
            'period_frames': period_frames,
            'num_transitions': num_transitions,
            'rtn_type': rtn_type,
            'fit_quality': fit_result['fit_quality']
        }
    
    def _fit_two_gaussians(self, x, y):
        """
        Fit two Gaussian distributions to histogram data.
        Returns fit parameters and quality metric.
        """
        try:
            from scipy.optimize import curve_fit
            
            # Define two-Gaussian model
            def two_gaussians(x, a1, mu1, sig1, a2, mu2, sig2):
                g1 = a1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2)
                g2 = a2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2)
                return g1 + g2
            
            # Initial guess - find two peaks in histogram
            peaks, _ = signal.find_peaks(y, height=np.max(y) * 0.1)
            
            if len(peaks) < 2:
                return {'success': False, 'fit_quality': 0.0}
            
            # Sort peaks by height and take top 2
            peak_heights = y[peaks]
            sorted_indices = np.argsort(peak_heights)[::-1]
            peak1_idx = peaks[sorted_indices[0]]
            peak2_idx = peaks[sorted_indices[1]]
            
            mu1_init = x[peak1_idx]
            mu2_init = x[peak2_idx]
            
            # Initial parameter guess
            p0 = [
                y[peak1_idx], mu1_init, np.std(x) / 4,  # First Gaussian
                y[peak2_idx], mu2_init, np.std(x) / 4   # Second Gaussian
            ]
            
            # Fit the curve
            popt, pcov = curve_fit(two_gaussians, x, y, p0=p0, maxfev=2000)
            
            # Calculate fit quality (R-squared)
            y_pred = two_gaussians(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'success': True,
                'mean1': popt[1],
                'mean2': popt[4],
                'sigma1': abs(popt[2]),
                'sigma2': abs(popt[5]),
                'amp1': popt[0],
                'amp2': popt[3],
                'fit_quality': r_squared
            }
            
        except Exception as e:
            self.logger.debug(f"Gaussian fitting failed: {e}")
            return {'success': False, 'fit_quality': 0.0}
    
    def _classify_rtn_type(self, frequency, duty_cycle, amplitude, num_transitions, time_series=None):
        """
        Classify RTN type based on paper characterization:
        - High frequency spike-like: short excursions from baseline, low duty cycle
        - Low frequency square-wave: longer periods, ~50% duty cycle
        """
        # Key characteristics from the paper:
        # High-freq spike: immediate recovery, low duty cycle (typically <20%)
        # Low-freq square: nearly equal time in both states (~50% duty cycle)
        
        square_wave_duty_tolerance = 0.15  # Within 15% of 50%
        spike_like_duty_threshold = 0.25   # Spike-like typically <25% duty cycle
        
        # Additional analysis if time series is available
        baseline_return_score = 0
        if time_series is not None:
            baseline_return_score = self._analyze_baseline_return_behavior(time_series)
        
        # Classification logic based on paper Figure 6
        if abs(duty_cycle - 0.5) < square_wave_duty_tolerance:
            # Nearly equal time in high and low states = square wave
            return 'low_freq_square'
        elif duty_cycle < spike_like_duty_threshold:
            # Low duty cycle suggests spike-like behavior
            if baseline_return_score > 0.7 or frequency > 0.05:
                return 'high_freq_spike'
            else:
                return 'irregular_spike'
        elif duty_cycle > (1 - spike_like_duty_threshold):
            # High duty cycle (inverted spikes - mostly high with low excursions)
            if baseline_return_score > 0.7:
                return 'high_freq_spike_inverted'
            else:
                return 'irregular_spike'
        else:
            # Intermediate duty cycle - need more analysis
            if frequency > 0.1 and baseline_return_score > 0.6:
                return 'high_freq_spike'
            elif frequency < 0.02:
                return 'low_freq_irregular'
            else:
                return 'irregular'
    
    def _validate_rtn_characteristics(self, amplitude, frequency, num_transitions, fit_quality, time_series=None):
        """Validate that detected signal meets RTN criteria from paper and is not cosmic rays."""
        
        # Amplitude check (paper shows most RTN < 300e-)
        amplitude_ok = (amplitude >= self.config.rtn_amplitude_range[0] and 
                       amplitude <= self.config.rtn_amplitude_range[1])
        
        # Frequency check
        frequency_ok = (frequency >= self.config.rtn_frequency_range[0] and
                       frequency <= self.config.rtn_frequency_range[1])
        
        # Minimum transitions
        transitions_ok = num_transitions >= self.config.rtn_min_transitions
        
        # Gaussian fit quality (must be reasonable bimodal distribution)
        fit_quality_threshold = getattr(self.config, 'rtn_fit_quality_threshold', 0.3)
        fit_ok = fit_quality >= fit_quality_threshold
        
        # Additional check: reject cosmic ray-like behavior
        cosmic_ray_rejection = True
        if time_series is not None:
            cosmic_ray_rejection = self._reject_cosmic_ray_like_behavior(time_series)
        
        return amplitude_ok and frequency_ok and transitions_ok and fit_ok and cosmic_ray_rejection
    
    def _reject_cosmic_ray_like_behavior(self, time_series):
        """
        Reject signals that look like cosmic rays rather than RTN.
        Cosmic rays: sudden large jumps that persist, high baseline drift
        RTN: oscillates around stable baseline
        """
        # Remove linear trend
        detrended = signal.detrend(time_series)
        
        # Check for sudden large jumps (cosmic ray characteristic)
        frame_to_frame_diff = np.abs(np.diff(detrended))
        large_jumps = frame_to_frame_diff > (3 * np.std(frame_to_frame_diff))
        large_jump_fraction = np.mean(large_jumps)
        
        # Too many large jumps suggests cosmic rays, not RTN
        if large_jump_fraction > 0.05:  # More than 5% large jumps
            return False
        
        # Check baseline stability over time
        # Split signal into chunks and check if baseline drifts significantly
        chunk_size = len(detrended) // 5
        if chunk_size < 10:
            return True  # Too short to analyze
        
        chunk_medians = []
        for i in range(0, len(detrended), chunk_size):
            chunk = detrended[i:i+chunk_size]
            if len(chunk) > 5:
                chunk_medians.append(np.median(chunk))
        
        if len(chunk_medians) < 3:
            return True
        
        # Check if baseline drifts too much (suggests cosmic ray persistence)
        baseline_drift = np.std(chunk_medians)
        signal_std = np.std(detrended)
        
        # If baseline drifts more than half the signal variation, reject
        if baseline_drift > 0.5 * signal_std:
            return False
        
        # Check for excessive signal range compared to typical RTN
        signal_range = np.ptp(detrended)
        if signal_range > 200:  # Arbitrary threshold - may need tuning
            return False
        
        return True
    
    def _failed_telegraph_analysis(self):
        """Return failed analysis result."""
        return {
            'is_telegraph': False,
            'amplitude': 0,
            'low_state': 0,
            'high_state': 0,
            'frequency': 0,
            'duty_cycle': 0,
            'period_frames': 0,
            'num_transitions': 0,
            'rtn_type': 'none',
            'fit_quality': 0.0
        }
    
    def classify(self, candidates: Dict) -> List[Dict]:
        """Classify RTN candidates with confidence scoring."""
        classified = []
        
        for candidate in candidates['candidates']:
            # Base validation
            valid_rtn = (
                candidate['frequency'] >= self.config.rtn_frequency_range[0] and
                candidate['frequency'] <= self.config.rtn_frequency_range[1] and
                candidate['amplitude'] >= self.config.rtn_amplitude_range[0] and
                candidate['amplitude'] <= self.config.rtn_amplitude_range[1]
            )
            
            if not valid_rtn:
                continue
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_rtn_confidence(candidate)
            
            if confidence > getattr(self.config, 'rtn_min_confidence', 0.5):
                classified_event = {
                    'type': 'telegraph_noise',
                    'confidence': confidence,
                    **candidate
                }
                
                classified.append(classified_event)
        
        return classified
    
    def _calculate_rtn_confidence(self, candidate):
        """Calculate confidence score for RTN detection."""
        confidence = 0.0
        
        # Gaussian fit quality (0-40% of confidence)
        fit_quality_weight = min(0.4, candidate['gaussian_fit_quality'])
        confidence += fit_quality_weight
        
        # Number of transitions (0-30% of confidence)
        transition_score = min(0.3, candidate['num_transitions'] / 20.0)
        confidence += transition_score
        
        # Amplitude in typical range (0-20% of confidence)
        if candidate['amplitude'] <= 100:  # Most RTN in paper < 100e-
            confidence += 0.2
        elif candidate['amplitude'] <= 200:
            confidence += 0.1
        
        # RTN type classification (0-10% of confidence)
        if candidate['rtn_type'] in ['high_freq_spike', 'low_freq_square']:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _analyze_baseline_return_behavior(self, time_series):
        """
        Analyze how quickly the signal returns to baseline after excursions.
        High-frequency spike-like RTN shows immediate recovery (paper characteristic).
        Also checks for true oscillatory behavior vs cosmic ray-like spikes.
        
        Returns a score 0-1 where 1 = perfect spike-like RTN behavior.
        """
        # Detrend the signal
        detrended = signal.detrend(time_series)
        
        # Define baseline as median (more robust than mean)
        baseline = np.median(detrended)
        baseline_std = np.std(detrended)
        
        # Check if this looks like RTN vs cosmic rays/other anomalies
        baseline_stability_score = self._check_baseline_stability(detrended, baseline, baseline_std)
        if baseline_stability_score < 0.3:
            return 0.0  # Not RTN-like behavior
        
        # Find excursions from baseline (both positive and negative)
        excursion_threshold = baseline_std * 1.5  # More sensitive threshold
        excursion_mask = np.abs(detrended - baseline) > excursion_threshold
        
        if np.sum(excursion_mask) < 3:
            return 0.0  # Not enough excursions to analyze
        
        # Analyze excursion characteristics
        immediate_return_score = self._analyze_immediate_return(detrended, baseline, excursion_mask)
        oscillatory_behavior_score = self._analyze_oscillatory_pattern(detrended, baseline)
        
        # Combine scores
        total_score = (baseline_stability_score * 0.4 + 
                      immediate_return_score * 0.4 + 
                      oscillatory_behavior_score * 0.2)
        
        return min(1.0, total_score)
    
    def _check_baseline_stability(self, detrended, baseline, baseline_std):
        """
        Check if the signal has a stable baseline with excursions, rather than
        being dominated by large spikes (cosmic rays) or drift.
        """
        # Calculate what fraction of points are near baseline
        near_baseline_mask = np.abs(detrended - baseline) < baseline_std
        baseline_fraction = np.mean(near_baseline_mask)
        
        # RTN should spend significant time near baseline
        if baseline_fraction < 0.5:
            return 0.0
        
        # Check for excessive outliers (suggests cosmic rays, not RTN)
        outlier_threshold = baseline + 5 * baseline_std
        outlier_fraction = np.mean(detrended > outlier_threshold)
        
        if outlier_fraction > 0.1:  # More than 10% outliers suggests not RTN
            return 0.0
        
        # Check signal range is reasonable for RTN
        signal_range = np.ptp(detrended)  # peak-to-peak
        if signal_range > 10 * baseline_std:  # Excessive range
            return baseline_fraction * 0.5  # Penalize but don't eliminate
        
        return baseline_fraction
    
    def _analyze_immediate_return(self, detrended, baseline, excursion_mask):
        """
        Analyze if excursions return immediately to baseline (RTN characteristic).
        """
        # Find excursion segments
        excursion_changes = np.diff(excursion_mask.astype(int))
        excursion_starts = np.where(excursion_changes == 1)[0] + 1
        excursion_ends = np.where(excursion_changes == -1)[0] + 1
        
        # Handle edge cases
        if len(excursion_starts) == 0:
            return 0.0
            
        if excursion_mask[0]:
            excursion_starts = np.concatenate([[0], excursion_starts])
        if excursion_mask[-1]:
            excursion_ends = np.concatenate([excursion_ends, [len(excursion_mask)]])
        
        min_length = min(len(excursion_starts), len(excursion_ends))
        excursion_starts = excursion_starts[:min_length]
        excursion_ends = excursion_ends[:min_length]
        
        if len(excursion_starts) == 0:
            return 0.0
        
        # Analyze excursion durations
        excursion_durations = excursion_ends - excursion_starts
        short_excursions = np.sum(excursion_durations <= 3)  # 3 frames or less
        
        # RTN spikes should be brief
        immediate_return_score = short_excursions / len(excursion_durations)
        
        return immediate_return_score
    
    def _analyze_oscillatory_pattern(self, detrended, baseline):
        """
        Check for true oscillatory behavior characteristic of RTN.
        """
        # Look for alternating pattern between states
        baseline_std = np.std(detrended)
        threshold = baseline + baseline_std
        
        # Create binary signal (above/below threshold)
        binary_signal = (detrended > threshold).astype(int)
        
        # Count runs (consecutive same values)
        runs = []
        current_run = 1
        for i in range(1, len(binary_signal)):
            if binary_signal[i] == binary_signal[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        if len(runs) < 4:  # Need at least a few transitions
            return 0.0
        
        # RTN should have relatively short runs (quick transitions)
        short_runs = np.sum(np.array(runs) <= 5)
        oscillatory_score = short_runs / len(runs)
        
        return oscillatory_score