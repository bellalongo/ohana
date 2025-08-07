from dataclasses import dataclass
from typing import Tuple

@dataclass
class DetectorConfig:
    """
        Dataclass holding detector and processing parameters
        Provides a single, structured object for configuration
        Attributes:
            sigma_threshold (float): sigma multiplier used for temporal thresholding
            min_anomaly_pixels (int): minimum connected pixels for a valid anomaly
            min_confidence (float): global minimum confidence required for classification

            cosmic_ray_min_intensity (float): minimum per-pixel peak intensity for screening
            cosmic_ray_max_spatial_extent (int): maximum size of a CR connected component
            cosmic_ray_min_step (float): minimum fitted step amplitude for CR acceptance

            rtn_min_transitions (int): minimum number of state transitions for RTN
            rtn_max_transitions (int): maximum number of state transitions for RTN
            rtn_amplitude_range (Tuple[float, float]): allowed amplitude range for RTN states
            rtn_frequency_range (Tuple[float, float]): allowed toggling frequency range (if used)
            rtn_fit_quality_threshold (float): minimum R^2 for two-Gaussian fit
            rtn_min_confidence (float): minimum confidence for RTN classification
    """
    # General detection parameters
    sigma_threshold: float = 5.0
    min_anomaly_pixels: int = 1
    min_confidence: float = 0.5

    # Cosmic ray parameters
    cosmic_ray_min_intensity: float = 40.0
    cosmic_ray_max_spatial_extent: int = 15
    cosmic_ray_min_step: float = 25.0

    # Telegraph noise parameters
    rtn_min_transitions: int = 2
    rtn_max_transitions: int = 50
    rtn_amplitude_range: Tuple[float, float] = (10.0, 300.0)
    rtn_frequency_range: Tuple[float, float] = (0.001, 0.5)
    rtn_fit_quality_threshold: float = 0.3
    rtn_min_confidence: float = 0.5

    # Snowball parameters
    snowball_min_confidence: float = 0.2
    snowball_min_intensity: float = 30.0
    snowball_max_intensity: float = 500.0
    snowball_min_radius: int = 3
    snowball_max_radius: int = 15
    snowball_circularity_threshold: float = 0.7
    snowball_expansion_rate: float = 0.1  # pixels per frame
    snowball_min_area: int = 75
    snowball_max_area: int = 120000