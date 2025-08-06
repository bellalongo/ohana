from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple
import logging

class BaseDetector(ABC):
    """Abstract base class for anomaly detectors."""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def detect(self, temporal_data: Dict, diff_stack: np.ndarray) -> Dict:
        """Detect anomalies in the data."""
        pass
    
    @abstractmethod
    def classify(self, candidates: Dict) -> List[Dict]:
        """Classify detected anomalies."""
        pass
    
    def _get_region_around_pixel(self, y: int, x: int, data: np.ndarray, 
                               radius: int = 10) -> np.ndarray:
        """Extract region around a pixel."""
        height, width = data.shape[:2]
        y_start = max(0, y - radius)
        y_end = min(height, y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(width, x + radius + 1)
        return data[y_start:y_end, x_start:x_end]