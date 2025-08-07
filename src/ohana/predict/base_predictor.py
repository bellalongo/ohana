from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
import logging

class BaseDetector(ABC):
    """
        Abstract base class for all anomaly detectors. It defines the common
        interface that all detectors must implement
    """
    def __init__(self, config):
        """
            Arguments:
                config (Any): configuration object (e.g., from YAML) containing
                    parameters for the detector
            Attributes:
                config: stored configuration reference
                logger: module logger for this detector (because prints are chaos)
        """
        # Init initialization
        super().__init__()

        # Store configuration (keep it around for subclasses)
        self.config = config

        # Create a logger scoped to the subclass name
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def detect(self, temporal_data, diff_stack):
        """
            Detect anomaly candidates based on temporal features and the data cube
            Arguments:
                temporal_data (Dict): dictionary of temporal features from TemporalAnalyzer
                diff_stack (np.ndarray): difference image stack of shape (T, H, W)
            Returns:
                Dict: dictionary containing the list of candidate detections
        """
        pass

    @abstractmethod
    def classify(self, candidates):
        """
            Classify and filter the detected candidates, returning the final list of events
            Arguments:
                candidates (Dict): dictionary of candidates produced by detect
            Returns:
                List[Dict]: final, classified anomaly events (aka the good stuff)
        """
        pass
