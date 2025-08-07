"""
    Ohana: A Python package for detecting signals in scientific data.
"""
__version__ = "0.1.0"

from .predict.predictor import Predictor

from .predict.base_predictor import BaseDetector
from .predict.predict_cosmic_rays import CosmicRayDetector
from .predict.predict_rtn import TelegraphNoiseDetector
from .predict.predict_snowball import SnowballDetector

from .models.unet_3d import UNet3D

from .preprocessing.data_loader import DataLoader
from .preprocessing.preprocessor import Preprocessor
from .training import injections

from .training.dataset import OhanaDataset
from .training.injections import *
from .training.training_set_creator import DataSetCreator

from .visualization.plotter import ResultVisualizer

from .config import DetectorConfig

__all__ = [
    "UNet3D",
    "Predictor",
    "BaseDetector",
    "CosmicRayDetector",
    "TelegraphNoiseDetector",
    "SnowballDetector",
    "DataLoader",
    "Preprocessor",
    "ReferencePixelCorrector",
    "OhanaDataset",
    "injections",
    "DataSetCreator",
    "ResultVisualizer",
    "DetectorConfig"
    "__version__",
]