import os
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Use relative imports for the new package structure
from ..preprocessing.data_loader import DataLoader
from ..preprocessing.preprocessor import Preprocessor
from ..models.unet_3d import UNet3D
from .predict_cosmic_rays import CosmicRayDetector
from .predict_rtn import TelegraphNoiseDetector
from .predict_snowball import SnowballDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Predictor:
    """
        Main class for running the anomaly detection pipeline on an exposure
    """
    def __init__(self, model_path, config):
        """
            Arguments:
                model_path (str): path to the trained model file
                config (DetectorConfig): dataclass object holding all parameters
            Attributes:
                config (DetectorConfig): stored configuration object
                preprocessor (Preprocessor): preprocessing pipeline utilities
                data_loader (DataLoader): exposure loading utility
                model (Any or None): placeholder for model (rule-based mode if None)
                cosmic_ray_detector (CosmicRayDetector): rule-based CR detector
                rtn_detector (TelegraphNoiseDetector): rule-based RTN detector
        """
        # Store the configuration
        self.config = config

        # Create the preprocessor with the provided configuration
        self.preprocessor = Preprocessor(config)

        # Instantiate the data loader
        self.data_loader = DataLoader()

        # Load or stub the model as configured
        self.model = self.load_model(model_path)

        """Initialize rule-based detectors"""
        # Cosmic ray detector with shared configuration
        self.cosmic_ray_detector = CosmicRayDetector(config)

        # Telegraph noise detector with shared configuration
        self.rtn_detector = TelegraphNoiseDetector(config)

        # Telegraph noise detector with shared configuration
        self.snowball_detector = SnowballDetector(config)

    def load_model(self, model_path):
        """
            Load the UNet3D model from the specified path
            Arguments:
                model_path (str): path to the trained model file
            Returns:
                None: placeholder for rule-based-only inference
            Notes:
                * rule-based detectors are active regardless of model presence
        """
        # Announce model path and operating mode
        logging.info(f"Model path provided: {model_path}. Rule-based detectors will be used.")

        # Return None to indicate rule-based-only pipeline
        return None

    def _get_output_paths(self, exposure_path, output_dir):
        """
            Helper to generate standardized output file paths
            Arguments:
                exposure_path (str): path to the input exposure
                output_dir (str): base output directory
            Returns:
                dict: mapping of artifact names to their resolved file paths
            Notes:
                * organizes outputs under raw/, processed/, temporal/ subfolders
        """
        # Derive a base name from the exposure path
        base_name = Path(exposure_path).stem

        # Construct the path bundle for artifacts
        return {
            "raw": Path(output_dir) / "raw" / f"{base_name}_raw.npy",
            "processed": Path(output_dir) / "processed" / f"{base_name}_processed.npy",
            "temporal": Path(output_dir) / "temporal" / f"{base_name}_temporal.npz",
        }

    def predict(self, exposure_path, output_dir="prediction_outputs"):
        """
            Run the full anomaly detection pipeline on a single exposure
            Arguments:
                exposure_path (str): path to the exposure file
                output_dir (str): output directory for cached artifacts
            Returns:
                list[dict]: consolidated list of detected and classified anomalies
            Notes:
                * uses caching to skip repeated preprocessing on the same exposure
                * runs rule-based detectors on the processed products
        """
        # Announce the exposure under analysis
        logging.info(f"--- Analyzing exposure: {exposure_path} ---")

        # Compute output paths and ensure directories exist
        paths = self._get_output_paths(exposure_path, output_dir)
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)

        """1) Load or compute raw stack"""
        # Load cached raw data if available
        if os.path.exists(paths["raw"]):
            logging.info(f"Loading cached raw data from {paths['raw']}...")
            raw_stack = np.load(paths["raw"])
        # Otherwise load from source and cache
        else:
            logging.info("Loading and preprocessing exposure...")
            raw_stack = self.data_loader.load_exposure(exposure_path)
            logging.info(f"Saving raw data to {paths['raw']}...")
            np.save(paths["raw"], raw_stack)

        """2) Load or compute processed stack (reference-pixel corrected)"""
        # Load cached processed stack if available
        if os.path.exists(paths["processed"]):
            logging.info(f"Loading cached processed data from {paths['processed']}...")
            processed_stack = np.load(paths["processed"])
        # Otherwise apply reference pixel correction and cache
        else:
            logging.info(f"Applying reference pixel correction to stack of shape {raw_stack.shape}...")
            processed_stack = self.preprocessor.correct_reference_pixels(raw_stack)
            logging.info(f"Saving processed data to {paths['processed']}...")
            np.save(paths["processed"], processed_stack)

        """3) Load or compute temporal features"""
        # Load cached temporal features if available
        if os.path.exists(paths["temporal"]):
            logging.info(f"Loading cached temporal features from {paths['temporal']}...")
            temporal_data = np.load(paths["temporal"], allow_pickle=True)
            temporal_features = {key: temporal_data[key] for key in temporal_data.files}
        # Otherwise analyze temporal patterns and cache
        else:
            logging.info("Performing temporal analysis...")
            temporal_features = self.preprocessor.analyze_temporal(processed_stack)
            logging.info(f"Saving temporal features to {paths['temporal']}...")
            np.savez(paths["temporal"], **temporal_features)

        """4) Run detection and classification"""
        # Execute configured detectors and collect events
        logging.info("Running advanced detection algorithms...")
        all_anomalies = self._detect_and_classify_anomalies(temporal_features, processed_stack)

        # Summarize the pipeline run
        logging.info("--- Analysis Complete ---")
        logging.info(f"Found a total of {len(all_anomalies)} anomalies across all types.")

        # Return the consolidated event list
        return all_anomalies

    def _detect_and_classify_anomalies(self, temporal_features, diff_stack):
        """
            Run all configured detectors and return a consolidated list of anomalies
            Arguments:
                temporal_features (dict): dictionary of temporal feature maps
                diff_stack (np.ndarray): processed difference stack (T, H, W)
            Returns:
                list[dict]: list of classified anomaly events across detectors
            Notes:
                * order: detect cosmic rays -> classify, then detect RTN -> classify
        """
        # Container for all classified events
        all_classified_events = []

        """Cosmic Rays"""
        # Detect cosmic ray candidates
        logging.info("Detecting cosmic ray candidates...")
        cr_candidates = self.cosmic_ray_detector.detect(temporal_features, diff_stack)

        # If candidates exist, classify them and collect results
        if cr_candidates and cr_candidates.get('candidates'):
            logging.info(f"Classifying {len(cr_candidates['candidates'])} cosmic ray candidates...")
            classified_crs = self.cosmic_ray_detector.classify(cr_candidates)
            all_classified_events.extend(classified_crs)
            logging.info(f"Finalized {len(classified_crs)} cosmic ray events.")

        """Telegraph Noise"""
        # Detect telegraph noise candidates
        logging.info("Detecting telegraph noise candidates...")
        rtn_candidates = self.rtn_detector.detect(temporal_features, diff_stack)

        # If candidates exist, classify them and collect results
        if rtn_candidates and rtn_candidates.get('candidates'):
            logging.info(f"Classifying {len(rtn_candidates['candidates'])} telegraph noise candidates...")
            classified_rtns = self.rtn_detector.classify(rtn_candidates)
            all_classified_events.extend(classified_rtns)
            logging.info(f"Finalized {len(classified_rtns)} telegraph noise events.")

        # --- Snowballs ---
        logging.info("Detecting snowball candidates...")
        sb_candidates = self.snowball_detector.detect(temporal_features, diff_stack)

        if sb_candidates and sb_candidates.get('candidates'):
            logging.info(f"Classifying {len(sb_candidates['candidates'])} snowball candidates...")
            classified_sbs = self.snowball_detector.classify(sb_candidates)
            all_classified_events.extend(classified_sbs)
            logging.info(f"Finalized {len(classified_sbs)} snowball events.")

        # Return the full set of classified events
        return all_classified_events
