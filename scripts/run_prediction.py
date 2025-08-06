import argparse
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ohana.predict.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection on a new H2RG exposure file."
    )

    parser.add_argument(
        "--exposure",
        required=True,
        help="Path to the new exposure file or directory."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained U-Net model file (.pth)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training configuration YAML file."
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to save the detection results (list of anomalies)."
    )
    parser.add_argument(
        "--output_mask",
        required=True,
        help="Path to save the full predicted segmentation mask as an .npy file."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--output_processed_data",
        required=True,
        help="Path to save the preprocessed data cube as an .npy file."
    )

    args = parser.parse_args()

    start_time = time.time()
    try:
        predictor = Predictor(model_path=args.model, config_path=args.config)
        anomalies = predictor.predict(exposure_path=args.exposure)

        print(f"\nSaving {len(anomalies)} detected anomalies to {args.output_json}...")
        with open(args.output_json, 'w') as f:
            json.dump(anomalies, f, indent=4)

        print(f"Saving full prediction mask to {args.output_mask}...")
        if predictor.prediction_mask is not None:
            np.save(args.output_mask, predictor.prediction_mask)
        else:
            print("Warning: No prediction mask was generated to save.")

        # --- NEW LOGIC TO SAVE THE PROCESSED CUBE ---
        print(f"Saving processed data cube to {args.output_processed_data}...")
        if predictor.processed_cube is not None:
            np.save(args.output_processed_data, predictor.processed_cube)
        else:
            print("Warning: No processed data cube was generated to save.")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()