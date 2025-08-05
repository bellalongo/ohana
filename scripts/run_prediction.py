import argparse
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ohana.predict.predictor import Predictor

def main():
    """Main entry-point for the prediction script."""
    parser = argparse.ArgumentParser(description="Run anomaly detection on a new H2RG exposure file.")
    parser.add_argument("--exposure", required=True, help="Path to the new exposure file or directory.")
    parser.add_argument("--model", required=True, help="Path to the trained PyTorch model file (.pth).")
    parser.add_argument("--config", required=True, help="Path to the training configuration YAML file.")
    parser.add_argument("--output", required=True, help="Path to save the detection results as a JSON file.")
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--save_processed_data",
        help="Optional. Path to save the corrected and differenced data cube as an .npy file for faster visualization."
    )
    args = parser.parse_args()

    start_time = time.time()
    try:
        predictor = Predictor(model_path=args.model, config_path=args.config)
        anomalies = predictor.predict(exposure_path=args.exposure)

        if anomalies:
            print(f"\nSuccess! Detected {len(anomalies)} anomalies.")
        else:
            print("\nSuccess! No anomalies were detected.")

        print(f"Saving detection results to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(anomalies, f, indent=4)
        print("Save complete.")

        # --- NEW LOGIC TO SAVE PROCESSED DATA ---
        if args.save_processed_data:
            print(f"Saving processed data cube to {args.save_processed_data}...")
            if predictor.processed_cube is not None:
                np.save(args.save_processed_data, predictor.processed_cube)
                print("Save complete.")
            else:
                print("Warning: No processed data cube was available to save.")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
