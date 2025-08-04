import argparse
import sys
import os
import time
import json

# Add the parent directory ('ohana-main') to the Python path
# This allows the script to find and import the 'ohana' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ohana.predict.predictor import Predictor

def main():
    """Main entry-point for the prediction script."""
    parser = argparse.ArgumentParser(
        description="Run anomaly detection on a new H2RG exposure file using a trained model."
    )
    parser.add_argument(
        "--exposure",
        required=True,
        help="Path to the new exposure file to be analyzed (e.g., data.h5, data.fits)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained PyTorch model file (.pth)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training configuration YAML file (creator_config.yaml)."
    )
    parser.add_argument(
        "--output",
        help="Optional. Path to save the detection results as a JSON file."
    )
    args = parser.parse_args()

    # --- Execution ---
    start_time = time.time()

    try:
        # 1. Initialize the predictor with the model and config
        predictor = Predictor(model_path=args.model, config_path=args.config)

        # 2. Run the prediction on the new exposure
        anomalies = predictor.predict(exposure_path=args.exposure)

        # 3. Print the results in a user-friendly format
        if anomalies:
            print(f"\nSuccess! Detected {len(anomalies)} anomalies:")
            anomalies.sort(key=lambda x: x['type'])
            for anomaly in anomalies:
                loc = anomaly['location_px']
                conf = anomaly.get('confidence', -1) * 100
                print(f"  - Type: {anomaly['type']:<12} | Location: (y={loc[0]}, x={loc[1]}) | Confidence: {conf:.2f}%")
        else:
            print("\nSuccess! No anomalies were detected in the exposure.")

        # 4. Save results to a file if an output path was provided
        if args.output:
            print(f"\nSaving results to {args.output}...")
            with open(args.output, 'w') as f:
                json.dump(anomalies, f, indent=4)
            print("Save complete.")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
