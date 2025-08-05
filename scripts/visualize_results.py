import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ohana.visualization.plotter import ResultVisualizer

def main():
    parser = argparse.ArgumentParser(description="Visualize anomaly detection results.")
    parser.add_argument("--processed_data", required=True, help="Path to the processed data cube (.npy file).")
    parser.add_argument("--prediction_mask", required=True, help="Path to the predicted segmentation mask (.npy file).")
    parser.add_argument("--detection_list", required=False, help="Optional. Path to the JSON file of detected object locations.")
    args = parser.parse_args()

    start_time = time.time()
    try:
        visualizer = ResultVisualizer(
            processed_data_path=args.processed_data,
            prediction_mask_path=args.prediction_mask
        )
        if args.detection_list:
            visualizer.load_detection_list(results_path=args.detection_list)
        visualizer.plot_full_mask_overlay()
    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        sys.exit(1)
    end_time = time.time()
    print(f"\nTotal visualization time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
