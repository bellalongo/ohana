import argparse
import sys
import os
import time

# Add the parent directory ('ohana-main') to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the new visualizer class
from ohana.visualization.plotter import ResultVisualizer

def main():
    """
    Main entry-point for the visualization script.
    Loads prediction results and the original data to generate diagnostic plots.
    """
    parser = argparse.ArgumentParser(
        description="Visualize anomaly detection results by plotting each detection."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the JSON file containing the detection results from run_prediction.py."
    )
    parser.add_argument(
        "--exposure",
        required=True,
        help="Path to the original exposure data source (FITS file or TIFF directory) that was analyzed."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the output plot images will be saved."
    )
    args = parser.parse_args()

    # --- Execution ---
    start_time = time.time()

    try:
        # 1. Initialize the visualizer with the path to the original data
        visualizer = ResultVisualizer(exposure_path=args.exposure)

        # 2. Generate and save plots for all detections found in the results file
        visualizer.plot_all_detections(results_path=args.results, output_dir=args.output_dir)

        print(f"\nVisualization complete. Plots are saved in '{args.output_dir}'.")

    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal visualization time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
