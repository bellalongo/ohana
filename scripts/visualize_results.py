import argparse
import sys
import os
import time

# Add the parent directory ('ohana-main') to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the visualizer class
from ohana.visualization.plotter import ResultVisualizer

def main():
    """
    Main entry-point for the visualization script.
    Loads detection results and pre-processed data to generate diagnostic plots.
    """
    parser = argparse.ArgumentParser(
        description="Visualize anomaly detection results by plotting each detection."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the JSON file containing the detection results from run_prediction.py."
    )
    # --- HELP TEXT CORRECTED ---
    parser.add_argument(
        "--processed_data",
        required=True,
        help="Path to the processed (e.g., corrected and differenced) data cube (.npy file) that was analyzed."
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
        # --- THIS LINE IS NOW CORRECTED ---
        # The script now correctly passes 'processed_data_path' to the visualizer.
        visualizer = ResultVisualizer(processed_data_path=args.processed_data)

        # Load the detection results from the JSON file
        visualizer.load_results(results_path=args.results)

        # Show individual plots interactively
        visualizer.show_individual_detections()

        print(f"\nVisualization complete.")

    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal visualization time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
