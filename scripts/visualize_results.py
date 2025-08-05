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
        help="Directory where the final summary plot image will be saved."
    )
    parser.add_argument(
        "--no_interactive",
        action='store_true',
        help="If set, skips the interactive one-by-one plotting and only saves the final summary plot."
    )
    args = parser.parse_args()

    # --- Execution ---
    start_time = time.time()

    try:
        # 1. Initialize the visualizer with the path to the original data
        visualizer = ResultVisualizer(exposure_path=args.exposure)

        # 2. Load the detection results from the JSON file
        visualizer.load_results(results_path=args.results)

        # 3. Show individual plots interactively, unless disabled
        if not args.no_interactive:
            visualizer.show_individual_detections()

        # 4. Generate and save the final summary plot
        visualizer.save_summary_plot(output_dir=args.output_dir)

        print(f"\n✅ Visualization complete.")

    except Exception as e:
        print(f"\n❌ An error occurred during visualization: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal visualization time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
