import argparse
import sys
import os

# Add the parent directory to the path to allow for relative imports
# This allows the script to find the 'ohana' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ohana.training.training_set_creator import DataSetCreator

def main():
    """Main entry-point for the data simulation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic detector data with multiple event types."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file (e.g., configs/config.yaml)."
    )
    args = parser.parse_args()

    # Instantiate the creator and run the dataset generation
    creator = DataSetCreator(config_path=args.config)
    creator.create_dataset()

if __name__ == "__main__":
    main()
