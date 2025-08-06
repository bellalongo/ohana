import argparse
import sys
import os

# Add the parent directory to the path to allow for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ohana.training.training_set_creator import DataSetCreator

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic detector data using an HPC job array."
    )
    
    # Argument for the main configuration file
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file (e.g., creator_config.yaml)."
    )
    
    # Argument for the starting index, provided by the SLURM script
    parser.add_argument(
        "--start_index",
        type=int,
        required=True,
        help="Starting index for the exposure generation, provided by the job scheduler."
    )
    
    # Argument for the ending index, provided by the SLURM script
    parser.add_argument(
        "--end_index",
        type=int,
        required=True,
        help="Ending index for the exposure generation, provided by the job scheduler."
    )
    
    args = parser.parse_args()

    # Intialize the creator with the configuration file
    creator = DataSetCreator(config_path=args.config)
    
    # Call the main creation method, passing the job-specific range
    creator.create_dataset(start=args.start_index, end=args.end_index)

if __name__ == "__main__":
    main()