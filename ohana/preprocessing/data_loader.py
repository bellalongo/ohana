import os
import glob
import numpy as np
from PIL import Image
from astropy.io import fits
from tqdm import tqdm

class DataLoader:
    """
    Handles loading H2RG exposure data from various real-world formats.
    """
    def load_exposure(self, path: str) -> np.ndarray:
        """
        Loads exposure data intelligently based on the input path.

        Args:
            path (str): Path to the data source. Can be a single file
                        (FITS, H5, NPY) or a directory containing TIFFs.

        Returns:
            np.ndarray: A 3D numpy array representing the data cube
                        (frames, height, width).
        """
        if os.path.isdir(path):
            return self._load_from_tif_directory(path)
        elif os.path.isfile(path):
            if path.lower().endswith(('.fits', '.fit')):
                return self._load_from_fits(path)
            # Add other single-file formats here if needed
            else:
                raise ValueError(f"Unsupported single file format for: {path}")
        else:
            raise FileNotFoundError(f"The specified path does not exist: {path}")

    def _load_from_fits(self, file_path: str) -> np.ndarray:
        """Loads a data cube from a FITS file."""
        print(f"Loading data from FITS file: {file_path}")
        with fits.open(file_path) as hdul:
            # Assumes data is in the primary HDU or the first extension
            data = hdul[0].data if hdul[0].data is not None else hdul[1].data
        return data.astype(np.float32)

    def _load_from_tif_directory(self, dir_path: str) -> np.ndarray:
        """
        Loads a sequence of TIFF files from a directory, sorts them,
        clips them to size, and stacks them into a data cube.
        """
        print(f"Loading data from TIFF directory: {dir_path}")
        # Find all .tif and .tiff files and sort them naturally
        tif_files = sorted(glob.glob(os.path.join(dir_path, '*.tif*')))
        if not tif_files:
            raise IOError(f"No TIFF files found in directory: {dir_path}")

        print(f"Found {len(tif_files)} TIFF files. Stacking into data cube...")
        
        frame_list = []
        for f_path in tqdm(tif_files, desc="Loading TIFF frames"):
            with Image.open(f_path) as img:
                frame = np.array(img, dtype=np.float32)
                
                # --- Clipping Logic ---
                # As requested, check if the frame is larger than the standard
                # 2048x2048 active pixel area and clip it.
                if frame.shape[0] > 2048 or frame.shape[1] > 2048:
                    print(f"  - Note: Clipping frame {os.path.basename(f_path)} from {frame.shape} to (2048, 2048)")
                    h, w = frame.shape
                    # This assumes the extra pixels are symmetric (e.g., reference pixels)
                    h_start = (h - 2048) // 2
                    w_start = (w - 2048) // 2
                    frame = frame[h_start:h_start+2048, w_start:w_start+2048]
                
                if frame.shape != (2048, 2048):
                    raise ValueError(f"Frame {os.path.basename(f_path)} has incorrect shape {frame.shape} after clipping.")

                frame_list.append(frame)
        
        return np.stack(frame_list, axis=0)
