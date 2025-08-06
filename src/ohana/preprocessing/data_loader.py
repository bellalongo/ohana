import os
import glob
import numpy as np
from PIL import Image
from astropy.io import fits
from tqdm import tqdm
import h5py

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
            # --- THIS SECTION IS NOW CORRECTED ---
            if path.lower().endswith(('.fits', '.fit')):
                return self._load_from_mef_fits(path)
            elif path.lower().endswith('.h5'):
                return self._load_from_h5(path)
            elif path.lower().endswith('.npy'):
                return self._load_from_npy(path)
            else:
                raise ValueError(f"Unsupported single file format: {os.path.basename(path)}")
        else:
            raise FileNotFoundError(f"The specified path does not exist: {path}")

    def _load_from_mef_fits(self, file_path: str) -> np.ndarray:
        """
        Loads a data cube from a multi-extension FITS (MEF) file where each
        HDU contains a single frame.
        """
        print(f"Loading data from Multi-Extension FITS file: {file_path}")
        try:
            with fits.open(file_path) as hdul:
                # Skip the primary HDU (index 0) as it usually contains no image data
                frame_list = [hdu.data.astype(np.float32) for hdu in tqdm(hdul[1:], desc="Loading FITS extensions")]
            
            if not frame_list:
                raise IOError("FITS file is valid, but no data extensions were found after the primary HDU.")
            
            # Stack the list of 2D frames into a single 3D data cube
            return np.stack(frame_list, axis=0)

        except Exception as e:
            raise IOError(f"Astropy failed to open or process the FITS file '{file_path}'. Original error: {e}")

    def _load_from_h5(self, file_path: str) -> np.ndarray:
        """Loads a data cube from an HDF5 file."""
        print(f"Loading data from HDF5 file: {file_path}")
        with h5py.File(file_path, 'r') as hf:
            if 'data' not in hf:
                raise KeyError("HDF5 file must contain a dataset named 'data'.")
            return hf['data'][:].astype(np.float32)

    def _load_from_npy(self, file_path: str) -> np.ndarray:
        """Loads a data cube from a NumPy .npy file."""
        print(f"Loading data from NumPy file: {file_path}")
        return np.load(file_path).astype(np.float32)

    def _load_from_tif_directory(self, dir_path: str) -> np.ndarray:
        """
        Loads a sequence of TIFF files from a directory, sorts them,
        clips them to size, and stacks them into a data cube.
        """
        print(f"Loading data from TIFF directory: {dir_path}")
        tif_files = sorted(glob.glob(os.path.join(dir_path, '*.tif*')))
        if not tif_files:
            raise IOError(f"No TIFF files found in directory: {dir_path}")

        print(f"Found {len(tif_files)} TIFF files. Stacking into data cube...")
        
        frame_list = []
        for f_path in tqdm(tif_files, desc="Loading TIFF frames"):
            with Image.open(f_path) as img:
                frame = np.array(img, dtype=np.float32)
                
                if frame.shape[0] > 2048 or frame.shape[1] > 2048:
                    h, w = frame.shape
                    h_start = (h - 2048) // 2
                    w_start = (w - 2048) // 2
                    frame = frame[h_start:h_start+2048, w_start:w_start+2048]
                
                if frame.shape != (2048, 2048):
                    raise ValueError(f"Frame {os.path.basename(f_path)} has incorrect shape {frame.shape} after clipping.")

                frame_list.append(frame)
        
        return np.stack(frame_list, axis=0)
