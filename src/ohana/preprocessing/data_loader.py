import os
import glob
import numpy as np
from PIL import Image
from astropy.io import fits
from tqdm import tqdm
import h5py

class DataLoader:
    """
        Handles loading H2RG exposure data from various real-world formats
    """
    def load_exposure(self, path):
        """
            Loads exposure data intelligently based on the input path
            Arguments:
                path (str): path to the data source; can be a single file
                    (FITS, H5, NPY) or a directory containing TIFFs
            Returns:
                np.ndarray: 3D numpy array (frames, height, width)
            Notes:
                * directories are treated as stacks of TIFF frames
                * single-file loaders handle MEF FITS, HDF5, and NumPy .npy
        """
        # Check if the path is a directory
        if os.path.isdir(path):
            # Load a stack of TIFFs from the directory
            return self._load_from_tif_directory(path)

        # Check if the path is a single file
        elif os.path.isfile(path):
            # Handle FITS files (MEF or simple)
            if path.lower().endswith(('.fits', '.fit')):
                return self._load_from_mef_fits(path)

            # Handle HDF5 files
            elif path.lower().endswith('.h5'):
                return self._load_from_h5(path)

            # Handle NumPy arrays
            elif path.lower().endswith('.npy'):
                return self._load_from_npy(path)

            # Unsupported single-file format
            else:
                raise ValueError(f"Unsupported single file format: {os.path.basename(path)}")

        # Path is neither a file nor a directory
        else:
            raise FileNotFoundError(f"The specified path does not exist: {path}")

    def _load_from_mef_fits(self, file_path):
        """
            Load a data cube from a multi-extension FITS (MEF) file
            Arguments:
                file_path (str): path to the FITS file
            Returns:
                np.ndarray: stacked 3D array of frames (T, H, W)
            Notes:
                * skips the primary HDU assuming no image data there
                * each extension HDU is treated as one frame
        """
        # Announce FITS loading operation
        print(f"Loading data from Multi-Extension FITS file: {file_path}")

        # Attempt to open and read the FITS file
        try:
            # Open FITS file with astropy
            with fits.open(file_path) as hdul:
                # Iterate over extensions starting from index 1
                frame_list = [hdu.data.astype(np.float32) for hdu in tqdm(hdul[1:], desc="Loading FITS extensions")]

            # Ensure that we found at least one frame
            if not frame_list:
                raise IOError("FITS file is valid, but no data extensions were found after the primary HDU.")

            # Stack 2D frames along a new temporal axis
            return np.stack(frame_list, axis=0)

        # Wrap any exception with a helpful message
        except Exception as e:
            raise IOError(f"Astropy failed to open or process the FITS file '{file_path}'. Original error: {e}")

    def _load_from_h5(self, file_path):
        """
            Load a data cube from an HDF5 file
            Arguments:
                file_path (str): path to the HDF5 file
            Returns:
                np.ndarray: stacked 3D array of frames (T, H, W)
            Notes:
                * expects a dataset named 'data' at the root of the file
        """
        # Announce HDF5 loading operation
        print(f"Loading data from HDF5 file: {file_path}")

        # Open the HDF5 file in read-only mode
        with h5py.File(file_path, 'r') as hf:
            # Verify that the 'data' dataset exists
            if 'data' not in hf:
                raise KeyError("HDF5 file must contain a dataset named 'data'.")

            # Read and convert to float32 for consistency
            return hf['data'][:].astype(np.float32)

    def _load_from_npy(self, file_path):
        """
            Load a data cube from a NumPy .npy file
            Arguments:
                file_path (str): path to the .npy file
            Returns:
                np.ndarray: stacked 3D array of frames (T, H, W)
            Notes:
                * assumes array is already shaped (T, H, W)
        """
        # Announce NumPy loading operation
        print(f"Loading data from NumPy file: {file_path}")

        # Load and cast to float32
        return np.load(file_path).astype(np.float32)

    def _load_from_tif_directory(self, dir_path):
        """
            Load a sequence of TIFF files from a directory, sort them,
            clip them to size, and stack them into a data cube
            Arguments:
                dir_path (str): path to the directory containing TIFF frames
            Returns:
                np.ndarray: stacked 3D array of frames (T, H, W)
            Notes:
                * frames are center-cropped to 2048x2048 if larger
                * all frames must end up exactly 2048x2048
        """
        # Announce TIFF directory loading operation
        print(f"Loading data from TIFF directory: {dir_path}")

        # Collect all TIFF files (both .tif and .tiff)
        tif_files = sorted(glob.glob(os.path.join(dir_path, '*.tif*')))

        # Verify that we found at least one TIFF file
        if not tif_files:
            raise IOError(f"No TIFF files found in directory: {dir_path}")

        # Announce stacking operation and frame count
        print(f"Found {len(tif_files)} TIFF files. Stacking into data cube...")

        # Accumulate 2D frames prior to stacking
        frame_list = []

        # Iterate over files with a progress bar
        for f_path in tqdm(tif_files, desc="Loading TIFF frames"):
            # Open the image using PIL
            with Image.open(f_path) as img:
                # Convert to numpy float32
                frame = np.array(img, dtype=np.float32)

                # If frame is larger than target, center-crop to 2048x2048
                if frame.shape[0] > 2048 or frame.shape[1] > 2048:
                    # Get current height and width
                    h, w = frame.shape
                    # Compute top-left crop origin
                    h_start = (h - 2048) // 2
                    w_start = (w - 2048) // 2
                    # Apply center crop
                    frame = frame[h_start:h_start+2048, w_start:w_start+2048]

                # Validate final shape
                if frame.shape != (2048, 2048):
                    raise ValueError(f"Frame {os.path.basename(f_path)} has incorrect shape {frame.shape} after clipping.")

                # Append to the list of frames
                frame_list.append(frame)

        # Stack frames along the temporal dimension
        return np.stack(frame_list, axis=0)
