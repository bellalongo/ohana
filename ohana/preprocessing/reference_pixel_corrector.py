import numpy as np
import logging
from numba import jit, prange

# This file contains the standalone logic for reference pixel correction.
# It is imported and used by the main Preprocessor class.

@jit(nopython=True, cache=True)
def _process_channel_numba(corrected_frame, ch, up_ref, down_ref, x_opt):
    """Numba-optimized channel processing."""
    # Define column ranges for each channel
    col_start = 4 + ch * 64
    col_end = 4 + (ch + 1) * 64
    if ch == 31:
        col_end = 2044

    # Up/down correction with sliding window
    for col in range(col_start, col_end):
        window_start = max(4, col - x_opt)
        window_end = min(2044, col + x_opt + 1)

        # Calculate averages for the window
        up_avg = np.mean(up_ref[:, window_start:window_end])
        down_avg = np.mean(down_ref[:, window_start:window_end])
        
        slope = (up_avg - down_avg) / 2040.0 # Based on 2044 - 4 active rows
        
        for row in range(4, 2044):
            ref_correction = down_avg + (row - 4) * slope
            corrected_frame[row, col] -= ref_correction
    return corrected_frame

@jit(nopython=True, cache=True)
def _perform_lr_correction_numba(corrected_frame, left_ref, right_ref, y_opt):
    """Numba-optimized left/right correction."""
    for row in range(4, 2044):
        window_start = max(4, row - y_opt)
        window_end = min(2044, row + y_opt + 1)
        
        left_avg = np.mean(left_ref[window_start:window_end, :])
        right_avg = np.mean(right_ref[window_start:window_end, :])
        
        lr_correction = (left_avg + right_avg) / 2.0
        corrected_frame[row, 4:2044] -= lr_correction
    return corrected_frame

@jit(nopython=True, parallel=True, cache=True)
def _batch_subtract_reference_pixels_numba(frame_stack, x_opt, y_opt):
    """Numba-optimized batch reference pixel subtraction."""
    n_frames, height, width = frame_stack.shape
    corrected_stack = np.empty_like(frame_stack, dtype=np.float32)

    for i in prange(n_frames):
        frame = frame_stack[i].astype(np.float64)
        corrected_frame = frame.copy()
        
        up_ref = frame[0:4, :]
        down_ref = frame[2044:2048, :]
        left_ref = frame[:, 0:4]
        right_ref = frame[:, 2044:2048]

        # Process each of the 32 channels
        for ch in range(32):
            corrected_frame = _process_channel_numba(corrected_frame, ch, up_ref, down_ref, x_opt)
        
        # Perform left/right correction
        corrected_frame = _perform_lr_correction_numba(corrected_frame, left_ref, right_ref, y_opt)
        
        corrected_stack[i] = corrected_frame.astype(np.float32)
        
    return corrected_stack


class ReferencePixelCorrector:
    """
    Handles reference pixel subtraction for H2RG detectors.
    This implementation is optimized with Numba for performance.
    """
    def __init__(self, x_opt=64, y_opt=4):
        """
        Args:
            x_opt (int): Sliding window radius for up/down correction.
            y_opt (int): Sliding window radius for left/right correction.
        """
        self.x_opt = x_opt
        self.y_opt = y_opt
        self.logger = logging.getLogger(__name__)
        print(f"ReferencePixelCorrector initialized with x_opt={x_opt}, y_opt={y_opt}.")

    def batch_correct(self, frame_stack: np.ndarray) -> np.ndarray:
        """
        Applies reference pixel subtraction to a stack of frames in parallel.

        Args:
            frame_stack (np.ndarray): A 3D numpy array of shape 
                                      (frames, height, width).

        Returns:
            np.ndarray: The corrected stack of frames.
        """
        if frame_stack.ndim != 3:
            raise ValueError("Input frame_stack must be a 3D array.")
        
        self.logger.info(f"Applying reference pixel correction to stack of shape {frame_stack.shape}...")
        
        # Ensure the input is compatible with Numba function
        if not frame_stack.flags['C_CONTIGUOUS']:
             frame_stack = np.ascontiguousarray(frame_stack, dtype=np.float32)

        return _batch_subtract_reference_pixels_numba(frame_stack, self.x_opt, self.y_opt)
