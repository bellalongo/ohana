import numpy as np
import logging
from numba import jit, prange


@jit(nopython=True, cache=True)
def _process_channel_numba(corrected_frame, ch, up_ref, down_ref, x_opt):
    """
        Numba-optimized per-channel up/down reference correction
        Arguments:
            corrected_frame (np.ndarray): working copy of the frame (H, W)
            ch (int): channel index in [0, 31]
            up_ref (np.ndarray): top reference rows (4, W)
            down_ref (np.ndarray): bottom reference rows (4, W)
            x_opt (int): sliding window radius in columns for averaging
        Returns:
            np.ndarray: corrected frame after up/down subtraction for this channel
        Notes:
            * columns are partitioned into 32 channels of width 64, with 4-pixel borders
            * slope is computed from average up/down refs over a horizontal window
    """
    # Define column range for this channel
    col_start = 4 + ch * 64
    col_end = 4 + (ch + 1) * 64
    if ch == 31:
        col_end = 2044

    # Apply per-column correction with a horizontal sliding window
    for col in range(col_start, col_end):
        # Compute horizontal window limits within active area
        window_start = max(4, col - x_opt)
        window_end = min(2044, col + x_opt + 1)

        # Compute average reference values across the window
        up_avg = np.mean(up_ref[:, window_start:window_end])
        down_avg = np.mean(down_ref[:, window_start:window_end])

        # Compute linear slope from top to bottom across active rows (4..2043)
        slope = (up_avg - down_avg) / 2040.0

        # Apply linear correction across rows
        for row in range(4, 2044):
            ref_correction = down_avg + (row - 4) * slope
            corrected_frame[row, col] -= ref_correction

    # Return the updated frame
    return corrected_frame

@jit(nopython=True, cache=True)
def _perform_lr_correction_numba(corrected_frame, left_ref, right_ref, y_opt):
    """
        Numba-optimized left/right reference correction
        Arguments:
            corrected_frame (np.ndarray): working copy of the frame (H, W)
            left_ref (np.ndarray): left reference columns (H, 4)
            right_ref (np.ndarray): right reference columns (H, 4)
            y_opt (int): sliding window radius in rows for averaging
        Returns:
            np.ndarray: corrected frame after left/right subtraction
        Notes:
            * averages left/right ref columns over a vertical window and removes mean
    """
    # Iterate over active rows
    for row in range(4, 2044):
        # Compute vertical window limits within active area
        window_start = max(4, row - y_opt)
        window_end = min(2044, row + y_opt + 1)

        # Average left/right reference over the vertical window
        left_avg = np.mean(left_ref[window_start:window_end, :])
        right_avg = np.mean(right_ref[window_start:window_end, :])

        # Use the mean of left/right as the correction value
        lr_correction = (left_avg + right_avg) / 2.0

        # Subtract from active columns
        corrected_frame[row, 4:2044] -= lr_correction

    # Return the updated frame
    return corrected_frame

@jit(nopython=True, parallel=True, cache=True)
def _batch_subtract_reference_pixels_numba(frame_stack, x_opt, y_opt):
    """
        Numba-optimized batch reference pixel subtraction
        Arguments:
            frame_stack (np.ndarray): stack of frames (T, H, W)
            x_opt (int): sliding window radius in columns for up/down correction
            y_opt (int): sliding window radius in rows for left/right correction
        Returns:
            np.ndarray: corrected stack of frames (T, H, W) as float32
        Notes:
            * processes each frame independently in parallel via prange
            * applies per-channel up/down then global left/right correction
    """
    # Unpack dimensions and prepare output buffer
    n_frames, height, width = frame_stack.shape
    corrected_stack = np.empty_like(frame_stack, dtype=np.float32)

    # Parallel loop over frames
    for i in prange(n_frames):
        # Promote to float64 for stable accumulation
        frame = frame_stack[i].astype(np.float64)

        # Create a working copy to hold corrections
        corrected_frame = frame.copy()

        # Extract reference regions
        up_ref = frame[0:4, :]
        down_ref = frame[2044:2048, :]
        left_ref = frame[:, 0:4]
        right_ref = frame[:, 2044:2048]

        # Apply per-channel up/down correction across all 32 channels
        for ch in range(32):
            corrected_frame = _process_channel_numba(corrected_frame, ch, up_ref, down_ref, x_opt)

        # Apply left/right correction across rows
        corrected_frame = _perform_lr_correction_numba(corrected_frame, left_ref, right_ref, y_opt)

        # Store as float32 in the output stack
        corrected_stack[i] = corrected_frame.astype(np.float32)

    # Return the corrected stack
    return corrected_stack

class ReferencePixelCorrector:
    """
        Handles reference pixel subtraction for H2RG detectors
        This implementation is optimized with Numba for performance
    """
    def __init__(self, x_opt=64, y_opt=4):
        """
            Arguments:
                x_opt (int): sliding window radius for up/down correction
                y_opt (int): sliding window radius for left/right correction
            Attributes:
                x_opt (int): stored horizontal window radius
                y_opt (int): stored vertical window radius
                logger (logging.Logger): module logger for this corrector
        """
        # Store correction window sizes
        self.x_opt = x_opt
        self.y_opt = y_opt

        # Create a logger scoped to the class
        self.logger = logging.getLogger(self.__class__.__name__)

        # Announce initialization
        print(f"ReferencePixelCorrector initialized with x_opt={x_opt}, y_opt={y_opt}.")

    def batch_correct(self, frame_stack):
        """
            Apply reference pixel subtraction to a stack of frames in parallel
            Arguments:
                frame_stack (np.ndarray): 3D array (T, H, W) of raw frames
            Returns:
                np.ndarray: corrected stack (T, H, W) as float32
            Notes:
                * ensures C-contiguous layout for Numba kernels
                * expects H2RG-like geometry with 4-pixel reference borders
        """
        # Validate input dimensionality
        if frame_stack.ndim != 3:
            raise ValueError("Input frame_stack must be a 3D array.")

        # Log the incoming stack shape
        self.logger.info(f"Applying reference pixel correction to stack of shape {frame_stack.shape}...")

        # Ensure C-contiguous memory layout for Numba kernels
        if not frame_stack.flags['C_CONTIGUOUS']:
            frame_stack = np.ascontiguousarray(frame_stack, dtype=np.float32)

        # Run the Numba-accelerated correction pipeline
        return _batch_subtract_reference_pixels_numba(frame_stack, self.x_opt, self.y_opt)
