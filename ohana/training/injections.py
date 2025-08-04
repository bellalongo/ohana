import numpy as np
from tqdm import tqdm

def get_saturation_level_elec(gain, saturation_level_pxs):
    """
        * return the saturation level in electrons for a given SCA ID
    """
    saturation_e = saturation_level_pxs * gain

    return saturation_e

def generate_baseline_ramp(shape, num_frames, gain, saturation_level_counts, 
                           dark_current, read_noise, extra_gaussian_noise_dn):
    """
    Simulates a baseline up-the-ramp signal (in counts) with dark current and noise.

    Args:
        shape (tuple): (height, width)
        num_frames (int): number of ramp frames
        gain (float): e-/ADU
        saturation_level_counts (float): DN at which saturation occurs
        dark_current (float): electrons per second
        read_noise (float): electrons (std dev)

    Returns:
        np.ndarray: ramp data of shape (num_frames, height, width), in counts
    """
    height, width = shape
    
    # Create a time array and reshape for broadcasting
    time_steps = np.arange(num_frames, dtype=np.float32).reshape(-1, 1, 1)
    
    # Generate the signal for all frames at once
    dark_signal = dark_current * time_steps
    noise = np.random.normal(0, read_noise, size=(num_frames, height, width))
    ramps_e = dark_signal + noise

    # Convert to counts
    ramps_dn = ramps_e / gain

    # Add extra Gaussian noise
    if extra_gaussian_noise_dn > 0:
        ramps_dn += np.random.normal(0, extra_gaussian_noise_dn, size=ramps_dn.shape)

    # Clip to saturation
    ramps_dn = np.clip(ramps_dn, 0, saturation_level_counts)

    return ramps_dn

def inject_cosmic_ray(ramps, position, frame_idx, 
                      charge_e, gain, saturation_level_counts):
    """
        Inject a cosmic ray as a step function at a specific frame and pixel
        Arguments:
            charge_e (float): total electrons to inject
    """
    charge_dn = charge_e / gain
    
    ramps[frame_idx:, position[0], position[1]] += charge_dn
    
    ramps[:] = np.clip(ramps, 0, saturation_level_counts)

def inject_rtn(ramps, position, high_offset_e, period, duty_fraction, gain, saturation_level_counts):
    """
    Injects a two-level Random Telegraph Noise (RTN) into a specified pixel of the ramp
    using a two-state Markov model.

    Args:
        ramps (np.ndarray): The data cube of shape (n_reads, height, width).
        position (tuple): (row, col) index of the pixel to modify.
        high_offset_e (float): The amplitude of the high state in electrons.
        period (int): The total period (T) of one on/off cycle in frames.
        duty_fraction (float): The fraction of the period (f) spent in the high state.
        gain (float): Gain in electrons per DN (e⁻/DN).
        saturation_level_counts (float): Maximum allowed DN value to clip the signal to.
    """
    high_offset_dn = high_offset_e / gain
    num_frames = ramps.shape[0]
    
    # Determine the time for up and down states
    up_time = int(period * duty_fraction)
    down_time = int(period * (1 - duty_fraction))

    if up_time == 0 or down_time == 0: # Avoid getting stuck in one state
        return

    is_high = np.random.choice([True, False]) # Randomly start in high or low state
    t = 0
    while t < num_frames:
        time_in_state = up_time if is_high else down_time
        end_frame = min(t + time_in_state, num_frames)
        
        if is_high:
            ramps[t:end_frame, position[0], position[1]] += high_offset_dn
        
        t = end_frame
        is_high = not is_high

    np.clip(ramps, 0, saturation_level_counts, out=ramps)

def inject_snowball(ramps, center, radius, 
                                 core_charge_e, halo_profile_e, gain, 
                                 saturation_level_counts, impact_frame):
    """
    Inject a realistic snowball into a ramp at a specific frame.

    Args:
        ramps (np.ndarray): Ramp cube (n_reads, height, width).
        center (tuple): (y, x) position of snowball.
        radius (float): Radius of saturated core in pixels.
        core_charge_e (float): Charge in e- for saturated core.
        halo_profile_e (callable): Function(distance) -> e- for halo shape.
        gain (float): Detector gain in e-/DN.
        saturation_level_counts (float): Max pixel value (DN).
        impact_frame (int): Frame index when snowball hits.
    """
    h, w = ramps.shape[1:]
    Y, X = np.ogrid[:h, :w]
    distance = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    core_mask = distance < radius
    halo_mask = (distance >= radius) & (distance < radius + 5)

    core_dn = core_charge_e / gain
    halo_e = halo_profile_e(distance[halo_mask])
    halo_dn = halo_e / gain

    # Inject into the impact frame only
    ramps[impact_frame:, core_mask] += core_dn
    ramps[impact_frame:, halo_mask] += halo_dn

    # Saturate the core from this point forward
    np.clip(ramps, 0, saturation_level_counts, out=ramps)