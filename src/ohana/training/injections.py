import numpy as np
from tqdm import tqdm


def get_saturation_level_elec(gain, saturation_level_pxs):
    """
        Convert a saturation level from DN (counts) to electrons
        Arguments:
            gain (float): detector gain in electrons per DN (e⁻/DN)
            saturation_level_counts (float): saturation threshold in DN (ADU)
        Returns:
            float: saturation threshold in electrons
        Notes:
            * this is a simple linear scaling -> electrons = DN x gain
    """
    # Multiple the count (DN) by gain (e-/DN) to get the electrons
    saturation_e = saturation_level_pxs * gain

    return saturation_e

def generate_baseline_ramp(shape, num_frames, gain, saturation_level_counts, 
                           dark_current, read_noise, extra_gaussian_noise_dn):
    """
        Simulates a baseline up-the-ramp signal (in counts) by linearly increasing dark 
        current and gaussian read noise, then converting to DN using the gain.
        Arguments:
            shape (tuple): spatial dim (height, width)
            num_frames (int): number of ramp frames
            gain (float): detector gain in e-/ADU
            saturation_level_counts (float): maximum DN before clippig (saturation)
            dark_current (float): electrons per second
            read_noise (float): per frame read noise std in electrons
            extra_gaussian_noise_dn (float): optional, extra gaussian noise std
        Returns:
            np.ndarray: ramp data of shape (num_frames, height, width), in counts (DN)
    """
    # Unpack the spatial dimensions
    height, width = shape
    
    # Create a time array and reshape for broadcasting
    time_steps = np.arange(num_frames, dtype=np.float32).reshape(-1, 1, 1)
    
    # Dark current increases linearly with frame index
    dark_signal = dark_current * time_steps

    # Per frame gaussina read noise in electrons
    noise = np.random.normal(0, read_noise, size=(num_frames, height, width))

    # Total signal in electsons
    ramps_e = dark_signal + noise

    # Convert to counts
    ramps_dn = ramps_e / gain

    # Add extra Gaussian noise
    if extra_gaussian_noise_dn > 0:
        ramps_dn += np.random.normal(0, extra_gaussian_noise_dn, size=ramps_dn.shape)

    # Clip to valid range (0, saturation)
    ramps_dn = np.clip(ramps_dn, 0, saturation_level_counts)

    return ramps_dn

def inject_cosmic_ray(ramps, position, frame_idx, 
                      charge_e, gain, saturation_level_counts):
    """
        Inject a cosmic ray as a step in a single pixel by applying a persisitent additive
        step starting at the frame index for the specified pixel
        Arguments:
            ramps (np.ndarray): ramp cube in DN of shape (F, H, W)
            position (tuple[int, int]): pixel coordinates as (row, col)
            frame_idx (int): index of the frame where the step begins 
            charge_e (float): total charge to inject in electrons (e⁻)
            gain (float): detector gain in e⁻/DN 
            saturation_level_counts (float): saturation threshold in DN for clipping
        Returns:
            None
    """
    # Convert the injected electrons to DN
    charge_dn = charge_e / gain
    
    # Add the sep from the frame_idx onward at the target pixel
    ramps[frame_idx:, position[0], position[1]] += charge_dn
    
    # Clip to valid DN range
    ramps[:] = np.clip(ramps, 0, saturation_level_counts)

def inject_rtn(ramps, position, high_offset_e, period, duty_fraction, gain, saturation_level_counts):
    """
        Injects a two-level Random Telegraph Noise (RTN) into a specified pixel of the ramp
        using a two-state Markov model by toggling between a low and high state that is 
        offset with the dwell time in the high vs low states within one period being
        controlled by the duty_fraction
        Arguments:
            ramps (np.ndarray): ramp cube in DN of shape (F, H, W)
            position (tuple[int, int]): pixel coordinates (row, col)
            high_offset_e (float): high-state additive offset amplitude in electrons (e⁻)
            period (int): total frames per on/off cycle (T) must be >= 2
            duty_fraction (float): fraction of the period spent in the high state (0-1)
            gain (float): detector gain in e⁻/DN
            saturation_level_counts (float): saturation threshold in DN for clipping
        Returns:
            None
    """
    # Convert the high-state offset from electrons to DN
    high_offset_dn = high_offset_e / gain

    # Total number of frames in the ramp
    num_frames = ramps.shape[0]
    
    # Determine the time for up and down states
    up_time = int(period * duty_fraction)
    down_time = int(period * (1 - duty_fraction))

    # Avoid degenerate cases where states are never switched
    if up_time == 0 or down_time == 0:
        return

    # Random initial state (either high or low)
    is_high = np.random.choice([True, False]) 

    # Iterate over the frames in blocks (high and low)
    t = 0
    while t < num_frames:
        # Caclulate the duration of the current state
        time_in_state = up_time if is_high else down_time
        end_frame = min(t + time_in_state, num_frames)
        
        # Apply the offset to the selected pixel fpr the high segments
        if is_high:
            ramps[t:end_frame, position[0], position[1]] += high_offset_dn
        
        # Advance the time and flip the states
        t = end_frame
        is_high = not is_high

    # Clip to the saturation
    np.clip(ramps, 0, saturation_level_counts, out=ramps)

def inject_snowball(ramps, center, radius, 
                                 core_charge_e, halo_profile_e, gain, 
                                 saturation_level_counts, impact_frame):
    """
        Inject a snowball event wtih a circulat saturated core with a halo whose
        amplitude follows a radial profle, persisting from the impact
        frame onwards
        Arguments:
            ramps (np.ndarray): ramp cube in DN of shape (F, H, W)
            center (tuple[int, int]): snowball center as (row, col)
            radius (float): radius of the saturated core (pixels)
            core_charge_e (float): core charge in electrons (e⁻) added per frame
            halo_profile_e (callable[[np.ndarray], np.ndarray]):
                function that maps radial distance (in pixels) to electrons to add
                (per frame) at those pixels and is called with an array of distances
                for the halo pixels and must return an array of equal shape in electrons
            gain (float): detector gain in e⁻/DN 
            saturation_level_counts (float): saturation threshold in DN for clipping
            impact_frame (int): fame index when the event begins
        Returns:
            None
    """
    # Grab the spatial dimensions (H, W)
    h, w = ramps.shape[1:]

    # Create the broadcastable coordinate grids
    Y, X = np.ogrid[:h, :w]

    # Calculate the radial distance from the snowball center
    distance = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    # Create a bool mask for the core and halo
    core_mask = distance < radius
    halo_mask = (distance >= radius) & (distance < radius + 5)

    # Convert core charge from electrons to DN
    core_dn = core_charge_e / gain

    # Evaluate the halo profule in electrons at the halo pxs
    halo_e = halo_profile_e(distance[halo_mask])
    halo_dn = halo_e / gain

    # Apply the core and halo additions from the impact frame onwards
    ramps[impact_frame:, core_mask] += core_dn
    ramps[impact_frame:, halo_mask] += halo_dn

    # Clip to be within the valid DN range
    np.clip(ramps, 0, saturation_level_counts, out=ramps)