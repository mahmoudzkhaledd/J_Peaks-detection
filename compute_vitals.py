import numpy as np
import pandas as pd
from beat_to_beat import compute_rate


def vitals(t1, t2, win_size, window_limit, sig, time, mpd, plot=0, signal_type="heart"):
    """
    Compute heart or respiratory rate from signal using windowed peak detection.
    Returns:
        - all_rate: list of computed rates (NaN if invalid)
        - all_peak_times: list of all detected peak timestamps
    """
    all_rate = []
    all_peak_times = []

    for j in range(window_limit):
        sub_signal = sig[t1:t2]
        sub_time = time[t1:t2]

        rate, indices = compute_rate(sub_signal, sub_time, mpd)

        # Debug logging
        if plot:
            print(f"[{signal_type.upper()}] Window {j}: "
                  f"{'Rate: {:.2f}'.format(rate) if not np.isnan(rate) else 'No valid rate'}, "
                  f"Peaks: {len(indices) if isinstance(indices, np.ndarray) else 0}")

        if not np.isnan(rate):
            all_rate.append(rate)

        if isinstance(indices, np.ndarray) and len(indices) > 0:
            # Handle both pandas Series and numpy array
            if hasattr(sub_time, 'iloc'):
                peak_times = sub_time.iloc[indices]
            else:
                peak_times = sub_time[indices]
                
            # Convert to numpy array if it's not already
            peak_times = np.array(peak_times)
            all_peak_times.extend(peak_times)

        # Move to next window
        t1 = t2
        t2 += win_size

    # Safe fallback
    if len(all_rate) == 0:
        all_rate = [np.nan]

    print("time type:", type(time))
    print("first 5 timestamps:", time[:5])

    return np.array(all_rate), np.array(all_peak_times)
