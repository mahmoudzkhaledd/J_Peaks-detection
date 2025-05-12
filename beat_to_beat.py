import numpy as np
from scipy.signal import find_peaks

def compute_rate(signal, time, mpd):
    """
    Detect peaks and compute heart or respiration rate from time intervals between peaks.
    """
    peaks, _ = find_peaks(signal, distance=mpd)

    # Optional: debug plot
    # import matplotlib.pyplot as plt
    # plt.plot(time, signal)
    # plt.plot(time[peaks], signal[peaks], "x")
    # plt.title("Detected Peaks")
    # plt.savefig(f"results/debug_peaks_{np.random.randint(10000)}.png")
    # plt.close()

    if len(peaks) > 1:
        # Handle both pandas Series and numpy array
        if hasattr(time, 'iloc'):
            peak_times = time.iloc[peaks]
        else:
            peak_times = time[peaks]
            
        # Convert to numpy array if it's not already
        peak_times = np.array(peak_times)
        
        # Calculate intervals in seconds
        if isinstance(peak_times[0], np.datetime64):
            ibi = np.diff(peak_times) / np.timedelta64(1, 's')
        else:
            ibi = np.diff(peak_times)
            
        ibi = ibi[ibi > 1e-5]
        rate = 60 / np.mean(ibi) if len(ibi) > 0 else np.nan
    else:
        rate = np.nan

    return rate, peaks
