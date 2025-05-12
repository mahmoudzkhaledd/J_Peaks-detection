"""
Function to perform a Chebyshev type I bandpass filter for heart rate and breathing.
"""

from scipy.signal import cheby1, filtfilt

def band_pass_filtering(data, fs, filter_type):
    if filter_type == "bcg":
        # Heart rate: 0.4–12 Hz, lower order for less attenuation
        [b_cheby_high, a_cheby_high] = cheby1(1, 0.5, [0.4 / (fs / 2)], btype='high', analog=False)
        bcg_ = filtfilt(b_cheby_high, a_cheby_high, data)
        [b_cheby_low, a_cheby_low] = cheby1(2, 0.5, [12.0 / (fs / 2)], btype='low', analog=False)
        filtered_data = filtfilt(b_cheby_low, a_cheby_low, bcg_)
    elif filter_type == "breath":
        # Respiratory: 0.08–0.6 Hz, lower order for less attenuation
        [b_cheby_high, a_cheby_high] = cheby1(1, 0.5, [0.08 / (fs / 2)], btype='high', analog=False)
        bcg_ = filtfilt(b_cheby_high, a_cheby_high, data)
        [b_cheby_low, a_cheby_low] = cheby1(2, 0.5, [0.6 / (fs / 2)], btype='low', analog=False)
        filtered_data = filtfilt(b_cheby_low, a_cheby_low, bcg_)
    else:
        filtered_data = data
    return filtered_data