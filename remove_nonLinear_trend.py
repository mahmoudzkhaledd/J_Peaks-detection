"""
Placeholder for remove_nonLinear_trend function.
Removes nonlinear trend from a signal using polynomial fitting.
"""

import numpy as np

def remove_nonLinear_trend(signal, degree):
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, degree)
    trend = np.polyval(coeffs, x)
    detrended_signal = signal - trend
    return detrended_signal