# Import required libraries
import math
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, resample, butter, filtfilt
import matplotlib.pyplot as plt

from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot


def synchronize_rr(bcg_rr, rr_ref, time_threshold='1s'):
    """
    Synchronize BCG-derived RR intervals with reference RR intervals.
    """
    # Convert reference timestamps to seconds since start
    rr_ref = rr_ref.copy()
    start_time = rr_ref['Timestamp'].min()
    rr_ref['Timestamp'] = (rr_ref['Timestamp'] - start_time).dt.total_seconds()
    rr_ref['RR_Interval_REF'] = rr_ref['RR Interval in seconds']
    rr_ref['Heart_Rate_REF'] = rr_ref['Heart Rate']

    # Sort both DataFrames by timestamp
    bcg_rr = bcg_rr.sort_values('Timestamp')
    rr_ref = rr_ref.sort_values('Timestamp')

    # Convert time threshold to seconds
    threshold_seconds = pd.Timedelta(time_threshold).total_seconds()

    # Perform the merge
    merged = pd.merge_asof(
        bcg_rr,
        rr_ref[['Timestamp', 'RR_Interval_REF', 'Heart_Rate_REF']],
        on='Timestamp',
        direction='nearest',
        tolerance=threshold_seconds)

    merged['RR_Difference'] = merged['RR_Interval'] - merged['RR_Interval_REF']
    merged['HR_Difference'] = merged['Heart_Rate'] - merged['Heart_Rate_REF']

    return merged


def resample_signal(signal, original_fs, target_fs):
    """
    Resample signal from original_fs to target_fs
    """
    number_of_samples = round(len(signal) * float(target_fs) / original_fs)
    resampled_signal = resample(signal, number_of_samples)
    return resampled_signal


def create_epochs(data, timestamps, epoch_length=140):
    """
    Create epochs of specified length from the data
    """
    n_epochs = len(data) // epoch_length
    epochs = []
    epoch_timestamps = []
    
    for i in range(n_epochs):
        start_idx = i * epoch_length
        end_idx = start_idx + epoch_length
        epochs.append(data[start_idx:end_idx])
        epoch_timestamps.append(timestamps[start_idx])
    
    return np.array(epochs), np.array(epoch_timestamps)


# Main program starts here
print('\nStart processing ...')

# File paths
file = r'dataset\data\01\BCG\01_20231104_BCG.csv'
rr_ref_file = r'dataset\data\01\Reference\RR\01_20231104_RR.csv'

# Load reference RR data
try:
    rr_ref = pd.read_csv(rr_ref_file, sep=",", header=0)
    rr_ref.columns = rr_ref.columns.str.strip()

    # Clean and parse timestamps
    if rr_ref['Timestamp'].dtype == object:
        rr_ref['Timestamp'] = rr_ref['Timestamp'].str.strip()
    rr_ref['Timestamp'] = pd.to_datetime(
        rr_ref['Timestamp'], format='%Y/%m/%d %H:%M:%S')

    print(f"Successfully loaded reference RR data with {len(rr_ref)} records")
except Exception as e:
    print(f"Error loading reference RR data: {str(e)}")
    raise

# Load BCG data
if file.endswith(".csv"):
    try:
        rawData = pd.read_csv(file, sep=",", header=0)
        rawData.columns = rawData.columns.str.strip()

        # Ensure required columns exist
        required_columns = ['Timestamp', 'BCG']
        for col in required_columns:
            if col not in rawData.columns:
                raise ValueError(
                    f"Expected column '{col}' not found in the file")

        # Handle missing values
        rawData['Timestamp'] = rawData['Timestamp'].ffill()
        rawData['BCG'] = rawData['BCG'].fillna(rawData['BCG'].mean())

        # Convert timestamps - handle both string and datetime inputs
        if rawData['Timestamp'].dtype == object:
            rawData['Timestamp'] = rawData['Timestamp'].str.strip()
            utc_time = pd.to_datetime(
                rawData['Timestamp'], format='%Y/%m/%d %H:%M:%S')
        else:
            utc_time = pd.to_datetime(rawData['Timestamp'])

        data_stream = rawData['BCG'].values
        print(f"Successfully loaded BCG data with {len(data_stream)} samples")

        # Handle NaNs
        if np.isnan(data_stream).any():
            data_stream[np.isnan(data_stream)] = np.nanmean(data_stream)

        # Signal processing parameters
        fs = 140  # Original sampling frequency
        target_fs = 50  # Target sampling frequency
        
        # Resample the data
        resampled_data = resample_signal(data_stream, fs, target_fs)
        resampled_time = np.linspace(0, len(resampled_data)/target_fs, len(resampled_data))
        
        # Apply low-pass filter to resampled BCG signal
        def apply_lowpass_filter(signal, cutoff_freq, fs, order=4):
            """
            Apply a low-pass filter to the signal
            cutoff_freq: cutoff frequency in Hz
            fs: sampling frequency in Hz
            order: filter order
            """
            nyquist = 0.5 * fs
            normal_cutoff = cutoff_freq / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, signal)
        
        # Apply low-pass filter with cutoff at 15 Hz (increased to preserve more signal content)
        cutoff_freq = 15  # Hz
        resampled_data = apply_lowpass_filter(resampled_data, cutoff_freq, target_fs)
        
        # Remove baseline drift with moderate polynomial order
        resampled_data = remove_nonLinear_trend(resampled_data, 4)
        
        # Normalize the resampled data
        resampled_data = (resampled_data - np.mean(resampled_data)) / np.std(resampled_data)
        
        # Create epochs (140 samples per epoch)
        epochs, epoch_timestamps = create_epochs(resampled_data, resampled_time)
        
        # Create DataFrame for epochs
        epoch_df = pd.DataFrame({
            'Timestamp': epoch_timestamps,
            'Epoch_Data': [epoch.tolist() for epoch in epochs]
        })
        
        # Save epochs to CSV
        epoch_csv_path = 'results/epochs.csv'
        epoch_df.to_csv(epoch_csv_path, index=False)
        print(f"Saved epochs to {epoch_csv_path}")

        # Continue with original processing using resampled data
        window_length = int(2800 * target_fs / fs)  # Adjust window length for new sampling rate
        window_shift = window_length

        # Process signals
        data_stream, utc_time = detect_patterns(
            0, window_length, window_shift, resampled_data, resampled_time, plot=1)
        
        # Enhanced BCG signal processing
        # 1. Apply band-pass filter with adjusted frequencies for 50 Hz
        # Adjust band-pass filter frequencies for better J-wave detection
        movement = band_pass_filtering(data_stream, target_fs, "bcg")
        
        # 2. Apply additional smoothing with parameters optimized for J-wave detection
        movement = savgol_filter(movement, window_length=13, polyorder=3)  # Increased window length
        
        # 3. Normalize the signal
        movement = (movement - np.mean(movement)) / np.std(movement)
        
        # Process breathing signal
        breathing = band_pass_filtering(data_stream, target_fs, "breath")
        breathing = remove_nonLinear_trend(breathing, 3)
        breathing = savgol_filter(breathing, 11, 3)
        
        # Ensure all signals have the same length
        min_length = min(len(resampled_time), len(data_stream), len(movement), len(breathing))
        resampled_time = resampled_time[:min_length]
        data_stream = data_stream[:min_length]
        movement = movement[:min_length]
        breathing = breathing[:min_length]
        
        # 4. Apply wavelet decomposition with adjusted parameters for better J-wave detection
        w = modwt(movement, 'db4', 4)  # Reduced decomposition level
        dc = modwtmra(w, 'db4')
        wavelet_cycle = dc[3]  # Using level 3 for better J-peak detection
        
        # 5. Additional smoothing of wavelet cycle
        wavelet_cycle = savgol_filter(wavelet_cycle, window_length=13, polyorder=3)  # Increased window length
        
        # 6. Normalize wavelet cycle
        wavelet_cycle = (wavelet_cycle - np.mean(wavelet_cycle)) / np.std(wavelet_cycle)
        
        # 7. Apply peak enhancement (invert if needed)
        if np.mean(wavelet_cycle) < 0:
            wavelet_cycle = -wavelet_cycle  # Invert if mean is negative
        
        # Ensure wavelet cycle has the same length
        wavelet_cycle = wavelet_cycle[:min_length]
        
        # Save processed signals with BCG-ECG relationship information
        processed_signals = pd.DataFrame({
            'Timestamp': resampled_time,
            'Raw_BCG': data_stream,
            'Filtered_BCG': movement,
            'Breathing': breathing,
            'Wavelet_Cycle': wavelet_cycle,
            'Low_Pass_Filtered_BCG': resampled_data[:min_length]  # Add low-pass filtered signal
        })
        processed_signals.to_csv('results/processed_signals.csv', index=False)

        # Heart rate detection with enhanced J-peak detection
        # Adjust parameters for better J-wave detection (corresponding to R-wave in ECG)
        limit = int(math.floor(breathing.size / window_shift))
        resampled_time_series = pd.Series(resampled_time)
        
        # Adjust minimum peak distance for 50 Hz sampling
        # Typical RR interval is 0.6-1.2 seconds (50-100 bpm)
        # For 50 Hz, this translates to 30-60 samples
        mpd = int(0.7 * target_fs)  # Increased minimum peak distance to 0.7 seconds
        
        # Use wavelet cycle for peak detection
        heart_rate_bcg, j_peak_times = vitals(
            0, window_length, window_shift, limit, wavelet_cycle, resampled_time_series, 
            mpd=mpd, plot=1, signal_type="heart"
        )

        # Ensure heart_rate_bcg and j_peak_times have the same length
        if isinstance(heart_rate_bcg, np.ndarray):
            heart_rate_bcg = heart_rate_bcg.flatten()
        else:
            heart_rate_bcg = np.array([heart_rate_bcg])

        # Create arrays of the same length
        min_length = min(len(heart_rate_bcg), len(j_peak_times))
        heart_rate_bcg = heart_rate_bcg[:min_length]
        j_peak_times = j_peak_times[:min_length]

        # Create sample indices and time since start arrays
        sample_indices = np.arange(min_length)
        time_since_start = j_peak_times - j_peak_times[0] if len(j_peak_times) > 0 else np.array([])

        # Save J-peak detection results with timing information
        j_peaks_df = pd.DataFrame({
            'Timestamp': j_peak_times,
            'Heart_Rate': heart_rate_bcg,
            'Sample_Index': sample_indices,
            'Time_Since_Start': time_since_start
        })
        j_peaks_df.to_csv('results/j_peaks.csv', index=False)

        # RR interval processing with BCG-ECG relationship consideration
        if len(j_peak_times) > 1:
            # Calculate RR intervals directly from the time differences
            rr_intervals = np.diff(j_peak_times)
            
            # Filter out physiologically impossible intervals
            # Normal heart rate range: 40-200 bpm
            # This corresponds to RR intervals of 0.3-1.5 seconds
            valid_intervals = (rr_intervals > 0.3) & (rr_intervals < 1.5)
            rr_intervals = rr_intervals[valid_intervals]
            j_peak_times = j_peak_times[1:][valid_intervals]

            if len(rr_intervals) > 0:
                # Create arrays of the same length for the RR DataFrame
                min_rr_length = min(len(j_peak_times), len(rr_intervals))
                j_peak_times = j_peak_times[:min_rr_length]
                rr_intervals = rr_intervals[:min_rr_length]
                heart_rates = 60 / rr_intervals
                expected_delays = np.full(min_rr_length, 0.15)  # Typical EMD delay in seconds

                bcg_rr = pd.DataFrame({
                    'Timestamp': j_peak_times,
                    'RR_Interval': rr_intervals,
                    'Heart_Rate': heart_rates,
                    'Expected_ECG_Delay': expected_delays
                })

                # Remove outliers from heart rate using MAD with moderate threshold
                median_hr = np.median(heart_rates)
                mad = np.median(np.abs(heart_rates - median_hr))
                bcg_rr = bcg_rr[np.abs(heart_rates - median_hr) <= 3.0 * mad]  # Moderate threshold

                # Synchronize with reference using a moderate threshold
                synchronized_rr = synchronize_rr(bcg_rr, rr_ref, time_threshold='0.25s')  # Moderate threshold
                valid_matches = synchronized_rr.dropna(subset=['RR_Interval_REF'])

                if len(valid_matches) > 0:
                    # Calculate error metrics
                    mae = np.mean(np.abs(valid_matches['HR_Difference']))
                    rmse = np.sqrt(np.mean(valid_matches['HR_Difference']**2))
                    mape = np.mean(np.abs(valid_matches['HR_Difference'] / valid_matches['Heart_Rate_REF'])) * 100
                    correlation = valid_matches['Heart_Rate'].corr(valid_matches['Heart_Rate_REF'])

                    # Save error metrics with BCG-ECG relationship information
                    with open('results/error_metrics.txt', 'w') as f:
                        f.write(f"Mean Absolute Error (MAE): {mae:.2f} bpm\n")
                        f.write(f"Root Mean Square Error (RMSE): {rmse:.2f} bpm\n")
                        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")
                        f.write(f"Number of valid matches: {len(valid_matches)}\n")
                        f.write(f"Correlation coefficient: {correlation:.3f}\n")
                        f.write(f"Sampling frequency: {target_fs} Hz\n")
                        f.write(f"Number of J-peaks detected: {len(j_peak_times)}\n")
                        f.write(f"Number of valid RR intervals: {len(rr_intervals)}\n")
                        f.write(f"Typical EMD delay: 150ms\n")
                        f.write(f"J-wave to R-wave relationship: J-wave follows R-wave by ~150ms\n")
                        f.write(f"Signal processing steps:\n")
                        f.write(f"1. Resampling from {fs} Hz to {target_fs} Hz\n")
                        f.write(f"2. Low-pass filtering (cutoff: {cutoff_freq} Hz)\n")
                        f.write(f"3. Baseline drift removal (order 4)\n")
                        f.write(f"4. Band-pass filtering for BCG\n")
                        f.write(f"5. Wavelet decomposition (level 4)\n")
                        f.write(f"6. Signal inversion check and correction\n")
                        f.write(f"7. J-peak detection with minimum distance: {mpd/target_fs:.2f} seconds\n")
                        f.write(f"8. Moderate outlier removal (3.0 * MAD)\n")
                        f.write(f"9. Moderate synchronization threshold (0.25s)\n")

                    # Create Bland-Altman plot
                    mean_hr = (valid_matches['Heart_Rate'] + valid_matches['Heart_Rate_REF']) / 2
                    hr_diff = valid_matches['Heart_Rate'] - valid_matches['Heart_Rate_REF']
                    mean_diff = np.mean(hr_diff)
                    std_diff = np.std(hr_diff)

                    plt.figure(figsize=(10, 6))
                    plt.scatter(mean_hr, hr_diff, alpha=0.5)
                    plt.axhline(y=mean_diff, color='r', linestyle='-', label=f'Mean difference: {mean_diff:.2f}')
                    plt.axhline(y=mean_diff + 1.96*std_diff, color='g', linestyle='--', 
                              label=f'Upper limit: {mean_diff + 1.96*std_diff:.2f}')
                    plt.axhline(y=mean_diff - 1.96*std_diff, color='g', linestyle='--', 
                              label=f'Lower limit: {mean_diff - 1.96*std_diff:.2f}')
                    plt.xlabel('Mean of BCG and Reference HR (bpm)')
                    plt.ylabel('Difference (BCG - Reference) (bpm)')
                    plt.title('Bland-Altman Plot')
                    plt.legend()
                    plt.savefig('results/bland_altman_plot.png')
                    plt.close()

                    # Create Pearson correlation plot
                    plt.figure(figsize=(10, 6))
                    plt.scatter(valid_matches['Heart_Rate_REF'], valid_matches['Heart_Rate'], alpha=0.5)
                    correlation = valid_matches['Heart_Rate'].corr(valid_matches['Heart_Rate_REF'])
                    
                    # Add regression line
                    z = np.polyfit(valid_matches['Heart_Rate_REF'], valid_matches['Heart_Rate'], 1)
                    p = np.poly1d(z)
                    plt.plot(valid_matches['Heart_Rate_REF'], p(valid_matches['Heart_Rate_REF']), 
                            "r--", alpha=0.8)
                    
                    # Add perfect correlation line (y=x)
                    min_hr = min(valid_matches['Heart_Rate_REF'].min(), valid_matches['Heart_Rate'].min())
                    max_hr = max(valid_matches['Heart_Rate_REF'].max(), valid_matches['Heart_Rate'].max())
                    plt.plot([min_hr, max_hr], [min_hr, max_hr], 'k--', alpha=0.5, label='Perfect Correlation')
                    
                    # Add labels and title
                    plt.xlabel('Reference Heart Rate (bpm)', fontsize=12)
                    plt.ylabel('BCG Heart Rate (bpm)', fontsize=12)
                    plt.title(f'BCG vs Reference Heart Rate Comparison\nCorrelation Coefficient (r) = {correlation:.3f}', 
                             fontsize=14, pad=20)
                    
                    # Add grid for better readability
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add legend
                    plt.legend(['Data Points', 'Regression Line', 'Perfect Correlation'], 
                              loc='upper left', fontsize=10)
                    
                    # Add text box with statistics
                    stats_text = f'Number of samples: {len(valid_matches)}\n'
                    stats_text += f'Mean BCG HR: {valid_matches["Heart_Rate"].mean():.1f} bpm\n'
                    stats_text += f'Mean Ref HR: {valid_matches["Heart_Rate_REF"].mean():.1f} bpm\n'
                    stats_text += f'RMSE: {rmse:.1f} bpm'
                    
                    plt.text(0.02, 0.98, stats_text,
                            transform=plt.gca().transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Adjust layout to prevent label cutoff
                    plt.tight_layout()
                    
                    plt.savefig('results/pearson_correlation_plot.png', dpi=300, bbox_inches='tight')
                    plt.close()

                    # Create Bland-Altman plot
                    plt.figure(figsize=(10, 6))
                    mean_hr = (valid_matches['Heart_Rate'] + valid_matches['Heart_Rate_REF']) / 2
                    hr_diff = valid_matches['Heart_Rate'] - valid_matches['Heart_Rate_REF']
                    mean_diff = np.mean(hr_diff)
                    std_diff = np.std(hr_diff)

                    plt.scatter(mean_hr, hr_diff, alpha=0.5)
                    plt.axhline(y=mean_diff, color='r', linestyle='-', 
                              label=f'Mean difference: {mean_diff:.2f} bpm')
                    plt.axhline(y=mean_diff + 1.96*std_diff, color='g', linestyle='--', 
                              label=f'Upper limit: {mean_diff + 1.96*std_diff:.2f} bpm')
                    plt.axhline(y=mean_diff - 1.96*std_diff, color='g', linestyle='--', 
                              label=f'Lower limit: {mean_diff - 1.96*std_diff:.2f} bpm')
                    
                    # Add labels and title
                    plt.xlabel('Mean of BCG and Reference HR (bpm)', fontsize=12)
                    plt.ylabel('Difference (BCG - Reference) (bpm)', fontsize=12)
                    plt.title('Bland-Altman Plot of Heart Rate Measurements', fontsize=14, pad=20)
                    
                    # Add grid for better readability
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add legend
                    plt.legend(loc='upper right', fontsize=10)
                    
                    # Add text box with statistics
                    stats_text = f'Number of samples: {len(valid_matches)}\n'
                    stats_text += f'Mean difference: {mean_diff:.2f} bpm\n'
                    stats_text += f'SD of difference: {std_diff:.2f} bpm\n'
                    stats_text += f'95% limits of agreement:\n'
                    stats_text += f'  Upper: {mean_diff + 1.96*std_diff:.2f} bpm\n'
                    stats_text += f'  Lower: {mean_diff - 1.96*std_diff:.2f} bpm'
                    
                    plt.text(0.02, 0.98, stats_text,
                            transform=plt.gca().transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Adjust layout to prevent label cutoff
                    plt.tight_layout()
                    
                    plt.savefig('results/bland_altman_plot.png', dpi=300, bbox_inches='tight')
                    plt.close()

        # Print processing summary
        print("\nProcessing Summary:")
        print(f"1. Generated {len(epochs)} epochs of 140 samples each")
        print(f"2. Resampled data from {fs} Hz to {target_fs} Hz")
        print(f"3. Applied low-pass filter with cutoff at {cutoff_freq} Hz")
        print(f"4. Processed {min_length} samples")
        print(f"5. Detected {len(j_peak_times)} J-peaks")
        print(f"6. Found {len(valid_matches) if 'valid_matches' in locals() else 0} synchronized RR intervals")
        print("\nOutput Files:")
        print("- results/epochs.csv: Epoch data")
        print("- results/pattern_detection.csv: Movement patterns")
        print("- results/processed_signals.csv: Filtered signals")
        print("- results/j_peaks.csv: Detected J-peaks")
        print("- results/error_metrics.txt: Performance metrics")
        print("- results/*.png: Visualization plots")

        # Visualization of signals
        os.makedirs("results", exist_ok=True)
        t1 = 0
        t2 = min(len(data_stream), window_length)
        
        # Ensure all signals are numpy arrays and have the same length
        data_stream = np.array(data_stream)
        movement = np.array(movement)
        breathing = np.array(breathing)
        wavelet_cycle = np.array(wavelet_cycle)
        
        # Print debug information
        print(f"Signal lengths - data_stream: {len(data_stream)}, movement: {len(movement)}, "
              f"breathing: {len(breathing)}, wavelet_cycle: {len(wavelet_cycle)}")
        
        # Call data_subplot with synchronized signals
        data_subplot(data_stream, movement, breathing, wavelet_cycle, t1, t2)
        print("Saved vitals subplot to results\\vitals.png")

        print('\nProcessing completed successfully')

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

print('\nEnd processing ...')
