import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter, filtfilt, welch
import pywt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


def calculate_snr(raw_signal, filtered_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR).
    
    Parameters:
        raw_signal (np.ndarray): The original noisy signal.
        filtered_signal (np.ndarray): The signal after filtering.

    Returns:
        float: SNR improvement in dB.
    """
    noise = raw_signal - filtered_signal
    signal_power = np.mean(filtered_signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


def optimize_savgol_filter(signals):
    """
    Optimizes Savitzky-Golay filter parameters over an array of signals to maximize SNR improvement.
    
    Parameters:
        signals (np.ndarray): 2D array of shape (n_signals, signal_length) containing noisy signals.
        
    Returns:
        dict: Dictionary containing the best parameters (window_length, polyorder)
              and their corresponding average SNR improvement.
    """
    best_params = {'window_length': None, 'polyorder': None, 'avg_snr_improvement': -float('inf')}
    window_lengths = range(5, 51, 2)  # Odd numbers from 5 to 50
    polyorders = range(2, 6)          # Polynomial orders from 2 to 5

    for window_length in window_lengths:
        for polyorder in polyorders:
            if polyorder < window_length:  # Ensure polyorder < window_length
                total_snr_improvement = 0
                valid_signals = 0

                for signal in signals:
                    try:
                        # Apply Savitzky-Golay filter to the current signal
                        filtered_signal = savgol_filter(signal, window_length, polyorder)

                        # Calculate SNR improvement
                        snr_improvement = calculate_snr(signal, filtered_signal)
                        total_snr_improvement += snr_improvement
                        valid_signals += 1
                    except ValueError:
                        # Skip invalid combinations (e.g., window_length > signal length)
                        continue

                # Calculate the average SNR improvement for this parameter combination
                avg_snr_improvement = total_snr_improvement / valid_signals if valid_signals > 0 else -float('inf')

                # Update the best parameters based on SNR improvement
                if avg_snr_improvement > best_params['avg_snr_improvement']:
                    best_params.update({
                        'window_length': window_length,
                        'polyorder': polyorder,
                        'avg_snr_improvement': avg_snr_improvement
                    })

    return best_params



def find_best_wavelet_and_level(data, labels, wavelet_list, signal_length):
    """
    Perform a grid search over wavelets and decomposition levels to find the best combination.
    """
    best_accuracy = 0
    best_params = None

    for wavelet in wavelet_list:
        try:
            # Determine the maximum allowable level for this wavelet and signal length
            max_level = pywt.dwt_max_level(signal_length, pywt.Wavelet(wavelet).dec_len)

            for level in range(1, max_level + 1):
                try:
                    # Extract features for the current wavelet and level
                    features = extract_waveletfeatures(data, wavelet, level)

                    # Scale the features
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        features_scaled, labels, test_size=0.3, random_state=42, stratify=labels
                    )

                    # Train SVM classifier
                    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                    svm.fit(X_train, y_train)

                    # Evaluate on the test set
                    y_pred = svm.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(accuracy)
                    # Update best parameters if current accuracy is higher
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'wavelet': wavelet, 'level': level, 'accuracy': accuracy}
                        print(best_params)
                except Exception as e:
                    print(f"Error with wavelet '{wavelet}' and level {level}: {e}")
                    continue

        except Exception as e:
            print(f"Error initializing wavelet '{wavelet}': {e}")
            continue

    return best_params


def extract_waveletfeatures(data,wavelet,level):        
    features = []
    for trial in data:
        coeffs = pywt.wavedec(trial, wavelet, level=level)
        trial_features = []
        for coeff in coeffs:
            trial_features.extend([
                np.mean(coeff), np.std(coeff), np.var(coeff),
                np.max(coeff), np.min(coeff), np.mean(np.abs(coeff)),
                np.sum(coeff**2), np.sqrt(np.mean(coeff**2))
            ])
        features.append(trial_features)
    return np.array(features)