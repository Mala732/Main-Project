
# Example: New raw EEG data for prediction (replace this with actual data)
# Assuming the input is a 2D array with shape (n_samples, n_features) where n_samples is the number of trials.
#new_data = loadmat("trial000.mat")['data'].transpose()  # 10 trials, each with 71680 features (adjust based on your input data)
import numpy as np
import joblib
from scipy.signal import savgol_filter, butter, filtfilt
import pywt
from Random_test_code import EEGProcessor,FeatureExtractor
from scipy.io import loadmat
# Load the trained model and scaler
svm_model = joblib.load("movement_classifier5.joblib")
scaler = joblib.load("scaler5.joblib")
selector = joblib.load("selector5.joblib")


#new_data = np.array([loadmat("A213.mat")['data'].transpose()[0][4337*3:433947*4]])
#new_data = loadmat("s01.mat")['eeg']['movement_left'][0][0]
#new_data = loadmat("trial_data.mat")['eeg']['movement_left'][0][0][88:]
new_data = loadmat("dataSet/right_trials.mat")['data'][:]
# 10 trials, each with 71680 features (adjust based on your input data)
#new_data = loadmat("trial000.mat")['data'].transpose()
# Preprocess the new data (apply the same filter as during training)


    
preprocessor = EEGProcessor()
filtered_new_data = preprocessor.preprocess_data(new_data)

# Extract features from the new data using wavelet decomposition
feature_extractor = FeatureExtractor()
new_wavelet_features = feature_extractor.extract_wavelet_features(filtered_new_data)

# Scale the features using the saved scaler
scaled_new_features = scaler.transform(new_wavelet_features)
scaled_new_features = selector.transform(scaled_new_features)

# Make predictions using the loaded SVM model
predictions = svm_model.predict(scaled_new_features)
pred_dict = {}

# Output predictions: 0 for Left Hand, 1 for Right Hand
for idx, pred in enumerate(predictions):
    pred_dict['Sample'+str(idx+1)] = 'Left Hand' if pred == 0 else "Right Hand"
    print(f"Sample {idx + 1}: {'Left Hand' if pred == 0 else 'Right Hand'}")


