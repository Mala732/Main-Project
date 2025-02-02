import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter, butter, filtfilt
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve, auc
)

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import (SelectKBest, f_classif, chi2,f_regression,
SelectPercentile, VarianceThreshold)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from scipy import stats
import joblib  # For saving and loading models

class EEGProcessor:
    def __init__(self, sampling_rate = 10000,window_length=5, polyorder = 4):
        self.sampling_rate = sampling_rate
        self.window_length =window_length
        self.polyorder = polyorder
        
    # def apply_bandpass_filter(self, data, lowcut=0.5, highcut=45.0, order=4):
    #     """Apply bandpass filter to remove noise and unwanted frequencies"""
    #     nyquist = 0.5 * self.sampling_rate
    #     low = lowcut / nyquist
    #     high = highcut / nyquist
    #     b, a = butter(order, [low, high], btype='band')
    #     return filtfilt(b, a, data, axis=1)

    def preprocess_data(self, data):
        filtered_data = np.zeros_like(data)
        #data = self.apply_bandpass_filter(data)
        """Preprocess data with filtering and smoothing"""
        for i in range(data.shape[0]):
            filtered_data[i] = savgol_filter(data[i], self.window_length, self.polyorder)
        return filtered_data

class FeatureExtractor:
    def extract_wavelet_features(self, data, wavelet='db18', level=3):
        """Extract wavelet-based features"""
        features = []
        for trial in data:
            coeffs = pywt.wavedec(trial, wavelet, level=level)
            trial_features = []
            for coeff in coeffs:
                stats_features = [
                    np.mean(coeff), np.std(coeff), np.var(coeff),
                    np.max(coeff), np.min(coeff), stats.kurtosis(coeff),
                    stats.skew(coeff), np.sum(coeff**2), np.sqrt(np.mean(coeff**2))
                ]
                trial_features.extend(stats_features)
            features.append(trial_features)
        return np.array(features)

class MovementClassifier:
    def __init__(self):
        self.model = SVC(probability=True, kernel='poly', C=68.13075385877796, gamma=1.1792127353952497,
                         degree = 3,random_state=42,class_weight='balanced')
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors = 10)
        self.selector = SequentialFeatureSelector(self.knn, n_features_to_select=4)

    def train(self, X, y):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        X_train = self.selector.fit_transform(X_train, y_train)
        X_test = self.selector.transform(X_test)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Save model and scaler
        joblib.dump(self.model, "movement_classifier5.joblib")
        joblib.dump(self.scaler, "scaler5.joblib")
        joblib.dump(self.selector, "selector5.joblib")
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Left", "Right"]))

        # ROC Curve
        y_prob = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend()
        plt.show()

        return X_test, y_test

    def load_model(self):
        """Load saved model and scaler"""
        self.model = joblib.load("movement_classifier5.joblib")
        self.scaler = joblib.load("scaler5.joblib")
        self.selector = joblib.load("selector5.joblib")

    def predict_real_time(self, data):
        """Simulate real-time EEG data classification"""
        processed_data = self.scaler.transform(data)
        processed_data = self.selector.transform(processed_data)
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        return predictions, probabilities

def main():
    # Load data
    file_path = "trial2_data.mat"
    mat_data = loadmat(file_path)
    movement_left = mat_data['eeg']['movement_left'][0][0]
    movement_right = mat_data['eeg']['movement_right'][0][0]

    # Combine data and create labels
    X = np.vstack([movement_left, movement_right])
    y = np.hstack([np.zeros(movement_left.shape[0]), np.ones(movement_right.shape[0])])

    # Initialize processor and classifier
    processor = EEGProcessor()
    feature_extractor = FeatureExtractor()
    classifier = MovementClassifier()

    # Preprocess data
    print("Preprocessing data...")
    processed_data = processor.preprocess_data(X)

    # Extract features
    print("Extracting features...")
    features = feature_extractor.extract_wavelet_features(processed_data)

    # Train and evaluate the model
    print("Training and evaluating model...")
    X_test, y_test = classifier.train(features, y)

    # Real-time simulation
    #print("\nSimulating real-time predictions...")
    # classifier.load_model()
    # random_sample =   # Simulate first 5 real-time samples
    # predictions, probabilities = classifier.predict_real_time(random_sample)

    # for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    #     print(f"Sample {i + 1}: Predicted = {'Right' if pred == 1 else 'Left'}, Probabilities = {prob}")
    # print(y_test)
if __name__ == "__main__":
    main()
