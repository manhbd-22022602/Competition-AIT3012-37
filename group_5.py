import os
import numpy as np
import pandas as pd
import mne
import joblib
import random
import argparse

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import hilbert, welch, coherence
from scipy.stats import spearmanr, skew, kurtosis
from itertools import combinations
from sklearn.model_selection import RandomizedSearchCV
from typing import Tuple, Dict, List
from joblib import Parallel, delayed

class BrainAgePredictor:
    """
    A class to predict brain age from EEG time series data using fractal dimension features.
    """

    def __init__(self, data_dir: str = "./data", n_test_subjects: int = 40, random_state: int = 42,
                use_fd: bool = True, use_freq: bool = True, use_range: bool = True,
                use_amplitude: bool = True, use_connectivity: bool = True, use_psd: bool = True):
        """
        Initialize the BrainAgePredictor.

        Args:
            data_dir (str): Directory containing the EEG data files
            n_test_subjects (int): Number of subjects to use for testing
            random_state (int): Random seed for reproducibility
            use_fd (bool): Whether to use fractal dimension features
            use_freq (bool): Whether to use frequency domain features
            use_range (bool): Whether to use range domain features
        """
        self.data_dir = data_dir
        self.n_test_subjects = n_test_subjects
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.use_fd = use_fd
        self.use_range = use_range
        self.use_amplitude = use_amplitude
        self.use_connectivity = use_connectivity
        self.use_psd = use_psd

    def compute_fractal_features(self, raw_data: mne.io.Raw) -> np.ndarray:
        """
        Compute fractal dimension features for each channel. (Bac)

        Args:
            raw_data (mne.io.Raw): Raw EEG data

        Returns:
            np.ndarray: Array of fractal dimension features

        Raises:
            ValueError: If NaN values are detected in signal processing
        """
        misc_data = raw_data.copy().pick('misc')
        data = misc_data.get_data()
        n_channels = data.shape[0]

        fd_features = []
        window_size = 1000  # 4 seconds at 250 Hz
        step_size = 500    # 50% overlap

        def higuchi_fd(signal: np.ndarray, kmax: int = 10) -> float:
            """
            Calculate Higuchi Fractal Dimension of a signal with improved numerical stability.
        
            Args:
                signal (np.ndarray): Input signal
                kmax (int): Maximum delay/lag value
        
            Returns:
                float: Fractal dimension
            """
            N = len(signal)
            L = []
            x = []
        
            for k in range(1, kmax + 1):
                Lk = 0
                for m in range(k):
                    # Calculate length Lm(k)
                    indices = np.arange(1, int((N-m)/k), dtype=np.int32)
        
                    # Skip if we don't have enough points
                    if len(indices) < 2:
                        continue
        
                    Lmk = np.abs(signal[m + indices*k] - signal[m + (indices-1)*k]).sum()
        
                    # Normalize by the scale factor
                    scale = (N - 1) / (((N - m) / k) * k)
                    Lmk = (Lmk * scale) / k
        
                    if not np.isnan(Lmk) and not np.isinf(Lmk):
                        Lk += Lmk
        
                # Add small constant to prevent log(0)
                eps = 1e-10
                if Lk > eps:  # Only compute log if Lk is significantly greater than 0
                    L.append(np.log(Lk / k))
                    x.append(np.log(1.0 / k))
        
            # Check if we have enough points for linear fit
            if len(L) < 2:
                return 0.0  # Return 0 if we can't compute FD
        
            # Fit line and get slope
            try:
                polyfit = np.polyfit(x, L, 1)
                return abs(polyfit[0])  # Return absolute value as FD should be positive
            except:
                return 0.0  # Return 0 if fitting fails

        for channel in range(n_channels):
            channel_fd = []
            signal = data[channel]

            # Compute FD for each window
            for start in range(0, len(signal) - window_size, step_size):
                window = signal[start:start + window_size]
                try:
                    fd = higuchi_fd(window)
                    if not np.isnan(fd) and not np.isinf(fd):
                        channel_fd.append(fd)
                except Exception as e:
                    print(f"Error computing FD for channel {channel}: {str(e)}")
                    continue

            if channel_fd:
                # Calculate statistical features of FD values
                fd_stats = [
                    np.mean(channel_fd),
                    np.std(channel_fd),
                    np.median(channel_fd),
                    np.max(channel_fd),
                    np.min(channel_fd)
                ]
                fd_features.extend(fd_stats)
            else:
                fd_features.extend([0.0] * 5)  # Add zeros if computation fails

        return np.array(fd_features)

    def compute_range_features(self, raw_data: mne.io.Raw) -> np.ndarray:
        """
        Compute range domain features for each channel. (Hung)
        """
        misc_data = raw_data.copy().pick('misc')
        data = misc_data.get_data()
        
        # Calculate statistical features
        mean_vals = np.mean(data, axis=1)
        median_vals = np.median(data, axis=1)
        percentile_5_vals = np.percentile(data, 5, axis=1)
        percentile_95_vals = np.percentile(data, 95, axis=1)
        std_vals = np.std(data, axis=1)
        coef_var_vals = std_vals / (mean_vals + 1e-8)
        skewness_vals = skew(data, axis=1)
        kurtosis_vals = kurtosis(data, axis=1)
        
        # Combine all features
        range_features = np.hstack([
            mean_vals, median_vals, percentile_5_vals, percentile_95_vals,
            std_vals, coef_var_vals, skewness_vals, kurtosis_vals
        ])
        
        return range_features
    
    def compute_amplitude_features(self, raw_data: mne.io.Raw) -> np.ndarray:
        """
        Compute amplitude domain features for each channel. (Ng Manh)
        
        Args:
            raw_data (mne.io.Raw): Raw EEG data
            
        Returns:
            np.ndarray: Array of amplitude domain features
        """
        misc_data = raw_data.copy().pick('misc')
        picks = mne.pick_types(raw_data.info, misc=True)
        
        # Filter signals into different frequency bands
        delta_signal = misc_data.copy().filter(l_freq=0.5, h_freq=4, picks=picks, method='iir', verbose=False)
        theta_signal = misc_data.copy().filter(l_freq=4, h_freq=7, picks=picks, method='iir', verbose=False)
        alpha_signal = misc_data.copy().filter(l_freq=7, h_freq=13, picks=picks, method='iir', verbose=False)
        beta_signal = misc_data.copy().filter(l_freq=13, h_freq=30, picks=picks, method='iir', verbose=False)
        wide_signal = misc_data.copy().filter(l_freq=0.5, h_freq=30, picks=picks, method='iir', verbose=False)
        
        def compute_band_features(signal):
            data = signal.get_data()
            # Statistical features
            mean = np.mean(data, axis=1)
            std = np.std(data, axis=1)
            skewness = skew(data, axis=1)
            kurt = kurtosis(data, axis=1)
            # Envelope
            analytic_signal = hilbert(data)
            envelope = np.abs(analytic_signal)
            env_mean = np.mean(envelope, axis=1)
            
            features = np.concatenate([mean, std, skewness, kurt, env_mean])
            return features
        
        # Compute features for each frequency band
        delta_features = compute_band_features(delta_signal)
        theta_features = compute_band_features(theta_signal)
        alpha_features = compute_band_features(alpha_signal)
        beta_features = compute_band_features(beta_signal)
        wide_features = compute_band_features(wide_signal)
        
        # Combine all features
        amplitude_features = np.concatenate([
            delta_features, theta_features, alpha_features, 
            beta_features, wide_features
        ])
        
        return amplitude_features
    
    def compute_connectivity_features(self, raw_data: mne.io.Raw) -> np.ndarray:
        """
        Compute connectivity domain features between brain regions. (Bui Manh)
        
        Args:
            raw_data (mne.io.Raw): Raw EEG data
            
        Returns:
            np.ndarray: Array of connectivity features
        """
        misc_data = raw_data.copy().pick('misc')
        data = misc_data.get_data()
        sfreq = raw_data.info['sfreq']
        n_channels = data.shape[0]
        
        # Calculate PSD for BSI
        freqs, psds = welch(data, fs=sfreq, nperseg=int(4*sfreq))
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),    # δ band
            'theta': (4, 8),      # θ band
            'alpha': (8, 13),     # α band
            'beta': (13, 30),     # β band
            'gamma': (30, 100)    # γ band
        }
        
        # Calculate BSI for each frequency band
        bsi_features = []
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_freqs = freqs[freq_mask]
            
            # Calculate P_left[K] and P_right[K] according to formula (2)
            P_left = np.mean(psds[:n_channels//2][:, freq_mask], axis=0)  # Average over left channels
            P_right = np.mean(psds[n_channels//2:][:, freq_mask], axis=0)  # Average over right channels
            
            # Calculate BSI according to formula (1)
            bsi_values = np.abs(P_left - P_right) / (P_left + P_right)
            C_bsi = np.sum(bsi_values) / (len(band_freqs))  # 1/(b_i - a_i) * sum(...)
            
            bsi_features.append(C_bsi)

        # Get paired channels (left and right hemispheres)
        left_channels = data[:n_channels//2]
        right_channels = data[n_channels//2:]
        
        # Calculate coherence features
        coherence_features = []

        for left, right in zip(left_channels, right_channels):
            # Calculate coherence using scipy.signal.coherence
            freqs_coh, coh_values = coherence(left, right, fs=sfreq, nperseg=int(4*sfreq))
            
            # Calculate average coherence in each frequency band
            for band_name, (fmin, fmax) in bands.items():
                freq_mask = (freqs_coh >= fmin) & (freqs_coh <= fmax)
                band_coherence = np.mean(coh_values[freq_mask])
                coherence_features.append(band_coherence)

        # Combine all features
        connectivity_features = np.concatenate([
            bsi_features,
            coherence_features
        ])
        
        return connectivity_features
    
    def compute_psd_features(self, raw_data: mne.io.Raw) -> np.ndarray:
        """
        Compute Power Spectral Density features for each channel. (Ninh)
        
        Args:
            raw_data (mne.io.Raw): Raw EEG data
            
        Returns:
            np.ndarray: Array of PSD features for all channels
        """
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Get misc channel data
        misc_data = raw_data.copy().pick('misc')
        data = misc_data.get_data()
        
        # Compute PSD for all channels
        psds = []
        freqs = None
        for channel_data in data:
            freqs, psd = welch(channel_data, fs=raw_data.info['sfreq'], nperseg=1024)
            psds.append(psd)
        psds = np.array(psds)
        
        # Extract features for each channel
        all_features = []
        for channel_psd in psds:
            channel_features = []
            
            # Basic PSD features
            channel_features.extend([
                np.mean(channel_psd),  # Mean PSD
                np.max(channel_psd),   # Max PSD
                freqs[np.argmax(channel_psd)],  # Frequency at max PSD
                np.sum(channel_psd)    # Total power
            ])
            
            # Band power features
            for band_name, (low_freq, high_freq) in bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.mean(channel_psd[band_mask])
                channel_features.append(band_power)
            
            all_features.extend(channel_features)
        
        return np.array(all_features)
        
    def compute_features(self, raw_data: mne.io.Raw) -> np.ndarray:
        """
        Compute all enabled features for the input data.
        """
        features = []
        
        if self.use_fd:
            try:
                fd_features = self.compute_fractal_features(raw_data)
                features.append(fd_features)
            except Exception as e:
                print("Dac trung mien phan dang:", str(e))
            
        if self.use_range:
            try:
                range_features = self.compute_range_features(raw_data)
                features.append(range_features)
            except Exception as e:
                print("Dac trung mien pham vi:", str(e))
            
        if self.use_amplitude:
            try:
                amplitude_features = self.compute_amplitude_features(raw_data)
                features.append(amplitude_features)
            except Exception as e:
                print("Dac trung mien bien do:", str(e))
            
        if self.use_connectivity:
            try:
                connectivity_features = self.compute_connectivity_features(raw_data)
                features.append(connectivity_features)
            except Exception as e:
                print("Dac trung mien ket noi:", str(e))
            
        if self.use_psd:
            try:
                psd_features = self.compute_psd_features(raw_data)
                features.append(psd_features)
            except Exception as e:
                print("Dac trung PSD:", str(e))
        
        if not features:
            raise ValueError("No features enabled. Enable at least one feature type.")
            
        return np.concatenate(features)

    def load_age_mapping(self, tsv_file: str) -> Dict[str, float]:
        """
        Load subject age mapping from TSV file.

        Args:
            tsv_file (str): Path to TSV file containing subject IDs and ages

        Returns:
            Dict[str, float]: Dictionary mapping subject IDs to ages
        """
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            # return dict(zip(df['participant_id'], df['age']))
            label = dict(zip(df['participant_id'], df['age']))
            label_df = pd.DataFrame(list(label.items()), columns=['participant_id', 'age'])
            filtered_label_df = label_df.groupby('age').apply(lambda x: x.sample(n=min(len(x), 2))).reset_index(drop=True)
            filtered_label = dict(zip(filtered_label_df['participant_id'], filtered_label_df['age']))
            return dict(random.sample(filtered_label.items(), min(60, len(filtered_label))))
        except Exception as e:
            raise ValueError(f"Error reading age mapping from {tsv_file}: {str(e)}")

    def process_single_file(self, filename: str, fif_dir: str, age_mapping: dict):
        """
        Process a single EEG file and compute its features.
        
        Args:
            filename (str): Name of the file to process
            fif_dir (str): Directory containing the files
            age_mapping (dict): Mapping of subject IDs to ages
            
        Returns:
            tuple: (features, age, subject_id) or None if processing fails
        """
        subject_id = filename.split('_')[0]
        try:
            file_path = os.path.join(fif_dir, filename)
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    
            if raw.info['sfreq'] != 250.0:
                print(f"Warning: Unexpected sampling frequency in {filename}")
    
            features = self.compute_features(raw)
    
            if len(features) > 0:
                return features, age_mapping[subject_id], subject_id
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return None
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and preprocess EEG data files using parallel processing.
    
        Returns:
            Tuple containing:
                - np.ndarray: Feature matrix X
                - np.ndarray: Target vector y (ages)
                - List[str]: List of subject IDs
        """
        processed_count = 0
        error_count = 0
    
        tsv_file = os.path.join(self.data_dir, "filtered_subjects_with_age.tsv")
        age_mapping = self.load_age_mapping(tsv_file)
    
        fif_dir = os.path.join(self.data_dir)
        if not os.path.exists(fif_dir):
            raise FileNotFoundError(f"Directory not found: {fif_dir}")
    
        # Filter valid files
        valid_files = [f for f in os.listdir(fif_dir) 
                    if f.endswith('_sflip_parc-raw.fif') 
                    and f.split('_')[0] in age_mapping]
    
        # Process files in parallel
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self.process_single_file)(filename, fif_dir, age_mapping)
            for filename in valid_files
        )
    
        # Separate successful results
        X, y, subject_ids = [], [], []
        for result in results:
            if result is not None:
                features, age, subject_id = result
                X.append(features)
                y.append(age)
                subject_ids.append(subject_id)
                processed_count += 1
            else:
                error_count += 1
    
        print(f"\nProcessing summary:")
        print(f"Successfully processed: {processed_count} files")
        print(f"Errors encountered: {error_count} files")
    
        if len(X) == 0:
            raise ValueError("No data was successfully loaded")
    
        return np.array(X), np.array(y), subject_ids

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Optimize Random Forest hyperparameters using RandomizedSearchCV.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
        """
        param_dist = {
            'regressor__n_estimators': [100, 200, 300, 400, 500],
            'regressor__max_depth': [5, 10, 15, 20, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }

        base_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=self.random_state))
        ])

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring='neg_mean_absolute_error',
            random_state=self.random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        self.feature_importance = self.model.named_steps['regressor'].feature_importances_

    def train(self, X: np.ndarray, y: np.ndarray, subject_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Split data and train the model, ensuring exactly n_test_subjects in test set.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            subject_ids (List[str]): List of subject IDs

        Returns:
            Tuple containing train and test data splits and corresponding subject IDs
        """
        all_indices = np.arange(len(y))
        test_indices = np.random.RandomState(self.random_state).choice(
            all_indices, size=self.n_test_subjects, replace=False
        )
        train_indices = np.array([i for i in all_indices if i not in test_indices])

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        train_subjects = [subject_ids[i] for i in train_indices]
        test_subjects = [subject_ids[i] for i in test_indices]

        self.optimize_hyperparameters(X_train, y_train)

        return X_train, X_test, y_train, y_test, train_subjects, test_subjects

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Evaluate model performance.

        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True target values

        Returns:
            Tuple containing:
                - Dict[str, float]: Dictionary of evaluation metrics
                - np.ndarray: Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)
        mae = mean_absolute_error(y, predictions)
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }, predictions

    def save_results(self, y_test: np.ndarray, predictions: np.ndarray,
                    test_subjects: List[str], output_file: str = "results.csv"):
        """
        Save prediction results to CSV file.

        Args:
            y_test (np.ndarray): True ages
            predictions (np.ndarray): Predicted ages
            test_subjects (List[str]): List of test subject IDs
            output_file (str): Output file path
        """
        results_df = pd.DataFrame({
            'Subject_ID': test_subjects,
            'Actual_Age': y_test,
            'Predicted_Age': predictions,
            'Absolute_Error': np.abs(y_test - predictions)
        })
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

def main():
    """Main function to run the brain age prediction pipeline."""
    # Add argument parser
    parser = argparse.ArgumentParser(description='Brain Age Prediction from EEG data')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/1-sample-cnc',
                        help='Path to data directory containing .tsv label file and .fif files')
    args = parser.parse_args()
    
    # List of different feature combinations to test
    feature_configs = [
        # Test 1: Fractal Feature
        {
            'use_fd': True,
            'use_range': False, 
            'use_amplitude': False,
            'use_connectivity': False,
            'use_psd': False
        },
        # Test 2: Range Feature
        {
            'use_fd': False,
            'use_range': True,
            'use_amplitude': False,
            'use_connectivity': False,
            'use_psd': False
        },
        # Test 3: Amplitude Feature
        {
            'use_fd': False,
            'use_range': False,
            'use_amplitude': True,
            'use_connectivity': False,
            'use_psd': False
        },
        # Test 4: Connectivity Feature
        {
            'use_fd': False,
            'use_range': False,
            'use_amplitude': False,
            'use_connectivity': True,
            'use_psd': False
        },
        # Test 5: PSD Feature
        {
            'use_fd': False,
            'use_range': False,
            'use_amplitude': False,
            'use_connectivity': False,
            'use_psd': True
        }
    ]

    base_args = {
        'data_dir': args.data_dir,
        'n_test_subjects': 40,
        'random_state': 42
    }

    for i, feature_config in enumerate(feature_configs, 1):
        print(f"\nRunning Test {i}...")
        
        # Combine base args with feature config
        args = {**base_args, **feature_config}
        predictor = BrainAgePredictor(**args)

        print("Loading and preprocessing data...")
        cache_dir = '.'
        cache_file = f'./preprocessed_data_test{i}.joblib'
        
        try:
            # Try to load from cache
            data = joblib.load(cache_file)
            X, y, subject_ids = data['X'], data['y'], data['subject_ids']
            print("Loaded preprocessed data from cache")
        except Exception as e:
            # os.makedirs(cache_dir, exist_ok=True)
            # If cache doesn't exist, process data and save to cache
            X, y, subject_ids = predictor.load_and_preprocess_data()
            joblib.dump({'X': X, 'y': y, 'subject_ids': subject_ids}, f'{cache_dir}/preprocessed_data_test{i}.joblib')
            print("Saved preprocessed data to cache")

        print(f"\nTotal samples: {len(X)}")
        print(f"Feature vector size: {X.shape[1]}")

        # Split data and train model
        X_train, X_test, y_train, y_test, train_subjects, test_subjects = predictor.train(X, y, subject_ids)

        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # Evaluate model
        metrics, predictions = predictor.evaluate(X_test, y_test)

        print("\nResults:")
        print(f"Mean Absolute Error: {metrics['mae']:.2f} years")
        print(f"Root Mean Square Error: {metrics['rmse']:.2f} years")
        print(f"R² Score: {metrics['r2']:.3f}")

        # Save results
        predictor.save_results(y_test, predictions, test_subjects, f"results_test{i}.csv")

if __name__ == "__main__":
    main()