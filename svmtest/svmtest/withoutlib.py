import numpy as np
import os
import scipy.io.wavfile as wav
import scipy.fftpack
import joblib

# Function to extract MFCC-like features without librosa
def extract_features(file_path, n_mfcc=13):
    # Read audio file
    sr, audio = wav.read(file_path)
    
    # Normalize audio
    if audio.dtype != np.float32:
        audio = audio / np.max(np.abs(audio), axis=0)

    # Perform a Short-Time Fourier Transform (STFT)
    window_size = 1024
    hop_size = 512
    window = np.hamming(window_size)
    num_frames = 1 + (len(audio) - window_size) // hop_size
    frames = np.lib.stride_tricks.sliding_window_view(audio, window_size)[::hop_size].copy()  # Copy to make writable
    frames *= window
    stft = np.abs(np.fft.rfft(frames, axis=1))

    # Compute mel filterbanks
    mel_filters = np.zeros((n_mfcc, stft.shape[1]))
    mel_freq = np.linspace(0, sr / 2, stft.shape[1])
    mel_points = np.linspace(0, sr / 2, n_mfcc + 2)
    for i in range(1, len(mel_points) - 1):
        mel_filters[i - 1, :] = np.maximum(0, np.minimum(
            (mel_freq - mel_points[i - 1]) / (mel_points[i] - mel_points[i - 1]),
            (mel_points[i + 1] - mel_freq) / (mel_points[i + 1] - mel_points[i])
        ))

    # Apply mel filters
    mel_spectrum = np.dot(stft, mel_filters.T)

    # Compute logarithm and discrete cosine transform (DCT)
    log_mel_spectrum = np.log(np.maximum(mel_spectrum, 1e-10))
    mfcc = scipy.fftpack.dct(log_mel_spectrum, axis=1)[:, :n_mfcc]

    # Return mean of MFCCs across frames
    return np.mean(mfcc, axis=0)

# Function to predict for new test files
def predict_test_data(test_folder, model):
    test_features = []
    test_files = []

    for file in os.listdir(test_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(test_folder, file)
            features = extract_features(file_path)
            test_features.append(features)
            test_files.append(file)

    test_features = np.array(test_features)
    predictions = model.predict(test_features)

    # Print predictions
    for i, file in enumerate(test_files):
        print(f"File: {file}, Predicted Class: {predictions[i]}")

# Load the saved model
model_path = r"D:\svmtest\svm_model.joblib"
svm_model = joblib.load(model_path)
print("Model loaded successfully.")

# Path to test data folder
test_folder = r'D:\svmtest\testdata'

# Predict and display results
predict_test_data(test_folder, svm_model)


